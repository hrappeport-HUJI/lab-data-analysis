"""Serve Position Viewer caches over HTTP/1.1."""

import ipaddress
import json
import secrets
import socket
from functools import lru_cache
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import numpy as np

from ..workbench.server import access_key, local_addresses
from ._format import (
    LEVELS, NATIVE, PIXEL_DTYPE, box_resize, channel_name, component,
    natural_key, overview_pixels, overview_shape,
)

WEB = Path(__file__).parent / "web"
LOOPBACK = {"127.0.0.1", "localhost"}
COOKIE = "rawview_v2"
STATIC = {
    "/": ("position.html", "text/html"),
    "/position": ("position.html", "text/html"),
    "/position.js": ("position.js", "application/javascript"),
    "/position.css": ("position.css", "text/css"),
}
DATA_PREFIX = "/position-data/"
DATA_ROUTES = {"list", "meta", "a", "u", "hist", "overview", "zoom"}

_BITS = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1)
_POP_COUNTS = {
    k: _BITS.reshape(256, 8 // k, k).sum(axis=2, dtype=np.uint8)
    for k in (2, 4)
}


def _overview_mask(bits, height, width, shape):
    if (height, width) == (2048, 2048) and shape in {(512, 512), (1024, 1024)}:
        n = shape[0]
        k = 2048 // n
        count = _POP_COUNTS[k][bits].reshape(n, k, n).sum(axis=1, dtype=np.uint8)
        return (count >= k * k // 2).astype(np.uint8)
    mask = np.unpackbits(bits, axis=-1, count=width)
    if mask.shape == shape:
        return mask
    return (box_resize(mask, shape) >= 0.5).astype(np.uint8)


class FastThreadingHTTPServer(ThreadingHTTPServer):
    """Serve an ordered set of read-only cache roots."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, cache_dirs, *, positions_file=None, host="127.0.0.1",
                 port=0, allow_clients=(), no_auth=False):
        network = host not in LOOPBACK
        if network and no_auth:
            raise ValueError("--no-auth is available only on loopback")
        if network and not allow_clients:
            raise ValueError("Network mode requires --allow-client")
        self.networks = tuple(ipaddress.ip_network(net, strict=False) for net in allow_clients)
        self.cache_roots = tuple(Path(root).resolve() for root in cache_dirs)
        if not any(root.is_dir() for root in self.cache_roots):
            raise ValueError("Provide at least one existing cache directory")
        # Bind bounded caches to this server's lifetime, including its open memory maps.
        self.pixel_path = lru_cache(maxsize=2048)(self._pixel_path)
        self.stack = lru_cache(maxsize=128)(lambda path: np.load(path, mmap_mode="r"))
        self.position_data = lru_cache(maxsize=16)(self._position_data)
        self.overlay = lru_cache(maxsize=1024)(self._overlay)
        self.positions = self._load_positions(positions_file)
        self.position_ids = {p["pos"] for p in self.positions}
        self.key = None if no_auth else access_key()
        self.token = secrets.token_urlsafe(32)
        self.host = host
        names = set(LOOPBACK)
        if network:
            names.update(local_addresses() if host == "0.0.0.0" else {host.lower()})
        super().__init__((host, port), Handler)
        self.hosts = {f"{name}:{self.server_port}" for name in names}

    @property
    def links(self):
        local = "127.0.0.1" if self.host == "0.0.0.0" or self.host in LOOPBACK else self.host
        primary = f"{local}:{self.server_port}"
        hosts = [primary]
        if self.host == "0.0.0.0":
            hosts.extend(sorted(self.hosts - {primary, f"localhost:{self.server_port}"}))
        query = "?" + urlencode({"key": self.key}) if self.key is not None else ""
        return [f"http://{host}/position{query}" for host in hosts]

    def server_close(self):
        super().server_close()
        for cached in (self.pixel_path, self.stack, self.position_data, self.overlay):
            cached.cache_clear()

    def _pixel_path(self, pos, filename):
        for root in self.cache_roots:
            folder = root / pos
            path = folder / filename
            if path.is_file():
                resolved = folder.resolve()
                if resolved.is_relative_to(root) and path.resolve().is_relative_to(resolved):
                    return path
        return None

    def _load_positions(self, positions_file):
        default = self.cache_roots[0] / "positions.json"
        if positions_file is not None or default.is_file():
            path = Path(positions_file) if positions_file is not None else default
            entries = json.loads(path.read_bytes())
            if not isinstance(entries, list):
                raise ValueError("positions.json must contain a list of positions")
            candidates = [(component(str(p["pos"])), p) for p in entries]
        else:
            ids = {p.name for root in self.cache_roots if root.is_dir()
                   for p in root.iterdir() if p.is_dir()}
            candidates = [(pos, {}) for pos in sorted(ids, key=lambda pos: (natural_key(pos), pos))]
        ready = []
        seen = set()
        for pos, entry in candidates:
            tracking = self.pixel_path(pos, "tracking.json")
            if pos in seen or tracking is None or self.pixel_path(pos, "channels.json") is None:
                continue
            seen.add(pos)
            info = json.loads(tracking.read_bytes()) if not entry else {}
            ready.append({
                "pos": pos,
                "strain": entry.get("strain", info.get("condition") or info.get("strain") or ""),
                "label": entry.get("label", info.get("label") or f"Pos {pos}"),
            })
        return ready

    def _overlay(self, pos, kind, frame):
        path = self.pixel_path(pos, f"{kind}-{frame}.json")
        return path.read_bytes() if path else (b"{}" if kind == "a" else b"[]")

    def _position_data(self, pos):
        data = json.loads(self.pixel_path(pos, "tracking.json").read_bytes())
        channels = json.loads(self.pixel_path(pos, "channels.json").read_bytes())
        acquired = {
            channel_name(ch): {frame: i for i, frame in enumerate(frames)}
            for ch, frames in channels.items()
        }
        height, width = data.get("height"), data.get("width")
        if height is None or width is None:
            inferred = (NATIVE, NATIVE)
            for ch in channels:
                path = self.pixel_path(pos, f"{ch}-{NATIVE}.npy")
                if path is not None:
                    inferred = self.stack(path).shape[-2:]
                    break
            height = inferred[0] if height is None else height
            width = inferred[1] if width is None else width
        height, width = int(height), int(width)
        if min(height, width) < 1:
            raise ValueError("Invalid frame dimensions")
        previous = [[record["x"], record["y"]] for record in data["records"]]
        anchors = []
        for frame in range(len(data["times"])):
            for i, polys in json.loads(self.overlay(pos, "a", frame)).items():
                if polys and polys[0]:
                    previous[int(i)] = np.mean(polys[0], axis=0).round(1).tolist()
            anchors.append(previous.copy())
        return dict(data, anchors=anchors, channels=channels, height=height, width=width), acquired

    def _array(self, pos, filename):
        path = self.pixel_path(pos, filename)
        if path is None:
            raise FileNotFoundError(f"Missing cache file {filename} for position {pos}")
        return self.stack(path)

    def pixels(self, pos, ch, idx, overview, level, shape, x, y, n, native_shape):
        if overview and level != NATIVE:
            path = self.pixel_path(pos, f"{ch}-{level}.npy")
            if path is not None:
                plane = self.stack(path)[idx]
                if plane.shape == shape:
                    return plane
        native = self._array(pos, f"{ch}-{NATIVE}.npy")[idx]
        if native.shape != native_shape:
            raise ValueError(f"Native stack for {ch!r} does not match the frame dimensions")
        if overview:
            return overview_pixels(native, shape)
        return native[y:y + n, x:x + n]

    def mask(self, pos, frame, overview, shape, height, width, x, y, n):
        bits = self._array(pos, "colony.npy")[frame]
        if bits.shape != (height, (width + 7) // 8):
            raise ValueError("Packed colony mask does not match the frame dimensions")
        if overview:
            return _overview_mask(bits, height, width, shape)
        crop = bits[y:y + n, x // 8:(x + n + 7) // 8]
        return np.unpackbits(crop, axis=-1)[:, x % 8:x % 8 + n]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    def log_message(self, *args):
        pass

    def reply(self, status, mime="text/plain", data=b"", *, headers=None, cache_control="no-store"):
        self.reply_chunks(status, mime, (data,), headers=headers, cache_control=cache_control)

    def reply_chunks(self, status, mime, chunks, *, headers=None, cache_control="no-store"):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(sum(map(len, chunks))))
        self.send_header("Cache-Control", cache_control)
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            for chunk in chunks:
                if chunk:
                    self.wfile.write(chunk)

    def json_reply(self, value):
        data = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
        self.reply(200, "application/json", data)

    def do_GET(self):
        try:
            self.dispatch()
        except (ValueError, KeyError, IndexError) as exc:
            self.reply(400, data=f"Invalid request or cache: {exc}".encode())
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True
        except OSError:
            self.reply(500, data=b"Could not read position cache")

    do_HEAD = do_GET

    def _authorized(self):
        try:
            token = SimpleCookie(self.headers.get("Cookie", "")).get(COOKIE)
        except CookieError:
            return False
        return token is not None and secrets.compare_digest(
            token.value.encode(), self.server.token.encode()
        )

    def dispatch(self):
        s = self.server
        ip = ipaddress.ip_address(self.client_address[0])
        hosts = self.headers.get_all("Host", [])
        origins = self.headers.get_all("Origin", [])
        if (
            not (ip.is_loopback or any(ip in net for net in s.networks))
            or len(hosts) != 1 or hosts[0].lower() not in s.hosts
            or (origins and [origin.lower() for origin in origins] != ["http://" + hosts[0].lower()])
        ):
            return self.reply(403)
        url = urlsplit(self.path)
        query = {k: v[-1] for k, v in parse_qs(url.query, keep_blank_values=True).items()}
        if s.key is not None:
            if "key" in query and secrets.compare_digest(query["key"].encode(), s.key.encode()):
                return self.reply(303, headers={
                    "Location": "/position",
                    "Set-Cookie": f"{COOKIE}={s.token}; HttpOnly; SameSite=Strict; Path=/",
                })
            if not self._authorized():
                return self.reply(403, data=b"Open the viewer using the access link printed at startup.")
        if url.path in STATIC:
            name, mime = STATIC[url.path]
            return self.reply(200, mime + "; charset=utf-8", (WEB / name).read_bytes())
        if url.path.startswith(DATA_PREFIX):
            route = url.path[len(DATA_PREFIX):]
            if route in DATA_ROUTES:
                return self.position(route, query)
        self.reply(404)

    def position(self, route, query):
        s = self.server
        if route == "list":
            return self.json_reply(s.positions)
        pos = query["pos"]
        if pos not in s.position_ids:
            raise ValueError("Unknown position")
        meta, acquired = s.position_data(pos)
        if route == "meta":
            return self.json_reply(meta)
        frame = int(query["f"])
        if not 0 <= frame < len(meta["times"]):
            raise ValueError("Frame out of range")
        if route in {"a", "u"}:
            return self.reply(200, "application/json", s.overlay(pos, route, frame),
                              cache_control="private, max-age=300")
        if route == "hist":
            ch = query["ch"]
            if frame not in acquired.get(ch, ()):
                return self.json_reply(None)
            path = s.pixel_path(pos, f"{ch}-{frame}-hist.json")
            if path is None:
                raise FileNotFoundError("Missing histogram")
            return self.reply(200, "application/json", path.read_bytes(),
                              cache_control="private, max-age=86400")
        self.image(route, query, pos, frame, meta, acquired)

    def image(self, route, query, pos, frame, meta, acquired):
        s = self.server
        height, width = meta["height"], meta["width"]
        overview = route == "overview"
        if overview:
            level = int(query.get("s", 1024))
            if level not in LEVELS:
                raise ValueError("Invalid overview size")
            shape = overview_shape(height, width, level)
            x = y = n = 0
        else:
            x, y, n = (int(query[k]) for k in ("x", "y", "n"))
            if min(x, y) < 0 or not 1 <= n <= 1024 or x + n > width or y + n > height:
                raise ValueError("Invalid zoom window")
            level, shape = NATIVE, (n, n)
        requested = query["ch"].split(",") if "ch" in query else acquired
        chunks = []
        for ch in requested:
            idx = acquired.get(ch, {}).get(frame)
            if idx is not None:
                plane = s.pixels(pos, ch, idx, overview, level, shape, x, y, n, (height, width))
                chunks.append(memoryview(np.ascontiguousarray(plane, dtype=PIXEL_DTYPE)).cast("B"))
        if query.get("mask") == "1":
            mask = s.mask(pos, frame, overview, shape, height, width, x, y, n)
            chunks.append(memoryview(np.ascontiguousarray(mask)).cast("B"))
        # Legacy clients omit ch and use channel order from the metadata.
        self.reply_chunks(200, "application/octet-stream", chunks, cache_control="private, max-age=86400")
