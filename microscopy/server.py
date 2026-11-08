"""Loopback-only review application. No raw files are exposed as a directory."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor
import io
import json
import secrets
import threading
import numpy as np
from PIL import Image
from scipy import ndimage
from .storage import Project, Conflict, read_json
from .images import read_frame
from .review import materialize, observation_mask, save
from .events import summarize
from .analysis import analyze


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, project, port=0, run_id=None):
        super().__init__(("127.0.0.1", port), Handler)
        self.project = Project(project)
        self.run_id = run_id or self.project.latest_run()
        self.project.load_run(self.run_id)
        self.token = secrets.token_urlsafe(32)
        self.ranges = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.job = None

    @property
    def origin(self):
        return f"http://127.0.0.1:{self.server_port}"

    def server_close(self):
        super().server_close()
        if hasattr(self, "pool"):
            self.pool.shutdown(wait=True)

    def snapshot(self):
        p = Project(self.project.root)
        review = p.review(self.run_id)
        data = materialize(p, self.run_id, review["operations"])
        return p, review, data


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, value, status=200, mime="application/json"):
        body = (
            json.dumps(value, allow_nan=False).encode()
            if mime == "application/json"
            else value
        )
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def authorized(self, write=False):
        if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
            self.send({"error": "Open the printed 127.0.0.1 URL"}, 403)
            return False
        if write and (
            self.headers.get("Origin") != self.server.origin
            or self.headers.get("X-Review-Token") != self.server.token
            or self.headers.get_content_type() != "application/json"
        ):
            self.send({"error": "Invalid local review session"}, 403)
            return False
        return True

    def do_GET(self):
        if not self.authorized():
            return
        try:
            self.get()
        except (ValueError, KeyError, FileNotFoundError, TypeError) as e:
            self.send({"error": str(e)}, 400)

    def get(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        path = url.path
        if path in {"/", "/app.js", "/style.css"}:
            name = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}[
                path
            ]
            self.send(
                (Path(__file__).parent / "web" / name).read_bytes(),
                mime={
                    "/": "text/html",
                    "/app.js": "text/javascript",
                    "/style.css": "text/css",
                }[path],
            )
            return
        p = self.server.project
        run_id = self.server.run_id
        if path == "/api/state":
            p, review, data = self.server.snapshot()
            founders = summarize(p, data)
            self.send(
                {
                    "token": self.server.token,
                    "run_id": run_id,
                    "revision": review["revision"],
                    "founders": founders,
                    "frames": [
                        {
                            k: r[k]
                            for k in ["field", "frame", "channel", "time_s", "shape"]
                        }
                        for r in p.manifest
                    ],
                    "structure_channel": p.config["structure_channel"],
                    "channels": p.config.get("fluorescence_channels", []),
                    "warnings": list(dict.fromkeys(data["needs_review"])),
                }
            )
            return
        if path == "/api/reviews":
            self.send(p.export_review(run_id))
            return
        if path == "/api/frame":
            p, review, data = self.server.snapshot()
            field = q["field"]
            frame = int(q["frame"])
            nodes = []
            for n in data["observations"]:
                if n["field"] == field and n["frame"] == frame:
                    mask = observation_mask(p, run_id, n)
                    border = mask & ~ndimage.binary_erosion(mask)
                    yy, xx = np.where(border)
                    nodes.append(
                        {
                            **{k: n[k] for k in ["id", "x", "y", "area_px"]},
                            "outline": np.column_stack([xx, yy]).tolist(),
                        }
                    )
            self.send({"nodes": nodes, "revision": review["revision"]})
            return
        if path == "/api/image":
            field = q["field"]
            frame = int(q["frame"])
            channel = q["channel"]
            image = read_frame(p, field, frame, channel).astype(float)
            key = (field, channel)
            if key not in self.server.ranges:
                first = min(
                    r["frame"]
                    for r in p.manifest
                    if r["field"] == field and r["channel"] == channel
                )
                reference = read_frame(p, field, first, channel)
                self.server.ranges[key] = [
                    float(np.percentile(reference, 0.1)),
                    float(np.percentile(reference, 99.9)),
                ]
            lo, hi = self.server.ranges[key]
            lo = float(q.get("min", lo))
            hi = float(q.get("max", hi))
            if not np.isfinite([lo, hi]).all() or hi <= lo:
                hi = lo + 1
            v = np.clip((image - lo) / (hi - lo), 0, 1)
            if q.get("scale") == "log":
                v = np.log1p(99 * v) / np.log(100)
            v = np.round(v * 255).astype("uint8")
            rgb = np.zeros((*v.shape, 3), np.uint8)
            if q.get("color") == "red":
                rgb[:, :, 0] = v
            else:
                rgb[:] = v[:, :, None]
            stream = io.BytesIO()
            Image.fromarray(rgb).save(stream, format="PNG")
            self.send(stream.getvalue(), mime="image/png")
            return
        if path == "/api/results":
            latest = p.root / "latest_results.json"
            result = (
                read_json(
                    p.resolve(
                        "results/" + read_json(latest)["result_id"] + "/summary.json"
                    )
                )
                if latest.exists()
                else None
            )
            job = self.server.job
            self.send(
                {
                    "result": result,
                    "running": bool(job and not job.done()),
                    "error": (
                        str(job.exception())
                        if job and job.done() and job.exception()
                        else None
                    ),
                }
            )
            return
        if path.startswith("/results/"):
            file = p.resolve(path.lstrip("/"))
            if not file.is_relative_to(p.root / "results") or file.suffix not in {
                ".csv",
                ".json",
                ".png",
                ".svg",
            }:
                raise ValueError("Unsupported result path")
            mime = {
                ".csv": "text/csv",
                ".json": "application/json",
                ".png": "image/png",
                ".svg": "image/svg+xml",
            }[file.suffix]
            self.send(file.read_bytes(), mime=mime)
            return
        self.send({"error": "Not found"}, 404)

    def do_POST(self):
        if not self.authorized(write=True):
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 2_000_000:
                raise ValueError("Invalid request size")
            body = json.loads(self.rfile.read(size))
            p = Project(self.server.project.root)
            if self.path == "/api/review":
                result = save(
                    p, self.server.run_id, body["operation"], body["revision"]
                )
                self.send({"revision": result["revision"]})
            elif self.path == "/api/analyze":
                with self.server.lock:
                    if self.server.job and not self.server.job.done():
                        raise Conflict("Analysis is already running")
                    self.server.job = self.server.pool.submit(
                        analyze,
                        p,
                        self.server.run_id,
                        bool(body.get("reviewed_only", False)),
                    )
                self.send({"started": True}, 202)
            else:
                self.send({"error": "Not found"}, 404)
        except Conflict as e:
            self.send({"error": str(e)}, 409)
        except (ValueError, KeyError, TypeError) as e:
            self.send({"error": str(e)}, 400)


def serve(project, port=8766, run_id=None):
    server = ReviewServer(project, port, run_id)
    print(f"Review locally: {server.origin}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
