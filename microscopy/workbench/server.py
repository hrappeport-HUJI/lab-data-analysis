"""Local server for the restored founder-review viewer.

Same endpoints and request flow as the previous viewer. Images are rendered as small
windows on the server. This preview is read-only: endpoints that would change reviews
answer with an explanation instead.
"""

from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import csv
import io
import json
import mimetypes
import secrets

from .adapter import RunReviewModel
from ..storage import Conflict

WEB = Path(__file__).parent / "web"
STATIC = {"/": "index.html", "/viewer.js": "viewer.js", "/style.css": "style.css", "/manual_ui.js": "manual_ui.js"}
WINDOWS = {"auto", "128", "192", "256", "384", "512", "768", "1024", "2048"}


def load_tagging_sets(path, fields):
    """Optional JSON {"sets": {"name": ["2", "11", ...]}} naming groups of positions to tag."""
    if not path:
        return []
    data = json.loads(Path(path).read_text())
    sets = []
    for name, positions in data["sets"].items():
        positions = [str(p) for p in positions]
        unknown = [p for p in positions if p not in fields]
        if unknown:
            raise ValueError(f"Tagging set {name!r} names positions not in this project: {unknown}")
        sets.append({"name": name, "positions": positions})
    return sets


class WorkbenchServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, project, port=0, run_id=None, read_only=False, tagging_sets=None):
        self.model = RunReviewModel(project, run_id, read_only=read_only)
        self.model.state()  # fail early on an unreadable run or review store
        self.tagging_sets = load_tagging_sets(tagging_sets, self.model.fields)
        self.token = secrets.token_urlsafe(32)
        super().__init__(("127.0.0.1", port), Handler)

    @property
    def origin(self):
        return f"http://127.0.0.1:{self.server_address[1]}"

    def server_close(self):
        super().server_close()
        self.model.prefetcher.shutdown(wait=False, cancel_futures=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    @property
    def m(self):
        return self.server.model

    def guards(self, api=False):
        port = self.server.server_address[1]
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in allowed:
            raise PermissionError("Only the loopback address is accepted")
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{v}" for v in allowed}:
            raise PermissionError("Cross-origin requests are not permitted")
        if api and urlparse(self.path).path != "/api/session" and not secrets.compare_digest(
            self.headers.get("X-Review-Token", ""), self.server.token
        ):
            raise PermissionError("Missing review-session token")

    def send(self, status, data, ctype="application/json", headers=None):
        body = json.dumps(data, allow_nan=False).encode() if ctype == "application/json" and not isinstance(data, bytes) else data
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, str(v))
        self.end_headers()
        self.wfile.write(body)

    def error(self, e):
        if isinstance(e, PermissionError):
            code = 403
        elif isinstance(e, (ValueError, KeyError)):
            code = 400
        else:
            code = 500
        self.send(code, {"error": str(e)})

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            q = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            self.guards(path.startswith("/api/"))
            if path == "/api/session":
                return self.send(200, {"token": self.server.token, "project_id": self.m.config()["project_id"],
                                       "project_path": str(self.m.root), "viewer_version": "4-run-adapter"})
            if path == "/api/project":
                st = self.m.state()
                return self.send(200, {
                    "config": self.m.config(),
                    "revision": st["revision"],
                    "founders": st["records"],
                    "excluded_frames": self.m.excluded_frames(),
                    "warnings": st["warnings"],
                    "latest_results": self.m.latest_results(),
                    "latest_pdf": None,
                    "tagging_sets": self.server.tagging_sets,
                })
            if path == "/api/frame":
                return self.frame(q)
            if path == "/api/export/json":
                body = json.dumps(self.m.export_review(), indent=2).encode()
                return self.send(200, body, "application/octet-stream",
                                 {"Content-Disposition": 'attachment; filename="reviews.json"'})
            if path == "/api/export/csv":
                keys = ["id", "position", "founder_id", "condition", "decision", "last_single_frame", "division_frame",
                        "last_valid_frame", "event_frame", "reviewed", "reviewer", "confidence", "note"]
                f = io.StringIO()
                w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                w.writeheader()
                w.writerows(self.m.all_effective())
                return self.send(200, f.getvalue().encode(), "text/csv; charset=utf-8",
                                 {"Content-Disposition": 'attachment; filename="reviewed_decisions.csv"'})
            if path.startswith("/artifacts/"):
                target = (self.m.root / path[len("/artifacts/"):]).resolve()
                if not target.is_relative_to(self.m.root / "results") or not target.is_file():
                    raise PermissionError("Artifact is outside the results folder")
                return self.send(200, target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            if path not in STATIC:
                return self.send(404, {"error": "Not found"})
            target = WEB / STATIC[path]
            ctype = mimetypes.guess_type(target.name)[0] or "text/plain"
            return self.send(200, target.read_bytes(), ctype + "; charset=utf-8")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # report every failure to the viewer instead of dropping the request
            try:
                self.error(e)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def frame(self, q):
        record = dict(self.m.effective(q["id"]))
        frame = int(q["frame"])
        mode = q.get("mode", "overlay")
        if mode not in {"phase", "overlay", "fluor"}:
            raise ValueError("Invalid display mode")
        channel = q.get("channel") or None
        width = q.get("width", "auto")
        if width not in WINDOWS:
            raise ValueError("Invalid window width")
        width = None if width == "auto" else float(width)
        gain = float(q.get("gain", "1"))
        alpha = float(q.get("alpha", "1"))
        if not 0.1 <= gain <= 16 or not 0 <= alpha <= 1:
            raise ValueError("Invalid display enhancement")
        if "anchor_x" in q:  # unsaved recentering from a draft, display only
            ax, ay = float(q["anchor_x"]), float(q["anchor_y"])
            if not (0 <= ax <= self.m.config()["image_width"] and 0 <= ay <= self.m.config()["image_height"]):
                raise ValueError("Invalid window center")
            record.update(review_anchor_x=ax, review_anchor_y=ay)
        png, meta = self.m.render(
            record, frame, mode=mode, channel=channel, width=width, gain=gain, alpha=alpha,
            outlines=q.get("outlines", "1") == "1", full=q.get("full", "0") == "1",
            auto_contrast=q.get("contrast") == "auto",
        )
        self.m.prefetch(record, frame, channel)
        self.send(200, png, "image/png", {"X-Frame-Meta": json.dumps(meta)})

    def do_POST(self):
        try:
            self.guards(True)
            if self.m.read_only:
                raise PermissionError("This viewer was started read-only; nothing was changed.")
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("JSON body required")
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 6_000_000:
                raise ValueError("Review payload too large or empty")
            x = json.loads(self.rfile.read(size))
            path = urlparse(self.path).path
            if path == "/api/review":
                revision, record = self.m.save_override(x["id"], x["changes"], x["expected_revision"])
                return self.send(200, {"revision": revision, "record": record})
            if path == "/api/revert":
                revision, record = self.m.revert(x["id"], x["expected_revision"])
                return self.send(200, {"revision": revision, "record": record})
            if path == "/api/import":
                revision, founders = self.m.import_reviews(x["reviews"], x["expected_revision"])
                return self.send(200, {"revision": revision, "founders": founders})
            if path in ("/api/split", "/api/undo-split", "/api/jobs"):
                raise ValueError("Not available in this version yet.")
            self.send(404, {"error": "Not found"})
        except Conflict as e:
            self.send(409, {"error": str(e) + " (another tab or viewer saved first)"})
        except Exception as e:
            self.error(e)


def serve(project, port=0, run_id=None, read_only=False, open_browser=False, tagging_sets=None):
    server = WorkbenchServer(project, port, run_id, read_only=read_only, tagging_sets=tagging_sets)
    print(f"Founder review ({'read-only preview' if read_only else 'editable'}): {server.origin}/", flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(server.origin + "/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
