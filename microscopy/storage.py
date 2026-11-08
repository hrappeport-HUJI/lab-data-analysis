"""Portable, versioned project storage. Raw acquisitions are never written."""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Conflict(ValueError):
    """Another reviewer saved a newer revision."""


class Project:
    def __init__(self, root):
        self.root = Path(root).resolve()
        from .config import validate

        self.config = validate(read_json(self.root / "project.json"))
        if self.config.get("schema_version") != 1:
            raise ValueError("Unsupported project schema")
        self.manifest = read_json(self.root / "manifest.json")
        if digest(self.manifest) != self.config["manifest_digest"]:
            raise ValueError("Image manifest changed; import a new project")
        self.raw = Path(self.config["raw_root"])
        self.frames = {(r["field"], r["frame"], r["channel"]): r for r in self.manifest}

    def resolve(self, relative):
        p = (self.root / relative).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError("Path outside project")
        return p

    def run_path(self, run_id):
        if (
            not isinstance(run_id, str)
            or not run_id
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in run_id)
        ):
            raise ValueError("Invalid run ID")
        return self.root / "runs" / run_id

    def load_run(self, run_id):
        run = read_json(self.run_path(run_id) / "run.json")
        if run["manifest_digest"] != self.config["manifest_digest"]:
            raise ValueError("Run belongs to another image manifest")
        if run["config"]["structure_channel"] != self.config["structure_channel"]:
            raise ValueError("Structure channel changed; initialize a new project")
        return run

    def latest_run(self):
        return read_json(self.root / "latest_run.json")["run_id"]

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.root / "reviews.sqlite", timeout=10)
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(
            "CREATE TABLE IF NOT EXISTS revisions (run_id TEXT, revision INTEGER, state TEXT NOT NULL, operation TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(run_id,revision))"
        )
        try:
            with db:
                yield db
        finally:
            db.close()

    def review(self, run_id):
        self.load_run(run_id)
        with self.connect() as db:
            row = db.execute(
                "SELECT revision,state FROM revisions WHERE run_id=? ORDER BY revision DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        return (
            {"revision": row[0], "operations": json.loads(row[1])}
            if row
            else {"revision": 0, "operations": []}
        )

    def save_review(self, run_id, operation, expected_revision, validate):
        # SQLite serializes writers across processes, not merely across threads.
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT revision,state FROM revisions WHERE run_id=? ORDER BY revision DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            revision, operations = (row[0], json.loads(row[1])) if row else (0, [])
            if type(expected_revision) is not int or expected_revision != revision:
                raise Conflict("Reviews changed in another tab. Reload before saving.")
            if operation.get("type") == "undo":
                if not operations:
                    raise ValueError("Nothing to undo")
                operations = operations[:-1]
            elif operation.get("type") == "import":
                imported = operation["review"]
                if (
                    imported.get("run_id") != run_id
                    or imported.get("manifest_digest") != self.config["manifest_digest"]
                ):
                    raise ValueError(
                        "Review import belongs to another run or acquisition"
                    )
                operations = imported["operations"]
            else:
                operations = operations + [operation]
            validate(operations)
            db.execute(
                "INSERT INTO revisions VALUES (?,?,?,?,?)",
                (
                    run_id,
                    revision + 1,
                    json.dumps(operations, allow_nan=False),
                    json.dumps(operation, allow_nan=False),
                    utc(),
                ),
            )
        return {"revision": revision + 1, "operations": operations}

    def export_review(self, run_id):
        return {
            "schema_version": 1,
            "run_id": run_id,
            "manifest_digest": self.config["manifest_digest"],
            **self.review(run_id),
        }

    def new_id(self, prefix):
        return prefix + "-" + uuid.uuid4().hex[:12]
