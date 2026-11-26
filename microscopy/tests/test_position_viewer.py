import json
import subprocess
import sys
from contextlib import contextmanager
from http.client import HTTPConnection
from queue import Queue
from threading import Thread
from urllib.parse import urlsplit

import numpy as np
import pytest
import tifffile

from microscopy.importing import initialize, template
from microscopy.position_viewer import server as viewer
from microscopy.position_viewer.cache import build_position_cache, build_project_viewer_cache
from microscopy.storage import write_json


DATA = "/position-data/"


@contextmanager
def listening(*roots, **options):
    with viewer.FastThreadingHTTPServer(roots, port=0, **options) as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


def request(port, path, headers=None, method="GET"):
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        assert response.version == 11
        body = response.read()
        if method != "HEAD":
            assert len(body) == int(response.getheader("Content-Length"))
        return response.status, response.headers, body
    finally:
        connection.close()


def payload(server, path):
    status, _, body = request(server.server_port, DATA + path)
    assert status == 200, body
    return body


def read(server, path):
    return json.loads(payload(server, path))


def tracking(root, pos, *, channels=None, **metadata):
    folder = root / pos
    write_json(folder / "tracking.json", {"records": [], "times": [0, 60], **metadata})
    write_json(folder / "channels.json", channels if channels is not None else {"Phase": [0, 1]})
    return folder


def test_discovery_and_position_files(tmp_path):
    root = tmp_path / "cache"
    tracking(root, "P10", strain="S10", label="Tenth position")
    tracking(root, "P2", condition="S2", strain="fallback")
    tracking(root, "P1")
    write_json(root / "unfinished" / "tracking.json", {})
    with listening(root, no_auth=True) as server:
        assert read(server, "list") == [
            {"pos": "P1", "strain": "", "label": "Pos P1"},
            {"pos": "P2", "strain": "S2", "label": "Pos P2"},
            {"pos": "P10", "strain": "S10", "label": "Tenth position"},
        ]
        meta = read(server, "meta?pos=P1")
        assert (meta["height"], meta["width"]) == (2048, 2048)
    default = [{"pos": "P10", "strain": "Z", "label": "Selected by default"}]
    explicit = [{"pos": "P2", "strain": "X", "label": "Second"},
                {"pos": "P1", "strain": "Y", "label": "First"}]
    write_json(root / "positions.json", default)
    selection = tmp_path / "explicit.json"
    write_json(selection, explicit)
    with listening(root, no_auth=True) as auto, listening(root, positions_file=selection, no_auth=True) as custom:
        assert read(auto, "list") == default
        assert read(custom, "list") == explicit
        assert request(custom.server_port, DATA + "meta?pos=P10")[0] == 400
        assert request(custom.server_port, DATA + "meta?pos=..%2f..")[0] == 400


def test_multi_root_file_fallback(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    primary = tracking(first, "2", condition="Primary")
    fallback = tracking(second, "2", condition="Secondary")
    tracking(second, "10")
    np.save(primary / "Phase-2048.npy", np.full((2, 10, 13), 5, dtype="uint16"))
    np.save(fallback / "Phase-2048.npy", np.full((2, 10, 13), 7, dtype="uint16"))
    np.save(fallback / "Phase-512.npy", np.full((2, 10, 13), 9, dtype="uint16"))
    write_json(fallback / "Phase-1-hist.json", [[5], [130]])
    write_json(fallback / "u-1.json", [[[1, 1], [2, 2], [3, 1]]])
    (primary / "channels.json").unlink()
    with listening(first, second, no_auth=True) as server:
        assert [p["pos"] for p in read(server, "list")] == ["2", "10"]
        meta = read(server, "meta?pos=2")
        assert (meta["condition"], meta["height"], meta["width"]) == ("Primary", 10, 13)
        assert read(server, "hist?pos=2&f=1&ch=Phase") == [[5], [130]]
        assert read(server, "u?pos=2&f=1") == [[[1, 1], [2, 2], [3, 1]]]
        assert payload(server, "zoom?pos=2&f=1&x=0&y=0&n=2") == b"\x05\x00" * 4
        assert payload(server, "overview?pos=2&f=1&s=512") == b"\x09\x00" * 130
        assert payload(server, "overview?pos=2&f=1&s=1024") == b"\x05\x00" * 130


@pytest.fixture
def legacy_cache(tmp_path):
    root = tmp_path / "legacy"
    folder = tracking(root, "3", channels={"Phase": [0, 1], "PI": [1]},
                      records=[{"x": 5, "y": 4, "founder_id": "f3", "decision": "divided"}])
    for n in (512, 1024, 2048):
        np.save(folder / f"Phase-{n}.npy", np.full((2, n, n), n, dtype="uint16"))
        np.save(folder / f"PI-{n}.npy", np.full((1, n, n), n + 1, dtype="uint16"))
    packed = np.zeros((2, 2048, 256), dtype="uint8")
    packed[1, :8, :2] = 0b10101010  # half foreground in each 2x2 or 4x4 output pixel
    np.save(folder / "colony.npy", packed)
    polygon = [[1, 2], [3, 2], [2, 5]]
    write_json(folder / "a-1.json", {"0": [polygon]})
    write_json(folder / "u-1.json", [polygon])
    write_json(folder / "PI-1-hist.json", [[2049], [2048 ** 2]])
    return root, polygon


def test_legacy_http_endpoints(legacy_cache):
    root, polygon = legacy_cache
    with listening(root, no_auth=True) as server:
        assert read(server, "list") == [{"pos": "3", "strain": "", "label": "Pos 3"}]
        meta = read(server, "meta?pos=3")
        assert meta["channels"] == {"Phase": [0, 1], "PI": [1]}
        assert meta["anchors"] == [[[5, 4]], [[2.0, 3.0]]]
        assert read(server, "a?pos=3&f=0") == {}
        assert read(server, "u?pos=3&f=0") == []
        assert read(server, "a?pos=3&f=1") == {"0": [polygon]}
        assert read(server, "u?pos=3&f=1") == [polygon]
        assert read(server, "hist?pos=3&f=0&ch=PI") is None
        assert read(server, "hist?pos=3&f=1&ch=PI") == [[2049], [2048 ** 2]]
        for path, mime in (("/", "text/html"), ("/position", "text/html"),
                           ("/position.js", "application/javascript"), ("/position.css", "text/css")):
            status, headers, body = request(server.server_port, path)
            assert status == 200 and body and headers["Content-Type"].startswith(mime)
            status, headers, head = request(server.server_port, path, method="HEAD")
            assert status == 200 and not head and int(headers["Content-Length"]) == len(body)
        assert request(server.server_port, DATA + "meta?pos=3&f=3")[0] == 200
        assert request(server.server_port, DATA + "u?pos=3&f=3")[0] == 400
        assert request(server.server_port, DATA + "overview?pos=3&f=1&s=800")[0] == 400
        assert request(server.server_port, DATA + "zoom?pos=3&f=1&x=2047&y=0&n=2")[0] == 400
        assert request(server.server_port, DATA + "unknown")[0] == 404


@pytest.mark.parametrize("size", [512, 1024, 2048])
def test_legacy_overview_planes_and_popcounts(legacy_cache, size):
    root, _ = legacy_cache
    area = size * size
    step = 2048 // size
    with listening(root, no_auth=True) as server:
        raw = payload(server, f"overview?pos=3&f=1&s={size}&ch=PI&mask=1")
        assert len(raw) == 3 * area
        assert np.all(np.frombuffer(raw, "<u2", area) == size + 1)
        expected = np.zeros((size, size), dtype="uint8")
        expected[:8 // step, :16 // step:(2 if step == 1 else 1)] = 1
        np.testing.assert_array_equal(np.frombuffer(raw, "u1", area, 2 * area).reshape(size, size), expected)
        assert payload(server, f"overview?pos=3&f=0&s={size}&ch=PI") == b""


def test_legacy_zoom_channel_order_and_mask(legacy_cache):
    root, _ = legacy_cache
    with listening(root, no_auth=True) as server:
        route = "zoom?pos=3&f=1&x=7&y=2&n=4"
        both = np.frombuffer(payload(server, route), "<u2").reshape(2, 16)
        assert np.all(both[0] == 2048) and np.all(both[1] == 2049)
        assert payload(server, route + "&ch=PI,Phase") == both[::-1].tobytes()
        assert payload(server, route + "&ch=&mask=1") == bytes([0, 1, 0, 1]) * 4


def test_network_auth_and_loopback(tmp_path, monkeypatch):
    tracking(tmp_path, "1")
    monkeypatch.setattr(viewer, "access_key", lambda: "a" * 32)
    monkeypatch.setattr(viewer, "local_addresses", lambda: {"test-pc"})
    with pytest.raises(ValueError, match="requires --allow-client"):
        viewer.FastThreadingHTTPServer([tmp_path], host="0.0.0.0")
    with pytest.raises(ValueError, match="only on loopback"):
        viewer.FastThreadingHTTPServer([tmp_path], host="0.0.0.0", allow_clients=["127.0.0.1"], no_auth=True)
    with listening(tmp_path, host="0.0.0.0", allow_clients=["192.0.2.7/24"]) as server:
        port = server.server_port
        login = "/position?key=" + "a" * 32
        route = DATA + "list"
        assert server.links == [f"http://{host}:{port}{login}" for host in ("127.0.0.1", "test-pc")]
        assert request(port, route)[0] == 403
        assert request(port, "/position?key=wrong")[0] == 403
        assert request(port, login, {"Host": f"other.invalid:{port}"})[0] == 403
        assert request(port, login, {"Origin": "http://other.invalid"})[0] == 403
        status, headers, _ = request(port, login, {"Origin": f"http://127.0.0.1:{port}"})
        assert status == 303 and headers["Location"] == "/position"
        cookie = headers["Set-Cookie"]
        assert {"HttpOnly", "SameSite=Strict", "Path=/"} <= set(cookie.split("; ")[1:])
        assert request(port, route, {"Cookie": cookie.split(";")[0]})[0] == 200
        assert request(port, route, {"Cookie": f"{viewer.COOKIE}=wrong"})[0] == 403
    with listening(tmp_path) as default, listening(tmp_path, host="localhost", no_auth=True) as local:
        assert request(default.server_port, DATA + "list")[0] == 403
        assert "?key=" in default.links[0]
        assert "?" not in local.links[0]
        assert request(local.server_port, DATA + "list")[0] == 200
        assert request(local.server_port, DATA + "list", {"Origin": "http://other.invalid"})[0] == 403


@pytest.mark.parametrize("subcommand", [[], ["serve"]], ids=["direct", "serve"])
def test_cli_prints_actual_ephemeral_listening_url(tmp_path, subcommand):
    tracking(tmp_path, "1")
    args = [sys.executable, "-m", "microscopy.position_viewer", *subcommand,
            "--cache-dir", str(tmp_path), "--port", "0", "--no-auth"]
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        lines = Queue()
        reader = Thread(target=lambda: lines.put(process.stdout.readline()), daemon=True)
        reader.start()
        try:
            url = urlsplit(lines.get(timeout=15).strip().split()[-1])
            assert (url.hostname, url.path) == ("127.0.0.1", "/position")
            assert url.port > 0
            assert request(url.port, url.path)[0] == 200
        finally:
            process.terminate()
            process.wait(timeout=5)
            reader.join(timeout=5)


def test_build_rectangular_cache_with_sparse_channels(tmp_path):
    output = tmp_path / "cache"
    height, width = 9, 1025
    phase = np.arange(height * width, dtype=np.uint16).reshape(height, width)
    masks = np.zeros_like(phase, dtype=bool)
    masks[:, :width - 1] = True
    polygon = [[1, 2], [3, 2], [2, 5]]
    build_position_cache(
        output, lambda idx: phase + idx, pos="sample-2", strain="Control",
        label="Control field", times=[0, None],
        channels={"far red": {1: phase + 20}},
        colony_masks=lambda idx: masks, assigned={0: {"0": [polygon]}},
        unassigned={1: [polygon]}, records=[{"x": 2, "y": 3}],
    )
    assert np.load(output / "sample-2" / "colony.npy").shape == (2, height, 129)
    with listening(output, no_auth=True) as server:
        assert read(server, "list") == [{"pos": "sample-2", "strain": "Control", "label": "Control field"}]
        meta = read(server, "meta?pos=sample-2")
        assert (meta["height"], meta["width"], meta["times"]) == (height, width, [0, None])
        assert meta["channels"] == {"Phase": [0, 1], "far red": [1]}
        assert read(server, "hist?pos=sample-2&f=0&ch=far+red") is None
        values, counts = read(server, "hist?pos=sample-2&f=1&ch=far+red")
        assert (values[0], values[-1], sum(counts)) == (20, 20 + phase.size - 1, phase.size)
        assert read(server, "u?pos=sample-2&f=1") == [polygon]
        route = "overview?pos=sample-2&f=1&ch=far+red&mask=1"
        raw = payload(server, route + "&s=2048")
        assert raw == (phase + 20).astype("<u2").tobytes() + masks.astype("u1").tobytes()
        for size, rows in ((512, 4), (1024, 9)):
            raw = payload(server, route + f"&s={size}")
            assert len(raw) == rows * size * 3
            assert np.frombuffer(raw[2 * rows * size:], "u1").reshape(rows, size)[0, 0] == 1
        zoom = payload(server, "zoom?pos=sample-2&f=0&x=1022&y=2&n=3&ch=Phase,far+red&mask=1")
        assert zoom == phase[2:5, 1022:1025].tobytes() + bytes([1, 1, 0]) * 3
    with pytest.raises(FileExistsError):
        build_position_cache(output, [phase])


def test_failed_build_does_not_publish_partial_cache(tmp_path):
    out = tmp_path / "cache"
    with pytest.raises(ValueError, match="Colony mask and frame dimensions differ"):
        build_position_cache(out, [np.zeros((3, 4), dtype="uint16")], colony_masks=[np.zeros((1, 1))])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode", ["standard", "founder_v2"])
def test_project_builder_uses_run_graph_and_label_planes(tmp_path, mode):
    config = template()
    config.update(structure_channel="phase", conditions={"7": "Control"}, frame_interval_s=60, expected_frame_step=2,
                  fluorescence_channels=["FarRed"])
    config["tracking"]["confirmation_frames"] = 0
    if mode == "founder_v2":
        config["tracking"]["events"] = mode
    write_json(tmp_path / "config.json", config)
    raw = tmp_path / "raw"
    shape = (12, 19)
    for frame, channel in [(7, "phase"), (8, "FarRed"), (9, "phase"), (11, "phase")]:
        path = raw / "Pos7" / channel / f"img_{frame:03}.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(path, np.full(shape, frame, dtype="uint16"))
    project = initialize(raw, tmp_path / "config.json", tmp_path / "project")
    run_id = "synthetic"
    run_dir = project.run_path(run_id)
    (run_dir / "labels").mkdir(parents=True)
    planes = {}
    for frame in [7, 9, 11]:
        labels = np.zeros(shape, dtype=np.uint32)
        if frame != 11:
            labels[2:4, 2:6] = 20
            labels[7:9, 10:12] = 50_000  # orphan, absent from the run's graph
        if frame == 9:
            labels[2:4, 6:8] = 30
        name = f"labels/0000_{frame:09}.npz"
        np.savez(run_dir / name, labels=labels)
        planes[f"7|{frame}"] = name
    nodes = [
        {"id": oid, "field": "7", "frame": frame, "label": label, "x": x, "y": 3,
         "labels_file": planes[f"7|{frame}"]}
        for oid, frame, label, x in [("root", 7, 20, 4), ("daughter-a", 9, 20, 4), ("daughter-b", 9, 30, 7)]
    ]
    run = {"run_id": run_id, "manifest_digest": project.config["manifest_digest"],
           "fields": ["7"], "config": project.config, "quality": [], "observations": nodes,
           "edges": [["root", "daughter-a"], ["root", "daughter-b"]],
           "founders": [{"id": "root", "field": "7", "observation_id": "root",
                        "baseline_frame": 7, "delayed_start": False}]}
    if mode == "founder_v2":
        run.update(edges=[], label_planes=planes, founder_events={
            "root": {"decision": "pi_positive_death", "lower_frame": 7, "upper_frame": 9,
                     "event_frame": 9, "pi_surge_frame": 9, "last_identity_frame": 9,
                     "tracking_end_reason": "pi_positive_death",
                     "observation_ids": ["root", "daughter-a", "daughter-b"]},
        })
    write_json(run_dir / "run.json", run)
    write_json(project.root / "latest_run.json", {"run_id": run_id})
    before = (run_dir / "run.json").read_bytes()
    output = tmp_path / "cache"
    if mode == "founder_v2":
        subprocess.run([sys.executable, "-m", "microscopy.position_viewer", "build",
                        str(project.root), str(output), "--run-id", run_id, "--field", "7"],
                       check=True, timeout=15)
    else:
        assert build_project_viewer_cache(project.root, output, fields=["7"]) == output
    assert (run_dir / "run.json").read_bytes() == before
    assert not (project.root / "reviews.sqlite").exists()
    with listening(output, no_auth=True) as server:
        assert read(server, "list") == [{"pos": "7", "strain": "Control", "label": "Pos 7"}]
        meta = read(server, "meta?pos=7")
        assert meta["frames"] == [7, 8, 9, 11]
        assert meta["channels"] == {"phase": [0, 2, 3], "FarRed": [1]}
        record, = meta["records"]
        assert record["reviewed"] is False
        key = "division_frame" if mode == "standard" else "pi_surge_frame"
        assert record[key] == 2
        assert record["decision"] == ("divided" if mode == "standard" else "pi_positive_death")
        assert len(read(server, "a?pos=7&f=2")["0"]) == 2
        assert len(read(server, "u?pos=7&f=2")) == 1
        assert read(server, "a?pos=7&f=1") == {}
        assert payload(server, "zoom?pos=7&f=1&x=0&y=0&n=2&ch=FarRed") == b"\x08\x00" * 4
        mask = payload(server, "overview?pos=7&f=2&ch=&mask=1")
        assert sum(mask) == 16
