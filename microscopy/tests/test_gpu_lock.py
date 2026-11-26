"""Tests for the OS GPU lock and the CUDA session boundaries in FastOmnipose."""

import getpass
import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from microscopy import fast_omnipose
from microscopy.gpu_lock import default_gpu_lock_path, gpu_lock


@pytest.fixture(autouse=True)
def clean_gpu_lock_environment(monkeypatch):
    monkeypatch.delenv("LAB_MICROSCOPY_DISABLE_GPU_LOCK", raising=False)
    monkeypatch.delenv("LAB_MICROSCOPY_GPU_LOCK", raising=False)


@pytest.fixture
def lock_path(tmp_path):
    return tmp_path / "gpu.lock"


@pytest.fixture
def holder_path(lock_path):
    return lock_path.with_suffix(".lock.holder.json")


def test_acquire_and_release(lock_path, holder_path):
    with gpu_lock(lock_path, job="position 42"):
        assert lock_path.exists()
        if os.name == "nt":
            assert lock_path.stat().st_size >= 1
        holder = json.loads(holder_path.read_text(encoding="utf-8"))
        assert holder["pid"] == os.getpid()
        assert holder["user"] == getpass.getuser()
        assert holder["host"] == socket.gethostname()
        assert datetime.fromisoformat(holder["started_at"]).utcoffset() == timedelta(0)
        assert holder["job"] == "position 42"
    assert not holder_path.exists()
    assert lock_path.exists()


def test_reentrant_acquisition(lock_path, holder_path):
    with gpu_lock(lock_path, job="outer"):
        outer_holder = holder_path.read_bytes()
        with gpu_lock(str(lock_path), timeout_s=0, job="inner"):
            assert holder_path.read_bytes() == outer_holder
        assert holder_path.read_bytes() == outer_holder
    assert not holder_path.exists()


def test_releases_on_exception(lock_path, holder_path):
    with pytest.raises(RuntimeError, match="inference failed"):
        with gpu_lock(lock_path):
            raise RuntimeError("inference failed")
    assert not holder_path.exists()
    with gpu_lock(lock_path, timeout_s=0):
        assert holder_path.exists()


@pytest.mark.parametrize("value", ["1", " TRUE ", "yes"])
def test_disabled_lock(monkeypatch, lock_path, holder_path, value):
    monkeypatch.setenv("LAB_MICROSCOPY_DISABLE_GPU_LOCK", value)
    with gpu_lock(lock_path, timeout_s=0):
        assert not lock_path.exists()
        assert not holder_path.exists()


def test_environment_and_explicit_paths(monkeypatch, lock_path, holder_path):
    monkeypatch.setenv("LAB_MICROSCOPY_GPU_LOCK", str(lock_path))
    assert default_gpu_lock_path() == lock_path
    with gpu_lock(job="configured"):
        assert holder_path.exists()

    explicit_path = lock_path.parent / "explicit.lock"
    assert default_gpu_lock_path(explicit_path) == explicit_path
    with gpu_lock(explicit_path):
        assert explicit_path.exists()
        assert not holder_path.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows ProgramData default")
def test_windows_default_falls_back_if_programdata_cannot_be_opened(monkeypatch, tmp_path):
    from microscopy import gpu_lock as lock_module

    programdata = tmp_path / "ProgramData"
    programdata.touch()
    monkeypatch.setenv("PROGRAMDATA", str(programdata))
    monkeypatch.setattr(lock_module.tempfile, "gettempdir", lambda: str(tmp_path))
    assert default_gpu_lock_path() == programdata / "lab-microscopy" / "gpu.lock"
    with gpu_lock():
        assert (tmp_path / "lab-microscopy-gpu.lock.holder.json").exists()


_CHILD = """
import os
import sys
import time
from pathlib import Path
from microscopy.gpu_lock import gpu_lock

with gpu_lock(sys.argv[1], job="child Omnipose"):
    ready = Path(sys.argv[2])
    temporary = ready.with_suffix(".tmp")
    temporary.write_text(str(os.getpid()))
    temporary.replace(ready)
    time.sleep(float(sys.argv[3]))
"""


@contextmanager
def _child_holding_lock(lock_path, hold_s):
    ready = lock_path.parent / "child.ready"
    holder_pid = None
    with subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(lock_path), str(ready), str(hold_s)],
        cwd=Path(__file__).resolve().parents[2],
    ) as child:
        try:
            deadline = time.monotonic() + 10
            while not ready.exists():
                assert child.poll() is None, f"Lock holder exited with code {child.returncode}"
                assert time.monotonic() < deadline, "Child did not acquire the GPU lock"
                time.sleep(0.01)
            # A Windows venv executable can launch Python in a second process.
            holder_pid = int(ready.read_text())
            yield child, holder_pid
        finally:
            if child.poll() is None:
                if holder_pid is None:
                    child.kill()
                else:
                    try:
                        os.kill(holder_pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                child.wait(timeout=5)


def test_waits_for_another_process(lock_path, holder_path):
    notices = []
    with _child_holding_lock(lock_path, 1.0) as (child, holder_pid):
        started = time.monotonic()
        with gpu_lock(lock_path, timeout_s=5, poll_interval_s=0.025, notify=notices.append):
            assert time.monotonic() - started >= 0.6
            assert any(f"PID {holder_pid}" in notice and "child Omnipose" in notice for notice in notices)
            assert json.loads(holder_path.read_text(encoding="utf-8"))["pid"] == os.getpid()
        assert child.wait(timeout=5) == 0
    assert not holder_path.exists()


def test_timeout_and_recovery_after_process_is_killed(lock_path, holder_path):
    with _child_holding_lock(lock_path, 30.0) as (child, holder_pid):
        started = time.monotonic()
        with pytest.raises(TimeoutError, match=rf"PID {holder_pid} .*child Omnipose"):
            with gpu_lock(lock_path, timeout_s=0.3, poll_interval_s=1.0):
                pytest.fail("The child still holds the GPU lock")
        assert time.monotonic() - started >= 0.3
        os.kill(holder_pid, signal.SIGTERM)
        child.wait(timeout=5)
        assert json.loads(holder_path.read_text(encoding="utf-8"))["pid"] == holder_pid

    with gpu_lock(lock_path, timeout_s=0, job="replacement"):
        holder = json.loads(holder_path.read_text(encoding="utf-8"))
        assert holder["pid"] == os.getpid()
        assert holder["job"] == "replacement"
    assert not holder_path.exists()


@pytest.fixture
def fake_omnipose(monkeypatch):
    events = []

    class FakeModel:
        nchan = 2
        nclasses = 3
        dim = 2
        device = None

        def __init__(self, *, gpu, model_type):
            self.gpu = gpu
            events.append("load")

        def _run_nets(self, image, **kwargs):
            events.append("inference")
            return np.zeros((*image.shape, self.nclasses), dtype=np.float32), None

    def compute_masks(flow, probability, **kwargs):
        events.append("postprocess")
        return (np.zeros_like(probability, dtype=np.int32),)

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: events.append("empty_cache"),
    )
    cellpose = ModuleType("cellpose_omni")
    cellpose.models = SimpleNamespace(CellposeModel=FakeModel)
    cellpose.transforms = SimpleNamespace(convert_image=lambda image, *args, **kwargs: image)
    omnipose = ModuleType("omnipose")
    omnipose.core = ModuleType("omnipose.core")
    omnipose.core.compute_masks = compute_masks
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "cellpose_omni", cellpose)
    monkeypatch.setitem(sys.modules, "omnipose", omnipose)
    monkeypatch.setitem(sys.modules, "omnipose.core", omnipose.core)
    return events


@pytest.fixture
def lock_calls(monkeypatch, fake_omnipose):
    calls = []

    @contextmanager
    def record_lock(**kwargs):
        calls.append(kwargs)
        fake_omnipose.append("acquire")
        try:
            yield
        finally:
            fake_omnipose.append("release")

    monkeypatch.setattr(fast_omnipose, "gpu_lock", record_lock)
    return calls


@pytest.mark.parametrize(
    "entry_point, operations",
    [
        ("load_model", ["load"]),
        ("warmup", ["load", "inference", "postprocess"]),
        ("segment_normalized_fp32", ["load", "inference", "postprocess"]),
        ("run_stack", ["load", "inference", "postprocess"]),
    ],
)
def test_gpu_entry_points_acquire_one_session(
    fake_omnipose, lock_calls, lock_path, entry_point, operations,
):
    runner = fast_omnipose.FastOmniposeRunner(gpu_lock_path=lock_path, gpu_lock_timeout_s=2.5)
    image = np.zeros((32, 32), dtype=np.float32)
    args = {
        "load_model": (),
        "warmup": (image,),
        "segment_normalized_fp32": (image,),
        "run_stack": (1, lambda _: image),
    }
    getattr(runner, entry_point)(*args[entry_point])
    assert lock_calls == [{
        "lock_path": lock_path,
        "timeout_s": 2.5,
        "job": f"FastOmniposeRunner.{entry_point}",
    }]
    assert fake_omnipose == ["acquire", *operations, "empty_cache", "release"]
    assert runner._model is None


@pytest.mark.parametrize(
    "gpu, use_gpu_lock, disable_env", [(False, True, False), (True, False, False), (True, True, True)],
)
def test_runner_bypasses_lock(monkeypatch, fake_omnipose, lock_calls, gpu, use_gpu_lock, disable_env):
    if disable_env:
        monkeypatch.setenv("LAB_MICROSCOPY_DISABLE_GPU_LOCK", "1")
    runner = fast_omnipose.FastOmniposeRunner(gpu=gpu, use_gpu_lock=use_gpu_lock)
    runner.warmup()
    assert lock_calls == []
    assert fake_omnipose == ["load", "inference", "postprocess"]
    assert runner.load_model() == 0.0


def test_cli_holds_one_lock_from_load_through_inference(
    monkeypatch, fake_omnipose, lock_calls, lock_path, tmp_path,
):
    import tifffile

    monkeypatch.setattr(tifffile, "imread", lambda path: np.zeros((32, 32), dtype=np.float32))
    out = tmp_path / "results.json"
    assert fast_omnipose.main([
        "--pos-dir", str(tmp_path),
        "--num-frames", "1",
        "--out-json", str(out),
        "--gpu-lock-path", str(lock_path),
        "--gpu-lock-timeout-s", "3",
    ]) == 0
    assert lock_calls == [{"lock_path": lock_path, "timeout_s": 3.0, "job": f"FastOmnipose {tmp_path}"}]
    assert fake_omnipose == [
        "acquire", "load", "inference", "postprocess", "inference", "postprocess", "empty_cache", "release",
    ]
    assert len(json.loads(out.read_text())["frames"]) == 1
