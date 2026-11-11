import pytest
import torch

from scanlag.device import autocast, torch_device


def test_device_order_and_override(monkeypatch):
    monkeypatch.delenv("SCANLAG_DEVICE", raising=False)
    for cuda, mps, expected in ((True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
        assert torch_device().type == expected
    monkeypatch.setenv("SCANLAG_DEVICE", " CPU ")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert torch_device().type == "cpu"
    monkeypatch.setenv("SCANLAG_DEVICE", "gpu")
    with pytest.raises(ValueError, match="cuda, mps, cpu"):
        torch_device()


def test_no_mixed_precision_on_the_cpu():
    """bfloat16 is emulated on most CPUs: a U-Net forward pass took ~1000x longer than in float32."""
    with autocast(torch.device("cpu")):
        y = torch.nn.Linear(4, 4)(torch.ones(1, 4))
    assert y.dtype == torch.float32
    assert autocast(torch.device("mps")).fast_dtype == torch.float16
    assert autocast(torch.device("cuda")).fast_dtype == torch.bfloat16
