from types import SimpleNamespace

import resource_inspector
from resource_inspector import inspect_current_resources


def test_inspector_returns_expected_sections():
    result = inspect_current_resources()
    assert set(result) == {"cpu", "memory", "gpu", "torch"}


def test_inspector_maps_v5000_portal_type(monkeypatch):
    monkeypatch.setattr(resource_inspector.shutil, "which", lambda _: "/bin/nvidia-smi")
    monkeypatch.setattr(
        resource_inspector.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="0, V5000, 98304, 90000, 1.0\n"
        ),
    )
    result = resource_inspector._gpu_info()
    assert result["devices"][0]["portalGpuType"] == "V5000"
