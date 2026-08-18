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


def test_cpu_reports_smallest_container_limit(monkeypatch):
    values = {
        "/sys/fs/cgroup/cpuset.cpus.effective": "0-7",
        "/sys/fs/cgroup/cpu.max": "250000 100000",
    }
    monkeypatch.setattr(resource_inspector, "_read_text", values.get)
    monkeypatch.setattr(resource_inspector.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(
        resource_inspector.os, "sched_getaffinity", lambda _: set(range(6))
    )

    result = resource_inspector._cpu_info()

    assert result == {"logicalCores": 2.5}


def test_memory_reports_cgroup_total_and_remaining(monkeypatch):
    gib = 1024**3
    values = {
        "/sys/fs/cgroup/memory.max": str(16 * gib),
    }
    monkeypatch.setattr(resource_inspector, "_read_text", values.get)
    monkeypatch.setattr(
        resource_inspector.os,
        "sysconf",
        lambda key: 128 * gib // 4096 if key == "SC_PHYS_PAGES" else 4096,
    )

    result = resource_inspector._memory_info()

    assert result == {"totalGiB": 16.0}


def test_cpuset_parser_handles_ranges_and_duplicates():
    assert resource_inspector._count_cpuset("0-3,6,8-9,3") == 7
