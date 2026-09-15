from types import SimpleNamespace

from hooks.block_portal_bash import build_hook_output
import resource_inspector
from resource_inspector import inspect_current_resources


def test_inspector_returns_expected_sections():
    result = inspect_current_resources()
    assert set(result) == {"cpu", "memory", "gpu", "torch"}


def test_inspector_uses_name_free_nvidia_query(monkeypatch):
    commands = []

    def which(name):
        return "/bin/nvidia-smi" if name == "nvidia-smi" else None

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="0, 98304, 90000\n")

    monkeypatch.setattr(resource_inspector.shutil, "which", which)
    monkeypatch.setattr(resource_inspector.subprocess, "run", run)

    result = resource_inspector._gpu_info()

    assert commands[0][1] == "--query-gpu=index,memory.total,memory.free"
    assert result["devices"][0]["portalGpuType"] == "GPU-96G"
    assert "name" not in result["devices"][0]


def test_inspector_uses_safe_xpu_smi_query_for_gpu_96g(monkeypatch):
    commands = []
    sample = """
==============XPUSMI LOG==============
Timestamp                                 : Tue Sep 15 10:01:42 2026
Driver Version                            : 5.0.21.26
XPU-RT Version                            : 10.2

Attached XPUs                             : 1
XPU 00000000:4E:00.0
    Memory Usage
        Total                             : 98304 MiB
        Reserved                          : 0 MiB
        Used                              : 0 MiB
        Free                              : 98304 MiB
    L3 Usage
        Total                             : 96 MiB
        Reserved                          : 0 MiB
        Used                              : 0 MiB
        Free                              : 96 MiB
    Utilization
        Xpu                               : 0 %
    Temperature
        XPU Current Temp                  : 46 C
    Clocks
        Cluster                           : 1450 MHz
    Processes                             : None
"""

    def which(name):
        return "/usr/bin/xpu-smi" if name == "xpu-smi" else None

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=sample)

    monkeypatch.setattr(resource_inspector.shutil, "which", which)
    monkeypatch.setattr(resource_inspector.subprocess, "run", run)

    result = resource_inspector._gpu_info()

    assert commands[0] == [
        "xpu-smi",
        "-q",
        "-d",
        "MEMORY,UTILIZATION,TEMPERATURE,CLOCK,PIDS",
    ]
    assert result == {
        "available": True,
        "count": 1,
        "devices": [
            {
                "index": 0,
                "portalGpuType": "GPU-96G",
                "totalMemoryGiB": 96.0,
                "freeMemoryGiB": 96.0,
            }
        ],
    }
    assert "name" not in result["devices"][0]


def test_raw_smi_output_is_denied_but_name_free_query_is_allowed():
    raw = build_hook_output(
        {"tool_name": "Bash", "tool_input": {"command": "xpu-smi"}}
    )
    identifying = build_hook_output(
        {
            "tool_name": "Bash",
            "tool_input": {
                "command": (
                    "nvidia-smi --query-gpu=index,name,memory.total "
                    "--format=csv,noheader,nounits"
                )
            },
        }
    )
    safe_nvidia = build_hook_output(
        {
            "tool_name": "Bash",
            "tool_input": {
                "command": (
                    "nvidia-smi --query-gpu=index,memory.total,memory.free "
                    "--format=csv,noheader,nounits"
                )
            },
        }
    )

    safe_xpu = build_hook_output(
        {
            "tool_name": "Bash",
            "tool_input": {
                "command": (
                    "xpu-smi -q -d "
                    "MEMORY,UTILIZATION,TEMPERATURE,CLOCK,PIDS"
                )
            },
        }
    )

    assert raw["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert identifying["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert safe_nvidia == {}
    assert safe_xpu == {}


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
