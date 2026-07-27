from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


_GIB = 1024**3


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def _memory_info() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        result["hostTotalGiB"] = round((pages * page_size) / _GIB, 3)
    except (ValueError, OSError, AttributeError):
        result["hostTotalGiB"] = None

    mem_available_kib = None
    text = _read_text("/proc/meminfo")
    if text:
        for line in text.splitlines():
            if line.startswith("MemAvailable:"):
                mem_available_kib = int(line.split()[1])
                break
    result["availableGiB"] = (
        round(mem_available_kib * 1024 / _GIB, 3)
        if mem_available_kib is not None
        else None
    )

    raw_limit = _read_text("/sys/fs/cgroup/memory.max")
    if raw_limit and raw_limit != "max":
        try:
            result["cgroupLimitGiB"] = round(int(raw_limit) / _GIB, 3)
        except ValueError:
            result["cgroupLimitGiB"] = None
    else:
        result["cgroupLimitGiB"] = None
    return result


def _cpu_info() -> dict[str, Any]:
    result: dict[str, Any] = {"logicalCores": os.cpu_count()}
    raw = _read_text("/sys/fs/cgroup/cpu.max")
    quota_cores = None
    if raw:
        parts = raw.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                quota_cores = round(int(parts[0]) / int(parts[1]), 3)
            except (ValueError, ZeroDivisionError):
                pass
    result["cgroupQuotaCores"] = quota_cores
    return result


def _gpu_info() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "count": 0, "devices": []}

    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=10, check=True
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return {
            "available": False,
            "count": 0,
            "devices": [],
            "error": str(exc),
        }

    devices = []
    for line in completed.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != 5:
            continue
        index, name, total_mib, free_mib, driver = values
        devices.append(
            {
                "index": int(index),
                "name": name,
                "portalGpuType": "Z1120" if "V100" in name.upper() else None,
                "totalMemoryGiB": round(float(total_mib) / 1024, 3),
                "freeMemoryGiB": round(float(free_mib) / 1024, 3),
                "driverVersion": driver,
            }
        )
    return {"available": bool(devices), "count": len(devices), "devices": devices}


def _torch_info() -> dict[str, Any]:
    script = """
import json
try:
    import torch
    print(json.dumps({
        'installed': True,
        'version': torch.__version__,
        'cudaAvailable': torch.cuda.is_available(),
        'deviceCount': torch.cuda.device_count(),
        'cudaRuntime': torch.version.cuda,
    }))
except Exception as exc:
    print(json.dumps({'installed': False, 'error': str(exc)}))
"""
    try:
        completed = subprocess.run(
            ["python3", "-c", script],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        return json.loads(completed.stdout.strip())
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as exc:
        return {"installed": False, "error": str(exc)}


def inspect_current_resources() -> dict[str, Any]:
    return {
        "cpu": _cpu_info(),
        "memory": _memory_info(),
        "gpu": _gpu_info(),
        "torch": _torch_info(),
    }
