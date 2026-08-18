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
    host_total_bytes = None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        host_total_bytes = pages * page_size
    except (ValueError, OSError, AttributeError):
        pass

    raw_limit = _read_text("/sys/fs/cgroup/memory.max")
    if raw_limit is None:
        raw_limit = _read_text("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    limit_bytes = None
    if raw_limit and raw_limit != "max":
        try:
            parsed_limit = int(raw_limit)
            # cgroup v1 sometimes represents "unlimited" with a huge sentinel.
            if parsed_limit < (1 << 60):
                limit_bytes = parsed_limit
        except ValueError:
            pass
    effective_total = host_total_bytes
    if limit_bytes is not None:
        effective_total = (
            min(effective_total, limit_bytes)
            if effective_total is not None
            else limit_bytes
        )

    result["totalGiB"] = (
        round(effective_total / _GIB, 3) if effective_total is not None else None
    )
    return {"totalGiB": result["totalGiB"]}


def _count_cpuset(raw: str | None) -> int | None:
    if not raw:
        return None
    cpus: set[int] = set()
    try:
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                start_raw, end_raw = item.split("-", 1)
                start, end = int(start_raw), int(end_raw)
                if end < start:
                    return None
                cpus.update(range(start, end + 1))
            else:
                cpus.add(int(item))
    except ValueError:
        return None
    return len(cpus) or None


def _cpu_info() -> dict[str, Any]:
    host_cores = os.cpu_count()
    affinity_cores = None
    try:
        affinity_cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass

    cpuset_raw = _read_text("/sys/fs/cgroup/cpuset.cpus.effective")
    if cpuset_raw is None:
        cpuset_raw = _read_text("/sys/fs/cgroup/cpuset/cpuset.cpus")
    cpuset_cores = _count_cpuset(cpuset_raw)

    raw = _read_text("/sys/fs/cgroup/cpu.max")
    quota_cores = None
    if raw:
        parts = raw.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                quota_cores = round(int(parts[0]) / int(parts[1]), 3)
            except (ValueError, ZeroDivisionError):
                pass
    else:
        quota_raw = _read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period_raw = _read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        try:
            if quota_raw and period_raw and int(quota_raw) > 0:
                quota_cores = round(int(quota_raw) / int(period_raw), 3)
        except (ValueError, ZeroDivisionError):
            pass

    candidates = [
        value
        for value in (affinity_cores, cpuset_cores, quota_cores, host_cores)
        if value is not None and value > 0
    ]
    effective_cores = min(candidates) if candidates else None
    return {"logicalCores": effective_cores}


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
        normalized_name = name.upper()
        portal_gpu_type = None
        if "V5000" in normalized_name:
            portal_gpu_type = "V5000"
        elif "V100" in normalized_name:
            portal_gpu_type = "Z1120"
        devices.append(
            {
                "index": int(index),
                "name": name,
                "portalGpuType": portal_gpu_type,
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
