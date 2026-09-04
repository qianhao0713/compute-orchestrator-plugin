#!/usr/bin/env python3
"""Block unsafe Bash commands before execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
from typing import Any


GPU_96G_ENV = "python310_torch29_cuda"
PORTAL_PATH_PATTERN = re.compile(r"/scientist/deployment(?:/|\b)", re.I)
NETWORK_CLIENT_PATTERN = re.compile(
    r"(?:^|[;&|()\s])(?:curl|wget|http|https)(?:\s|$)|"
    r"\b(?:requests|httpx|aiohttp)\s*\.\s*"
    r"(?:request|get|post|put|patch|delete)\s*\(|"
    r"\burllib\.request\b|"
    r"\b(?:Invoke-WebRequest|Invoke-RestMethod)\b",
    re.I,
)
BROAD_KILL_PATTERN = re.compile(
    r"\bpkill\b[^\n;&|]*\bpython(?:\d+(?:\.\d+)?)?\b|"
    r"\bkillall\b[^\n;&|]*\bpython(?:\d+(?:\.\d+)?)?\b|"
    r"\bfuser\b[^\n;&|]*(?:-k|--kill)\b",
    re.I,
)
PYTHON_COMMAND_PATTERN = re.compile(
    r"(?:^|[;&|()\s])(?:python\d*(?:\.\d+)?|pip\d*|torchrun|"
    r"deepspeed)(?:\s|$)",
    re.I,
)
TRAINING_SCRIPT_PATTERN = re.compile(
    r"\b(?:torchrun|deepspeed|pretrain|fine[ _-]?tun|training|"
    r"xmegatron|nhmegatron)\b",
    re.I,
)
UNSUPPORTED_OPERATOR_PATTERN = re.compile(
    r"\b(?:complex64|complex128|float8_e4m3fn|float8_e5m2|"
    r"view_as_complex|view_as_real|to_sparse|sparse_coo_tensor|"
    r"quantize_per_tensor|quantize_per_channel|dequantize|ctc_loss)\b|"
    r"torch\.sparse\.|torch\.quantization\.|"
    r"torch\.fft\.(?:fftn|rfft2|fftshift|ifftn|irfft2|rfftn|irfftn)\b|"
    r"""scatter_reduce\s*\([^\n]*reduce\s*=\s*['"]add['"]""",
    re.I,
)
CONDA_RUN_PATTERN = re.compile(
    rf"\bconda\s+run\b[^\n;&|]*\s(?:-n|--name)\s+{GPU_96G_ENV}\b",
    re.I,
)
CONDA_ACTIVATE_PATTERN = re.compile(
    rf"\bconda\s+activate\s+{GPU_96G_ENV}\b",
    re.I,
)
PORTAL_DENIAL = (
    "Direct Portal requests from Bash are forbidden. Use the compute-orchestrator "
    "MCP tools so URL normalization, validation, session injection, availability "
    "checks, and request logging are preserved."
)
KILL_DENIAL = (
    "Broad process-killing commands are forbidden because they can terminate "
    "agent infrastructure. Copy the training script, replace broad kills with a "
    "training-only pattern, verify the copy, and run only that derived script."
)
ENV_DENIAL = (
    "GPU-96G Python work must use the python310_torch29_cuda Conda environment. "
    "Activate it in this same shell command or use "
    "conda run -n python310_torch29_cuda; activation from an earlier tool call "
    "is not trusted."
)
OPERATOR_DENIAL = (
    "The referenced GPU-96G workload contains a known unsupported PyTorch "
    "operator. Apply the GPU-96G runtime checklist; if the operation is optional "
    "and an equivalent exists, explain the substitution to the user, modify the "
    "implementation, and smoke-test before running."
)


def _portal_markers() -> tuple[str, ...]:
    markers = ["PORTAL_BASE_URL"]
    configured = os.environ.get("PORTAL_BASE_URL", "").strip().rstrip("/")
    if configured:
        markers.append(configured)
    return tuple(markers)


def is_direct_portal_request(command: str) -> bool:
    if not NETWORK_CLIENT_PATTERN.search(command):
        return False
    return bool(
        PORTAL_PATH_PATTERN.search(command)
        or any(marker in command for marker in _portal_markers())
    )


def _referenced_scripts(command: str, cwd: str | None) -> list[Path]:
    try:
        tokens = shlex.split(command, comments=False, posix=True)
    except ValueError:
        return []
    root = Path(cwd or os.getcwd())
    paths: list[Path] = []
    for token in tokens:
        candidate = Path(token)
        if candidate.suffix not in {".sh", ".py"}:
            continue
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file():
            paths.append(resolved)
    return paths


def _script_texts(command: str, cwd: str | None) -> list[str]:
    texts: list[str] = []
    for path in _referenced_scripts(command, cwd):
        try:
            if path.stat().st_size <= 2_000_000:
                texts.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return texts


def contains_broad_kill(command: str, cwd: str | None = None) -> bool:
    return bool(
        BROAD_KILL_PATTERN.search(command)
        or any(BROAD_KILL_PATTERN.search(text) for text in _script_texts(command, cwd))
    )


def contains_known_unsupported_operator(command: str, cwd: str | None = None) -> bool:
    return any(
        UNSUPPORTED_OPERATOR_PATTERN.search(text)
        for text in _script_texts(command, cwd)
    )


def is_gpu_96g_runtime() -> bool:
    for name in ("GPU_TYPE", "CLUSTER_TYPE", "PORTAL_GPU_TYPE"):
        value = os.environ.get(name, "").strip().lower()
        if value in {"gpu-96g", "v5000"}:
            return True
    return shutil.which("xpu-smi") is not None


def _requires_gpu_96g_conda(command: str, cwd: str | None) -> bool:
    if PYTHON_COMMAND_PATTERN.search(command):
        return True
    return any(TRAINING_SCRIPT_PATTERN.search(text) for text in _script_texts(command, cwd))


def _uses_gpu_96g_conda(command: str, cwd: str | None) -> bool:
    if CONDA_RUN_PATTERN.search(command) or CONDA_ACTIVATE_PATTERN.search(command):
        return True
    return any(
        CONDA_RUN_PATTERN.search(text) or CONDA_ACTIVATE_PATTERN.search(text)
        for text in _script_texts(command, cwd)
    )


def _deny(message: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
        },
        "systemMessage": message,
    }


def build_hook_output(
    event: dict[str, Any],
    *,
    gpu_96g_runtime: bool | None = None,
) -> dict[str, Any]:
    if event.get("tool_name") != "Bash":
        return {}
    tool_input = event.get("tool_input")
    if not isinstance(tool_input, dict):
        return _deny("Cannot validate Bash command; execution was denied.")
    command = tool_input.get("command")
    if not isinstance(command, str):
        return _deny("Cannot validate Bash command; execution was denied.")
    cwd = event.get("cwd")
    if not isinstance(cwd, str):
        cwd = None

    if contains_broad_kill(command, cwd):
        return _deny(KILL_DENIAL)
    if is_direct_portal_request(command):
        return _deny(PORTAL_DENIAL)

    on_gpu_96g = is_gpu_96g_runtime() if gpu_96g_runtime is None else gpu_96g_runtime
    if on_gpu_96g and contains_known_unsupported_operator(command, cwd):
        return _deny(OPERATOR_DENIAL)
    if (
        on_gpu_96g
        and _requires_gpu_96g_conda(command, cwd)
        and not _uses_gpu_96g_conda(command, cwd)
    ):
        return _deny(ENV_DENIAL)
    return {}


def main() -> int:
    try:
        event = json.load(sys.stdin)
        if not isinstance(event, dict):
            raise ValueError("hook input must be a JSON object")
        output = build_hook_output(event)
    except (json.JSONDecodeError, ValueError) as exc:
        output = _deny(f"Cannot validate Bash command; execution denied: {exc}")
    json.dump(output, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
