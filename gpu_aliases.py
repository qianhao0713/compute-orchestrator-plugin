"""Translate public GPU aliases at the Claude/MCP boundary."""

from __future__ import annotations

import re
from typing import Any


PUBLIC_GPU_32G = "gpu-32G"
PUBLIC_GPU_96G = "gpu-96G"
_BACKEND_GPU_32G = "Z1120"
_BACKEND_GPU_96G = "V5000"

_PUBLIC_TO_BACKEND = {
    PUBLIC_GPU_32G.lower(): _BACKEND_GPU_32G,
    PUBLIC_GPU_96G.lower(): _BACKEND_GPU_96G,
}
_BACKEND_TO_PUBLIC = {
    _BACKEND_GPU_32G.lower(): PUBLIC_GPU_32G,
    _BACKEND_GPU_96G.lower(): PUBLIC_GPU_96G,
}
_BACKEND_PATTERN = re.compile(
    "|".join(re.escape(value) for value in _BACKEND_TO_PUBLIC),
    re.IGNORECASE,
)


def to_backend_gpu_type(value: str | None) -> str | None:
    if value is None:
        return None
    backend = _PUBLIC_TO_BACKEND.get(value.strip().lower())
    if backend is None:
        raise ValueError("gpu_type must be 'gpu-32G' or 'gpu-96G'")
    return backend


def public_gpu_type(value: str | None) -> str | None:
    if value is None:
        return None
    return _BACKEND_TO_PUBLIC.get(value.strip().lower(), value)


def _sanitize_string(value: str) -> str:
    return _BACKEND_PATTERN.sub(
        lambda match: _BACKEND_TO_PUBLIC[match.group(0).lower()],
        value,
    )


def to_public(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_public(child) for key, child in value.items()}
    if isinstance(value, list):
        return [to_public(child) for child in value]
    if isinstance(value, tuple):
        return tuple(to_public(child) for child in value)
    if isinstance(value, str):
        return _sanitize_string(value)
    return value
