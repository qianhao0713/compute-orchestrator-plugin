#!/usr/bin/env python3
"""Block Bash commands that bypass the Portal MCP client."""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any


PORTAL_PATH_PATTERN = re.compile(r"/scientist/deployment(?:/|\b)", re.I)
NETWORK_CLIENT_PATTERN = re.compile(
    r"(?:^|[;&|()\s])(?:curl|wget|http|https)(?:\s|$)|"
    r"\b(?:requests|httpx|aiohttp)\s*\.\s*"
    r"(?:request|get|post|put|patch|delete)\s*\(|"
    r"\burllib\.request\b|"
    r"\b(?:Invoke-WebRequest|Invoke-RestMethod)\b",
    re.I,
)
DENIAL_MESSAGE = (
    "Direct Portal requests from Bash are forbidden. Use the "
    "compute-orchestrator MCP tools (get_resource_status, "
    "get_available_clusters, or ensure_resource) so URL normalization, "
    "validation, session injection, and request logging are preserved. "
    "Load the compute-orchestrator skill first when the task involves compute "
    "resource assessment or switching."
)


def _portal_markers() -> tuple[str, ...]:
    markers = ["PORTAL_BASE_URL"]
    configured_base_url = os.environ.get("PORTAL_BASE_URL", "").strip().rstrip("/")
    if configured_base_url:
        markers.append(configured_base_url)
    return tuple(markers)


def is_direct_portal_request(command: str) -> bool:
    if not NETWORK_CLIENT_PATTERN.search(command):
        return False
    if PORTAL_PATH_PATTERN.search(command):
        return True
    return any(marker in command for marker in _portal_markers())


def _deny(message: str = DENIAL_MESSAGE) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
        },
        "systemMessage": message,
    }


def build_hook_output(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("tool_name") != "Bash":
        return {}
    tool_input = event.get("tool_input")
    if not isinstance(tool_input, dict):
        return _deny("Cannot validate Bash command; direct execution was denied.")
    command = tool_input.get("command")
    if not isinstance(command, str):
        return _deny("Cannot validate Bash command; direct execution was denied.")
    return _deny() if is_direct_portal_request(command) else {}


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
