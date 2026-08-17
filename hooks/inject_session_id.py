#!/usr/bin/env python3
"""Inject Claude Code's authoritative session ID into ensure_resource."""

from __future__ import annotations

import json
import sys
from typing import Any


EXPECTED_TOOLS = {
    "mcp__plugin_compute-orchestrator_compute-orchestrator-resource__ensure_resource",
    "mcp__compute-orchestrator-resource__ensure_resource",
}
MAX_SESSION_ID_LENGTH = 128


def build_hook_output(event: dict[str, Any]) -> dict[str, Any]:
    tool_name = event.get("tool_name")
    if tool_name not in EXPECTED_TOOLS:
        raise ValueError(f"unexpected tool for session injection: {tool_name!r}")

    session_id = event.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("Claude Code hook input is missing session_id")
    if len(session_id) > MAX_SESSION_ID_LENGTH:
        raise ValueError("Claude Code session_id exceeds 128 characters")

    tool_input = event.get("tool_input")
    if not isinstance(tool_input, dict):
        raise ValueError("Claude Code hook input is missing tool_input")

    updated_input = {**tool_input, "session_id": session_id}
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "updatedInput": updated_input,
        }
    }


def main() -> int:
    try:
        event = json.load(sys.stdin)
        if not isinstance(event, dict):
            raise ValueError("hook input must be a JSON object")
        json.dump(build_hook_output(event), sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"Cannot inject Claude Code session_id: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
