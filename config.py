from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    portal_base_url: str
    portal_access_token: str
    portal_project_id: int
    stable_workspace_root: Path
    plugin_log_dir: Path = Path("/home/scientist/.claude/logs")
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 30.0
    write_timeout_seconds: float = 30.0
    pool_timeout_seconds: float = 5.0

    @classmethod
    def from_env(cls) -> "Settings":
        base_url = _normalize_portal_base_url(
            os.environ.get("PORTAL_BASE_URL", "")
        )
        token = os.environ.get("PORTAL_ACCESS_TOKEN", "").strip()
        project_id_raw = os.environ.get("PORTAL_PROJECT_ID", "").strip()
        workspace = os.environ.get(
            "AI_SCIENTIST_STABLE_WORKSPACE_ROOT", "/home/scientist"
        ).strip()
        plugin_log_dir = os.environ.get(
            "PLUGIN_LOG_DIR", "/home/scientist/.claude/logs/"
        ).strip()

        if not base_url:
            raise RuntimeError("PORTAL_BASE_URL is required")
        if not token:
            raise RuntimeError("PORTAL_ACCESS_TOKEN is required")
        if not workspace:
            raise RuntimeError("AI_SCIENTIST_STABLE_WORKSPACE_ROOT is required")
        if not plugin_log_dir:
            raise RuntimeError("PLUGIN_LOG_DIR must not be empty")

        project_id = _resolve_project_id(project_id_raw, workspace)

        return cls(
            portal_base_url=base_url,
            portal_access_token=token,
            portal_project_id=project_id,
            stable_workspace_root=Path(workspace).expanduser().resolve(),
            plugin_log_dir=Path(plugin_log_dir).expanduser().resolve(),
            connect_timeout_seconds=float(
                os.environ.get("PORTAL_CONNECT_TIMEOUT_SECONDS", "5")
            ),
            read_timeout_seconds=float(
                os.environ.get("PORTAL_READ_TIMEOUT_SECONDS", "30")
            ),
            write_timeout_seconds=float(
                os.environ.get("PORTAL_WRITE_TIMEOUT_SECONDS", "30")
            ),
            pool_timeout_seconds=float(
                os.environ.get("PORTAL_POOL_TIMEOUT_SECONDS", "5")
            ),
        )


def _normalize_portal_base_url(raw_value: str) -> str:
    base_url = raw_value.strip().rstrip("/")
    if not base_url:
        return ""
    if base_url.lower().startswith("https://"):
        return f"https://{base_url[8:]}"
    if base_url.lower().startswith("http://"):
        return f"https://{base_url[7:]}"
    return f"https://{base_url}"


def _resolve_project_id(project_id_raw: str, workspace: str) -> int:
    if project_id_raw:
        try:
            project_id = int(project_id_raw)
        except ValueError as exc:
            raise RuntimeError(
                "PORTAL_PROJECT_ID must be a positive integer"
            ) from exc
        if project_id <= 0:
            raise RuntimeError("PORTAL_PROJECT_ID must be a positive integer")
        return project_id

    directory_name = Path(workspace).expanduser().name
    match = re.fullmatch(r"project([1-9][0-9]*)", directory_name)
    if match is None:
        raise RuntimeError(
            "PORTAL_PROJECT_ID is not set and "
            "AI_SCIENTIST_STABLE_WORKSPACE_ROOT must end with project{project_id}"
        )
    return int(match.group(1))
