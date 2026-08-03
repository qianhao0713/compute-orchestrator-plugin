from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    portal_base_url: str
    portal_access_token: str
    portal_project_id: int
    stable_workspace_root: Path
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 30.0
    write_timeout_seconds: float = 30.0
    pool_timeout_seconds: float = 5.0

    @classmethod
    def from_env(cls) -> "Settings":
        base_url = os.environ.get("PORTAL_BASE_URL", "").strip().rstrip("/")
        token = os.environ.get("PORTAL_ACCESS_TOKEN", "").strip()
        project_id_raw = os.environ.get("PORTAL_PROJECT_ID", "").strip()
        workspace = os.environ.get(
            "AI_SCIENTIST_STABLE_WORKSPACE_ROOT", "/home/scientist"
        ).strip()

        if not base_url:
            raise RuntimeError("PORTAL_BASE_URL is required")
        if not token:
            raise RuntimeError("PORTAL_ACCESS_TOKEN is required")
        if not project_id_raw:
            raise RuntimeError("PORTAL_PROJECT_ID is required")
        try:
            project_id = int(project_id_raw)
        except ValueError as exc:
            raise RuntimeError(
                "PORTAL_PROJECT_ID must be a positive integer"
            ) from exc
        if project_id <= 0:
            raise RuntimeError("PORTAL_PROJECT_ID must be a positive integer")
        if not workspace:
            raise RuntimeError("AI_SCIENTIST_STABLE_WORKSPACE_ROOT is required")

        return cls(
            portal_base_url=base_url,
            portal_access_token=token,
            portal_project_id=project_id,
            stable_workspace_root=Path(workspace).expanduser().resolve(),
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
