from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    portal_base_url: str
    portal_access_token: str
    stable_workspace_root: Path
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 30.0
    write_timeout_seconds: float = 30.0
    pool_timeout_seconds: float = 5.0

    @classmethod
    def from_env(cls) -> "Settings":
        base_url = os.environ.get("PORTAL_BASE_URL", "").strip().rstrip("/")
        token = os.environ.get("PORTAL_ACCESS_TOKEN", "").strip()
        workspace = os.environ.get(
            "AI_SCIENTIST_STABLE_WORKSPACE_ROOT", "/home/scientist"
        ).strip()

        if not base_url:
            raise RuntimeError("PORTAL_BASE_URL is required")
        if not token:
            raise RuntimeError("PORTAL_ACCESS_TOKEN is required")
        if not workspace:
            raise RuntimeError("AI_SCIENTIST_STABLE_WORKSPACE_ROOT is required")

        return cls(
            portal_base_url=base_url,
            portal_access_token=token,
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
