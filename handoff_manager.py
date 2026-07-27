from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class HandoffRecord(BaseModel):
    project_id: int = Field(gt=0, alias="projectId")
    client_message_id: str = Field(min_length=1, max_length=128, alias="clientMessageId")
    objective: str = Field(min_length=1)
    repository_path: str | None = Field(default=None, alias="repositoryPath")
    repository_revision: str | None = Field(default=None, alias="repositoryRevision")
    completed_steps: list[str] = Field(default_factory=list, alias="completedSteps")
    model_paths: list[str] = Field(default_factory=list, alias="modelPaths")
    dataset_paths: list[str] = Field(default_factory=list, alias="datasetPaths")
    config_paths: list[str] = Field(default_factory=list, alias="configPaths")
    remaining_environment_steps: list[str] = Field(
        default_factory=list, alias="remainingEnvironmentSteps"
    )
    next_command: str | None = Field(default=None, alias="nextCommand")
    expected_outputs: list[str] = Field(default_factory=list, alias="expectedOutputs")
    notes: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


def ensure_stable_path(path: Path, stable_root: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = stable_root.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Path must be under stable workspace root {root}: {resolved}"
        ) from exc
    return resolved


def write_handoff(
    *, working_directory: str, stable_root: Path, record: HandoffRecord
) -> dict[str, Any]:
    workdir = ensure_stable_path(Path(working_directory), stable_root)
    workdir.mkdir(parents=True, exist_ok=True)
    target_dir = workdir / ".ai-scientist"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "handoff.json"

    payload = record.model_dump(by_alias=True)
    payload["updatedAt"] = datetime.now(timezone.utc).isoformat()

    fd, temporary_name = tempfile.mkstemp(
        prefix="handoff-", suffix=".json", dir=target_dir
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)

    return {"handoffPath": str(target), "record": payload}


def inspect_artifacts(paths: list[str], stable_root: Path) -> dict[str, Any]:
    results = []
    all_ready = True
    for raw in paths:
        try:
            path = ensure_stable_path(Path(raw), stable_root)
            exists = path.exists()
            partial = path.name.endswith((".partial", ".part", ".tmp"))
            size = path.stat().st_size if exists and path.is_file() else None
            ready = exists and not partial
            all_ready = all_ready and ready
            results.append(
                {
                    "path": str(path),
                    "exists": exists,
                    "isFile": path.is_file() if exists else False,
                    "isDirectory": path.is_dir() if exists else False,
                    "sizeBytes": size,
                    "looksPartial": partial,
                    "ready": ready,
                }
            )
        except (OSError, ValueError) as exc:
            all_ready = False
            results.append({"path": raw, "ready": False, "error": str(exc)})
    return {"allReady": all_ready, "artifacts": results}
