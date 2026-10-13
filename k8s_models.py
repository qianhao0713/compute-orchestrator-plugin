"""Pydantic models for the K8s CPU distributed-job Portal API.

Covers submit, detail, list, cancel, attempts, logs, and cluster resources.
The unified envelope (code/msg/data/traceId) is shared with the existing
portal_client envelope; this module defines the K8s-specific request and
response structures that plug into it.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
RESERVED_ENV_KEYS = {"JOB_ID", "POD_UID", "JOB_COMPLETION_INDEX"}

MAX_COMMAND_ITEMS = 64
MAX_ARGS_ITEMS = 256
MAX_COMMAND_SINGLE_LENGTH = 8192
MAX_COMMAND_TOTAL_BYTES = 32768
MAX_ENV_ITEMS = 32
MAX_ENV_TOTAL_BYTES = 8192
MAX_SHARD_COUNT = 5000
MAX_PARALLELISM = 64
MAX_BACKOFF_LIMIT = 20
MAX_ACTIVE_DEADLINE = 172800


class SubmitK8sJobRequest(BaseModel):
    """Request body for POST /scientist/project/{projectId}/jobs/submit."""

    model_config = ConfigDict(populate_by_name=True)

    request_id: str = Field(
        alias="requestId", min_length=1, max_length=64
    )
    session_id: str | None = Field(
        default=None, alias="sessionId", max_length=128
    )
    execution_profile: str | None = Field(
        default=None, alias="executionProfile", max_length=64
    )
    is_vip: bool | None = Field(default=None, alias="isVip")
    image: str | None = Field(default=None, max_length=512)
    command: list[str] | None = Field(default=None, min_length=1)
    args: list[str] | None = Field(default=None)
    shard_count: int = Field(default=1, alias="shardCount", ge=1)
    parallelism: int = Field(default=1, ge=1)
    cpu: int = Field(default=1, ge=1)
    memory_gib: int = Field(default=4, alias="memoryGiB", ge=1)
    ephemeral_storage_gib: int | None = Field(
        default=None, alias="ephemeralStorageGiB", ge=1
    )
    job_backoff_limit: int = Field(
        default=6, alias="jobBackoffLimit", ge=0
    )
    active_deadline_seconds: int = Field(
        default=21600, alias="activeDeadlineSeconds", ge=1
    )
    description: str | None = Field(default=None, max_length=1024)
    env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_submit(self) -> "SubmitK8sJobRequest":
        if self.command is not None:
            if len(self.command) > MAX_COMMAND_ITEMS:
                raise ValueError(
                    f"command must have at most {MAX_COMMAND_ITEMS} items"
                )
            for item in self.command:
                if not item or not item.strip():
                    raise ValueError("command items must be non-blank")
                if len(item) > MAX_COMMAND_SINGLE_LENGTH:
                    raise ValueError(
                        f"command item exceeds {MAX_COMMAND_SINGLE_LENGTH} chars"
                    )
        if self.args is not None:
            if len(self.args) > MAX_ARGS_ITEMS:
                raise ValueError(
                    f"args must have at most {MAX_ARGS_ITEMS} items"
                )
            for item in self.args:
                if len(item) > MAX_COMMAND_SINGLE_LENGTH:
                    raise ValueError(
                        f"args item exceeds {MAX_COMMAND_SINGLE_LENGTH} chars"
                    )
        combined_bytes = 0
        if self.command is not None:
            combined_bytes += sum(
                len(item.encode("utf-8")) for item in self.command
            )
        if self.args is not None:
            combined_bytes += sum(
                len(item.encode("utf-8")) for item in self.args
            )
        if combined_bytes > MAX_COMMAND_TOTAL_BYTES:
            raise ValueError(
                f"command+args total exceeds {MAX_COMMAND_TOTAL_BYTES} bytes"
            )
        if self.shard_count > MAX_SHARD_COUNT:
            raise ValueError(
                f"shardCount must be at most {MAX_SHARD_COUNT}"
            )
        if self.parallelism > MAX_PARALLELISM:
            raise ValueError(
                f"parallelism must be at most {MAX_PARALLELISM}"
            )
        if self.parallelism > self.shard_count:
            raise ValueError(
                "parallelism must not exceed shardCount"
            )
        if self.job_backoff_limit > MAX_BACKOFF_LIMIT:
            raise ValueError(
                f"jobBackoffLimit must be at most {MAX_BACKOFF_LIMIT}"
            )
        if self.active_deadline_seconds > MAX_ACTIVE_DEADLINE:
            raise ValueError(
                f"activeDeadlineSeconds must be at most {MAX_ACTIVE_DEADLINE}"
            )
        self._validate_env()
        return self

    def _validate_env(self) -> None:
        if len(self.env) > MAX_ENV_ITEMS:
            raise ValueError(
                f"env must have at most {MAX_ENV_ITEMS} items"
            )
        total_bytes = 0
        for key, value in self.env.items():
            if not ENV_KEY_PATTERN.match(key):
                raise ValueError(
                    f"env key {key!r} does not match "
                    "[A-Za-z_][A-Za-z0-9_]{{0,63}}"
                )
            if key in RESERVED_ENV_KEYS:
                raise ValueError(
                    f"env key {key!r} is reserved by the platform"
                )
            if not isinstance(value, str):
                raise ValueError(f"env value for {key!r} must be a string")
            if "\x00" in key or "\x00" in value:
                raise ValueError(
                    f"env key/value must not contain NUL: {key!r}"
                )
            total_bytes += len(key.encode("utf-8")) + len(
                value.encode("utf-8")
            )
        if total_bytes > MAX_ENV_TOTAL_BYTES:
            raise ValueError(
                f"env total exceeds {MAX_ENV_TOTAL_BYTES} bytes"
            )


class EffectiveSpec(BaseModel):
    """The merged execution specification returned in TaskView."""

    model_config = ConfigDict(populate_by_name=True)

    cpu: int
    memory_gib: int = Field(alias="memoryGiB")
    parallelism: int
    image: str
    command: list[str] | None = None
    args: list[str] | None = None
    ephemeral_storage_gib: int = Field(alias="ephemeralStorageGiB")


class TaskView(BaseModel):
    """Common task object returned by submit, detail, list, and cancel."""

    model_config = ConfigDict(populate_by_name=True)

    job_id: str = Field(alias="jobId")
    state: str
    reason_code: str | None = Field(default=None, alias="reasonCode")
    last_error_type: str | None = Field(default=None, alias="lastErrorType")
    total: int
    succeeded_indexes_count: int = Field(alias="succeededIndexesCount")
    active_pods: int = Field(alias="activePods")
    running_pods: int = Field(alias="runningPods")
    failed_attempts: int = Field(alias="failedAttempts")
    created_at: str | None = Field(default=None, alias="createdAt")
    started_at: str | None = Field(default=None, alias="startedAt")
    finished_at: str | None = Field(default=None, alias="finishedAt")
    observed_at: str | None = Field(default=None, alias="observedAt")
    stale: bool = False
    resources_released: bool = Field(alias="resourcesReleased")
    storage_released: bool = Field(alias="storageReleased")
    execution_profile: str | None = Field(
        default=None, alias="executionProfile"
    )
    is_vip: bool | None = Field(default=None, alias="isVip")
    env: dict[str, str] = Field(default_factory=dict)
    mount_path: str = Field(alias="mountPath")
    sub_path: str = Field(alias="subPath")
    effective_spec: EffectiveSpec = Field(alias="effectiveSpec")


class K8sSubmitResponse(BaseModel):
    """Submit response: TaskView + replayed flag."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    replayed: bool = False


class K8sCancelResponse(BaseModel):
    """Cancel response: TaskView + cancelApplied flag."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    cancel_applied: bool = Field(alias="cancelApplied")


class K8sNodeResource(BaseModel):
    """A single node's resource availability snapshot."""

    model_config = ConfigDict(populate_by_name=True)

    node_name: str = Field(alias="nodeName")
    ready: bool
    schedulable: bool
    allocatable_cpu_cores: float = Field(alias="allocatableCpuCores")
    requested_cpu_cores: float = Field(alias="requestedCpuCores")
    available_cpu_cores: float = Field(alias="availableCpuCores")
    allocatable_memory_bytes: int = Field(alias="allocatableMemoryBytes")
    requested_memory_bytes: int = Field(alias="requestedMemoryBytes")
    available_memory_bytes: int = Field(alias="availableMemoryBytes")


class K8sResourcesData(BaseModel):
    """Response data for GET /resources."""

    model_config = ConfigDict(populate_by_name=True)

    cluster_id: str = Field(alias="clusterId")
    calculation: str
    observed_at: str = Field(alias="observedAt")
    nodes: list[K8sNodeResource] = Field(default_factory=list)


class K8sJobListData(BaseModel):
    """Response data for GET / (list jobs)."""

    items: list[TaskView] = Field(default_factory=list)
    next_cursor: str | None = Field(default=None, alias="nextCursor")


class K8sAttemptItem(BaseModel):
    """A single shard execution attempt."""

    model_config = ConfigDict(populate_by_name=True)

    job_id: str = Field(alias="jobId")
    pod_uid: str = Field(alias="podUid")
    pod_name: str = Field(alias="podName")
    job_uid: str = Field(alias="jobUid")
    shard_index: int = Field(alias="shardIndex")
    phase: str | None = None
    image_id: str | None = Field(default=None, alias="imageId")
    reason: str | None = None
    exit_code: int | None = Field(default=None, alias="exitCode")
    pod_created_at: str | None = Field(default=None, alias="podCreatedAt")
    started_at: str | None = Field(default=None, alias="startedAt")
    finished_at: str | None = Field(default=None, alias="finishedAt")
    terminal: bool = False


class K8sAttemptsData(BaseModel):
    """Response data for GET /{jobId}/attempts."""

    items: list[K8sAttemptItem] = Field(default_factory=list)
    next_cursor: str | None = Field(default=None, alias="nextCursor")


class K8sLogData(BaseModel):
    """Response data for GET /{jobId}/logs."""

    attempt_id: str = Field(alias="attemptId")
    shard_index: int = Field(alias="shardIndex")
    text: str
    truncated: bool = False


VALID_JOB_STATES = {
    "ACCEPTED",
    "SUBMITTING",
    "QUEUED",
    "RUNNING",
    "RETRYING",
    "FAILING",
    "CANCELLING",
    "SUCCEEDED",
    "FAILED",
    "CANCELLED",
    "UNKNOWN",
}

TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELLED"}


def is_terminal_state(state: str) -> bool:
    return state in TERMINAL_STATES


def is_fully_done(task: TaskView) -> bool:
    """True when the task has reached a terminal state and resources are released."""
    return is_terminal_state(task.state) and task.resources_released
