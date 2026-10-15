from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gpu_aliases import public_gpu_type


GPU_BACKEND_TYPES = {"Z1120", "Z3200", "V5000"}
Z1120_SPECS = {
    1: (8, 64),
    2: (16, 128),
    4: (32, 256),
    8: (64, 512),
}
Z3200_SPECS = {
    1: (8, 100),
    2: (16, 200),
    4: (32, 400),
    8: (64, 800),
}
V5000_SPECS = {
    8: (128, 900),
}
FORBIDDEN_KEYS = {
    "authorization",
    "cookie",
    "token",
    "accesstoken",
    "access_token",
    "refreshtoken",
    "refresh_token",
    "apikey",
    "api_key",
    "secret",
    "password",
    "passwd",
    "credential",
    "credentials",
}


class ResourceSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    resource_type: Literal["CPU", "GPU"] = Field(alias="resourceType")
    cpu: int = Field(ge=1, le=128)
    memory_gib: int = Field(alias="memoryGiB", ge=1)
    gpu_type: str | None = Field(default=None, alias="gpuType", max_length=64)
    gpu_count: int = Field(default=0, alias="gpuCount", ge=0)
    worker_num: int = Field(default=1, alias="workerNum", ge=1)

    @model_validator(mode="after")
    def validate_resource(self) -> "ResourceSpec":
        if self.worker_num != 1:
            raise ValueError("workerNum currently must be 1")
        if self.resource_type == "CPU":
            if self.cpu > 32:
                raise ValueError("CPU resource supports at most 32 cores")
            if self.gpu_count != 0:
                raise ValueError("CPU resource requires gpuCount=0")
            if self.gpu_type not in (None, ""):
                raise ValueError("CPU resource requires gpuType=null")
            self.gpu_type = None
        else:
            if self.gpu_type not in GPU_BACKEND_TYPES:
                raise ValueError("gpuType must identify a supported GPU cluster")
            fixed_specs = {
                "Z1120": Z1120_SPECS,
                "Z3200": Z3200_SPECS,
                "V5000": V5000_SPECS,
            }[self.gpu_type]
            if self.gpu_count not in fixed_specs:
                allowed_counts = ", ".join(str(count) for count in fixed_specs)
                raise ValueError(
                    f"{public_gpu_type(self.gpu_type)} GPU count must be one of {allowed_counts}"
                )
            expected_cpu, expected_memory = fixed_specs[self.gpu_count]
            if (self.cpu, self.memory_gib) != (expected_cpu, expected_memory):
                raise ValueError(
                    f"{public_gpu_type(self.gpu_type)} requires the fixed "
                    "(gpuCount, cpu, memoryGiB) specification "
                    f"({self.gpu_count}, {expected_cpu}, {expected_memory})"
                )
        return self


CPU_JOB_RESERVED_ENV = {"JOB_ID", "POD_UID", "JOB_COMPLETION_INDEX"}
CPU_JOB_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
CPU_JOB_SECRET_ENV_PATTERN = re.compile(
    r"(?:^|_)(?:ACCESS_?TOKEN|REFRESH_?TOKEN|TOKEN|SECRET|PASSWORD|PASSWD|"
    r"API_?KEY|CREDENTIALS?|COOKIE|AUTHORIZATION)(?:_|$)",
    re.I,
)


class CpuJobSubmitRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    request_id: str = Field(alias="requestId", min_length=1, max_length=64)
    session_id: str = Field(alias="sessionId", min_length=1, max_length=128)
    execution_profile: str | None = Field(
        default=None, alias="executionProfile", max_length=64
    )
    is_vip: bool = Field(default=False, alias="isVip")
    image: str | None = Field(default=None, max_length=512)
    command: list[str] = Field(min_length=1, max_length=64)
    args: list[str] = Field(default_factory=list, max_length=256)
    shard_count: int = Field(default=1, alias="shardCount", ge=1, le=5000)
    parallelism: int = Field(default=1, ge=1, le=64)
    cpu: int = Field(default=1, ge=1, le=32)
    memory_gib: int = Field(default=4, alias="memoryGiB", ge=1, le=128)
    ephemeral_storage_gib: int = Field(
        default=2, alias="ephemeralStorageGiB", ge=1, le=20
    )
    job_backoff_limit: int = Field(
        default=6, alias="jobBackoffLimit", ge=0, le=20
    )
    active_deadline_seconds: int = Field(
        default=21600, alias="activeDeadlineSeconds", ge=1, le=172800
    )
    description: str | None = Field(default=None, max_length=1024)
    env: dict[str, str] = Field(default_factory=dict, max_length=32)

    @model_validator(mode="after")
    def validate_cpu_job(self) -> "CpuJobSubmitRequest":
        if self.is_vip:
            raise ValueError(
                "VIP CPU jobs are not accepted by this plugin because stable-path "
                "visibility cannot be verified"
            )
        if self.execution_profile is not None and not self.execution_profile.strip():
            raise ValueError("executionProfile must be omitted or non-blank")
        if self.image is not None and not self.image.strip():
            raise ValueError("image must be omitted or non-blank")
        if self.execution_profile is None and self.image is None:
            raise ValueError("executionProfile or image is required")
        if self.parallelism > self.shard_count:
            raise ValueError("parallelism must not exceed shardCount")
        command_bytes = 0
        for field_name, values in (("command", self.command), ("args", self.args)):
            for value in values:
                if not isinstance(value, str) or not value.strip() or "\x00" in value:
                    raise ValueError(
                        f"{field_name} items must be non-empty strings without NUL"
                    )
                if len(value) > 8192:
                    raise ValueError(f"{field_name} item exceeds 8192 characters")
                command_bytes += len(value.encode("utf-8"))
        if command_bytes > 32768:
            raise ValueError("command and args exceed 32768 UTF-8 bytes")
        env_bytes = 0
        for key, value in self.env.items():
            if CPU_JOB_ENV_NAME_PATTERN.fullmatch(key) is None:
                raise ValueError(f"invalid CPU Job environment variable: {key}")
            if key in CPU_JOB_RESERVED_ENV:
                raise ValueError(f"reserved CPU Job environment variable: {key}")
            if CPU_JOB_SECRET_ENV_PATTERN.search(key):
                raise ValueError(f"Sensitive field is forbidden in cpuJob.env: {key}")
            if not isinstance(value, str) or "\x00" in value:
                raise ValueError("CPU Job environment values must be strings without NUL")
            env_bytes += len(key.encode("utf-8")) + len(value.encode("utf-8"))
        if env_bytes > 8192:
            raise ValueError("CPU Job environment exceeds 8192 UTF-8 bytes")
        _reject_forbidden_keys({"env": self.env}, path="cpuJob")
        return self


class PendingRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    message: str = Field(default="", max_length=200_000)
    parts: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    model: str | None = Field(default=None, max_length=256)
    working_directory: str | None = Field(
        default=None, alias="workingDirectory", max_length=1024
    )
    attachment_refs: list[Any] = Field(
        default_factory=list, alias="attachmentRefs", max_length=100
    )

    @model_validator(mode="after")
    def reject_secrets(self) -> "PendingRequest":
        _reject_forbidden_keys(self.model_dump(by_alias=True, exclude_none=True))
        return self


class EnsureResourceRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    request_id: str = Field(alias="requestId", min_length=1, max_length=128)
    reason: str | None = Field(default=None, max_length=512)
    project_id: int = Field(alias="projectId", gt=0)
    session_id: str = Field(alias="sessionId", min_length=1, max_length=128)
    pending_request: PendingRequest = Field(alias="pendingRequest")
    resource: ResourceSpec


class PortalEnvelope(BaseModel):
    code: str
    msg: str | None = None
    data: Any = None
    traceId: str | None = None


def _reject_forbidden_keys(value: Any, path: str = "pendingRequest") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).replace("-", "").lower()
            normalized_underscore = str(key).lower()
            if normalized in {k.replace("_", "") for k in FORBIDDEN_KEYS} or normalized_underscore in FORBIDDEN_KEYS:
                raise ValueError(f"Sensitive field is forbidden in {path}: {key}")
            _reject_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, f"{path}[{index}]")
