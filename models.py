from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gpu_aliases import public_gpu_type


GPU_BACKEND_TYPES = {"Z1120", "V5000"}
Z1120_SPECS = {
    1: (8, 64),
    2: (16, 128),
    4: (32, 256),
    8: (64, 512),
}
V5000_SPECS = {
    1: (16, 112),
    2: (32, 225),
    4: (64, 450),
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
            fixed_specs = (
                Z1120_SPECS if self.gpu_type == "Z1120" else V5000_SPECS
            )
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
