from __future__ import annotations

import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from config import Settings
from cpu_job_preflight import (
    load_verified_preflight,
    prepare_cpu_job as prepare_cpu_job_files,
)
from handoff_manager import HandoffRecord, inspect_artifacts, write_handoff
from gpu_aliases import public_gpu_type, to_backend_gpu_type, to_public
from models import (
    CpuJobSubmitRequest,
    EnsureResourceRequest,
    PendingRequest,
    ResourceSpec,
)
from portal_client import PortalClient
from resource_inspector import inspect_current_resources as inspect_local_resources


mcp = FastMCP("res")
_settings: Settings | None = None
_client: PortalClient | None = None


def settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def portal_client() -> PortalClient:
    global _client
    if _client is None:
        _client = PortalClient(settings())
    return _client


async def _require_available_gpu_cluster(gpu_type: str) -> None:
    result = await portal_client().get_available_clusters()
    clusters = result.get("clusters")
    if not isinstance(clusters, list):
        raise ValueError("Portal did not return a valid available-cluster list")
    cluster_names = (
        value if isinstance(value, str) else value.get("cluster")
        for value in clusters
        if isinstance(value, (str, dict))
    )
    available = {
        public_name.lower()
        for value in cluster_names
        if isinstance(value, str)
        for public_name in [public_gpu_type(value)]
        if public_name is not None
    }
    requested = public_gpu_type(gpu_type)
    if requested is None or requested.lower() not in available:
        raise ValueError(
            f"{requested or 'Requested GPU cluster'} is unavailable; "
            "the request was not submitted and must not be queued or waited for"
        )


@mcp.tool()
async def get_resource_status() -> dict[str, Any]:
    """Return Portal state plus the server-side handoff-message setting.

    `provisioning` is authoritative. The result also includes route switching and
    pending-prompt delivery fields when supplied by Portal.
    """
    result = await portal_client().get_current_provisioning()
    result["handoffEnabled"] = settings().enable_handoff
    return to_public(result)


@mcp.tool()
async def get_available_clusters(required_gpu_count: int) -> dict[str, Any]:
    """Return sanitized cluster capacity for the requested GPU count.

    Call this after classifying the workload and before selecting a GPU cluster.
    Exact remaining-card counts and backend resource specifications are never
    exposed. Current responses become capacityKnown/capacityAvailable/
    capacitySufficient booleans; legacy name-only responses have unknown
    capacity.
    """
    if (
        not isinstance(required_gpu_count, int)
        or isinstance(required_gpu_count, bool)
        or required_gpu_count < 1
        or required_gpu_count > 8
    ):
        raise ValueError("required_gpu_count must be an integer from 1 to 8")

    result = await portal_client().get_available_clusters()
    clusters = result.get("clusters")
    if not isinstance(clusters, list):
        raise ValueError("Portal did not return a valid available-cluster list")
    sanitized = []
    for entry in clusters:
        if isinstance(entry, str):
            sanitized.append(
                {
                    "cluster": public_gpu_type(entry),
                    "capacityKnown": False,
                    "capacityAvailable": None,
                    "capacitySufficient": None,
                }
            )
            continue
        if not isinstance(entry, dict):
            continue
        cluster = entry.get("cluster")
        remaining = entry.get("remainCardNum")
        if not isinstance(cluster, str) or not isinstance(remaining, str):
            continue
        sanitized.append(
            {
                "cluster": public_gpu_type(cluster),
                "capacityKnown": True,
                "capacityAvailable": int(remaining) > 0,
                "capacitySufficient": int(remaining) >= required_gpu_count,
            }
        )
    return {"clusters": sanitized, "traceId": result.get("traceId")}


@mcp.tool()
async def ensure_resource(
    request_id: str,
    resource_type: Literal["CPU", "GPU"],
    cpu: int,
    memory_gib: int,
    gpu_count: int,
    pending_message: str = "",
    reason: str | None = None,
    gpu_type: str | None = None,
    worker_num: int = 1,
    parts: list[dict[str, Any]] | None = None,
    model: str | None = None,
    working_directory: str | None = None,
    attachment_refs: list[Any] | None = None,
    session_id: str = "",
) -> dict[str, Any]:
    """Submit an idempotent Portal resource request with resumable task context.

    project_id is read from PORTAL_PROJECT_ID and is not a tool argument. Do not
    supply session_id yourself. The plugin's PreToolUse hook injects the current
    Claude Code session ID and missing trusted context fails validation closed.
    ENABLE_HANDOFF defaults to false; when false, pending_message may be omitted
    and the server sends pendingRequest.message as an empty string. When true,
    pending_message must be non-empty.

    Use gpu_type='GPU-32G' or 'GPU-40G' for the corresponding CUDA cluster, or
    'GPU-96G' for the domestic-accelerator cluster. On a network timeout, call
    again with exactly the same request ID and pending request content.
    """
    current_settings = settings()
    if current_settings.enable_handoff and not pending_message.strip():
        raise ValueError(
            "pending_message must be non-empty when ENABLE_HANDOFF is enabled"
        )
    effective_pending_message = (
        pending_message if current_settings.enable_handoff else ""
    )
    request = EnsureResourceRequest(
        requestId=request_id,
        projectId=current_settings.portal_project_id,
        sessionId=session_id,
        reason=reason,
        resource=ResourceSpec(
            resourceType=resource_type,
            cpu=cpu,
            memoryGiB=memory_gib,
            gpuType=to_backend_gpu_type(gpu_type) if resource_type == "GPU" else None,
            gpuCount=gpu_count,
            workerNum=worker_num,
        ),
        pendingRequest=PendingRequest(
            message=effective_pending_message,
            parts=parts or [],
            model=model,
            workingDirectory=working_directory,
            attachmentRefs=attachment_refs or [],
        ),
    )
    if request.resource.resource_type == "GPU":
        await _require_available_gpu_cluster(request.resource.gpu_type or "")
    result = await portal_client().ensure_resource(request)
    result["submittedRequestId"] = request_id
    result["requestIdMatches"] = result.get("requestId") == request_id
    return to_public(result)


_CPU_JOB_STATES = {
    "ACCEPTED", "SUBMITTING", "QUEUED", "RUNNING", "RETRYING", "FAILING",
    "CANCELLING", "SUCCEEDED", "FAILED", "CANCELLED", "UNKNOWN",
}
def _validate_cpu_job_id(job_id: str) -> str:
    if (
        not isinstance(job_id, str)
        or not job_id.strip()
        or len(job_id) > 512
        or any(ord(character) < 32 for character in job_id)
    ):
        raise ValueError("job_id must be a non-empty opaque CPU Job identifier")
    return job_id


def _validate_cpu_job_env_paths(
    env: dict[str, str],
    *,
    current_settings: Settings,
    env_path_keys: list[str],
) -> None:
    root = current_settings.stable_workspace_root.resolve(strict=True)
    mount = PurePosixPath(str(root))
    portal_mount = PurePosixPath("/mnt/nas")
    for key in env_path_keys:
        value = env.get(key)
        if not isinstance(value, str):
            raise ValueError(f"prepared path environment variable is missing: {key}")
        job_path = PurePosixPath(value)
        is_workspace_path = job_path == mount or mount in job_path.parents
        is_portal_storage_path = portal_mount in job_path.parents
        if (
            not job_path.is_absolute()
            or ".." in job_path.parts
            or not (is_workspace_path or is_portal_storage_path)
        ):
            raise ValueError(
                f"CPU Job path environment variable {key} must be under "
                f"{mount} or /mnt/nas"
            )
        if is_portal_storage_path:
            # Portal decides whether the K8s-only storage is mounted. It cannot
            # be checked for existence or permissions from this container.
            continue
        resolved = Path(value).resolve(strict=False)
        if root not in (resolved, *resolved.parents):
            raise ValueError(
                f"CPU Job path environment variable {key} escapes stable storage"
            )


def _persist_cpu_job_submission(
    *,
    current_settings: Settings,
    request: CpuJobSubmitRequest,
    result: dict[str, Any] | None = None,
) -> None:
    directory = (
        current_settings.stable_workspace_root
        / ".compute-orchestrator"
        / "cpu-jobs"
        / request.request_id
    )
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    request_body = request.model_dump(by_alias=True, exclude_none=True)
    target = directory / "job.json"
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("saved CPU Job request record is unreadable") from exc
        if existing.get("request") != request_body:
            raise ValueError(
                "request_id already has a different CPU Job request; "
                "retry only with the exact original request"
            )
    record = {
        "requestId": request.request_id,
        "request": request_body,
        "response": None if result is None else {
            "jobId": result.get("jobId"),
            "state": result.get("state"),
            "traceId": result.get("traceId"),
            "replayed": result.get("replayed"),
        },
    }
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)


@mcp.tool()
def prepare_cpu_job(
    request_id: str,
    command: list[str],
    shard_count: int,
    script_paths: list[str] | None = None,
    input_paths: list[str] | None = None,
    output_paths: list[str] | None = None,
    env: dict[str, str] | None = None,
    env_path_keys: list[str] | None = None,
    required_env_keys: list[str] | None = None,
    optional_env_keys: list[str] | None = None,
    required_python_packages: list[str] | None = None,
    independent_shards_confirmed: bool = False,
    uses_native_completion_index: bool = False,
    worker_index_env: str | None = None,
    worker_index_argument: str | None = None,
    worker_index_base: int = 0,
) -> dict[str, Any]:
    """Validate job paths and create a derived K8s worker-index launcher.

    The model must confirm independent shards, explicitly declare whether user
    code consumes JOB_COMPLETION_INDEX natively or map it to the user's actual
    environment variable/argument, classify all consumed environment variables
    as required or optional, and list all scripts/configuration and business
    input/output paths. The model owns their semantic classification and access
    checks; the plugin enforces only stable-workspace path boundaries.
    Explicit /mnt/nas business paths are recorded as K8s-only and locally
    unverified because Portal determines VIP mounting. The plugin creates a
    launcher without modifying user source. The model must declare required
    Python distribution names; preparation verifies their metadata directly in
    the shared .venv rather than accepting an import resolved from system paths.
    K8s Python commands run through
    /opt/conda/bin/python3 and load the shared .venv site-packages via a Pod-local
    sitecustomize bootstrap; the shared virtual environment is never modified.
    """
    current_settings = settings()
    return prepare_cpu_job_files(
        stable_root=current_settings.stable_workspace_root,
        mount_path=str(current_settings.stable_workspace_root.parent),
        request_id=request_id,
        command=command,
        shard_count=shard_count,
        script_paths=script_paths or [],
        input_paths=input_paths or [],
        output_paths=output_paths or [],
        env=env or {},
        env_path_keys=env_path_keys or [],
        required_env_keys=required_env_keys or [],
        optional_env_keys=optional_env_keys or [],
        required_python_packages=(
            [] if required_python_packages is None else required_python_packages
        ),
        independent_shards_confirmed=independent_shards_confirmed,
        uses_native_completion_index=uses_native_completion_index,
        worker_index_env=worker_index_env,
        worker_index_argument=worker_index_argument,
        worker_index_base=worker_index_base,
    )


@mcp.tool()
async def estimate_cpu_job_capacity(
    cpu_per_worker: int,
    memory_gib_per_worker: int,
    parallelism: int,
) -> dict[str, Any]:
    """Return sanitized immediate-capacity booleans for a proposed CPU Job."""
    if not 1 <= cpu_per_worker <= 32:
        raise ValueError("cpu_per_worker must be from 1 to 32")
    if not 1 <= memory_gib_per_worker <= 128:
        raise ValueError("memory_gib_per_worker must be from 1 to 128")
    if not 1 <= parallelism <= 64:
        raise ValueError("parallelism must be from 1 to 64")
    result = await portal_client().get_cpu_job_resources()
    nodes = result.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("Portal did not return a valid CPU Job resource snapshot")
    slots = 0
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("Portal returned an invalid CPU Job node entry")
        if not (node.get("ready") is True and node.get("schedulable") is True):
            continue
        available_cpu = node.get("availableCpuCores")
        available_memory = node.get("availableMemoryBytes")
        if not isinstance(available_cpu, (int, float)) or isinstance(available_cpu, bool):
            raise ValueError("Portal returned invalid available CPU capacity")
        if not isinstance(available_memory, (int, float)) or isinstance(available_memory, bool):
            raise ValueError("Portal returned invalid available memory capacity")
        cpu_slots = math.floor(max(0.0, float(available_cpu)) / cpu_per_worker)
        memory_slots = math.floor(
            max(0.0, float(available_memory))
            / (memory_gib_per_worker * 1024**3)
        )
        slots += min(cpu_slots, memory_slots)
    return {
        "capacityKnown": True,
        "capacityAvailable": slots > 0,
        "capacitySufficient": slots >= parallelism,
        "requestedParallelism": parallelism,
        "observedAt": result.get("observedAt"),
        "traceId": result.get("traceId"),
        "advisoryOnly": True,
    }


@mcp.tool()
async def submit_cpu_job(
    request_id: str,
    shard_count: int,
    parallelism: int,
    cpu_per_worker: int,
    memory_gib_per_worker: int,
    image: str | None = None,
    ephemeral_storage_gib: int = 2,
    job_backoff_limit: int = 6,
    active_deadline_seconds: int = 21600,
    description: str | None = None,
    session_id: str = "",
) -> dict[str, Any]:
    """Submit a prepared, confirmed, idempotent K8s Indexed CPU Job.

    Call prepare_cpu_job first, then estimate_cpu_job_capacity, then use the
    fixed AskUserQuestion contract in the Skill. Environment variables come only
    from the verified preflight manifest and cannot be added at submission time.
    The trusted PreToolUse hook injects session_id. This tool never switches the
    Claude Code container.
    """
    current_settings = settings()
    manifest = load_verified_preflight(
        stable_root=current_settings.stable_workspace_root,
        request_id=request_id,
        shard_count=shard_count,
    )
    effective_env = manifest.get("env")
    if not isinstance(effective_env, dict):
        raise ValueError("prepared CPU Job environment is missing or invalid")
    env_path_keys = manifest.get("envPathKeys")
    if not isinstance(env_path_keys, list) or not all(
        isinstance(value, str) for value in env_path_keys
    ):
        raise ValueError("prepared CPU Job environment path keys are missing or invalid")
    _validate_cpu_job_env_paths(
        effective_env,
        current_settings=current_settings,
        env_path_keys=env_path_keys,
    )
    request = CpuJobSubmitRequest(
        requestId=request_id,
        sessionId=session_id,
        image=image,
        command=manifest["command"],
        args=manifest["args"],
        shardCount=shard_count,
        parallelism=parallelism,
        cpu=cpu_per_worker,
        memoryGiB=memory_gib_per_worker,
        ephemeralStorageGiB=ephemeral_storage_gib,
        jobBackoffLimit=job_backoff_limit,
        activeDeadlineSeconds=active_deadline_seconds,
        description=description,
        env=effective_env,
    )
    _persist_cpu_job_submission(
        current_settings=current_settings, request=request
    )
    result = await portal_client().submit_cpu_job(request)
    if not isinstance(result.get("jobId"), str) or not result["jobId"].strip():
        raise ValueError("Portal returned an invalid CPU Job submission response")
    if not isinstance(result.get("state"), str):
        raise ValueError("Portal returned an invalid CPU Job submission state")
    _persist_cpu_job_submission(
        current_settings=current_settings, request=request, result=result
    )
    return to_public(result)


@mcp.tool()
async def get_cpu_job(job_id: str) -> dict[str, Any]:
    """Return Portal's latest persisted CPU Job state."""
    return to_public(await portal_client().get_cpu_job(_validate_cpu_job_id(job_id)))


@mcp.tool()
async def list_cpu_jobs(
    request_id: str | None = None,
    states: list[str] | None = None,
    cursor: str | None = None,
    page_size: int = 50,
) -> dict[str, Any]:
    """List CPU Jobs using request/state filters and opaque cursor pagination."""
    if request_id is not None and (
        not request_id.strip() or not 1 <= len(request_id) <= 64
    ):
        raise ValueError("request_id must be from 1 to 64 characters")
    if states is not None:
        if len(states) > 8 or any(state not in _CPU_JOB_STATES for state in states):
            raise ValueError("states must contain at most 8 recognized CPU Job states")
    if cursor is not None and not cursor:
        raise ValueError("cursor must be omitted or non-empty")
    if not 1 <= page_size <= 200:
        raise ValueError("page_size must be from 1 to 200")
    return to_public(
        await portal_client().list_cpu_jobs(
            request_id=request_id,
            states=states,
            cursor=cursor,
            page_size=page_size,
        )
    )


@mcp.tool()
async def cancel_cpu_job(job_id: str) -> dict[str, Any]:
    """Record cancellation intent; poll until a terminal state releases resources."""
    return to_public(await portal_client().cancel_cpu_job(_validate_cpu_job_id(job_id)))


@mcp.tool()
async def get_cpu_job_attempts(
    job_id: str,
    shard_index: int | None = None,
    cursor: str | None = None,
    page_size: int = 50,
) -> dict[str, Any]:
    """Return paginated Pod attempts, optionally for one shard."""
    validated_job_id = _validate_cpu_job_id(job_id)
    if shard_index is not None and shard_index < 0:
        raise ValueError("shard_index must be non-negative")
    if cursor is not None and not cursor:
        raise ValueError("cursor must be omitted or non-empty")
    if not 1 <= page_size <= 200:
        raise ValueError("page_size must be from 1 to 200")
    return to_public(
        await portal_client().get_cpu_job_attempts(
            validated_job_id,
            shard_index=shard_index,
            cursor=cursor,
            page_size=page_size,
        )
    )


@mcp.tool()
async def get_cpu_job_logs(
    job_id: str,
    shard_index: int,
    attempt_id: str | None = None,
    tail_lines: int = 200,
) -> dict[str, Any]:
    """Return bounded logs for one CPU Job shard attempt."""
    validated_job_id = _validate_cpu_job_id(job_id)
    if shard_index < 0:
        raise ValueError("shard_index must be non-negative")
    if attempt_id is not None and not attempt_id.strip():
        raise ValueError("attempt_id must be omitted or non-empty")
    if not 1 <= tail_lines <= 2000:
        raise ValueError("tail_lines must be from 1 to 2000")
    return to_public(
        await portal_client().get_cpu_job_logs(
            validated_job_id,
            shard_index=shard_index,
            attempt_id=attempt_id,
            tail_lines=tail_lines,
        )
    )


@mcp.tool()
def inspect_current_resources() -> dict[str, Any]:
    """Inspect effective CPU, RAM, GPU, VRAM, CUDA visibility, and PyTorch state."""
    return to_public(inspect_local_resources())


@mcp.tool()
def persist_handoff(
    client_message_id: str,
    working_directory: str,
    objective: str,
    repository_path: str | None = None,
    repository_revision: str | None = None,
    completed_steps: list[str] | None = None,
    model_paths: list[str] | None = None,
    dataset_paths: list[str] | None = None,
    config_paths: list[str] | None = None,
    remaining_environment_steps: list[str] | None = None,
    next_command: str | None = None,
    expected_outputs: list[str] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Persist a handoff using the project ID configured for this MCP server."""
    record = HandoffRecord(
        projectId=settings().portal_project_id,
        clientMessageId=client_message_id,
        objective=objective,
        repositoryPath=repository_path,
        repositoryRevision=repository_revision,
        completedSteps=completed_steps or [],
        modelPaths=model_paths or [],
        datasetPaths=dataset_paths or [],
        configPaths=config_paths or [],
        remainingEnvironmentSteps=remaining_environment_steps or [],
        nextCommand=next_command,
        expectedOutputs=expected_outputs or [],
        notes=notes or [],
    )
    return write_handoff(
        working_directory=working_directory,
        stable_root=settings().stable_workspace_root,
        record=record,
    )


@mcp.tool()
def verify_persistent_artifacts(paths: list[str]) -> dict[str, Any]:
    """Verify that prepared files live under stable storage and are not partial."""
    return inspect_artifacts(paths, settings().stable_workspace_root)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
