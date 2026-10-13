from __future__ import annotations

from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from config import Settings
from handoff_manager import HandoffRecord, inspect_artifacts, write_handoff
from gpu_aliases import public_gpu_type, to_backend_gpu_type, to_public
from k8s_client import K8sPortalClient
from k8s_models import SubmitK8sJobRequest
from models import EnsureResourceRequest, PendingRequest, ResourceSpec
from portal_client import PortalClient
from resource_inspector import inspect_current_resources as inspect_local_resources


mcp = FastMCP("res")
_settings: Settings | None = None
_client: PortalClient | None = None
_k8s_client: K8sPortalClient | None = None


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


def k8s_client() -> K8sPortalClient:
    global _k8s_client
    if _k8s_client is None:
        _k8s_client = K8sPortalClient(settings())
    return _k8s_client


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


# ---------------------------------------------------------------------------
# K8s CPU distributed-job MCP tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_k8s_resources() -> dict[str, Any]:
    """Query the CPU job cluster's per-node available CPU cores and memory.

    Returns allocatable minus Pod requests for each schedulable node. This is
    an estimate of requestable capacity, not real-time usage. Kubernetes
    scheduling also considers taints, affinity, and quotas.

    Use this before choosing per-worker CPU/memory, shardCount, and parallelism
    for a distributed CPU job.
    """
    result = await k8s_client().get_resources()
    return result


@mcp.tool()
async def submit_k8s_job(
    request_id: str,
    image: str,
    command: list[str] | None = None,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    shard_count: int = 1,
    parallelism: int = 1,
    cpu: int = 1,
    memory_gib: int = 4,
    ephemeral_storage_gib: int | None = None,
    job_backoff_limit: int = 6,
    active_deadline_seconds: int = 21600,
    description: str | None = None,
    execution_profile: str | None = None,
    is_vip: bool | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Submit an idempotent distributed CPU job to the K8s cluster.

    The job runs as a Kubernetes Indexed Job. Each worker gets the same env and
    can distinguish shards via the JOB_COMPLETION_INDEX environment variable
    (0-indexed). Inputs and outputs must be on the shared NAS mount visible at
    /mnt/nas inside the container.

    request_id: Idempotency key (max 64 chars). On timeout, retry with the same
    request_id and identical parameters; do not generate a new one.
    image: Container image. Must be pullable by the K8s cluster.
    command/args: Container entrypoint. Use ['/bin/sh', '-c'] for shell syntax.
    env: Environment variables for all workers. Keys must match
    [A-Za-z_][A-Za-z0-9_]{0,63}. JOB_ID, POD_UID, JOB_COMPLETION_INDEX are
    reserved. Values must be strings. Max 32 items, 8192 bytes total.
    shard_count: Total shards (1-5000). Each shard index runs exactly once.
    parallelism: Max concurrent workers (1-64, must not exceed shard_count).
    cpu: CPU cores per worker (integer, request == limit).
    memory_gib: Memory GiB per worker (integer, request == limit).
    ephemeral_storage_gib: /tmp emptyDir size limit per worker (default 2 GiB).
    job_backoff_limit: Total failure retry budget for the entire job (0-20).
    active_deadline_seconds: Max queue+run time in seconds (1-172800).
    description: Optional human-readable note (max 1024 chars).
    execution_profile: Optional platform preset. Must match a configured profile.
    is_vip: True to use the VIP pre-built PVC root. Only for whitelisted users.
    session_id: Injected by the PreToolUse hook. Do not supply yourself.

    Returns ACCEPTED on success (not yet running). Poll get_k8s_job_details for
    progress. On timeout, retry with the same request_id.
    """
    request = SubmitK8sJobRequest(
        requestId=request_id,
        sessionId=session_id,
        executionProfile=execution_profile,
        isVip=is_vip,
        image=image,
        command=command,
        args=args,
        shardCount=shard_count,
        parallelism=parallelism,
        cpu=cpu,
        memoryGiB=memory_gib,
        ephemeralStorageGiB=ephemeral_storage_gib,
        jobBackoffLimit=job_backoff_limit,
        activeDeadlineSeconds=active_deadline_seconds,
        description=description,
        env=env or {},
    )
    result = await k8s_client().submit_job(request)
    result["submittedRequestId"] = request_id
    result["requestIdMatches"] = result.get("jobId") is not None
    return result


@mcp.tool()
async def get_k8s_job_details(job_id: str) -> dict[str, Any]:
    """Query the latest observed status and progress of a K8s CPU job.

    Returns a TaskView with state, succeededIndexesCount/total, runningPods,
    effectiveSpec, and timing. This reads persisted observations; it does not
    synchronously fetch live K8s state. The reconcile loop updates roughly every
    15 seconds.

    Use this to poll after submit_k8s_job. A task is fully done when state is
    SUCCEEDED/FAILED/CANCELLED and resourcesReleased is true.
    """
    if not job_id or not job_id.strip():
        raise ValueError("job_id must be non-empty")
    return await k8s_client().get_job_details(job_id.strip())


@mcp.tool()
async def list_k8s_jobs(
    cursor: str | None = None,
    request_id: str | None = None,
    states: list[str] | None = None,
    page_size: int = 50,
) -> dict[str, Any]:
    """List K8s CPU jobs with optional filtering and cursor pagination.

    cursor: Opaque pagination token from a previous response's nextCursor.
    request_id: Filter by exact requestId to recover a task after a lost submit.
    states: Filter by state, up to 8 values (e.g. ["RUNNING", "QUEUED"]).
    page_size: Items per page, 1-200, default 50.

    Returns items (TaskView[]) sorted by jobId ascending, and nextCursor (null
    when no more pages). Items do not include submit-only replayed or
    cancel-only cancelApplied fields.
    """
    return await k8s_client().list_jobs(
        cursor=cursor,
        request_id=request_id,
        states=states,
        page_size=page_size,
    )


@mcp.tool()
async def get_k8s_job_attempts(
    job_id: str,
    shard_index: int | None = None,
    cursor: str | None = None,
    page_size: int = 50,
) -> dict[str, Any]:
    """Query per-shard execution attempts for a K8s CPU job.

    job_id: The job ID from submit_k8s_job.
    shard_index: Optional filter for a specific shard (0 to total-1).
    cursor: Opaque pagination token from a previous response's nextCursor.
    page_size: Items per page, 1-200, default 50.

    Returns items with podUid (use as attemptId for logs), shardIndex, phase,
    exitCode, timing, and terminal flag. Retry or rescheduling may produce
    multiple attempts per shard.
    """
    if not job_id or not job_id.strip():
        raise ValueError("job_id must be non-empty")
    return await k8s_client().get_job_attempts(
        job_id.strip(),
        shard_index=shard_index,
        cursor=cursor,
        page_size=page_size,
    )


@mcp.tool()
async def get_k8s_job_logs(
    job_id: str,
    shard_index: int,
    attempt_id: str | None = None,
    tail_lines: int = 200,
) -> dict[str, Any]:
    """Query bounded log text for a specific shard attempt.

    job_id: The job ID from submit_k8s_job.
    shard_index: The shard index (0 to total-1).
    attempt_id: The podUid from get_k8s_job_attempts. When omitted, the latest
    attempt is selected by pod creation time; for old records this may not be
    the truly latest, so pass attempt_id explicitly when precision matters.
    tail_lines: Number of trailing log lines, 1-2000, default 200.

    Returns text (log content), truncated (whether Portal's 256 KiB fetch limit
    was hit), attemptId, and shardIndex.
    """
    if not job_id or not job_id.strip():
        raise ValueError("job_id must be non-empty")
    return await k8s_client().get_job_logs(
        job_id.strip(),
        shard_index=shard_index,
        attempt_id=attempt_id,
        tail_lines=tail_lines,
    )


@mcp.tool()
async def cancel_k8s_job(job_id: str) -> dict[str, Any]:
    """Record cancellation intent for a K8s CPU job.

    Returns the task view with cancelApplied=true, meaning the desired state is
    now CANCELLED. This does NOT mean Pods have exited. Continue polling
    get_k8s_job_details until state is CANCELLED and resourcesReleased is true.

    SUCCEEDED/FAILED tasks with resourcesReleased=true will not accept another
    cancel. A task with state FAILED but resourcesReleased=false still accepts
    cancel to drain remaining execution.
    """
    if not job_id or not job_id.strip():
        raise ValueError("job_id must be non-empty")
    return await k8s_client().cancel_job(job_id.strip())


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
