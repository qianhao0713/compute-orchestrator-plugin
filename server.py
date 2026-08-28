from __future__ import annotations

from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from config import Settings
from handoff_manager import HandoffRecord, inspect_artifacts, write_handoff
from gpu_aliases import to_backend_gpu_type, to_public
from models import EnsureResourceRequest, PendingRequest, ResourceSpec
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
async def get_available_clusters() -> dict[str, Any]:
    """Return Portal cluster types currently enabled for resource requests.

    Call this after classifying the workload and before selecting a GPU cluster.
    An enabled cluster may still queue because this list is not live capacity.
    """
    return to_public(await portal_client().get_available_clusters())


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

    Use gpu_type='GPU-32G' for the 32 GiB CUDA cluster or 'GPU-96G' for the
    96 GiB domestic-accelerator cluster. On a network timeout, call again with
    exactly the same request ID and pending request content.
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


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
