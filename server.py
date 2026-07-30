from __future__ import annotations

from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from config import Settings
from handoff_manager import HandoffRecord, inspect_artifacts, write_handoff
from models import EnsureResourceRequest, PendingRequest, ResourceSpec
from portal_client import PortalClient
from resource_inspector import inspect_current_resources as inspect_local_resources


mcp = FastMCP("ai-scientist-resource")
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
    """Return the current Portal provisioning state and current/target resources.

    `provisioning` is authoritative. The result also includes route switching and
    pending-prompt delivery fields when supplied by Portal.
    """
    return await portal_client().get_current_provisioning()


@mcp.tool()
async def ensure_resource(
    request_id: str,
    project_id: int,
    client_message_id: str,
    pending_message: str,
    resource_type: Literal["CPU", "GPU"],
    cpu: int,
    memory_gib: int,
    gpu_count: int,
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

    Do not supply session_id yourself. The plugin's Claude Code PreToolUse hook
    injects the current session's real ID immediately before this tool executes.
    A missing hook leaves it empty and request validation fails closed.

    For physical V100 GPUs, pass gpu_type='Z1120'. On a network timeout, call
    again with exactly the same IDs and pending request content.
    """
    request = EnsureResourceRequest(
        requestId=request_id,
        projectId=project_id,
        sessionId=session_id,
        clientMessageId=client_message_id,
        reason=reason,
        resource=ResourceSpec(
            resourceType=resource_type,
            cpu=cpu,
            memoryGiB=memory_gib,
            gpuType=gpu_type,
            gpuCount=gpu_count,
            workerNum=worker_num,
        ),
        pendingRequest=PendingRequest(
            message=pending_message,
            parts=parts or [],
            model=model,
            workingDirectory=working_directory,
            attachmentRefs=attachment_refs or [],
        ),
    )
    result = await portal_client().ensure_resource(request)
    result["submittedRequestId"] = request_id
    result["requestIdMatches"] = result.get("requestId") == request_id
    return result


@mcp.tool()
def inspect_current_resources() -> dict[str, Any]:
    """Inspect effective CPU, RAM, GPU, VRAM, CUDA visibility, and PyTorch state."""
    return inspect_local_resources()


@mcp.tool()
def persist_handoff(
    project_id: int,
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
    """Atomically persist a machine-switch handoff record on stable storage."""
    record = HandoffRecord(
        projectId=project_id,
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
