from pathlib import Path

import pytest

import server
from config import Settings
from handoff_manager import HandoffRecord, inspect_artifacts, write_handoff


def test_handoff_written_atomically_under_stable_root(tmp_path: Path):
    workdir = tmp_path / "project"
    result = write_handoff(
        working_directory=str(workdir),
        stable_root=tmp_path,
        record=HandoffRecord(
            projectId=1,
            clientMessageId="msg-1",
            objective="reproduce paper",
        ),
    )
    assert Path(result["handoffPath"]).exists()


def test_path_outside_stable_root_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        write_handoff(
            working_directory="/tmp/outside",
            stable_root=tmp_path,
            record=HandoffRecord(
                projectId=1,
                clientMessageId="msg-1",
                objective="reproduce paper",
            ),
        )


def test_partial_artifact_not_ready(tmp_path: Path):
    partial = tmp_path / "model.part"
    partial.write_text("partial", encoding="utf-8")
    result = inspect_artifacts([str(partial)], tmp_path)
    assert result["allReady"] is False


class _CapturingPortalClient:
    def __init__(self):
        self.request = None

    async def ensure_resource(self, request):
        self.request = request
        return {"requestId": request.request_id}

    async def get_current_provisioning(self):
        return {"provisioning": False}


def _settings(tmp_path: Path, *, enable_handoff: bool) -> Settings:
    return Settings(
        portal_base_url="https://portal.example.com",
        portal_access_token="token",
        portal_project_id=1,
        stable_workspace_root=tmp_path,
        plugin_log_dir=tmp_path / "logs",
        enable_handoff=enable_handoff,
    )


@pytest.mark.asyncio
async def test_ensure_forces_empty_pending_message_when_handoff_disabled(
    monkeypatch, tmp_path
):
    client = _CapturingPortalClient()
    monkeypatch.setattr(
        server, "settings", lambda: _settings(tmp_path, enable_handoff=False)
    )
    monkeypatch.setattr(server, "portal_client", lambda: client)

    await server.ensure_resource(
        request_id="request-1",
        pending_message="model-generated handoff",
        resource_type="CPU",
        cpu=1,
        memory_gib=2,
        gpu_count=0,
        session_id="session-1",
    )

    assert client.request.pending_request.message == ""


@pytest.mark.asyncio
async def test_ensure_preserves_pending_message_when_handoff_enabled(
    monkeypatch, tmp_path
):
    client = _CapturingPortalClient()
    monkeypatch.setattr(
        server, "settings", lambda: _settings(tmp_path, enable_handoff=True)
    )
    monkeypatch.setattr(server, "portal_client", lambda: client)

    await server.ensure_resource(
        request_id="request-1",
        pending_message="continue from prepared state",
        resource_type="CPU",
        cpu=1,
        memory_gib=2,
        gpu_count=0,
        session_id="session-1",
    )

    assert (
        client.request.pending_request.message
        == "continue from prepared state"
    )


@pytest.mark.asyncio
async def test_ensure_rejects_empty_message_when_handoff_enabled(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        server, "settings", lambda: _settings(tmp_path, enable_handoff=True)
    )
    with pytest.raises(ValueError, match="pending_message"):
        await server.ensure_resource(
            request_id="request-1",
            resource_type="CPU",
            cpu=1,
            memory_gib=2,
            gpu_count=0,
            session_id="session-1",
        )


@pytest.mark.asyncio
async def test_resource_status_exposes_handoff_setting(monkeypatch, tmp_path):
    client = _CapturingPortalClient()
    monkeypatch.setattr(
        server, "settings", lambda: _settings(tmp_path, enable_handoff=False)
    )
    monkeypatch.setattr(server, "portal_client", lambda: client)

    result = await server.get_resource_status()

    assert result["handoffEnabled"] is False
