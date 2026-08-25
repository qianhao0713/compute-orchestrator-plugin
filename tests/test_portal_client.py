import json
from pathlib import Path

import httpx
import pytest
import respx

from config import Settings
from models import EnsureResourceRequest, PendingRequest, ResourceSpec
from portal_client import PortalAPIError, PortalClient


def test_parse_preserves_trace_id():
    response = httpx.Response(
        200,
        json={"code": "00000", "msg": None, "data": {"status": "IDLE"}, "traceId": "t1"},
    )
    result = PortalClient._parse(response)
    assert result == {"status": "IDLE", "traceId": "t1"}


def test_http_200_business_error_is_rejected():
    response = httpx.Response(
        200,
        json={"code": "0400", "msg": "bad request", "data": None, "traceId": "t2"},
    )
    with pytest.raises(PortalAPIError) as caught:
        PortalClient._parse(response)
    assert caught.value.trace_id == "t2"


def test_parse_available_cluster_list():
    response = httpx.Response(
        200,
        json={
            "code": "00000",
            "msg": None,
            "data": ["z1120", "v5000"],
            "traceId": "t3",
        },
    )
    result = PortalClient._parse_list(response, key="clusters")
    assert result == {"clusters": ["z1120", "v5000"], "traceId": "t3"}


def test_cluster_list_rejects_non_list_data():
    response = httpx.Response(
        200,
        json={"code": "00000", "msg": None, "data": {}, "traceId": "t4"},
    )
    with pytest.raises(PortalAPIError) as caught:
        PortalClient._parse_list(response, key="clusters")
    assert caught.value.code == "INVALID_DATA"


def _client(tmp_path):
    return PortalClient(
        Settings(
            portal_base_url="https://portal.example.com",
            portal_access_token="secret-token",
            portal_project_id=1,
            stable_workspace_root=tmp_path,
            plugin_log_dir=tmp_path / "logs",
        )
    )


def test_authorization_header_uses_bearer(tmp_path):
    client = PortalClient(
        Settings(
            portal_base_url="https://portal.example.com",
            portal_access_token="secret-token",
            portal_project_id=1,
            stable_workspace_root=tmp_path,
        )
    )
    assert client._headers()["Authorization"] == "Bearer secret-token"


def test_authorization_header_does_not_duplicate_bearer(tmp_path):
    client = PortalClient(
        Settings(
            portal_base_url="https://portal.example.com",
            portal_access_token="Bearer secret-token",
            portal_project_id=1,
            stable_workspace_root=tmp_path,
        )
    )
    assert client._headers()["Authorization"] == "Bearer secret-token"


def test_authorization_header_accepts_case_insensitive_scheme(tmp_path):
    client = PortalClient(
        Settings(
            portal_base_url="https://portal.example.com",
            portal_access_token="bearer secret-token",
            portal_project_id=1,
            stable_workspace_root=tmp_path,
        )
    )
    assert client._headers()["Authorization"] == "bearer secret-token"


@pytest.mark.asyncio
@respx.mock
async def test_every_portal_request_is_logged_before_sending(tmp_path):
    client = _client(tmp_path)
    current_url = (
        "https://portal.example.com/scientist/deployment/provisioning/current"
    )
    clusters_url = (
        "https://portal.example.com/scientist/deployment/clusters/available"
    )
    ensure_url = "https://portal.example.com/scientist/deployment/ensure"
    respx.get(current_url).mock(
        return_value=httpx.Response(
            200,
            json={
                "code": "00000",
                "data": {"status": "IDLE"},
                "traceId": "trace-current",
            },
        )
    )
    respx.get(clusters_url).mock(
        return_value=httpx.Response(
            200,
            json={
                "code": "00000",
                "data": ["z1120"],
                "traceId": "trace-clusters",
            },
        )
    )
    respx.post(ensure_url).mock(
        return_value=httpx.Response(
            200,
            json={
                "code": "00000",
                "data": {"requestId": "request-1"},
                "traceId": "trace-ensure",
            },
        )
    )
    request = EnsureResourceRequest(
        requestId="request-1",
        projectId=1,
        sessionId="session-1",
        pendingRequest=PendingRequest(message="continue"),
        resource=ResourceSpec(
            resourceType="CPU",
            cpu=1,
            memoryGiB=2,
            gpuType=None,
            gpuCount=0,
        ),
    )

    await client.get_current_provisioning()
    await client.get_available_clusters()
    await client.ensure_resource(request)

    log_path = tmp_path / "logs" / "portal_requests.jsonl"
    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["method"] for record in records] == ["GET", "GET", "POST"]
    assert [record["url"] for record in records] == [
        current_url,
        clusters_url,
        ensure_url,
    ]
    assert all(
        record["headers"]["Authorization"] == "[REDACTED]"
        for record in records
    )
    assert records[0]["body"] is None
    assert records[1]["body"] is None
    assert records[2]["body"]["requestId"] == "request-1"
    assert records[2]["body"]["sessionId"] == "session-1"
    assert records[2]["body"]["pendingRequest"]["message"] == "continue"
    assert log_path.stat().st_mode & 0o777 == 0o600


def test_plugin_log_dir_defaults_and_accepts_override(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTAL_BASE_URL", "portal.example.com")
    monkeypatch.setenv("PORTAL_ACCESS_TOKEN", "token")
    monkeypatch.setenv("PORTAL_PROJECT_ID", "1")
    monkeypatch.setenv(
        "AI_SCIENTIST_STABLE_WORKSPACE_ROOT", str(tmp_path / "project1")
    )
    monkeypatch.delenv("PLUGIN_LOG_DIR", raising=False)
    assert Settings.from_env().plugin_log_dir == Path(
        "/home/scientist/.claude/logs"
    )

    custom_log_dir = tmp_path / "custom-logs"
    monkeypatch.setenv("PLUGIN_LOG_DIR", str(custom_log_dir))
    assert Settings.from_env().plugin_log_dir == custom_log_dir.resolve()


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [("true", True), ("1", True), ("yes", True), ("on", True),
     ("false", False), ("0", False), ("no", False), ("off", False)],
)
def test_enable_handoff_parsing(monkeypatch, tmp_path, raw_value, expected):
    monkeypatch.setenv("PORTAL_BASE_URL", "portal.example.com")
    monkeypatch.setenv("PORTAL_ACCESS_TOKEN", "token")
    monkeypatch.setenv("PORTAL_PROJECT_ID", "1")
    monkeypatch.setenv(
        "AI_SCIENTIST_STABLE_WORKSPACE_ROOT", str(tmp_path / "project1")
    )
    monkeypatch.setenv("ENABLE_HANDOFF", raw_value)
    assert Settings.from_env().enable_handoff is expected


def test_enable_handoff_defaults_to_false(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTAL_BASE_URL", "portal.example.com")
    monkeypatch.setenv("PORTAL_ACCESS_TOKEN", "token")
    monkeypatch.setenv("PORTAL_PROJECT_ID", "1")
    monkeypatch.setenv(
        "AI_SCIENTIST_STABLE_WORKSPACE_ROOT", str(tmp_path / "project1")
    )
    monkeypatch.delenv("ENABLE_HANDOFF", raising=False)
    assert Settings.from_env().enable_handoff is False


def test_invalid_enable_handoff_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTAL_BASE_URL", "portal.example.com")
    monkeypatch.setenv("PORTAL_ACCESS_TOKEN", "token")
    monkeypatch.setenv("PORTAL_PROJECT_ID", "1")
    monkeypatch.setenv(
        "AI_SCIENTIST_STABLE_WORKSPACE_ROOT", str(tmp_path / "project1")
    )
    monkeypatch.setenv("ENABLE_HANDOFF", "sometimes")
    with pytest.raises(RuntimeError, match="ENABLE_HANDOFF"):
        Settings.from_env()


def test_empty_plugin_log_dir_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTAL_BASE_URL", "portal.example.com")
    monkeypatch.setenv("PORTAL_ACCESS_TOKEN", "token")
    monkeypatch.setenv("PORTAL_PROJECT_ID", "1")
    monkeypatch.setenv(
        "AI_SCIENTIST_STABLE_WORKSPACE_ROOT", str(tmp_path / "project1")
    )
    monkeypatch.setenv("PLUGIN_LOG_DIR", " ")
    with pytest.raises(RuntimeError, match="PLUGIN_LOG_DIR"):
        Settings.from_env()
