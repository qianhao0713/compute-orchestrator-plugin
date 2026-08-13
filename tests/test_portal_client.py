import httpx
import pytest

from config import Settings
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
