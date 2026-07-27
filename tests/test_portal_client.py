import httpx
import pytest

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
