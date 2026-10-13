"""Authenticated Portal client for the K8s CPU distributed-job API.

Mirrors portal_client.py for audit logging and envelope parsing, but targets
the /scientist/project/{projectId}/jobs/* endpoint family. All seven endpoints
are covered: submit, detail, list, cancel, attempts, logs, and resources.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from typing import Any

import httpx

from config import Settings
from k8s_models import (
    K8sAttemptsData,
    K8sJobListData,
    K8sLogData,
    K8sResourcesData,
    SubmitK8sJobRequest,
    TaskView,
)
from portal_client import PortalAPIError


class K8sPortalClient:
    """HTTP client for the K8s CPU job Portal API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._timeout = httpx.Timeout(
            connect=settings.connect_timeout_seconds,
            read=settings.read_timeout_seconds,
            write=settings.write_timeout_seconds,
            pool=settings.pool_timeout_seconds,
        )

    def _base_path(self) -> str:
        return (
            f"{self._settings.portal_base_url}"
            f"/scientist/project/{self._settings.portal_project_id}/jobs"
        )

    def _headers(self) -> dict[str, str]:
        token = self._settings.portal_access_token.strip()
        authorization = (
            token if token.lower().startswith("bearer ") else f"Bearer {token}"
        )
        return {
            "Authorization": authorization,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _log_request(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        body: Any = None,
    ) -> None:
        """Persist one sanitized K8s Portal request before sending it."""
        log_dir = self._settings.plugin_log_dir
        log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": method.upper(),
            "url": url,
            "headers": {
                key: "[REDACTED]" if key.lower() == "authorization" else value
                for key, value in headers.items()
            },
            "body": body,
        }
        payload = (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode(
            "utf-8"
        )
        log_path = log_dir / "k8s_portal_requests.jsonl"
        descriptor = os.open(
            log_path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)

    async def get_resources(self) -> dict[str, Any]:
        """GET /resources — query cluster node available CPU and memory."""
        url = f"{self._base_path()}/resources"
        headers = self._headers()
        self._log_request(method="GET", url=url, headers=headers)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, headers=headers)
        data, _trace_id = _parse_envelope(response)
        if not isinstance(data, dict):
            raise PortalAPIError(
                code="INVALID_DATA",
                message="K8s resources response data must be an object",
                trace_id=_trace_id,
                http_status=response.status_code,
            )
        resources = K8sResourcesData.model_validate(data)
        result = resources.model_dump(by_alias=True)
        result["traceId"] = _trace_id
        return result

    async def submit_job(
        self, request: SubmitK8sJobRequest
    ) -> dict[str, Any]:
        """POST /submit — submit a distributed CPU job (idempotent)."""
        url = f"{self._base_path()}/submit"
        headers = self._headers()
        body = request.model_dump(by_alias=True, exclude_none=True)
        self._log_request(method="POST", url=url, headers=headers, body=body)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, headers=headers, json=body)
        data, trace_id = _parse_envelope(response)
        if not isinstance(data, dict):
            raise PortalAPIError(
                code="INVALID_DATA",
                message="K8s submit response data must be an object",
                trace_id=trace_id,
                http_status=response.status_code,
            )
        data["traceId"] = trace_id
        return data

    async def get_job_details(self, job_id: str) -> dict[str, Any]:
        """GET /{jobId} — query task detail and progress."""
        url = f"{self._base_path()}/{job_id}"
        headers = self._headers()
        self._log_request(method="GET", url=url, headers=headers)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, headers=headers)
        data, trace_id = _parse_envelope(response)
        if not isinstance(data, dict):
            raise PortalAPIError(
                code="INVALID_DATA",
                message="K8s job detail response data must be an object",
                trace_id=trace_id,
                http_status=response.status_code,
            )
        data["traceId"] = trace_id
        return data

    async def list_jobs(
        self,
        *,
        cursor: str | None = None,
        request_id: str | None = None,
        states: list[str] | None = None,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """GET / — list jobs with optional filtering and cursor pagination."""
        if page_size < 1 or page_size > 200:
            raise ValueError("pageSize must be between 1 and 200")
        if states is not None and len(states) > 8:
            raise ValueError("state filter must have at most 8 items")

        url = self._base_path()
        params: list[tuple[str, str]] = []
        if cursor is not None:
            params.append(("cursor", cursor))
        if request_id is not None:
            params.append(("requestId", request_id))
        if states:
            for state in states:
                params.append(("state", state))
        if page_size != 50:
            params.append(("pageSize", str(page_size)))

        headers = self._headers()
        self._log_request(method="GET", url=url, headers=headers)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(
                url, headers=headers, params=params if params else None
            )
        data, trace_id = _parse_envelope(response)
        if not isinstance(data, dict):
            raise PortalAPIError(
                code="INVALID_DATA",
                message="K8s job list response data must be an object",
                trace_id=trace_id,
                http_status=response.status_code,
            )
        data["traceId"] = trace_id
        return data

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        """POST /{jobId}/cancel — record cancellation intent."""
        url = f"{self._base_path()}/{job_id}/cancel"
        headers = self._headers()
        self._log_request(method="POST", url=url, headers=headers)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, headers=headers)
        data, trace_id = _parse_envelope(response)
        if not isinstance(data, dict):
            raise PortalAPIError(
                code="INVALID_DATA",
                message="K8s cancel response data must be an object",
                trace_id=trace_id,
                http_status=response.status_code,
            )
        data["traceId"] = trace_id
        return data

    async def get_job_attempts(
        self,
        job_id: str,
        *,
        shard_index: int | None = None,
        cursor: str | None = None,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """GET /{jobId}/attempts — query shard execution attempts."""
        if page_size < 1 or page_size > 200:
            raise ValueError("pageSize must be between 1 and 200")
        if shard_index is not None and shard_index < 0:
            raise ValueError("shardIndex must be non-negative")

        url = f"{self._base_path()}/{job_id}/attempts"
        params: list[tuple[str, str]] = []
        if shard_index is not None:
            params.append(("shardIndex", str(shard_index)))
        if cursor is not None:
            params.append(("cursor", cursor))
        if page_size != 50:
            params.append(("pageSize", str(page_size)))

        headers = self._headers()
        self._log_request(method="GET", url=url, headers=headers)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(
                url, headers=headers, params=params if params else None
            )
        data, trace_id = _parse_envelope(response)
        if not isinstance(data, dict):
            raise PortalAPIError(
                code="INVALID_DATA",
                message="K8s attempts response data must be an object",
                trace_id=trace_id,
                http_status=response.status_code,
            )
        data["traceId"] = trace_id
        return data

    async def get_job_logs(
        self,
        job_id: str,
        *,
        shard_index: int,
        attempt_id: str | None = None,
        tail_lines: int = 200,
    ) -> dict[str, Any]:
        """GET /{jobId}/logs — query bounded log text for a shard attempt."""
        if tail_lines < 1 or tail_lines > 2000:
            raise ValueError("tailLines must be between 1 and 2000")
        if shard_index < 0:
            raise ValueError("shardIndex must be non-negative")

        url = f"{self._base_path()}/{job_id}/logs"
        params: list[tuple[str, str]] = [
            ("shardIndex", str(shard_index)),
            ("tailLines", str(tail_lines)),
        ]
        if attempt_id is not None:
            params.append(("attemptId", attempt_id))

        headers = self._headers()
        self._log_request(method="GET", url=url, headers=headers)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, headers=headers, params=params)
        data, trace_id = _parse_envelope(response)
        if not isinstance(data, dict):
            raise PortalAPIError(
                code="INVALID_DATA",
                message="K8s logs response data must be an object",
                trace_id=trace_id,
                http_status=response.status_code,
            )
        data["traceId"] = trace_id
        return data


def _parse_envelope(response: httpx.Response) -> tuple[Any, str | None]:
    """Parse the unified Portal envelope for K8s endpoints."""
    try:
        raw = response.json()
    except ValueError as exc:
        raise PortalAPIError(
            code="INVALID_JSON",
            message="Portal returned a non-JSON response",
            trace_id=None,
            http_status=response.status_code,
        ) from exc

    try:
        from models import PortalEnvelope

        envelope = PortalEnvelope.model_validate(raw)
    except Exception as exc:
        raise PortalAPIError(
            code="INVALID_ENVELOPE",
            message="Portal response does not match the unified envelope",
            trace_id=raw.get("traceId") if isinstance(raw, dict) else None,
            http_status=response.status_code,
        ) from exc

    if response.is_error or envelope.code != "00000":
        raise PortalAPIError(
            code=envelope.code,
            message=envelope.msg,
            trace_id=envelope.traceId,
            http_status=response.status_code,
        )

    return envelope.data, envelope.traceId
