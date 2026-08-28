from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from typing import Any

import httpx

from config import Settings
from gpu_aliases import to_public
from models import EnsureResourceRequest, PortalEnvelope


class PortalAPIError(RuntimeError):
    def __init__(
        self,
        *,
        code: str,
        message: str | None,
        trace_id: str | None,
        http_status: int | None = None,
    ) -> None:
        self.code = code
        self.message = to_public(message)
        self.trace_id = trace_id
        self.http_status = http_status
        super().__init__(
            f"Portal API error code={code}, message={self.message!r}, "
            f"traceId={trace_id!r}, httpStatus={http_status!r}"
        )


class PortalClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._timeout = httpx.Timeout(
            connect=settings.connect_timeout_seconds,
            read=settings.read_timeout_seconds,
            write=settings.write_timeout_seconds,
            pool=settings.pool_timeout_seconds,
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
        """Persist one sanitized Portal request before sending it."""
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
        log_path = log_dir / "portal_requests.jsonl"
        descriptor = os.open(
            log_path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)

    async def get_current_provisioning(self) -> dict[str, Any]:
        url = (
            f"{self._settings.portal_base_url}"
            "/scientist/deployment/provisioning/current"
        )
        headers = self._headers()
        self._log_request(method="GET", url=url, headers=headers)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, headers=headers)
        return self._parse(response)

    async def get_available_clusters(self) -> dict[str, Any]:
        url = (
            f"{self._settings.portal_base_url}"
            "/scientist/deployment/clusters/available"
        )
        headers = self._headers()
        self._log_request(method="GET", url=url, headers=headers)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, headers=headers)
        return self._parse_list(response, key="clusters")

    async def ensure_resource(
        self, request: EnsureResourceRequest
    ) -> dict[str, Any]:
        # On a transport timeout, retry this exact object so requestId and
        # pendingRequest remain unchanged.
        url = f"{self._settings.portal_base_url}/scientist/deployment/ensure"
        headers = self._headers()
        body = request.model_dump(by_alias=True, exclude_none=False)
        self._log_request(method="POST", url=url, headers=headers, body=body)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, headers=headers, json=body)
        return self._parse(response)

    @classmethod
    def _parse_list(cls, response: httpx.Response, *, key: str) -> dict[str, Any]:
        data, trace_id = cls._parse_envelope(response)
        if not isinstance(data, list):
            raise PortalAPIError(
                code="INVALID_DATA",
                message="Portal response data must be a list",
                trace_id=trace_id,
                http_status=response.status_code,
            )
        return {key: data, "traceId": trace_id}

    @staticmethod
    def _parse(response: httpx.Response) -> dict[str, Any]:
        data, trace_id = PortalClient._parse_envelope(response)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise PortalAPIError(
                code="INVALID_DATA",
                message="Portal response data must be an object",
                trace_id=trace_id,
                http_status=response.status_code,
            )
        return {**data, "traceId": trace_id}

    @staticmethod
    def _parse_envelope(response: httpx.Response) -> tuple[Any, str | None]:
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
