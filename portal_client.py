from __future__ import annotations

from typing import Any

import httpx

from config import Settings
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
        self.message = message
        self.trace_id = trace_id
        self.http_status = http_status
        super().__init__(
            f"Portal API error code={code}, message={message!r}, "
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
        return {
            "Authorization": f"Bearer {self._settings.portal_access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def get_current_provisioning(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(
                f"{self._settings.portal_base_url}/scientist/deployment/provisioning/current",
                headers=self._headers(),
            )
        return self._parse(response)

    async def ensure_resource(
        self, request: EnsureResourceRequest
    ) -> dict[str, Any]:
        # On a transport timeout, the caller must retry this exact object so
        # requestId/clientMessageId/pendingRequest remain unchanged.
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._settings.portal_base_url}/scientist/deployment/ensure",
                headers=self._headers(),
                json=request.model_dump(by_alias=True, exclude_none=False),
            )
        return self._parse(response)

    @staticmethod
    def _parse(response: httpx.Response) -> dict[str, Any]:
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

        data = envelope.data or {}
        # Preserve traceId in successful tool results for diagnostics.
        return {**data, "traceId": envelope.traceId}
