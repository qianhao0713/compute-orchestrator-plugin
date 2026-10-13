"""Tests for the K8s Portal client — envelope parsing, audit logging, errors."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from config import Settings
from k8s_client import K8sPortalClient, _parse_envelope
from k8s_models import SubmitK8sJobRequest
from portal_client import PortalAPIError


# ---------------------------------------------------------------------------
# Envelope parsing
# ---------------------------------------------------------------------------


class TestParseEnvelope:
    def test_success_parses_data_and_trace_id(self):
        response = httpx.Response(
            200,
            json={
                "code": "00000",
                "msg": None,
                "data": {"jobId": "cj-test", "state": "ACCEPTED"},
                "traceId": "trace-1",
            },
        )
        data, trace_id = _parse_envelope(response)
        assert data == {"jobId": "cj-test", "state": "ACCEPTED"}
        assert trace_id == "trace-1"

    def test_http_200_business_error_is_rejected(self):
        response = httpx.Response(
            200,
            json={
                "code": "REQUEST_CONFLICT",
                "msg": "requestId already used with different parameters",
                "data": None,
                "traceId": "trace-2",
            },
        )
        with pytest.raises(PortalAPIError) as caught:
            _parse_envelope(response)
        assert caught.value.code == "REQUEST_CONFLICT"
        assert caught.value.trace_id == "trace-2"

    def test_http_500_is_rejected(self):
        response = httpx.Response(
            500,
            json={
                "code": "0001",
                "msg": "internal error",
                "data": None,
                "traceId": "trace-3",
            },
        )
        with pytest.raises(PortalAPIError) as caught:
            _parse_envelope(response)
        assert caught.value.http_status == 500

    def test_non_json_response_raises_invalid_json(self):
        response = httpx.Response(200, text="not json")
        with pytest.raises(PortalAPIError) as caught:
            _parse_envelope(response)
        assert caught.value.code == "INVALID_JSON"

    def test_malformed_envelope_raises_invalid_envelope(self):
        response = httpx.Response(200, json={"no_code": True})
        with pytest.raises(PortalAPIError) as caught:
            _parse_envelope(response)
        assert caught.value.code == "INVALID_ENVELOPE"

    def test_http_400_validation_error(self):
        response = httpx.Response(
            400,
            json={
                "code": "0400",
                "msg": "bean validation failed",
                "data": None,
                "traceId": "trace-4",
            },
        )
        with pytest.raises(PortalAPIError) as caught:
            _parse_envelope(response)
        assert caught.value.code == "0400"
        assert caught.value.http_status == 400


# ---------------------------------------------------------------------------
# Client integration with mock HTTP
# ---------------------------------------------------------------------------


def _client(tmp_path: Path) -> K8sPortalClient:
    return K8sPortalClient(
        Settings(
            portal_base_url="https://portal.example.com",
            portal_access_token="secret-token",
            portal_project_id=1001,
            stable_workspace_root=tmp_path,
            plugin_log_dir=tmp_path / "logs",
        )
    )


BASE = "https://portal.example.com/scientist/project/1001/jobs"


class TestGetResources:
    @pytest.mark.asyncio
    @respx.mock
    async def test_get_resources_parses_nodes(self, tmp_path):
        client = _client(tmp_path)
        url = f"{BASE}/resources"
        respx.get(url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "code": "00000",
                    "data": {
                        "clusterId": "cpu-batch",
                        "calculation": "allocatable-minus-pod-requests",
                        "observedAt": "1791500000000",
                        "nodes": [
                            {
                                "nodeName": "worker-1",
                                "ready": True,
                                "schedulable": True,
                                "allocatableCpuCores": 32,
                                "requestedCpuCores": 10,
                                "availableCpuCores": 22,
                                "allocatableMemoryBytes": 137438953472,
                                "requestedMemoryBytes": 42949672960,
                                "availableMemoryBytes": 94489280512,
                            }
                        ],
                    },
                    "traceId": "t-res",
                },
            )
        )
        result = await client.get_resources()
        assert result["clusterId"] == "cpu-batch"
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["availableCpuCores"] == 22


class TestSubmitJob:
    @pytest.mark.asyncio
    @respx.mock
    async def test_submit_job_returns_task_view(self, tmp_path):
        client = _client(tmp_path)
        url = f"{BASE}/submit"
        respx.post(url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "code": "00000",
                    "data": {
                        "jobId": "cj-test-001",
                        "state": "ACCEPTED",
                        "total": 3,
                        "succeededIndexesCount": 0,
                        "activePods": 0,
                        "runningPods": 0,
                        "failedAttempts": 0,
                        "resourcesReleased": False,
                        "storageReleased": False,
                        "mountPath": "/mnt/nas",
                        "subPath": "data/users/user1",
                        "effectiveSpec": {
                            "cpu": 2,
                            "memoryGiB": 8,
                            "parallelism": 3,
                            "image": "registry.example/worker:v1",
                            "ephemeralStorageGiB": 2,
                        },
                        "replayed": False,
                    },
                    "traceId": "t-submit",
                },
            )
        )
        request = SubmitK8sJobRequest(
            requestId="compute-001",
            image="registry.example/worker:v1",
            command=["/bin/sh", "-c"],
            args=["echo hello"],
            shardCount=3,
            parallelism=3,
            cpu=2,
            memoryGiB=8,
        )
        result = await client.submit_job(request)
        assert result["jobId"] == "cj-test-001"
        assert result["state"] == "ACCEPTED"
        assert result["replayed"] is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_submit_logs_request_before_sending(self, tmp_path):
        client = _client(tmp_path)
        url = f"{BASE}/submit"
        respx.post(url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "code": "00000",
                    "data": {
                        "jobId": "cj-log",
                        "state": "ACCEPTED",
                        "total": 1,
                        "succeededIndexesCount": 0,
                        "activePods": 0,
                        "runningPods": 0,
                        "failedAttempts": 0,
                        "resourcesReleased": False,
                        "storageReleased": False,
                        "mountPath": "/mnt/nas",
                        "subPath": "",
                        "effectiveSpec": {
                            "cpu": 1,
                            "memoryGiB": 4,
                            "parallelism": 1,
                            "image": "img",
                            "ephemeralStorageGiB": 2,
                        },
                        "replayed": False,
                    },
                    "traceId": "t-log",
                },
            )
        )
        request = SubmitK8sJobRequest(
            requestId="log-req-1", image="img"
        )
        await client.submit_job(request)

        log_path = tmp_path / "logs" / "k8s_portal_requests.jsonl"
        records = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
        ]
        assert len(records) == 1
        assert records[0]["method"] == "POST"
        assert records[0]["url"] == url
        assert records[0]["headers"]["Authorization"] == "[REDACTED]"
        assert records[0]["body"]["requestId"] == "log-req-1"
        assert log_path.stat().st_mode & 0o777 == 0o600


class TestGetJobDetails:
    @pytest.mark.asyncio
    @respx.mock
    async def test_get_job_details(self, tmp_path):
        client = _client(tmp_path)
        job_id = "cj-test-001"
        url = f"{BASE}/{job_id}"
        respx.get(url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "code": "00000",
                    "data": {
                        "jobId": job_id,
                        "state": "SUCCEEDED",
                        "total": 5,
                        "succeededIndexesCount": 5,
                        "activePods": 0,
                        "runningPods": 0,
                        "failedAttempts": 0,
                        "resourcesReleased": True,
                        "storageReleased": False,
                        "mountPath": "/mnt/nas",
                        "subPath": "data/users/user1",
                        "effectiveSpec": {
                            "cpu": 1,
                            "memoryGiB": 4,
                            "parallelism": 5,
                            "image": "img",
                            "ephemeralStorageGiB": 2,
                        },
                    },
                    "traceId": "t-detail",
                },
            )
        )
        result = await client.get_job_details(job_id)
        assert result["jobId"] == job_id
        assert result["state"] == "SUCCEEDED"


class TestCancelJob:
    @pytest.mark.asyncio
    @respx.mock
    async def test_cancel_job(self, tmp_path):
        client = _client(tmp_path)
        job_id = "cj-cancel"
        url = f"{BASE}/{job_id}/cancel"
        respx.post(url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "code": "00000",
                    "data": {
                        "jobId": job_id,
                        "state": "CANCELLING",
                        "total": 5,
                        "succeededIndexesCount": 0,
                        "activePods": 5,
                        "runningPods": 5,
                        "failedAttempts": 0,
                        "resourcesReleased": False,
                        "storageReleased": False,
                        "mountPath": "/mnt/nas",
                        "subPath": "data/users/user1",
                        "effectiveSpec": {
                            "cpu": 1,
                            "memoryGiB": 4,
                            "parallelism": 5,
                            "image": "img",
                            "ephemeralStorageGiB": 2,
                        },
                        "cancelApplied": True,
                    },
                    "traceId": "t-cancel",
                },
            )
        )
        result = await client.cancel_job(job_id)
        assert result["state"] == "CANCELLING"
        assert result["cancelApplied"] is True


class TestGetJobLogs:
    @pytest.mark.asyncio
    @respx.mock
    async def test_get_logs(self, tmp_path):
        client = _client(tmp_path)
        job_id = "cj-logs"
        url = f"{BASE}/{job_id}/logs"
        respx.get(url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "code": "00000",
                    "data": {
                        "attemptId": "pod-uid-001",
                        "shardIndex": 0,
                        "text": "hello from shard 0\n",
                        "truncated": False,
                    },
                    "traceId": "t-logs",
                },
            )
        )
        result = await client.get_job_logs(
            job_id, shard_index=0, tail_lines=100
        )
        assert result["text"] == "hello from shard 0\n"
        assert result["truncated"] is False

    def test_rejects_tail_lines_out_of_range(self, tmp_path):
        client = _client(tmp_path)
        with pytest.raises(ValueError, match="tailLines"):
            import asyncio

            asyncio.get_event_loop().run_until_complete(
                client.get_job_logs("cj-1", shard_index=0, tail_lines=0)
            )


class TestListJobs:
    def test_rejects_page_size_out_of_range(self, tmp_path):
        client = _client(tmp_path)
        with pytest.raises(ValueError, match="pageSize"):
            import asyncio

            asyncio.get_event_loop().run_until_complete(
                client.list_jobs(page_size=0)
            )

    def test_rejects_too_many_state_filters(self, tmp_path):
        client = _client(tmp_path)
        with pytest.raises(ValueError, match="state filter"):
            import asyncio

            asyncio.get_event_loop().run_until_complete(
                client.list_jobs(states=["A"] * 9)
            )
