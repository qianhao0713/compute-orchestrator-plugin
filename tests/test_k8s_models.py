"""Tests for K8s job Pydantic models — validation, constraints, and helpers."""

import pytest
from pydantic import ValidationError

from k8s_models import (
    K8sAttemptItem,
    K8sLogData,
    K8sNodeResource,
    K8sResourcesData,
    SubmitK8sJobRequest,
    TaskView,
    is_fully_done,
    is_terminal_state,
)


# ---------------------------------------------------------------------------
# SubmitK8sJobRequest validation
# ---------------------------------------------------------------------------


class TestSubmitK8sJobRequest:
    def test_minimal_valid_request(self):
        req = SubmitK8sJobRequest(
            requestId="req-1",
            image="registry.example/worker:v1",
        )
        assert req.request_id == "req-1"
        assert req.image == "registry.example/worker:v1"
        assert req.shard_count == 1
        assert req.parallelism == 1
        assert req.cpu == 1
        assert req.memory_gib == 4
        assert req.env == {}

    def test_full_valid_request(self):
        req = SubmitK8sJobRequest(
            requestId="req-2",
            image="registry.example/worker:v1",
            command=["/bin/sh", "-c"],
            args=["echo hello"],
            env={"INPUT_DIR": "/mnt/nas/input", "OUTPUT_DIR": "/mnt/nas/output"},
            shardCount=10,
            parallelism=5,
            cpu=4,
            memoryGiB=8,
            ephemeralStorageGiB=10,
            jobBackoffLimit=3,
            activeDeadlineSeconds=3600,
            description="Test job",
            sessionId="session-1",
        )
        payload = req.model_dump(by_alias=True, exclude_none=True)
        assert payload["shardCount"] == 10
        assert payload["parallelism"] == 5
        assert payload["cpu"] == 4
        assert payload["memoryGiB"] == 8
        assert payload["env"] == {
            "INPUT_DIR": "/mnt/nas/input",
            "OUTPUT_DIR": "/mnt/nas/output",
        }

    def test_rejects_empty_request_id(self):
        with pytest.raises(ValidationError, match="requestId"):
            SubmitK8sJobRequest(requestId="", image="img")

    def test_rejects_request_id_too_long(self):
        with pytest.raises(ValidationError, match="requestId"):
            SubmitK8sJobRequest(requestId="x" * 65, image="img")

    def test_rejects_image_too_long(self):
        with pytest.raises(ValidationError, match="image"):
            SubmitK8sJobRequest(requestId="r", image="x" * 513)

    def test_rejects_parallelism_exceeding_shard_count(self):
        with pytest.raises(ValidationError, match="parallelism must not exceed"):
            SubmitK8sJobRequest(
                requestId="r",
                image="img",
                shardCount=3,
                parallelism=5,
            )

    def test_rejects_shard_count_too_large(self):
        with pytest.raises(ValidationError, match="shardCount must be at most"):
            SubmitK8sJobRequest(
                requestId="r", image="img", shardCount=5001
            )

    def test_rejects_parallelism_too_large(self):
        with pytest.raises(ValidationError, match="parallelism must be at most"):
            SubmitK8sJobRequest(
                requestId="r", image="img", parallelism=65, shardCount=100
            )

    def test_rejects_cpu_zero(self):
        with pytest.raises(ValidationError):
            SubmitK8sJobRequest(requestId="r", image="img", cpu=0)

    def test_rejects_memory_zero(self):
        with pytest.raises(ValidationError):
            SubmitK8sJobRequest(requestId="r", image="img", memoryGiB=0)

    def test_rejects_backoff_limit_too_large(self):
        with pytest.raises(ValidationError, match="jobBackoffLimit"):
            SubmitK8sJobRequest(
                requestId="r", image="img", jobBackoffLimit=21
            )

    def test_rejects_active_deadline_too_large(self):
        with pytest.raises(ValidationError, match="activeDeadlineSeconds"):
            SubmitK8sJobRequest(
                requestId="r", image="img", activeDeadlineSeconds=172801
            )

    def test_rejects_command_with_blank_item(self):
        with pytest.raises(ValidationError, match="command items must be non-blank"):
            SubmitK8sJobRequest(
                requestId="r",
                image="img",
                command=["/bin/sh", "  "],
            )

    def test_rejects_command_item_too_long(self):
        with pytest.raises(ValidationError, match="command item exceeds"):
            SubmitK8sJobRequest(
                requestId="r",
                image="img",
                command=["/bin/sh", "x" * 8193],
            )

    def test_rejects_too_many_command_items(self):
        with pytest.raises(ValidationError, match="command must have at most"):
            SubmitK8sJobRequest(
                requestId="r",
                image="img",
                command=["cmd"] * 65,
            )

    def test_rejects_too_many_args_items(self):
        with pytest.raises(ValidationError, match="args must have at most"):
            SubmitK8sJobRequest(
                requestId="r",
                image="img",
                args=["arg"] * 257,
            )

    def test_rejects_combined_command_args_too_large(self):
        # Each item is under the single limit but total exceeds 32768 bytes
        big_item = "x" * 8000
        with pytest.raises(ValidationError, match="command\\+args total exceeds"):
            SubmitK8sJobRequest(
                requestId="r",
                image="img",
                command=["/bin/sh", "-c"],
                args=[big_item] * 5,
            )

    def test_rejects_reserved_env_key_job_id(self):
        with pytest.raises(ValidationError, match="reserved"):
            SubmitK8sJobRequest(
                requestId="r",
                image="img",
                env={"JOB_ID": "123"},
            )

    def test_rejects_reserved_env_key_pod_uid(self):
        with pytest.raises(ValidationError, match="reserved"):
            SubmitK8sJobRequest(
                requestId="r",
                image="img",
                env={"POD_UID": "abc"},
            )

    def test_rejects_reserved_env_key_job_completion_index(self):
        with pytest.raises(ValidationError, match="reserved"):
            SubmitK8sJobRequest(
                requestId="r",
                image="img",
                env={"JOB_COMPLETION_INDEX": "0"},
            )

    def test_rejects_invalid_env_key(self):
        with pytest.raises(ValidationError, match="env key"):
            SubmitK8sJobRequest(
                requestId="r",
                image="img",
                env={"123invalid": "value"},
            )

    def test_rejects_env_with_nul_character(self):
        with pytest.raises(ValidationError, match="NUL"):
            SubmitK8sJobRequest(
                requestId="r",
                image="img",
                env={"KEY": "val\x00ue"},
            )

    def test_rejects_too_many_env_items(self):
        with pytest.raises(ValidationError, match="env must have at most"):
            SubmitK8sJobRequest(
                requestId="r",
                image="img",
                env={f"KEY_{i}": "v" for i in range(33)},
            )

    def test_rejects_description_too_long(self):
        with pytest.raises(ValidationError, match="description"):
            SubmitK8sJobRequest(
                requestId="r",
                image="img",
                description="x" * 1025,
            )

    def test_defaults_serialized_correctly(self):
        req = SubmitK8sJobRequest(requestId="r", image="img")
        payload = req.model_dump(by_alias=True, exclude_none=True)
        assert payload["shardCount"] == 1
        assert payload["parallelism"] == 1
        assert payload["cpu"] == 1
        assert payload["memoryGiB"] == 4
        assert payload["jobBackoffLimit"] == 6
        assert payload["activeDeadlineSeconds"] == 21600
        assert "ephemeralStorageGiB" not in payload
        assert "description" not in payload


# ---------------------------------------------------------------------------
# TaskView and helpers
# ---------------------------------------------------------------------------


class TestTaskView:
    def _make_task(self, **overrides) -> TaskView:
        defaults = {
            "jobId": "cj-test",
            "state": "RUNNING",
            "total": 5,
            "succeededIndexesCount": 3,
            "activePods": 2,
            "runningPods": 2,
            "failedAttempts": 0,
            "resourcesReleased": False,
            "storageReleased": False,
            "mountPath": "/mnt/nas",
            "subPath": "data/users/user1",
            "effectiveSpec": {
                "cpu": 2,
                "memoryGiB": 8,
                "parallelism": 5,
                "image": "registry.example/worker:v1",
                "ephemeralStorageGiB": 2,
            },
        }
        defaults.update(overrides)
        return TaskView(**defaults)

    def test_is_terminal_state(self):
        assert is_terminal_state("SUCCEEDED")
        assert is_terminal_state("FAILED")
        assert is_terminal_state("CANCELLED")
        assert not is_terminal_state("RUNNING")
        assert not is_terminal_state("ACCEPTED")
        assert not is_terminal_state("CANCELLING")

    def test_is_fully_done_requires_both(self):
        task = self._make_task(state="SUCCEEDED", resourcesReleased=True)
        assert is_fully_done(task)

        task_running = self._make_task(state="RUNNING", resourcesReleased=False)
        assert not is_fully_done(task_running)

        task_succeeded_not_released = self._make_task(
            state="SUCCEEDED", resourcesReleased=False
        )
        assert not is_fully_done(task_succeeded_not_released)

    def test_task_view_parses_from_api_response(self):
        data = {
            "jobId": "cj-1790035200000-abc",
            "state": "SUCCEEDED",
            "reasonCode": None,
            "lastErrorType": None,
            "total": 5,
            "succeededIndexesCount": 5,
            "activePods": 0,
            "runningPods": 0,
            "failedAttempts": 0,
            "createdAt": "1790035200000",
            "startedAt": "1790035210000",
            "finishedAt": "1790035530000",
            "observedAt": "1790035530000",
            "stale": False,
            "resourcesReleased": True,
            "storageReleased": False,
            "executionProfile": None,
            "isVip": False,
            "env": {},
            "mountPath": "/mnt/nas",
            "subPath": "data/ai-scientist/users/user42",
            "effectiveSpec": {
                "cpu": 1,
                "memoryGiB": 4,
                "parallelism": 5,
                "image": "registry.example/worker:v1",
                "command": ["python", "-c"],
                "args": ["print('hello')"],
                "ephemeralStorageGiB": 2,
            },
        }
        task = TaskView.model_validate(data)
        assert task.job_id == "cj-1790035200000-abc"
        assert task.state == "SUCCEEDED"
        assert task.total == 5
        assert task.succeeded_indexes_count == 5
        assert task.resources_released is True
        assert task.effective_spec.cpu == 1
        assert task.effective_spec.parallelism == 5


# ---------------------------------------------------------------------------
# K8sResourcesData
# ---------------------------------------------------------------------------


class TestK8sResourcesData:
    def test_parses_resources_response(self):
        data = {
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
        }
        resources = K8sResourcesData.model_validate(data)
        assert resources.cluster_id == "cpu-batch"
        assert len(resources.nodes) == 1
        assert resources.nodes[0].available_cpu_cores == 22
        assert resources.nodes[0].schedulable is True


# ---------------------------------------------------------------------------
# K8sAttemptItem
# ---------------------------------------------------------------------------


class TestK8sAttemptItem:
    def test_parses_attempt_item(self):
        data = {
            "jobId": "cj-example",
            "podUid": "pod-uid-001",
            "podName": "cpu-cj-example-0-abcde",
            "jobUid": "k8s-job-uid-001",
            "shardIndex": 0,
            "phase": "Succeeded",
            "imageId": "registry.example/worker@sha256:aaa",
            "reason": "Completed",
            "exitCode": 0,
            "podCreatedAt": "1790035205000",
            "startedAt": "1790035210000",
            "finishedAt": "1790035510000",
            "terminal": True,
        }
        attempt = K8sAttemptItem.model_validate(data)
        assert attempt.pod_uid == "pod-uid-001"
        assert attempt.shard_index == 0
        assert attempt.exit_code == 0
        assert attempt.terminal is True


# ---------------------------------------------------------------------------
# K8sLogData
# ---------------------------------------------------------------------------


class TestK8sLogData:
    def test_parses_log_response(self):
        data = {
            "attemptId": "pod-uid-001",
            "shardIndex": 0,
            "text": "running on: cpu-cj-example-0-abcde, rank: 0\n",
            "truncated": False,
        }
        log = K8sLogData.model_validate(data)
        assert log.attempt_id == "pod-uid-001"
        assert log.shard_index == 0
        assert "rank: 0" in log.text
        assert log.truncated is False
