import pytest
from pydantic import ValidationError

from models import CpuJobSubmitRequest, EnsureResourceRequest, PendingRequest, ResourceSpec


@pytest.mark.parametrize(
    ("gpu_count", "cpu", "memory_gib"),
    [(1, 8, 64), (2, 16, 128), (4, 32, 256), (8, 64, 512)],
)
def test_z1120_accepts_only_fixed_specs(gpu_count, cpu, memory_gib):
    resource = ResourceSpec(
        resourceType="GPU",
        cpu=cpu,
        memoryGiB=memory_gib,
        gpuType="Z1120",
        gpuCount=gpu_count,
    )
    assert resource.gpu_type == "Z1120"


def test_z1120_rejects_unsupported_gpu_count():
    with pytest.raises(
        ValidationError, match="GPU-32G GPU count must be one of 1, 2, 4, 8"
    ):
        ResourceSpec(
            resourceType="GPU",
            cpu=24,
            memoryGiB=192,
            gpuType="Z1120",
            gpuCount=3,
        )


def test_v100_display_name_is_rejected_as_backend_enum():
    with pytest.raises(ValidationError):
        ResourceSpec(
            resourceType="GPU", cpu=8, memoryGiB=64, gpuType="V100", gpuCount=1
        )


def test_v5000_accepts_only_eight_card_spec():
    resource = ResourceSpec(
        resourceType="GPU",
        cpu=128,
        memoryGiB=900,
        gpuType="V5000",
        gpuCount=8,
    )
    assert resource.gpu_type == "V5000"


def test_v5000_rejects_unsupported_gpu_count():
    with pytest.raises(
        ValidationError, match="GPU-96G GPU count must be one of 8"
    ):
        ResourceSpec(
            resourceType="GPU",
            cpu=48,
            memoryGiB=338,
            gpuType="V5000",
            gpuCount=3,
        )


def test_v5000_rejects_non_fixed_spec():
    with pytest.raises(ValidationError):
        ResourceSpec(
            resourceType="GPU",
            cpu=16,
            memoryGiB=128,
            gpuType="V5000",
            gpuCount=1,
        )


def test_z1120_rejects_non_fixed_spec():
    with pytest.raises(ValidationError):
        ResourceSpec(
            resourceType="GPU",
            cpu=64,
            memoryGiB=128,
            gpuType="Z1120",
            gpuCount=1,
        )


def test_cpu_still_rejects_more_than_32_cores():
    with pytest.raises(ValidationError):
        ResourceSpec(
            resourceType="CPU", cpu=64, memoryGiB=128, gpuType=None, gpuCount=0
        )


def test_ensure_request_requires_session_id_but_no_client_message_id():
    request = EnsureResourceRequest(
        requestId="request-1",
        projectId=1,
        sessionId="session-1",
        pendingRequest=PendingRequest(message="continue"),
        resource=ResourceSpec(
            resourceType="CPU", cpu=1, memoryGiB=2, gpuType=None, gpuCount=0
        ),
    )
    payload = request.model_dump(by_alias=True)
    assert payload["sessionId"] == "session-1"
    assert "clientMessageId" not in payload


def test_ensure_request_rejects_missing_session_id():
    with pytest.raises(ValidationError):
        EnsureResourceRequest(
            requestId="request-1",
            projectId=1,
            pendingRequest=PendingRequest(message="continue"),
            resource=ResourceSpec(
                resourceType="CPU", cpu=1, memoryGiB=2, gpuType=None, gpuCount=0
            ),
        )


def test_cpu_rejects_gpu_count():
    with pytest.raises(ValidationError):
        ResourceSpec(
            resourceType="CPU", cpu=8, memoryGiB=32, gpuType=None, gpuCount=1
        )


def test_pending_request_allows_empty_message():
    assert PendingRequest().message == ""


def test_pending_request_rejects_secret_key():
    with pytest.raises(ValidationError):
        PendingRequest(message="continue", parts=[{"accessToken": "secret"}])


def _cpu_job(**overrides):
    values = {
        "requestId": "cpu-job-1",
        "sessionId": "session-1",
        "executionProfile": "python-cpu",
        "command": ["/bin/sh", "/mnt/nas/job/launch.sh"],
        "shardCount": 4,
        "parallelism": 2,
        "cpu": 2,
        "memoryGiB": 4,
    }
    values.update(overrides)
    return CpuJobSubmitRequest(**values)


def test_cpu_job_request_accepts_valid_normal_job():
    request = _cpu_job()
    assert request.parallelism == 2
    assert request.is_vip is False


def test_cpu_job_request_rejects_parallelism_above_shards():
    with pytest.raises(ValidationError, match="parallelism must not exceed"):
        _cpu_job(shardCount=2, parallelism=3)


def test_cpu_job_request_rejects_reserved_or_secret_env():
    with pytest.raises(ValidationError, match="reserved CPU Job"):
        _cpu_job(env={"JOB_COMPLETION_INDEX": "fake"})
    with pytest.raises(ValidationError, match="Sensitive field"):
        _cpu_job(env={"API_TOKEN": "secret"})


def test_cpu_job_request_rejects_vip_until_visibility_can_be_verified():
    with pytest.raises(ValidationError, match="VIP CPU jobs"):
        _cpu_job(isVip=True)
