import pytest
from pydantic import ValidationError

from models import EnsureResourceRequest, PendingRequest, ResourceSpec


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


def test_v100_display_name_is_rejected_as_backend_enum():
    with pytest.raises(ValidationError):
        ResourceSpec(
            resourceType="GPU", cpu=8, memoryGiB=64, gpuType="V100", gpuCount=1
        )


@pytest.mark.parametrize(
    ("gpu_count", "cpu", "memory_gib"),
    [(1, 16, 112), (2, 32, 225), (4, 64, 450), (8, 128, 900)],
)
def test_v5000_accepts_only_fixed_specs(gpu_count, cpu, memory_gib):
    resource = ResourceSpec(
        resourceType="GPU",
        cpu=cpu,
        memoryGiB=memory_gib,
        gpuType="V5000",
        gpuCount=gpu_count,
    )
    assert resource.gpu_type == "V5000"


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


def test_ensure_request_has_no_legacy_session_fields():
    request = EnsureResourceRequest(
        requestId="request-1",
        projectId=1,
        pendingRequest=PendingRequest(message="continue"),
        resource=ResourceSpec(
            resourceType="CPU", cpu=1, memoryGiB=2, gpuType=None, gpuCount=0
        ),
    )
    payload = request.model_dump(by_alias=True)
    assert "sessionId" not in payload
    assert "clientMessageId" not in payload


def test_cpu_rejects_gpu_count():
    with pytest.raises(ValidationError):
        ResourceSpec(
            resourceType="CPU", cpu=8, memoryGiB=32, gpuType=None, gpuCount=1
        )


def test_pending_request_rejects_secret_key():
    with pytest.raises(ValidationError):
        PendingRequest(message="continue", parts=[{"accessToken": "secret"}])
