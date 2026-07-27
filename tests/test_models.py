import pytest
from pydantic import ValidationError

from models import PendingRequest, ResourceSpec


def test_v100_uses_z1120():
    resource = ResourceSpec(
        resourceType="GPU", cpu=8, memoryGiB=64, gpuType="Z1120", gpuCount=1
    )
    assert resource.gpu_type == "Z1120"


def test_v100_display_name_is_rejected_as_backend_enum():
    with pytest.raises(ValidationError):
        ResourceSpec(
            resourceType="GPU", cpu=8, memoryGiB=64, gpuType="V100", gpuCount=1
        )


def test_cpu_rejects_gpu_count():
    with pytest.raises(ValidationError):
        ResourceSpec(
            resourceType="CPU", cpu=8, memoryGiB=32, gpuType=None, gpuCount=1
        )


def test_pending_request_rejects_secret_key():
    with pytest.raises(ValidationError):
        PendingRequest(message="continue", parts=[{"accessToken": "secret"}])
