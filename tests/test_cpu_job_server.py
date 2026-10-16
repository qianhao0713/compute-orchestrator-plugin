import json
from pathlib import Path

import pytest

import server
from config import Settings


class _CpuJobClient:
    def __init__(self):
        self.submission = None

    async def get_cpu_job_resources(self):
        return {
            "nodes": [
                {
                    "ready": True,
                    "schedulable": True,
                    "availableCpuCores": 8,
                    "availableMemoryBytes": 16 * 1024**3,
                    "name": "private-node-name",
                }
            ],
            "observedAt": "2026-10-15T00:00:00Z",
            "traceId": "trace-capacity",
        }

    async def submit_cpu_job(self, request):
        self.submission = request
        return {
            "jobId": "job-1",
            "state": "ACCEPTED",
            "replayed": False,
            "traceId": "trace-submit",
        }


def _settings(root: Path) -> Settings:
    return Settings(
        portal_base_url="https://portal.example.com",
        portal_access_token="token",
        portal_project_id=7,
        stable_workspace_root=root,
        plugin_log_dir=root / "logs",
    )


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "project7"
    root.mkdir()
    (root / "worker.py").write_text(
        'import os\nprint(os.environ["MY_SHARD"])\n', encoding="utf-8"
    )
    (root / "input.csv").write_text("x\n1\n", encoding="utf-8")
    return root


@pytest.mark.asyncio
async def test_capacity_response_is_sanitized(monkeypatch, tmp_path):
    root = _workspace(tmp_path)
    client = _CpuJobClient()
    monkeypatch.setattr(server, "portal_client", lambda: client)

    result = await server.estimate_cpu_job_capacity(2, 4, 4)

    assert result["capacitySufficient"] is True
    assert "nodes" not in result
    assert "private-node-name" not in str(result)


@pytest.mark.asyncio
async def test_submit_uses_verified_launcher_and_trusted_session(monkeypatch, tmp_path):
    root = _workspace(tmp_path)
    current_settings = _settings(root)
    client = _CpuJobClient()
    monkeypatch.setattr(server, "settings", lambda: current_settings)
    monkeypatch.setattr(server, "portal_client", lambda: client)
    server.prepare_cpu_job(
        request_id="request-1",
        command=["python", str(root / "worker.py")],
        shard_count=3,
        script_paths=[str(root / "worker.py")],
        input_paths=[str(root / "input.csv")],
        output_paths=[str(root / "results")],
        env={
            "INPUT_FILE": str(root / "input.csv"),
            "CACHE_DIR": str(root / "results"),
        },
        env_path_keys=["INPUT_FILE", "CACHE_DIR"],
        required_env_keys=["MY_SHARD", "INPUT_FILE", "CACHE_DIR"],
        independent_shards_confirmed=True,
        worker_index_env="MY_SHARD",
    )

    result = await server.submit_cpu_job(
        request_id="request-1",
        shard_count=3,
        parallelism=2,
        cpu_per_worker=2,
        memory_gib_per_worker=4,
        session_id="trusted-session",
    )

    assert result["jobId"] == "job-1"
    assert client.submission.session_id == "trusted-session"
    assert client.submission.env == {
        "INPUT_FILE": "/mnt/nas/project7/input.csv",
        "CACHE_DIR": "/mnt/nas/project7/results",
    }
    assert client.submission.command[0] == "/bin/sh"
    assert client.submission.command[1].startswith(
        "/mnt/nas/project7/.compute-orchestrator/cpu-jobs/request-1/"
    )
    record_path = root / ".compute-orchestrator/cpu-jobs/request-1/job.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["request"]["sessionId"] == "trusted-session"
    assert record["request"]["cpu"] == 2
    assert record["request"]["env"] == {
        "INPUT_FILE": "/mnt/nas/project7/input.csv",
        "CACHE_DIR": "/mnt/nas/project7/results",
    }
    assert "executionProfile" not in record["request"]
    assert "image" not in record["request"]
    assert record["response"]["jobId"] == "job-1"

    with pytest.raises(ValueError, match="exact original request"):
        await server.submit_cpu_job(
            request_id="request-1",
            shard_count=3,
            parallelism=2,
            cpu_per_worker=3,
            memory_gib_per_worker=4,
            session_id="trusted-session",
        )


@pytest.mark.asyncio
async def test_submit_without_preflight_fails_closed(monkeypatch, tmp_path):
    root = _workspace(tmp_path)
    monkeypatch.setattr(server, "settings", lambda: _settings(root))
    with pytest.raises(ValueError, match="prepare_cpu_job must succeed"):
        await server.submit_cpu_job(
            request_id="missing",
            shard_count=2,
            parallelism=2,
            cpu_per_worker=1,
            memory_gib_per_worker=2,
            session_id="trusted-session",
        )


@pytest.mark.asyncio
async def test_submit_rejects_env_path_without_project_directory(monkeypatch, tmp_path):
    root = _workspace(tmp_path)
    monkeypatch.setattr(server, "settings", lambda: _settings(root))
    with pytest.raises(ValueError, match="missing or unreadable|escapes"):
        server.prepare_cpu_job(
            request_id="request-2",
            command=["python", str(root / "worker.py")],
            shard_count=2,
            script_paths=[str(root / "worker.py")],
            input_paths=[str(root / "input.csv")],
            output_paths=[str(root / "results")],
            env={"INPUT_FILE": "/mnt/nas/input.csv"},
            env_path_keys=["INPUT_FILE"],
            required_env_keys=["MY_SHARD", "INPUT_FILE"],
            independent_shards_confirmed=True,
            worker_index_env="MY_SHARD",
        )
