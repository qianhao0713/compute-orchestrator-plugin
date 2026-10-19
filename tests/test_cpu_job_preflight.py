import json
from pathlib import Path

import pytest

from cpu_job_preflight import load_verified_preflight, prepare_cpu_job


def _prepare(root: Path, **overrides):
    values = {
        "stable_root": root,
        "mount_path": str(root.parent),
        "request_id": "request-1",
        "command": ["python", str(root / "worker.py")],
        "shard_count": 4,
        "script_paths": [str(root / "worker.py")],
        "input_paths": [str(root / "data.csv")],
        "output_paths": [str(root / "results")],
        "env": {},
        "env_path_keys": [],
        "required_env_keys": ["SHARD_ID"],
        "optional_env_keys": [],
        "independent_shards_confirmed": True,
        "uses_native_completion_index": False,
        "worker_index_env": "SHARD_ID",
        "worker_index_argument": None,
        "worker_index_base": 0,
    }
    values.update(overrides)
    return prepare_cpu_job(**values)


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "project1"
    root.mkdir()
    (root / "data.csv").write_text("value\n1\n", encoding="utf-8")
    (root / "worker.py").write_text(
        'import os\nprint(os.environ["SHARD_ID"])\n', encoding="utf-8"
    )
    return root


def test_preflight_maps_k8s_index_to_user_environment(tmp_path):
    root = _workspace(tmp_path)
    result = _prepare(root, worker_index_base=1)

    launcher = Path(result["launcherPath"]).read_text(encoding="utf-8")
    assert 'raw_index="${JOB_COMPLETION_INDEX:' in launcher
    assert 'worker_index="$((raw_index + 1))"' in launcher
    assert 'export SHARD_ID="$worker_index"' in launcher
    assert "export VIRTUAL_ENV=/home/scientist/.venv" in launcher
    assert "export PATH=/home/scientist/.venv/bin:$PATH" in launcher
    assert "rm -f /home/scientist/.venv/bin/python3.12" in launcher
    assert (
        "ln -s /opt/conda/bin/python3 /home/scientist/.venv/bin/python3.12"
        in launcher
    )
    assert "exec /home/scientist/.venv/bin/python " in launcher
    assert str(root / "worker.py") in launcher
    assert f"cd {root}" in launcher
    assert result["jobCommand"][1].startswith(
        str(root / ".compute-orchestrator/cpu-jobs/request-1/")
    )


def test_k8s_project_path_maps_back_to_local_stable_root(tmp_path):
    root = _workspace(tmp_path)
    result = _prepare(
        root,
        command=["python", str(root / "worker.py")],
        script_paths=[str(root / "worker.py")],
        input_paths=[str(root / "data.csv")],
        output_paths=[str(root / "results")],
    )
    assert result["scriptPaths"] == [str(root / "worker.py")]
    assert result["inputPaths"] == [str(root / "data.csv")]


def test_k8s_path_missing_project_directory_is_rejected(tmp_path):
    root = _workspace(tmp_path)
    with pytest.raises(ValueError, match="missing or unreadable|escapes"):
        _prepare(root, input_paths=[str(root.parent / "data.csv")])


def test_preflight_maps_index_to_user_argument(tmp_path):
    root = _workspace(tmp_path)
    (root / "worker.py").write_text(
        'import argparse\nparser = argparse.ArgumentParser()\n'
        'parser.add_argument("--rank", type=int, required=True)\n'
        'print(parser.parse_args().rank)\n',
        encoding="utf-8",
    )
    result = _prepare(
        root,
        worker_index_env=None,
        worker_index_argument="--rank",
        required_env_keys=[],
    )
    launcher = Path(result["launcherPath"]).read_text(encoding="utf-8")
    assert '--rank "$worker_index"' in launcher


def test_native_k8s_index_needs_no_mapping(tmp_path):
    root = _workspace(tmp_path)
    (root / "worker.py").write_text(
        'import os\nprint(os.environ["JOB_COMPLETION_INDEX"])\n', encoding="utf-8"
    )
    result = _prepare(
        root,
        worker_index_env=None,
        uses_native_completion_index=True,
        required_env_keys=["JOB_COMPLETION_INDEX"],
    )
    assert result["workerIndex"]["environmentVariable"] is None


def test_multi_shard_job_rejects_missing_explicit_index_contract(tmp_path):
    root = _workspace(tmp_path)
    with pytest.raises(ValueError, match="must explicitly declare native"):
        _prepare(root, worker_index_env=None, required_env_keys=[])


def test_preflight_does_not_guess_input_output_paths_from_code(tmp_path):
    root = _workspace(tmp_path)
    (root / "worker.py").write_text(
        'from pathlib import Path\nPath("missing/input.csv").read_text()\n',
        encoding="utf-8",
    )
    result = _prepare(root)
    assert "discoveredPaths" not in result


def test_preflight_requires_model_confirmation_of_independent_shards(tmp_path):
    root = _workspace(tmp_path)
    (root / "worker.py").write_text(
        "import torch.distributed as dist\ndist.barrier()\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="explicit model confirmation"):
        _prepare(root, independent_shards_confirmed=False)


def test_preflight_does_not_guess_scripts_from_command_suffix(tmp_path):
    root = _workspace(tmp_path)
    result = _prepare(root, script_paths=[])
    assert result["scriptPaths"] == []


def test_preflight_rejects_path_outside_stable_workspace(tmp_path):
    root = _workspace(tmp_path)
    outside = tmp_path / "outside.csv"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        _prepare(root, input_paths=[str(outside)])


def test_submit_gate_detects_modified_launcher(tmp_path):
    root = _workspace(tmp_path)
    result = _prepare(root)
    Path(result["launcherPath"]).write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after preflight"):
        load_verified_preflight(
            stable_root=root, request_id="request-1", shard_count=4
        )


def test_submit_gate_rejects_obsolete_workspace_mapping(tmp_path):
    root = _workspace(tmp_path)
    result = _prepare(root)
    manifest_path = Path(result["manifestPath"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pathMapping"] = "legacy-prefix-v1"
    manifest["projectWorkspacePath"] = f"/mnt/nas/{root.name}"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="obsolete workspace or Python runtime"):
        load_verified_preflight(
            stable_root=root, request_id="request-1", shard_count=4
        )


def test_preflight_replaces_explicit_system_python_with_shared_venv(tmp_path):
    root = _workspace(tmp_path)
    result = _prepare(root, command=["/usr/bin/python3", str(root / "worker.py")])

    launcher = Path(result["launcherPath"]).read_text(encoding="utf-8")
    assert "exec /home/scientist/.venv/bin/python " in launcher
    assert "/usr/bin/python3" not in launcher


def test_preflight_uses_shared_venv_torchrun_entrypoint(tmp_path):
    root = _workspace(tmp_path)
    result = _prepare(root, command=["/usr/bin/torchrun", str(root / "worker.py")])

    launcher = Path(result["launcherPath"]).read_text(encoding="utf-8")
    assert "exec /home/scientist/.venv/bin/torchrun " in launcher
    assert "/usr/bin/torchrun" not in launcher


def test_preflight_discovers_required_env_and_maps_directory_values(tmp_path):
    root = _workspace(tmp_path)
    data_dir = root / "dataset"
    data_dir.mkdir()
    (root / "worker.py").write_text(
        'import os\n'
        'print(os.environ["SHARD_ID"])\n'
        'print(os.environ["DATA_DIR"])\n'
        'print(os.environ["RUN_MODE"])\n'
        'print(os.getenv("OPTIONAL_LABEL", "default"))\n',
        encoding="utf-8",
    )
    result = _prepare(
        root,
        env={
            "DATA_DIR": str(data_dir),
            "RUN_MODE": "evaluate",
            "OUTPUT_DIR": str(root / "results"),
        },
        input_paths=[str(root / "data.csv"), str(data_dir)],
        env_path_keys=["DATA_DIR", "OUTPUT_DIR"],
        required_env_keys=["SHARD_ID", "DATA_DIR", "RUN_MODE", "OUTPUT_DIR"],
        optional_env_keys=["OPTIONAL_LABEL"],
    )
    assert result["env"] == {
        "DATA_DIR": str(root / "dataset"),
        "RUN_MODE": "evaluate",
        "OUTPUT_DIR": str(root / "results"),
    }
    requirements = {item["name"]: item for item in result["environmentRequirements"]}
    assert requirements["SHARD_ID"]["providedBy"] == "worker-index-launcher"
    assert requirements["DATA_DIR"]["providedBy"] == "request"
    assert requirements["RUN_MODE"]["required"] is True
    assert requirements["OPTIONAL_LABEL"]["providedBy"] == "optional-unset"


def test_preflight_rejects_missing_required_environment_variable(tmp_path):
    root = _workspace(tmp_path)
    (root / "worker.py").write_text(
        'import os\nprint(os.environ["SHARD_ID"])\nprint(os.environ["RUN_MODE"])\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="RUN_MODE"):
        _prepare(root, required_env_keys=["SHARD_ID", "RUN_MODE"])


def test_preflight_rejects_undeclared_output_environment_path(tmp_path):
    root = _workspace(tmp_path)
    with pytest.raises(ValueError, match="input_paths or output_paths"):
        _prepare(
            root,
            env={"OUTPUT_DIR": str(root / "other-results")},
            env_path_keys=["OUTPUT_DIR"],
            required_env_keys=["SHARD_ID", "OUTPUT_DIR"],
        )


def test_declared_custom_directory_variable_is_mapped_as_output(tmp_path):
    root = _workspace(tmp_path)
    result = _prepare(
        root,
        env={"CACHE_DIR": str(root / "results")},
        env_path_keys=["CACHE_DIR"],
        required_env_keys=["SHARD_ID", "CACHE_DIR"],
    )
    assert result["env"]["CACHE_DIR"] == str(root / "results")


def test_workspace_env_path_must_be_declared_by_model(tmp_path):
    root = _workspace(tmp_path)
    with pytest.raises(ValueError, match="declare it in env_path_keys"):
        _prepare(
            root,
            env={"LOCATION": str(root / "data.csv")},
            required_env_keys=["SHARD_ID", "LOCATION"],
        )


def test_rank_environment_is_a_supported_worker_index_contract(tmp_path):
    root = _workspace(tmp_path)
    (root / "worker.py").write_text(
        'import os\nprint(os.environ["RANK"])\n', encoding="utf-8"
    )
    result = _prepare(
        root,
        worker_index_env="RANK",
        required_env_keys=["RANK"],
    )
    launcher = Path(result["launcherPath"]).read_text(encoding="utf-8")
    assert 'export RANK="$worker_index"' in launcher


def test_supplied_environment_must_be_semantically_classified(tmp_path):
    root = _workspace(tmp_path)
    with pytest.raises(ValueError, match="must be classified"):
        _prepare(root, env={"RUN_MODE": "evaluate"})
