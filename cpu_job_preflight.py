from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
from typing import Any

from models import (
    CPU_JOB_ENV_NAME_PATTERN,
    CPU_JOB_RESERVED_ENV,
    CPU_JOB_SECRET_ENV_PATTERN,
)


_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_PYTHON_EXECUTABLE_PATTERN = re.compile(r"python(?:\d+(?:\.\d+)*)?", re.I)
_PYTHON_VENV_ENTRYPOINTS = {"pip", "pip3", "pytest", "torchrun"}
_K8S_PYTHON_EXECUTABLE = "/home/scientist/.venv/bin/python"
_K8S_VIRTUAL_ENV = "/home/scientist/.venv"
_K8S_PYTHON312_LINK = "/home/scientist/.venv/bin/python3.12"
_K8S_BASE_PYTHON = "/opt/conda/bin/python3"
_IMPLICIT_ENV = {
    "HOME", "HOSTNAME", "LANG", "LC_ALL", "LOGNAME", "PATH", "PWD",
    "PYTHONPATH", "SHLVL", "TMPDIR", "USER",
}


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _resolve_local_path(
    raw: str,
    *,
    stable_root: Path,
    mount_path: PurePosixPath,
    must_exist: bool,
    require_file: bool = False,
) -> tuple[Path, str]:
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise ValueError("paths must be non-empty strings without NUL")
    value = raw.strip()
    posix = PurePosixPath(value)
    if posix.is_absolute() and (posix == mount_path or mount_path in posix.parents):
        relative = posix.relative_to(mount_path)
        candidate = stable_root.joinpath(*relative.parts)
    else:
        source = Path(value).expanduser()
        candidate = source if source.is_absolute() else stable_root / source

    stable_root_resolved = stable_root.resolve(strict=True)
    if must_exist:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"required stable path is missing or unreadable: {raw}") from exc
    else:
        parent = _nearest_existing_parent(candidate)
        try:
            resolved_parent = parent.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"output parent cannot be resolved: {raw}") from exc
        if not _inside(resolved_parent, stable_root_resolved):
            raise ValueError(f"path escapes AI_SCIENTIST_STABLE_WORKSPACE_ROOT: {raw}")
        resolved = resolved_parent.joinpath(*candidate.relative_to(parent).parts)

    if not _inside(resolved, stable_root_resolved):
        raise ValueError(f"path escapes AI_SCIENTIST_STABLE_WORKSPACE_ROOT: {raw}")
    if require_file and not resolved.is_file():
        raise ValueError(f"required script is not a file: {raw}")
    if must_exist and not os.access(resolved, os.R_OK):
        raise ValueError(f"required stable path is not readable: {raw}")
    if not must_exist:
        writable_parent = _nearest_existing_parent(resolved)
        if not os.access(writable_parent, os.W_OK):
            raise ValueError(f"output parent is not writable: {raw}")

    relative = resolved.relative_to(stable_root_resolved)
    job_path = str(mount_path.joinpath(*relative.parts))
    return resolved, job_path


def _prepare_environment(
    env: dict[str, str],
    *,
    stable_root: Path,
    project_mount: PurePosixPath,
    worker_index_env: str | None,
    input_roots: list[Path],
    output_roots: list[Path],
    env_path_keys: list[str],
    required_env_keys: list[str],
    optional_env_keys: list[str],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    if not isinstance(env, dict):
        raise ValueError("CPU Job env must be an object of string values")
    if len(env) > 32:
        raise ValueError("CPU Job environment supports at most 32 variables")
    if len(env_path_keys) != len(set(env_path_keys)):
        raise ValueError("env_path_keys must not contain duplicates")
    if len(required_env_keys) != len(set(required_env_keys)):
        raise ValueError("required_env_keys must not contain duplicates")
    if len(optional_env_keys) != len(set(optional_env_keys)):
        raise ValueError("optional_env_keys must not contain duplicates")
    required_names = set(required_env_keys)
    optional_names = set(optional_env_keys)
    overlap = sorted(required_names & optional_names)
    if overlap:
        raise ValueError(
            "environment variables cannot be both required and optional: "
            + ", ".join(overlap)
        )
    for key in required_names | optional_names:
        if CPU_JOB_ENV_NAME_PATTERN.fullmatch(key) is None:
            raise ValueError(f"invalid declared CPU Job environment variable: {key}")
    if worker_index_env and worker_index_env not in required_names:
        raise ValueError(
            "worker_index_env must be declared in required_env_keys: "
            + worker_index_env
        )
    path_keys = set(env_path_keys)
    for key in path_keys:
        if CPU_JOB_ENV_NAME_PATTERN.fullmatch(key) is None or key not in env:
            raise ValueError(
                f"env_path_keys entry must name a supplied env variable: {key}"
            )
    normalized: dict[str, str] = {}
    env_bytes = 0
    for key, value in env.items():
        if CPU_JOB_ENV_NAME_PATTERN.fullmatch(key) is None:
            raise ValueError(f"invalid CPU Job environment variable: {key}")
        if key in CPU_JOB_RESERVED_ENV:
            raise ValueError(f"reserved CPU Job environment variable: {key}")
        if CPU_JOB_SECRET_ENV_PATTERN.search(key):
            raise ValueError(f"sensitive CPU Job environment variable is forbidden: {key}")
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError("CPU Job environment values must be strings without NUL")
        if key == worker_index_env:
            raise ValueError(
                f"{key} is generated from JOB_COMPLETION_INDEX and must not be set in env"
            )
        path_named = key in path_keys
        raw_path = Path(value).expanduser()
        looks_like_workspace_path = False
        if raw_path.is_absolute():
            try:
                resolved_value = raw_path.resolve(strict=False)
            except OSError:
                resolved_value = raw_path
            looks_like_workspace_path = _inside(resolved_value, stable_root)
        posix_value = PurePosixPath(value)
        looks_like_workspace_path = looks_like_workspace_path or (
            posix_value.is_absolute()
            and (posix_value == project_mount or project_mount in posix_value.parents)
        )
        if looks_like_workspace_path and not path_named:
            raise ValueError(
                f"environment variable {key} contains a workspace path; "
                "declare it in env_path_keys"
            )
        effective_value = value
        if path_named:
            local_value, effective_value = _resolve_local_path(
                value,
                stable_root=stable_root,
                mount_path=project_mount,
                must_exist=False,
            )
            is_output = any(
                local_value == output or _inside(local_value, output)
                for output in output_roots
            )
            is_input = any(
                local_value == input_path
                or _inside(local_value, input_path)
                or _inside(input_path, local_value)
                for input_path in input_roots
            )
            if not is_output and not is_input:
                raise ValueError(
                    f"environment path must be declared in input_paths or output_paths: {key}"
                )
            if is_input and not is_output:
                _, effective_value = _resolve_local_path(
                    value,
                    stable_root=stable_root,
                    mount_path=project_mount,
                    must_exist=True,
                )
        normalized[key] = effective_value
        env_bytes += len(key.encode("utf-8")) + len(effective_value.encode("utf-8"))
    if env_bytes > 8192:
        raise ValueError("CPU Job environment exceeds 8192 UTF-8 bytes")

    undeclared_values = sorted(set(normalized) - required_names - optional_names)
    if undeclared_values:
        raise ValueError(
            "supplied environment variables must be classified in required_env_keys "
            "or optional_env_keys: " + ", ".join(undeclared_values)
        )
    provided_by_platform = CPU_JOB_RESERVED_ENV | _IMPLICIT_ENV
    if worker_index_env:
        provided_by_platform.add(worker_index_env)
    missing = sorted(
        name
        for name in required_names
        if name not in normalized and name not in provided_by_platform
    )
    if missing:
        raise ValueError(
            "required CPU Job environment variables are not set: " + ", ".join(missing)
        )
    inventory = [
        {
            "name": name,
            "required": name in required_names,
            "providedBy": (
                "platform"
                if name in CPU_JOB_RESERVED_ENV or name in _IMPLICIT_ENV
                else "worker-index-launcher"
                if name == worker_index_env
                else "request"
                if name in normalized
                else "optional-unset"
            ),
        }
        for name in sorted(required_names | optional_names)
    ]
    return normalized, inventory


def _translate_command_token(
    token: str, *, stable_root: Path, mount_path: PurePosixPath
) -> str:
    if "\x00" in token:
        raise ValueError("command tokens must not contain NUL")
    prefix = ""
    value = token
    if "=" in token and token.startswith("-"):
        prefix, value = token.split("=", 1)
        prefix += "="
    source = Path(value).expanduser()
    if source.is_absolute():
        try:
            resolved = source.resolve(strict=False)
        except OSError:
            return token
        stable = stable_root.resolve(strict=True)
        if _inside(resolved, stable):
            relative = resolved.relative_to(stable)
            return prefix + str(mount_path.joinpath(*relative.parts))
    return token


def _normalize_python_executable(command: list[str]) -> list[str]:
    normalized = list(command)
    executable_name = PurePosixPath(command[0]).name
    if _PYTHON_EXECUTABLE_PATTERN.fullmatch(executable_name):
        normalized[0] = _K8S_PYTHON_EXECUTABLE
    elif executable_name in _PYTHON_VENV_ENTRYPOINTS:
        normalized[0] = f"{_K8S_VIRTUAL_ENV}/bin/{executable_name}"
    return normalized


def _manifest_dir(stable_root: Path, request_id: str) -> Path:
    if _REQUEST_ID_PATTERN.fullmatch(request_id) is None:
        raise ValueError("request_id must be 1-64 safe filename characters")
    return stable_root / ".compute-orchestrator" / "cpu-jobs" / request_id


def prepare_cpu_job(
    *,
    stable_root: Path,
    mount_path: str,
    request_id: str,
    command: list[str],
    shard_count: int,
    script_paths: list[str],
    input_paths: list[str],
    output_paths: list[str],
    env: dict[str, str],
    env_path_keys: list[str],
    required_env_keys: list[str],
    optional_env_keys: list[str],
    independent_shards_confirmed: bool,
    uses_native_completion_index: bool = False,
    worker_index_env: str | None = None,
    worker_index_argument: str | None = None,
    worker_index_base: int = 0,
) -> dict[str, Any]:
    root = stable_root.expanduser().resolve(strict=True)
    mount_base = PurePosixPath(mount_path)
    if not mount_base.is_absolute():
        raise ValueError("K8s workspace parent path must be absolute")
    project_mount = mount_base / root.name
    if not command or len(command) > 64 or any(not isinstance(v, str) or not v for v in command):
        raise ValueError("command must contain 1-64 non-empty string tokens")
    if not 1 <= shard_count <= 5000:
        raise ValueError("shard_count must be from 1 to 5000")
    if worker_index_base not in {0, 1}:
        raise ValueError("worker_index_base must be 0 or 1")
    if worker_index_env is not None and _ENV_NAME_PATTERN.fullmatch(worker_index_env) is None:
        raise ValueError("worker_index_env must be a valid environment variable name")
    if worker_index_argument is not None and (
        not worker_index_argument.strip() or "\x00" in worker_index_argument
    ):
        raise ValueError("worker_index_argument must be a non-empty token")
    if shard_count > 1 and not independent_shards_confirmed:
        raise ValueError(
            "multi-shard jobs require an explicit model confirmation that shards "
            "do not use live cross-worker coordination"
        )
    has_index_mapping = bool(worker_index_env or worker_index_argument)
    if uses_native_completion_index and has_index_mapping:
        raise ValueError(
            "uses_native_completion_index cannot be combined with a worker index mapping"
        )
    if shard_count > 1 and not (uses_native_completion_index or has_index_mapping):
        raise ValueError(
            "multi-shard jobs must explicitly declare native JOB_COMPLETION_INDEX "
            "use or define worker_index_env/worker_index_argument mapping"
        )
    if uses_native_completion_index and "JOB_COMPLETION_INDEX" not in required_env_keys:
        raise ValueError(
            "native JOB_COMPLETION_INDEX use must be declared in required_env_keys"
        )

    job_dir = _manifest_dir(root, request_id)
    if (job_dir / "job.json").exists():
        raise ValueError(
            "request_id already has a submitted CPU Job; use a new request_id"
        )

    local_scripts: list[str] = []
    job_scripts: list[str] = []
    for raw in script_paths:
        local, job = _resolve_local_path(
            raw,
            stable_root=root,
            mount_path=project_mount,
            must_exist=True,
            require_file=True,
        )
        if local.stat().st_size > 2_000_000:
            raise ValueError(f"script/config is too large for preflight inspection: {raw}")
        local_scripts.append(str(local))
        job_scripts.append(job)

    local_inputs: list[str] = []
    job_inputs: list[str] = []
    for raw in input_paths:
        local, job = _resolve_local_path(
            raw, stable_root=root, mount_path=project_mount, must_exist=True
        )
        local_inputs.append(str(local))
        job_inputs.append(job)

    local_outputs: list[str] = []
    job_outputs: list[str] = []
    for raw in output_paths:
        local, job = _resolve_local_path(
            raw, stable_root=root, mount_path=project_mount, must_exist=False
        )
        local_outputs.append(str(local))
        job_outputs.append(job)

    input_roots = [Path(value) for value in local_inputs]
    output_roots = [Path(value) for value in local_outputs]

    prepared_env, environment_requirements = _prepare_environment(
        env,
        stable_root=root,
        project_mount=project_mount,
        worker_index_env=worker_index_env,
        input_roots=input_roots,
        output_roots=output_roots,
        env_path_keys=env_path_keys,
        required_env_keys=required_env_keys,
        optional_env_keys=optional_env_keys,
    )

    translated_command = _normalize_python_executable(
        [
            _translate_command_token(token, stable_root=root, mount_path=project_mount)
            for token in command
        ]
    )
    job_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    launcher = job_dir / "launch.sh"
    launcher_job_path = str(
        project_mount.joinpath(*launcher.resolve().relative_to(root).parts)
    )
    lines = [
        "#!/bin/sh",
        "set -eu",
        f"rm -f {shlex.quote(_K8S_PYTHON312_LINK)}",
        f"ln -s {shlex.quote(_K8S_BASE_PYTHON)} {shlex.quote(_K8S_PYTHON312_LINK)}",
        f"export VIRTUAL_ENV={shlex.quote(_K8S_VIRTUAL_ENV)}",
        f"export PATH={shlex.quote(_K8S_VIRTUAL_ENV + '/bin')}:$PATH",
        "unset PYTHONHOME || true",
        f"cd {shlex.quote(str(project_mount))}",
        'raw_index="${JOB_COMPLETION_INDEX:?JOB_COMPLETION_INDEX is required}"',
        f'worker_index="$((raw_index + {worker_index_base}))"',
    ]
    if worker_index_env:
        lines.append(f"export {worker_index_env}=\"$worker_index\"")
    exec_parts = [shlex.quote(token) for token in translated_command]
    if worker_index_argument:
        exec_parts.extend([shlex.quote(worker_index_argument), '"$worker_index"'])
    lines.append("exec " + " ".join(exec_parts))
    launcher_text = "\n".join(lines) + "\n"
    temporary_launcher = launcher.with_suffix(".sh.tmp")
    temporary_launcher.write_text(launcher_text, encoding="utf-8")
    temporary_launcher.chmod(0o700)
    os.replace(temporary_launcher, launcher)
    launcher_sha256 = hashlib.sha256(launcher_text.encode("utf-8")).hexdigest()

    manifest = {
        "requestId": request_id,
        "shardCount": shard_count,
        "pathMapping": "identity-venv-python312-v3",
        "pythonExecutable": _K8S_PYTHON_EXECUTABLE,
        "python312Target": _K8S_BASE_PYTHON,
        "mountPath": str(mount_base),
        "projectWorkspacePath": str(project_mount),
        "command": ["/bin/sh", launcher_job_path],
        "args": [],
        "env": prepared_env,
        "envPathKeys": env_path_keys,
        "requiredEnvKeys": required_env_keys,
        "optionalEnvKeys": optional_env_keys,
        "independentShardsConfirmed": independent_shards_confirmed,
        "environmentRequirements": environment_requirements,
        "launcherLocalPath": str(launcher),
        "launcherJobPath": launcher_job_path,
        "launcherSha256": launcher_sha256,
        "scriptLocalPaths": local_scripts,
        "scriptJobPaths": job_scripts,
        "inputLocalPaths": local_inputs,
        "inputJobPaths": job_inputs,
        "outputLocalPaths": local_outputs,
        "outputJobPaths": job_outputs,
        "workerIndex": {
            "source": "JOB_COMPLETION_INDEX",
            "usesNativeCompletionIndex": uses_native_completion_index,
            "environmentVariable": worker_index_env,
            "argument": worker_index_argument,
            "base": worker_index_base,
        },
    }
    manifest_path = job_dir / "preflight.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    return {
        "preflightId": request_id,
        "manifestPath": str(manifest_path),
        "launcherPath": str(launcher),
        "jobCommand": manifest["command"],
        "jobArgs": [],
        "scriptPaths": job_scripts,
        "inputPaths": job_inputs,
        "outputPaths": job_outputs,
        "env": prepared_env,
        "envPathKeys": env_path_keys,
        "requiredEnvKeys": required_env_keys,
        "optionalEnvKeys": optional_env_keys,
        "independentShardsConfirmed": independent_shards_confirmed,
        "environmentRequirements": environment_requirements,
        "workerIndex": manifest["workerIndex"],
    }


def load_verified_preflight(
    *, stable_root: Path, request_id: str, shard_count: int
) -> dict[str, Any]:
    root = stable_root.expanduser().resolve(strict=True)
    manifest_path = _manifest_dir(root, request_id) / "preflight.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("prepare_cpu_job must succeed before submit_cpu_job") from exc
    if manifest.get("requestId") != request_id or manifest.get("shardCount") != shard_count:
        raise ValueError("prepared request does not match request_id and shard_count")
    if (
        manifest.get("pathMapping") != "identity-venv-python312-v3"
        or manifest.get("pythonExecutable") != _K8S_PYTHON_EXECUTABLE
        or manifest.get("python312Target") != _K8S_BASE_PYTHON
        or manifest.get("projectWorkspacePath") != str(root)
    ):
        raise ValueError(
            "prepared CPU job uses an obsolete workspace or Python runtime; run "
            "prepare_cpu_job again"
        )
    launcher = Path(str(manifest.get("launcherLocalPath", "")))
    try:
        launcher_resolved = launcher.resolve(strict=True)
    except OSError as exc:
        raise ValueError("prepared CPU job launcher is missing") from exc
    if not _inside(launcher_resolved, root):
        raise ValueError("prepared CPU job launcher escaped stable storage")
    digest = hashlib.sha256(launcher_resolved.read_bytes()).hexdigest()
    if digest != manifest.get("launcherSha256"):
        raise ValueError("prepared CPU job launcher changed after preflight")
    for raw in manifest.get("scriptLocalPaths", []) + manifest.get("inputLocalPaths", []):
        resolved = Path(raw).resolve(strict=True)
        if not _inside(resolved, root) or not os.access(resolved, os.R_OK):
            raise ValueError(f"prepared input is no longer readable in stable storage: {raw}")
    for raw in manifest.get("outputLocalPaths", []):
        output = Path(raw)
        parent = _nearest_existing_parent(output)
        resolved_parent = parent.resolve(strict=True)
        if not _inside(resolved_parent, root) or not os.access(resolved_parent, os.W_OK):
            raise ValueError(f"prepared output parent is no longer writable: {raw}")
    return manifest
