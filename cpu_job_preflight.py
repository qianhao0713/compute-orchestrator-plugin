from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
from typing import Any


_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_DATA_SUFFIXES = {
    ".bin", ".cfg", ".ckpt", ".csv", ".h5", ".hdf5", ".ini", ".json",
    ".jsonl", ".npy", ".npz", ".parquet", ".pkl", ".pt", ".pth",
    ".safetensors", ".toml", ".tsv", ".txt", ".yaml", ".yml",
}
_DATA_COMPONENTS = {
    "checkpoint", "checkpoints", "config", "configs", "data", "dataset",
    "datasets", "input", "inputs", "model", "models", "output", "outputs",
}
_INSPECTED_COMMAND_SUFFIXES = {
    ".cfg", ".ini", ".json", ".py", ".sh", ".toml", ".yaml", ".yml",
}
_COLLECTIVE_PATTERNS = {
    "MPI": re.compile(r"\b(?:mpi4py|mpirun|mpiexec|MPI_COMM_WORLD)\b", re.I),
    "torch.distributed": re.compile(
        r"\btorch\.distributed\b|\bdist\.(?:all_reduce|all_gather|gather|scatter|"
        r"reduce_scatter|barrier)\s*\(", re.I
    ),
    "distributed runtime": re.compile(r"\b(?:ray\.init|dask\.distributed|pyspark)\b", re.I),
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


def _looks_like_data_path(value: str) -> bool:
    if not value or "\n" in value or "://" in value:
        return False
    if value.startswith(("$", "${")):
        return False
    path = PurePosixPath(value)
    if path.suffix.lower() in _DATA_SUFFIXES:
        return True
    lowered_parts = {part.lower() for part in path.parts}
    return bool(lowered_parts & _DATA_COMPONENTS) or value.startswith(("./", "../", "/mnt/nas/"))


def _quoted_candidates(text: str, suffix: str) -> set[str]:
    candidates: set[str] = set()
    if suffix == ".py":
        try:
            tree = ast.parse(text)
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if _looks_like_data_path(node.value):
                        candidates.add(node.value)
    for match in re.finditer(r'''["']([^"'\n]{1,1024})["']''', text):
        value = match.group(1)
        if _looks_like_data_path(value):
            candidates.add(value)
    return candidates


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
    worker_index_env: str | None = None,
    worker_index_argument: str | None = None,
    worker_index_base: int = 0,
) -> dict[str, Any]:
    root = stable_root.expanduser().resolve(strict=True)
    mount_base = PurePosixPath(mount_path)
    if not mount_base.is_absolute():
        raise ValueError("K8S_CPU_MOUNT_PATH must be absolute")
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

    job_dir = _manifest_dir(root, request_id)
    if (job_dir / "job.json").exists():
        raise ValueError(
            "request_id already has a submitted CPU Job; use a new request_id"
        )

    local_scripts: list[str] = []
    job_scripts: list[str] = []
    script_texts: list[tuple[str, str]] = []
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
        text = local.read_text(encoding="utf-8", errors="replace")
        local_scripts.append(str(local))
        job_scripts.append(job)
        script_texts.append((local.suffix.lower(), text))

    declared_script_set = {Path(value) for value in local_scripts}
    for token in command:
        candidate = token.split("=", 1)[1] if token.startswith("-") and "=" in token else token
        if any(character.isspace() for character in candidate):
            continue
        if PurePosixPath(candidate).suffix.lower() not in _INSPECTED_COMMAND_SUFFIXES:
            continue
        local, _ = _resolve_local_path(
            candidate,
            stable_root=root,
            mount_path=project_mount,
            must_exist=True,
            require_file=True,
        )
        if local not in declared_script_set:
            raise ValueError(
                f"command script/config must be declared in script_paths: {candidate}"
            )

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

    output_roots = [Path(value) for value in local_outputs]
    discovered_paths: list[dict[str, str]] = []
    for suffix, text in script_texts:
        for candidate in sorted(_quoted_candidates(text, suffix)):
            try:
                local, job = _resolve_local_path(
                    candidate,
                    stable_root=root,
                    mount_path=project_mount,
                    must_exist=True,
                )
            except ValueError:
                try:
                    local, job = _resolve_local_path(
                        candidate,
                        stable_root=root,
                        mount_path=project_mount,
                        must_exist=False,
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"script contains a data path that is not visible from "
                        f"AI_SCIENTIST_STABLE_WORKSPACE_ROOT: {candidate}"
                    ) from exc
                if not any(local == output or _inside(local, output) for output in output_roots):
                    raise ValueError(
                        f"script path does not exist and was not declared as output: {candidate}"
                    )
            discovered_paths.append({"source": candidate, "jobPath": job})

    combined_text = "\n".join(text for _, text in script_texts) + "\n" + " ".join(command)
    unsupported = [
        name for name, pattern in _COLLECTIVE_PATTERNS.items() if pattern.search(combined_text)
    ]
    if unsupported:
        raise ValueError(
            "K8s Indexed Job does not support cross-worker coordination: "
            + ", ".join(unsupported)
        )
    if shard_count > 1 and not (worker_index_env or worker_index_argument):
        if "JOB_COMPLETION_INDEX" not in combined_text:
            raise ValueError(
                "multi-shard jobs must use JOB_COMPLETION_INDEX directly or define "
                "worker_index_env/worker_index_argument mapping"
            )

    translated_command = [
        _translate_command_token(token, stable_root=root, mount_path=project_mount)
        for token in command
    ]
    job_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    launcher = job_dir / "launch.sh"
    launcher_job_path = str(
        project_mount.joinpath(*launcher.resolve().relative_to(root).parts)
    )
    lines = [
        "#!/bin/sh",
        "set -eu",
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
        "mountPath": str(mount_base),
        "projectWorkspacePath": str(project_mount),
        "command": ["/bin/sh", launcher_job_path],
        "args": [],
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
            "environmentVariable": worker_index_env,
            "argument": worker_index_argument,
            "base": worker_index_base,
        },
        "discoveredPaths": discovered_paths,
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
        "workerIndex": manifest["workerIndex"],
        "discoveredPaths": discovered_paths,
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
