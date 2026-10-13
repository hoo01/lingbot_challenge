#!/usr/bin/env python3
"""Fail-closed preflight for the two-rank H0/G2 full Action Expert run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import yaml


EXPECTED_BASE_IDENTITY = "2e727a4c9be84d1a9fe8952544fed421df65fd32b8e543f0bc468f6aeb508a81"
EXPECTED_QWEN_CONFIG_SHA256 = "edac7703329133edfc53e46ac0081835144c99d7eebf28b71c732694d435224d"

PHASE_RECIPES = {
    "h0-a": {
        "max_steps": 15000,
        "warmup_steps": 50,
        "lr_schedule": "cosine",
        "checkpoint_keep_steps": [5000, 10000, 15000],
        "parent_range": (1, 15000),
    },
    "h0-c1": {
        "max_steps": 25000,
        "warmup_steps": 0,
        "lr_schedule": "constant",
        "checkpoint_keep_steps": [25000],
        "parent_range": (15000, 25000),
    },
    "h0-c2": {
        "max_steps": 40000,
        "warmup_steps": 0,
        "lr_schedule": "constant",
        "checkpoint_keep_steps": [40000],
        "parent_range": (25000, 40000),
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"PRECHECK FAILED: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def write_json_atomic(path: Path, value: Any) -> None:
    require(path.parent.is_dir(), f"receipt parent directory does not exist: {path.parent}")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def resolve(root: Path, value: str) -> Path:
    expanded = value.replace("${TRAIN_ROOT}", str(root))
    path = Path(os.path.expandvars(expanded)).expanduser()
    return path if path.is_absolute() else root / path


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def manifest_content_sha256(manifest: dict[str, Any]) -> str:
    portable = json.loads(json.dumps(manifest))
    portable.pop("training_list", None)
    for entry in portable.get("datasets", []):
        entry.pop("path", None)
    return canonical_sha256(portable)


def safetensors_identity(checkpoint: Path) -> dict[str, Any]:
    files = sorted(checkpoint.glob("*.safetensors"))
    require(bool(files), f"no safetensors shards found in base checkpoint: {checkpoint}")
    inventory = [
        {"file": path.name, "size": path.stat().st_size, "sha256": sha256(path)}
        for path in files
    ]
    return {"sha256": canonical_sha256(inventory), "shards": inventory}


def validate_model_assets(
    root: Path,
    cfg: dict[str, Any],
    *,
    expected_base_identity: str = EXPECTED_BASE_IDENTITY,
    expected_qwen_config_sha256: str = EXPECTED_QWEN_CONFIG_SHA256,
) -> tuple[int, str, str]:
    """Verify the exact frozen base bytes and required local processor assets."""
    base_path = resolve(root, cfg["base_checkpoint"])
    qwen_path = resolve(root, cfg["qwen_path"])
    architecture_path = resolve(root, cfg["architecture_yaml"])
    require(base_path.is_dir(), f"base checkpoint directory missing: {base_path}")
    require(qwen_path.is_dir(), f"Qwen processor directory missing: {qwen_path}")
    require(architecture_path.is_file(), f"architecture YAML missing: {architecture_path}")

    base_config_path = base_path / "config.json"
    index_path = base_path / "model.safetensors.index.json"
    require(base_config_path.is_file(), f"base config missing: {base_config_path}")
    require(index_path.is_file(), f"base safetensors index missing: {index_path}")
    require(isinstance(load_json(base_config_path), dict), "base config.json must contain an object")
    index = load_json(index_path)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    require(isinstance(weight_map, dict) and weight_map, "base safetensors index has no weight_map")
    indexed_shards = set(weight_map.values())
    require(all(isinstance(name, str) and name and Path(name).name == name for name in indexed_shards),
            "base safetensors index contains an unsafe shard path")
    for shard_name in sorted(indexed_shards):
        shard_path = base_path / shard_name
        require(shard_path.is_file() and shard_path.stat().st_size > 0,
                f"base shard missing or empty: {shard_path}")

    identity = safetensors_identity(base_path)
    actual_shards = {item["file"] for item in identity["shards"]}
    require(actual_shards == indexed_shards,
            "base shard files do not exactly match model.safetensors.index.json")
    require(identity["sha256"] == expected_base_identity,
            f"base checkpoint identity is {identity['sha256']}, expected {expected_base_identity}")

    required_qwen_files = (
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    )
    for filename in required_qwen_files:
        path = qwen_path / filename
        require(path.is_file() and path.stat().st_size > 0,
                f"Qwen processor file missing or empty: {path}")
    qwen_config_sha = sha256(qwen_path / "config.json")
    require(qwen_config_sha == expected_qwen_config_sha256,
            f"Qwen config SHA is {qwen_config_sha}, expected {expected_qwen_config_sha256}")

    return len(actual_shards), identity["sha256"], qwen_config_sha


def validate_storage(output_dir: Path, min_free_gib: int) -> dict[str, Any]:
    require(isinstance(min_free_gib, int) and not isinstance(min_free_gib, bool) and min_free_gib > 0,
            "min_free_disk_gib must be a positive integer")
    output_dir = output_dir.resolve()
    require(not output_dir.exists() or output_dir.is_dir(),
            f"output path exists but is not a directory: {output_dir}")

    anchor = output_dir
    while not anchor.exists():
        parent = anchor.parent
        require(parent != anchor, f"cannot find an existing parent for output directory: {output_dir}")
        anchor = parent
    require(anchor.is_dir(), f"output ancestor is not a directory: {anchor}")
    require(os.access(anchor, os.W_OK | os.X_OK), f"output filesystem is not writable: {anchor}")

    usage = shutil.disk_usage(anchor)
    required_bytes = min_free_gib * 2**30
    require(usage.free >= required_bytes,
            f"output filesystem has {usage.free / 2**30:.1f} GiB free, requires at least {min_free_gib} GiB")
    return {
        "output_dir": str(output_dir),
        "checked_ancestor": str(anchor.resolve()),
        "device": int(anchor.stat().st_dev),
        "total_bytes": usage.total,
        "free_bytes": usage.free,
        "required_free_bytes": required_bytes,
    }


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def validate_parent_checkpoint(
    checkpoint: Path,
    *,
    phase: str,
    world_size: int,
    expected_step: int | None = None,
    expected_parent_phase: str | None = None,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    require(checkpoint.is_dir(), f"parent checkpoint directory missing: {checkpoint}")
    marker_path = checkpoint / "COMPLETED.json"
    require(marker_path.is_file(), f"parent checkpoint is incomplete: {checkpoint}")
    marker = load_json(marker_path)
    require(marker.get("format") == "lrvla_upstream_fsdp2_dcp_v1",
            "parent checkpoint format is not upstream FSDP2 DCP")
    require(marker.get("world_size") == world_size,
            "parent checkpoint world size differs from the requested topology")
    step = marker.get("global_step")
    require(isinstance(step, int), "parent checkpoint global_step is invalid")
    lower, upper = PHASE_RECIPES[phase]["parent_range"]
    require(lower <= step <= upper,
            f"parent step {step} is outside the allowed {phase} range {lower}..{upper}")
    if expected_step is not None:
        require(step == expected_step,
                f"parent step is {step}, expected the frozen stage boundary {expected_step}")
    require(marker.get("base_sha256") == EXPECTED_BASE_IDENTITY,
            "parent checkpoint was not created from the pinned original base")
    run_config_path = checkpoint.parent.parent / "training_config.yaml"
    require(run_config_path.is_file(), "parent run has no archived training_config.yaml")
    parent_phase = load_yaml(run_config_path).get("experiment_phase")
    require(parent_phase in PHASE_RECIPES, "parent run has no recognized experiment phase")
    if expected_parent_phase is not None:
        require(parent_phase == expected_parent_phase,
                f"parent run phase is {parent_phase}, expected {expected_parent_phase}")
    dcp_bytes = directory_size(checkpoint) - directory_size(checkpoint / "hf_ckpt_fp32")
    export_bytes = directory_size(checkpoint / "hf_ckpt_fp32")
    if expected_step is not None:
        require(export_bytes > 0, "frozen parent stage has no completed FP32 HF export")
        require((checkpoint.parent.parent / "action_audit_reference.pt").is_file(),
                "frozen parent stage has no action audit reference")
    return {
        "path": str(checkpoint),
        "global_step": step,
        "experiment_phase": parent_phase,
        "marker_sha256": sha256(marker_path),
        "dcp_bytes": dcp_bytes,
        "export_bytes": export_bytes,
    }


def validate_recipe(cfg: dict[str, Any], phase: str, world_size: int) -> int:
    require(phase in PHASE_RECIPES, f"unknown H0 phase: {phase}")
    expected = PHASE_RECIPES[phase]
    require(cfg.get("experiment_phase") == phase, f"config experiment_phase must be {phase}")
    require(cfg["mode"] == "expert", "mode must be expert")
    require(cfg["expert_last_n_layers"] == -1, "full Expert must use expert_last_n_layers=-1")
    require(cfg.get("distributed_strategy") == "upstream_fsdp2", "must use upstream_fsdp2")
    require(cfg.get("base_checkpoint") == "models/lingbot-vla-v2-6b", "H0 must use the pinned original base")
    require(cfg.get("qwen_path") == "models/Qwen3-VL-4B-Instruct", "Qwen processor path changed")
    require(cfg.get("architecture_yaml") == "vendor/lingbot-vla-v2/configs/vla/robotwin/robotwin.yaml",
            "robot architecture path changed")
    require(cfg.get("manifest") == "data/clean_manifest.json", "Clean manifest path changed")
    require(cfg.get("norm_stats") == "data/clean_norm_stats.json", "Clean norm path changed")
    require(cfg.get("action_mode") == "absolute", "action mode must be absolute")
    require(cfg.get("train_projections") is True, "projections must be trainable")
    require(cfg.get("gradient_checkpointing") is True, "gradient checkpointing must be enabled")
    require(cfg.get("image_augment") is False, "H0 must not add image augmentation")
    require(cfg.get("num_workers") == 2, "validated H0 recipe uses two workers per rank")
    require(cfg.get("chunk_size") == 50, "chunk_size must remain 50")
    require(cfg.get("action_chunk_tail") == "mask", "chunk tail must be masked")
    require(cfg.get("val_episodes") == 5, "each task must hold out episodes 45..49")
    require(cfg.get("action_audit") is True, "final export action audit must be enabled")
    require(cfg.get("export") is True, "formal FP32 export must be enabled")
    require(cfg.get("checkpoint") is True, "DCP checkpointing must be enabled")
    require(cfg.get("save_every") == 500, "formal checkpoints must be written every 500 steps")
    require(cfg.get("checkpoint_keep_steps") == expected["checkpoint_keep_steps"],
            f"{phase} milestone retention changed")
    require(cfg.get("warmup_steps") == expected["warmup_steps"], f"{phase} warmup changed")
    require(cfg.get("max_steps") == expected["max_steps"], f"{phase} target step changed")
    require(cfg.get("lr") == 5e-5, "base learning rate must remain 5e-5")
    require(cfg.get("weight_decay") == 0.01, "weight decay must remain 0.01")
    require(cfg.get("max_grad_norm") == 1.0, "gradient clipping must remain 1.0")
    require(cfg.get("lr_schedule") == expected["lr_schedule"], f"{phase} LR schedule changed")
    require(cfg.get("lr_minimum_multiplier") == 0.1, "minimum LR multiplier must remain 0.1")
    if phase == "h0-a":
        require("lr_constant_multiplier" not in cfg, "H0-A must use its cosine schedule")
        require("expected_resume_lr" not in cfg, "H0-A must not declare a continuation LR")
    else:
        require(cfg.get("lr_constant_multiplier") == 0.1,
                "continuation must use a constant 0.1 LR multiplier")
        require(cfg.get("expected_resume_lr") == 5e-6,
                "continuation resume LR must remain 5e-6")
    require(cfg.get("export_dtype") == "float32", "formal export must be float32")
    require(cfg.get("eval_every") == 250, "Clean validation interval must remain 250 steps")
    require(cfg.get("eval_batches") == 8, "formal Clean validation must use 8 batches")
    require(cfg.get("img_size") == 256, "image size must remain 256")
    require(cfg.get("seed") == 42, "training seed must remain 42")
    require(cfg.get("device") == "cuda", "formal training device must be CUDA")
    require(cfg.get("output_dir") is None, "output_dir must be supplied explicitly by the launcher")
    require(cfg.get("min_free_disk_gib") == 120, "formal minimum free disk must remain 120 GiB")
    require(cfg.get("batch_size") == 8, "validated per-rank batch size is 8")
    require(cfg.get("gradient_accumulation") == 4, "validated gradient accumulation is 4")
    global_batch = int(cfg["batch_size"]) * world_size * int(cfg["gradient_accumulation"])
    require(global_batch == 64, f"global batch is {global_batch}, expected 64")
    return global_batch


def validate_clean_manifest(root: Path, manifest: dict[str, Any]) -> tuple[int, int]:
    """Validate both the manifest declaration and the files it points to."""
    require(manifest.get("clean_only") is True, "manifest is not explicitly clean_only=true")
    require(manifest.get("complete_50_tasks") is True, "manifest is not the complete 50-task set")
    datasets = manifest.get("datasets")
    require(isinstance(datasets, list), "manifest datasets must be a list")
    require(len(datasets) == 50, f"manifest has {len(datasets)} datasets, expected 50")

    task_names = [entry.get("task") for entry in datasets]
    require(all(isinstance(name, str) and name for name in task_names), "every dataset needs a task name")
    require(len(set(task_names)) == 50, "manifest task names are not unique")
    require(manifest.get("total_episodes") == 2500, "manifest total_episodes must be 2500")

    total_frames = 0
    for entry in datasets:
        task = entry["task"]
        require(entry.get("episodes") == 50, f"{task}: episodes must be 50")
        require(entry.get("episode_ids") == list(range(50)), f"{task}: episode_ids must be 0..49")
        require(entry.get("format") == "v3.0", f"{task}: manifest format must be v3.0")

        frames = entry.get("frames")
        require(isinstance(frames, int) and frames > 0, f"{task}: frames must be a positive integer")
        total_frames += frames

        source = entry.get("source")
        require(isinstance(source, dict), f"{task}: source metadata is missing")
        require(source.get("setting") == "clean", f"{task}: source setting must be clean")
        require(source.get("episodes") == 50, f"{task}: source episodes must be 50")
        require(source.get("instructions") == "seen", f"{task}: source instructions must be seen")
        require("randomized" not in json.dumps(entry, ensure_ascii=False).lower(),
                f"{task}: randomized data reference found")

        dataset_path_value = entry.get("path")
        require(isinstance(dataset_path_value, str) and dataset_path_value,
                f"{task}: dataset path is missing")
        dataset_path = resolve(root, dataset_path_value)
        require(dataset_path.is_dir(), f"{task}: dataset directory missing: {dataset_path}")

        info_path = dataset_path / "meta" / "info.json"
        require(info_path.is_file(), f"{task}: LeRobot metadata missing: {info_path}")
        info = load_json(info_path)
        require(info.get("codebase_version") == "v3.0",
                f"{task}: LeRobot codebase_version must be v3.0")

        parquet_files = entry.get("parquet_files")
        require(isinstance(parquet_files, list) and parquet_files,
                f"{task}: parquet_files must be a non-empty list")
        for relative_path in parquet_files:
            require(isinstance(relative_path, str) and relative_path,
                    f"{task}: invalid parquet path in manifest")
            parquet_path = dataset_path / relative_path
            require(parquet_path.is_file(), f"{task}: parquet file missing: {parquet_path}")

    return len(datasets), total_frames


def validate_norm_stats(norm: dict[str, Any], provenance: dict[str, Any]) -> int:
    """Validate the expected absolute-action 12-arm + 2-effector schema."""
    count = norm.get("count")
    require(isinstance(count, int) and count > 0, "norm count must be a positive integer")
    require(count == provenance.get("count"), "norm count does not match provenance")

    groups = norm.get("norm_stats")
    require(isinstance(groups, dict), "norm_stats must be an object")
    expected_dimensions = {
        "observation.state.arm.position": 12,
        "observation.state.effector.position": 2,
        "action.arm.position": 12,
        "action.effector.position": 2,
    }
    require(set(groups) == set(expected_dimensions),
            "norm_stats keys do not match the absolute 14-D state/action schema")
    statistic_names = ("mean", "std", "q01", "q02", "q98", "q99", "min", "max")

    for group_name, dimension in expected_dimensions.items():
        statistics = groups[group_name]
        require(isinstance(statistics, dict), f"{group_name}: statistics must be an object")
        require(set(statistics) == set(statistic_names),
                f"{group_name}: statistic names are incomplete or unexpected")
        for statistic_name in statistic_names:
            values = statistics[statistic_name]
            require(isinstance(values, list) and len(values) == dimension,
                    f"{group_name}.{statistic_name}: expected {dimension} values")
            require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) for value in values),
                    f"{group_name}.{statistic_name}: values must be finite numbers")

        for index in range(dimension):
            ordered = [statistics[name][index] for name in ("min", "q01", "q02", "q98", "q99", "max")]
            require(ordered == sorted(ordered),
                    f"{group_name}: quantiles are not ordered at dimension {index}")
            require(statistics["std"][index] > 0,
                    f"{group_name}.std must be positive at dimension {index}")

    return count


def git_revision(repository: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise SystemExit(
            f"PRECHECK FAILED: cannot identify Git revision for {repository}: {error}"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--norm-provenance", required=True, type=Path)
    parser.add_argument("--world-size", default=2, type=int, choices=(2,))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--phase", choices=tuple(PHASE_RECIPES), default="h0-a")
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--expected-parent-step", type=int)
    parser.add_argument("--expected-parent-phase", choices=tuple(PHASE_RECIPES))
    args = parser.parse_args()

    root = args.training_root.resolve()
    config_path = args.config.resolve()
    provenance_path = args.norm_provenance.resolve()
    require(root.is_dir(), f"training root does not exist: {root}")
    require(config_path.is_file(), f"config does not exist: {config_path}")
    require(provenance_path.is_file(), f"norm provenance missing: {provenance_path}")

    required_runtime = (
        root / "scripts" / "train.py",
        root / "src" / "lrvla" / "training.py",
        root / "src" / "lrvla" / "training_checkpoint.py",
        root / "src" / "lrvla" / "training_distributed.py",
        root / "src" / "lrvla" / "training_schedule.py",
    )
    for path in required_runtime:
        require(path.is_file(), f"training runtime file missing: {path}")

    cfg = load_yaml(config_path)
    global_batch = validate_recipe(cfg, args.phase, args.world_size)
    require(args.phase == "h0-a" or args.parent_checkpoint is not None,
            f"{args.phase} requires a parent checkpoint")
    require(args.expected_parent_step is None or args.parent_checkpoint is not None,
            "expected parent step requires a parent checkpoint")
    require(args.expected_parent_phase is None or args.parent_checkpoint is not None,
            "expected parent phase requires a parent checkpoint")

    base_shards, base_identity, qwen_config_sha = validate_model_assets(root, cfg)
    output_dir = (args.output_dir.resolve() if args.output_dir is not None else (root / "runs").resolve())
    storage = validate_storage(output_dir, cfg["min_free_disk_gib"])
    parent = None
    if args.parent_checkpoint is not None:
        parent = validate_parent_checkpoint(
            args.parent_checkpoint,
            phase=args.phase,
            world_size=args.world_size,
            expected_step=args.expected_parent_step,
            expected_parent_phase=args.expected_parent_phase,
        )
        if args.phase != "h0-a":
            continuation_required = math.ceil(1.2 * (2 * parent["dcp_bytes"] + parent["export_bytes"]))
            require(storage["free_bytes"] >= continuation_required,
                    "output filesystem lacks measured continuation checkpoint/export headroom")
            storage["continuation_required_free_bytes"] = continuation_required

    manifest_path = resolve(root, cfg["manifest"])
    norm_path = resolve(root, cfg["norm_stats"])
    robot_path = root / "vendor" / "lingbot-vla-v2" / "configs" / "robot_configs" / "robotwin.yaml"
    for label, path in (("manifest", manifest_path), ("norm", norm_path), ("robot config", robot_path)):
        require(path.is_file(), f"{label} missing: {path}")

    vendor_root = root / "vendor" / "lingbot-vla-v2"
    expected_vendor_revision = "be969b8fd117fb70550c5d4bf4bc328211b5b1b6"
    if (vendor_root / ".git").exists():
        actual_vendor_revision = git_revision(vendor_root)
        require(
            actual_vendor_revision == expected_vendor_revision,
            f"LingBot vendor revision is {actual_vendor_revision}, expected {expected_vendor_revision}",
        )
    else:
        snapshot_path = vendor_root / ".lrvla_snapshot.json"
        require(snapshot_path.is_file(), "vendor tree has neither Git metadata nor snapshot identity")
        import sys

        sys.path.insert(0, str(root / "src"))
        from lrvla.vendor_sources import verify_snapshot

        try:
            verify_snapshot(vendor_root, expected_vendor_revision)
        except (OSError, ValueError) as error:
            raise SystemExit(f"PRECHECK FAILED: invalid LingBot vendor snapshot: {error}") from error

    manifest = load_json(manifest_path)
    task_count, total_frames = validate_clean_manifest(root, manifest)

    provenance = load_json(provenance_path)
    require(provenance.get("clean_only") is True, "norm provenance is not clean-only")
    manifest_sha = sha256(manifest_path)
    content_sha = manifest_content_sha256(manifest)
    require(
        provenance.get("manifest_sha256") == manifest_sha
        or provenance.get("manifest_content_sha256") == content_sha,
        "norm provenance does not match the manifest bytes or portable content identity",
    )
    require(provenance.get("train_episode_ids") == list(range(45)), "norm train split must be episodes 0..44")
    require(provenance.get("val_episode_ids") == list(range(45, 50)), "norm val split must be episodes 45..49")
    expected_norm_sha = provenance.get("norm_stats_sha256")
    require(expected_norm_sha == sha256(norm_path), "norm SHA does not match provenance")
    norm = load_json(norm_path)
    norm_count = validate_norm_stats(norm, provenance)

    robot = load_yaml(robot_path)
    actions = robot.get("actions", [])
    require(len(actions) == 2, "robot config must contain arm and effector action groups")
    for group in actions:
        spec = next(iter(group.values()))
        require(spec.get("subtract_state") is False, "all H0 actions must be absolute")
        require("relative_type" not in spec, "relative_type is forbidden in H0")

    receipt = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "commit": git_revision(root),
            "dirty": False,
        },
        "config": {
            "path": str(config_path),
            "sha256": sha256(config_path),
            "base_checkpoint": cfg["base_checkpoint"],
            "distributed_strategy": cfg["distributed_strategy"],
            "expert_last_n_layers": cfg["expert_last_n_layers"],
            "experiment_phase": args.phase,
        },
        "base": {
            "path": str(resolve(root, cfg["base_checkpoint"])),
            "identity_sha256": base_identity,
            "shards": base_shards,
        },
        "qwen": {
            "path": str(resolve(root, cfg["qwen_path"])),
            "config_sha256": qwen_config_sha,
        },
        "vendor": {
            "path": str(vendor_root),
            "revision": expected_vendor_revision,
        },
        "data": {
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "manifest_content_sha256": content_sha,
            "clean_tasks": task_count,
            "clean_frames": total_frames,
        },
        "norm": {
            "path": str(norm_path),
            "sha256": sha256(norm_path),
            "count": norm_count,
            "provenance": str(provenance_path),
            "provenance_sha256": sha256(provenance_path),
        },
        "robot_config": {
            "path": str(robot_path),
            "sha256": sha256(robot_path),
            "action_mode": "absolute",
        },
        "distributed": {
            "world_size": args.world_size,
            "global_batch_size": global_batch,
        },
        "storage": storage,
    }
    if parent is not None:
        receipt["parent_checkpoint"] = parent
    if args.receipt is not None:
        receipt_path = args.receipt.resolve()
        write_json_atomic(receipt_path, receipt)
        print(f"receipt={receipt_path}")

    print("PRECHECK PASSED")
    print(f"config={config_path}")
    print(f"manifest_sha256={manifest_sha}")
    print(f"manifest_content_sha256={content_sha}")
    print(f"norm_sha256={sha256(norm_path)}")
    print(f"robot_config_sha256={sha256(robot_path)}")
    print(f"vendor_revision={expected_vendor_revision}")
    print(f"base_shards={base_shards}")
    print(f"base_identity_sha256={base_identity}")
    print(f"qwen_config_sha256={qwen_config_sha}")
    print(f"output_dir={storage['output_dir']}")
    print(f"disk_free_gib={storage['free_bytes'] / 2**30:.1f}")
    print(f"disk_required_gib={storage['required_free_bytes'] / 2**30:.1f}")
    print(f"clean_tasks={task_count}")
    print(f"clean_frames={total_frames}")
    print(f"norm_count={norm_count}")
    print(f"world_size={args.world_size}")
    print(f"global_batch_size={global_batch}")
    print("required_post-start_check=trainable inventory must include expert layers 0..35")


if __name__ == "__main__":
    main()
