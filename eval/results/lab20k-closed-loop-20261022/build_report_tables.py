#!/usr/bin/env python3
"""Rebuild and verify the four Lab20k closed-loop evaluation archives."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = {
    "step6000-official": {"job_id": "48a4a918600d", "label": "6k official"},
    "step6000-pace": {"job_id": "95834709f412", "label": "6k PACE"},
    "step7000-official": {"job_id": "7ee63959db2b", "label": "7k official"},
    "step8000-pace": {"job_id": "5eb52c93c126", "label": "8k PACE"},
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setting_successes(report: dict, setting: str) -> int:
    return sum(group["successes"] for group in report["groups"] if group["setting"] == setting)


def main() -> None:
    overall_rows = []
    summaries = []
    checks = {}

    for dirname, identity in RUNS.items():
        run_dir = ROOT / dirname
        job = load(run_dir / "job.json")
        request = load(run_dir / "request.json")
        report = load(run_dir / "report.json")
        checkpoint = load(run_dir / "checkpoint.json")
        runtime = load(run_dir / "server-runtime.json")
        source = load(run_dir / "official-defaults-manifest.json")
        timing = load(run_dir / "timing.json")

        groups = report["groups"]
        episodes = report["episodes"]
        tasks = request["tasks"]
        run_checks = {
            "job_id_matches": job["id"] == identity["job_id"],
            "job_completed": job["status"] == "completed" and job["error"] is None,
            "planned_500": report["planned"] == 500,
            "valid_500": report["valid"] == 500,
            "groups_100": len(groups) == 100,
            "tasks_50": len(tasks) == len(set(tasks)) == 50,
            "two_settings": request["settings"] == ["demo_clean", "demo_randomized"],
            "five_valid_each_group": all(group["valid"] == 5 for group in groups),
            "group_valid_sum": sum(group["valid"] for group in groups) == report["valid"],
            "group_success_sum": sum(group["successes"] for group in groups) == report["successes"],
            "group_setup_error_sum": sum(group["setup_errors"] for group in groups) == report["setup_errors"],
            "episode_attempt_count": len(episodes) == report["attempted"],
            "norm_binding_matches": (
                checkpoint.get("norm_sha256")
                == runtime.get("normalization_sha256")
                == source.get("normalization_sha256")
            ),
            "robot_binding_matches": (
                checkpoint.get("robot_config_sha256")
                == runtime.get("robot_config_sha256")
                == source.get("robot_config_sha256")
            ),
            "fp32": runtime.get("dtype") == "fp32",
            "relative_arm_absolute_gripper": runtime.get("action_subtract_state")
            == {"action.arm.position": True, "action.effector.position": False},
        }
        if not all(run_checks.values()):
            failed = [name for name, passed in run_checks.items() if not passed]
            raise AssertionError(f"{dirname}: failed checks: {failed}")

        write_csv(
            run_dir / "by-group.csv",
            groups,
            [
                "task", "setting", "planned", "attempted", "valid", "successes",
                "setup_errors", "execution_errors", "mean_seconds",
            ],
        )

        by_task = defaultdict(lambda: defaultdict(float))
        for group in groups:
            row = by_task[group["task"]]
            for key in ("planned", "attempted", "valid", "successes", "setup_errors", "execution_errors"):
                row[key] += group[key]
            row[f"{group['setting']}_successes"] += group["successes"]
        task_rows = []
        for task in sorted(by_task):
            row = {"task": task, **{key: int(value) for key, value in by_task[task].items()}}
            row["success_rate"] = row["successes"] / row["valid"]
            task_rows.append(row)
        write_csv(
            run_dir / "by-task.csv",
            task_rows,
            [
                "task", "planned", "attempted", "valid", "successes", "success_rate",
                "demo_clean_successes", "demo_randomized_successes", "setup_errors", "execution_errors",
            ],
        )

        episode_fields = [
            "task", "setting", "seed", "status", "success", "actions", "action_limit", "chunks",
            "mean_inference_ms", "mean_execution_horizon", "min_execution_horizon",
            "max_execution_horizon", "horizon_selector_fallbacks", "autohorizon_fallbacks",
            "seconds", "expert_attempts", "execution_segment", "instruction",
        ]
        write_csv(run_dir / "episodes.csv", episodes, episode_fields)

        clean_successes = setting_successes(report, "demo_clean")
        randomized_successes = setting_successes(report, "demo_randomized")
        pace = runtime.get("pace") or {}
        auto = report.get("autohorizon")
        summary = {
            "archive": dirname,
            "label": identity["label"],
            "job_id": job["id"],
            "title": job["title"],
            "status": job["status"],
            "created_utc": job["created"],
            "completed_utc": job["updated"],
            "checkpoint_id_at_run": request["checkpoint_id"],
            "checkpoint_name": checkpoint["name"],
            "checkpoint_path_at_run": runtime["checkpoint"],
            "checkpoint_original_source": checkpoint.get("original_source"),
            "protocol": request["protocol"],
            "inference_profile": runtime["inference_profile"],
            "official_revision": source["official_revision"],
            "pace_commit": pace.get("commit"),
            "precision": runtime["dtype"],
            "execute": runtime["execute"],
            "flow_steps": runtime["num_steps"],
            "image_size": runtime["image_size"],
            "compile": runtime["use_compile"],
            "moe_backend": runtime["moe_backend"],
            "normalization_sha256": runtime["normalization_sha256"],
            "robot_config_sha256": runtime["robot_config_sha256"],
            "sampling_seed_at_process_start": runtime["sampling_seed_at_process_start"],
            "primary_seeds": request.get("primary_seeds", request.get("seeds")),
            "seed_start": request.get("seed_start"),
            "setup_error_policy": request.get("setup_error_policy", request.get("seed_policy")),
            "planned": report["planned"],
            "attempted": report["attempted"],
            "valid": report["valid"],
            "successes": report["successes"],
            "success_rate": report["success_rate"],
            "clean_successes": clean_successes,
            "randomized_successes": randomized_successes,
            "setup_errors": report["setup_errors"],
            "execution_errors": report["execution_errors"],
            "autohorizon": auto,
            "timing_seconds": timing,
        }
        dump(run_dir / "summary.json", summary)
        summaries.append(summary)
        checks[dirname] = run_checks
        overall_rows.append({
            "archive": dirname,
            "job_id": job["id"],
            "checkpoint": checkpoint["name"],
            "protocol": request["protocol"],
            "valid": report["valid"],
            "successes": report["successes"],
            "success_rate": report["success_rate"],
            "clean_successes": clean_successes,
            "randomized_successes": randomized_successes,
            "attempted": report["attempted"],
            "setup_errors": report["setup_errors"],
            "execution_errors": report["execution_errors"],
            "mean_execution_horizon": auto.get("mean_execution_horizon") if auto else runtime["execute"],
            "evaluation_wall_hours": timing["evaluation_wall_seconds"] / 3600,
        })

    write_csv(
        ROOT / "comparison.csv",
        overall_rows,
        [
            "archive", "job_id", "checkpoint", "protocol", "valid", "successes", "success_rate",
            "clean_successes", "randomized_successes", "attempted", "setup_errors", "execution_errors",
            "mean_execution_horizon", "evaluation_wall_hours",
        ],
    )
    dump(ROOT / "summary.json", {"report_date": "2026-10-22", "runs": summaries})
    dump(
        ROOT / "verification.json",
        {
            "passed": all(all(run.values()) for run in checks.values()),
            "report_date": "2026-10-22",
            "runs": checks,
            "scope": "Offline verification of archived lightweight run evidence; no simulation or model inference was rerun.",
        },
    )

    manifest = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name == "manifest.json" or "__pycache__" in path.parts:
            continue
        manifest.append({
            "file": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    dump(ROOT / "manifest.json", manifest)
    print(f"verified {len(RUNS)} completed runs; wrote {len(manifest)} manifest entries")


if __name__ == "__main__":
    main()
