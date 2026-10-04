#!/usr/bin/env python3
"""Fail closed unless the prepared dataset is the complete clean RoboTwin set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/clean_manifest.json"))
    parser.add_argument("--norm-stats", type=Path, default=Path("data/norm_stats_clean.json"))
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args()

    provenance_path = args.provenance or args.norm_stats.with_suffix(".provenance.json")
    missing = [path for path in (args.manifest, args.norm_stats) if not path.is_file()]
    if missing:
        raise SystemExit("Missing required file(s): " + ", ".join(map(str, missing)))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    datasets = manifest.get("datasets", [])

    errors: list[str] = []
    if manifest.get("clean_only") is not True:
        errors.append("manifest.clean_only must be true")
    if manifest.get("complete_50_tasks") is not True:
        errors.append("manifest.complete_50_tasks must be true")
    if len(datasets) != 50:
        errors.append(f"expected 50 tasks, found {len(datasets)}")

    tasks = [item.get("task") for item in datasets]
    if len(set(tasks)) != len(tasks):
        errors.append("task names are not unique")
    for item in datasets:
        task = item.get("task", "<unknown>")
        if item.get("episodes") != 50 or len(item.get("episode_ids", [])) != 50:
            errors.append(f"{task}: expected exactly 50 episodes")
        serialized = json.dumps(item).lower()
        if "randomized" in serialized:
            errors.append(f"{task}: randomized data reference found")
        info = Path(item.get("path", "")) / "meta" / "info.json"
        if info.is_file():
            version = json.loads(info.read_text(encoding="utf-8")).get("codebase_version")
            if version != "v3.0":
                errors.append(f"{task}: expected LeRobot v3.0, found {version!r}")

    norm = json.loads(args.norm_stats.read_text(encoding="utf-8"))
    if not norm.get("norm_stats") or not norm.get("count"):
        errors.append("norm stats is missing norm_stats or count")

    if not provenance_path.is_file():
        errors.append(f"missing provenance sidecar: {provenance_path}")
    else:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("clean_only") is not True:
            errors.append("provenance.clean_only must be true")
        expected = provenance.get("norm_sha256")
        actual = digest(args.norm_stats)
        if expected != actual:
            errors.append(f"norm SHA mismatch: provenance={expected}, actual={actual}")

    if errors:
        raise SystemExit("Clean-data verification failed:\n- " + "\n- ".join(errors))

    print(json.dumps({
        "tasks": len(datasets),
        "episodes": sum(item["episodes"] for item in datasets),
        "manifest_sha256": digest(args.manifest),
        "norm_sha256": digest(args.norm_stats),
        "provenance_sha256": digest(provenance_path),
    }, indent=2))


if __name__ == "__main__":
    main()
