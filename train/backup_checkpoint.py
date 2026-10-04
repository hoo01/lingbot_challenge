#!/usr/bin/env python3
"""Upload a recoverable run snapshot to a private Hugging Face model repo."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

DEFAULT_PATTERNS = [
    "assets/**",
    "configs/**",
    "source/**",
    "training_config.yaml",
    "run_metadata.json",
    "episode_splits.json",
    "lingbotvla_cli.yaml",
    "swanlab_run_id.txt",
    "metrics.jsonl",
    "backup_sha256.txt",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("path_in_repo")
    parser.add_argument("--commit-message", default="backup recoverable training checkpoint")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    run_dir = args.run_dir.resolve()
    states = list(run_dir.glob("checkpoints/global_step_*/training.pt"))
    if not states:
        raise SystemExit(f"No training.pt found below {run_dir}")

    checksum_file = run_dir / "backup_sha256.txt"
    latest_state = max(states, key=lambda path: int(path.parent.name.removeprefix("global_step_")))
    checksum_file.write_text(
        f"{sha256(latest_state)}  {latest_state.relative_to(run_dir).as_posix()}\n",
        encoding="utf-8",
    )

    api = HfApi()
    info = api.repo_info(args.repo_id, repo_type="model")
    if not info.private:
        raise SystemExit("Refusing to upload checkpoints: destination repository is not private")

    result = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=run_dir,
        path_in_repo=args.path_in_repo,
        allow_patterns=DEFAULT_PATTERNS + [latest_state.relative_to(run_dir).as_posix()],
        commit_message=args.commit_message,
    )
    print(result)


if __name__ == "__main__":
    main()
