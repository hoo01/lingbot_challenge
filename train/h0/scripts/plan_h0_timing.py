#!/usr/bin/env python3
"""Build the frozen H0 milestone budget from JSONL training metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import tempfile
from typing import Any


DEFAULT_TARGETS = (5000, 10000, 15000, 25000, 40000)


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if isinstance(value, dict) and isinstance(value.get("step"), int):
                records.append(value)
    if not records:
        raise ValueError(f"no step records found in {path}")
    return records


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def positive_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
            values.append(float(value))
    return values


def build_budget(
    records: list[dict[str, Any]],
    *,
    checkpoint_records: list[dict[str, Any]] | None = None,
    stable_start: int = 21,
    stable_end: int = 99,
    validation_scale: float = 1.0,
    checkpoint_seconds: float | None = None,
    setup_seconds: float | None = None,
    uncertainty: float = 1.2,
    rental_hours: float = 144.0,
    finalization_hours: float = 24.0,
    fault_hours: float = 12.0,
    current_step: int = 0,
    elapsed_rental_hours: float = 0.0,
    targets: tuple[int, ...] = DEFAULT_TARGETS,
) -> dict[str, Any]:
    stable_by_step = {}
    for record in records:
        value = record.get("step_seconds")
        if (stable_start <= record["step"] <= stable_end
                and isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value > 0):
            stable_by_step[record["step"]] = float(value)
    expected_steps = set(range(stable_start, stable_end + 1))
    missing_steps = sorted(expected_steps - set(stable_by_step))
    if missing_steps:
        raise ValueError(
            f"frozen step_seconds window {stable_start}..{stable_end} is incomplete; "
            f"missing {missing_steps[:8]}"
        )
    stable = [stable_by_step[step] for step in range(stable_start, stable_end + 1)]
    validation_values = positive_values(records, "validation_seconds")
    validation_seconds = (
        statistics.median(validation_values) * validation_scale if validation_values else None
    )
    checkpoint_source = checkpoint_records if checkpoint_records is not None else records
    checkpoint_values = positive_values(checkpoint_source, "checkpoint_seconds")
    if checkpoint_seconds is None and checkpoint_values:
        checkpoint_seconds = statistics.median(checkpoint_values)
    setup_values = positive_values(records, "setup_seconds")
    if setup_seconds is None and setup_values:
        setup_seconds = setup_values[0]
    usable_hours = rental_hours - finalization_hours - fault_hours - elapsed_rental_hours
    complete = validation_seconds is not None and checkpoint_seconds is not None and setup_seconds is not None
    step_median = statistics.median(stable)
    rows = []
    for target in targets:
        remaining_steps = max(0, target - current_step)
        remaining_validations = max(0, target // 250 - current_step // 250)
        remaining_checkpoints = max(0, target // 500 - current_step // 500)
        known_seconds = (0.0 if current_step else (setup_seconds or 0.0)) + remaining_steps * step_median
        if validation_seconds is not None:
            known_seconds += remaining_validations * validation_seconds
        if checkpoint_seconds is not None:
            known_seconds += remaining_checkpoints * checkpoint_seconds
        budget_hours = uncertainty * known_seconds / 3600
        rows.append({
            "target_step": target,
            "remaining_steps": remaining_steps,
            "validation_events": remaining_validations,
            "checkpoint_events": remaining_checkpoints,
            "budget_hours": budget_hours,
            "hours_of_usable_window_remaining": usable_hours - budget_hours,
            "fits_rental_window": bool(complete and budget_hours <= usable_hours),
        })
    return {
        "schema_version": 1,
        "timing_gate_complete": complete and current_step == 0,
        "status": "complete" if complete else "provisional_missing_setup_validation_or_checkpoint_timing",
        "stable_window": {"first_step": stable_start, "last_step": stable_end, "samples": len(stable)},
        "step_seconds": {
            "median": step_median,
            "p90": percentile(stable, 0.90),
            "mean": statistics.fmean(stable),
        },
        "validation_seconds": validation_seconds,
        "validation_scale": validation_scale,
        "checkpoint_seconds": checkpoint_seconds,
        "setup_seconds": setup_seconds,
        "uncertainty_multiplier": uncertainty,
        "rental": {
            "total_hours": rental_hours,
            "elapsed_hours": elapsed_rental_hours,
            "finalization_reserve_hours": finalization_hours,
            "fault_reserve_hours": fault_hours,
            "usable_hours_remaining": usable_hours,
        },
        "current_step": current_step,
        "targets": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    step = report["step_seconds"]
    lines = [
        "# H0 timing budget",
        "",
        f"Status: `{report['status']}`",
        "",
        f"Steady step time: median {step['median']:.3f}s, p90 {step['p90']:.3f}s, "
        f"mean {step['mean']:.3f}s, n={report['stable_window']['samples']}.",
        f"Validation estimate: {report['validation_seconds'] if report['validation_seconds'] is not None else 'MISSING'} s.",
        f"Checkpoint estimate: {report['checkpoint_seconds'] if report['checkpoint_seconds'] is not None else 'MISSING'} s.",
        f"Formal setup estimate: {report['setup_seconds'] if report['setup_seconds'] is not None else 'MISSING'} s.",
        "",
        "| Target | Budget hours | Usable-window margin | Fits |",
        "| ---: | ---: | ---: | :---: |",
    ]
    for row in report["targets"]:
        lines.append(
            f"| {row['target_step']:,} | {row['budget_hours']:.2f} | "
            f"{row['hours_of_usable_window_remaining']:.2f} | "
            f"{'yes' if row['fits_rental_window'] else 'no'} |"
        )
    if report["timing_gate_complete"]:
        lines.extend(("", "Timing gate complete; recalibrate this budget at formal step 500.", ""))
    else:
        lines.extend(("", "A provisional report is not permission to launch formal H0-A.", ""))
    return "\n".join(lines)


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary_name = stream.name
            stream.write(text)
            stream.flush()
        Path(temporary_name).replace(path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--checkpoint-metrics", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validation-scale", type=float, default=1.0)
    parser.add_argument("--checkpoint-seconds", type=float)
    parser.add_argument("--setup-seconds", type=float)
    parser.add_argument("--uncertainty", type=float, default=1.2)
    parser.add_argument("--rental-hours", type=float, default=144.0)
    parser.add_argument("--finalization-hours", type=float, default=24.0)
    parser.add_argument("--fault-hours", type=float, default=12.0)
    parser.add_argument("--current-step", type=int, default=0)
    parser.add_argument("--elapsed-rental-hours", type=float, default=0.0)
    args = parser.parse_args()
    if args.validation_scale <= 0 or args.uncertainty < 1:
        parser.error("validation scale must be positive and uncertainty must be at least 1")
    if args.checkpoint_seconds is not None and args.checkpoint_seconds <= 0:
        parser.error("checkpoint seconds must be positive")
    non_negative = [args.rental_hours, args.finalization_hours, args.fault_hours,
                    args.elapsed_rental_hours, args.current_step]
    if args.setup_seconds is not None:
        non_negative.append(args.setup_seconds)
    if min(non_negative) < 0:
        parser.error("time and step arguments must be non-negative")
    records = load_records(args.metrics)
    checkpoint_records = load_records(args.checkpoint_metrics) if args.checkpoint_metrics else None
    report = build_budget(
        records,
        checkpoint_records=checkpoint_records,
        validation_scale=args.validation_scale,
        checkpoint_seconds=args.checkpoint_seconds,
        setup_seconds=args.setup_seconds,
        uncertainty=args.uncertainty,
        rental_hours=args.rental_hours,
        finalization_hours=args.finalization_hours,
        fault_hours=args.fault_hours,
        current_step=args.current_step,
        elapsed_rental_hours=args.elapsed_rental_hours,
    )
    output = args.output or args.metrics.with_name("h0_timing_budget.json")
    write_atomic(output, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    markdown = output.with_suffix(".md")
    write_atomic(markdown, render_markdown(report))
    print(render_markdown(report), end="")
    print(f"json={output.resolve()}")
    print(f"markdown={markdown.resolve()}")


if __name__ == "__main__":
    main()
