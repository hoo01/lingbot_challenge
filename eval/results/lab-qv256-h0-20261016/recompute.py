"""Recount archived receipts and CSVs; never run a model or contact a server."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REFERENCE_SHA = "ec14ae2092bca38a689feca77b98dc89317bf9add20b820dd67e2d4bd503dd20"
SETTINGS = ("demo_clean", "demo_randomized")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def key(row):
    return row["task"], row["setting"], int(row["seed"])


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def manifest_hashes(value):
    files = value["files"]
    if isinstance(files, list):
        return {entry["name"]: entry["sha256"] for entry in files}
    return {name: entry["sha256"] if isinstance(entry, dict) else entry
            for name, entry in files.items()}


def recount():
    sources = json.loads((ROOT / "sources.json").read_bytes())
    for name, expected in sources["files"].items():
        require(sha((ROOT / name).read_bytes()) == expected, f"Source SHA mismatch: {name}")
    reference_raw = (ROOT / "evidence/paired-reference.json").read_bytes()
    require(sha(reference_raw) == REFERENCE_SHA, "Original E0 reference changed")
    reference = json.loads(reference_raw)["scenes"]
    keys = {key(row) for row in reference}
    require(len(reference) == len(keys) == 500, "Expected 500 unique reference keys")
    require(len(Counter(k[:2] for k in keys)) == 100, "Expected 100 reference groups")

    runs, indexed = [], {}
    for spec in sources["runs"]:
        folder = ROOT / "evidence" / spec["id"]
        receipt = json.loads((folder / "verification.json").read_bytes())
        manifest = json.loads((folder / "receipt-manifest.json").read_bytes())
        hashes = manifest_hashes(manifest)
        for name in hashes:
            require(Path(name).name == name, "Receipt member must be a local filename")
            require(sha((folder / name).read_bytes()) == hashes[name], f"Receipt SHA: {spec['id']}/{name}")
        require(receipt["passed"] is True, f"Completion audit failed: {spec['id']}")
        require(receipt["job_id"] == spec["job_id"], f"Job identity: {spec['id']}")
        require(receipt["model_id"] == spec["model_id"], f"Model identity: {spec['id']}")
        rows = list(csv.DictReader(io.StringIO((folder / "paired-outcomes.csv").read_text(encoding="utf-8-sig"))))
        table = {key(row): row for row in rows}
        require(len(rows) == len(table) == 500 and table.keys() == keys, f"Original 500-key coverage: {spec['id']}")
        groups = Counter(k[:2] for k in table)
        require(len(groups) == 100 and set(groups.values()) == {5}, f"Group coverage: {spec['id']}")
        require(all(row["success"] in ("True", "False") for row in rows), "Invalid success value")
        statuses = Counter(row["status"] for row in rows)
        require(statuses == {"policy_completed": 500}, f"Unexpected completion status: {spec['id']}")
        completed, timeout = statuses["policy_completed"], statuses["policy_timeout"]
        successes = sum(row["success"] == "True" for row in rows)
        by_setting = {setting: sum(row["success"] == "True" for row in rows if row["setting"] == setting)
                      for setting in SETTINGS}
        require(receipt["completed"] == completed and receipt["timeout_zero"] == timeout, "Receipt/CSV completion mismatch")
        require(receipt.get("success", receipt.get("successes")) == successes, "Receipt/CSV success mismatch")
        declared = receipt.get("success_by_setting") or {s: receipt["by_setting"][s]["successes"] for s in SETTINGS}
        require(declared == by_setting, "Receipt/CSV setting mismatch")
        require(all(sum(row["setting"] == s for row in rows) == 250 for s in SETTINGS), "Setting denominator mismatch")
        require(receipt["unique_keys"] == 500 and receipt["groups"] == 100, "Receipt key counts mismatch")
        runs.append(dict(id=spec["id"], label=spec["label"], step=spec["step"], job_id=spec["job_id"],
                         model_id=spec["model_id"], completed=completed, timeout_zero=timeout,
                         success=successes, total=500, success_rate=successes / 500,
                         clean_success=by_setting[SETTINGS[0]], clean_total=250,
                         randomized_success=by_setting[SETTINGS[1]], randomized_total=250,
                         finite_blocks=receipt.get("finite_prediction_chunks", receipt.get("finite_50x14_blocks")),
                         initial_rgb_pngs=receipt.get("initial_rgb_pngs_checked", receipt.get("initial_RGB_PNG_verified")),
                         terminal_pngs=receipt.get("terminal_pngs_checked", receipt.get("terminal_PNG_verified")),
                         video_metadata=receipt.get("nonempty_video_metadata_checked",
                                            receipt.get("nonempty_video_files_metadata_checked",
                                                        receipt.get("nonempty_video_metadata"))),
                         verification_sha256=sha((folder / "verification.json").read_bytes())))
        indexed[spec["id"]] = table

    lab, h0 = indexed["lab-20000"], indexed["h0-15000"]
    paired_rows = [dict(task=k[0], setting=k[1], seed=k[2],
                        lab20k_success=lab[k]["success"] == "True", h0_success=h0[k]["success"] == "True")
                   for k in sorted(keys)]
    pairs = []
    for setting in ("all", *SETTINGS):
        selected = [r for r in paired_rows if setting == "all" or r["setting"] == setting]
        counts = Counter((r["lab20k_success"], r["h0_success"]) for r in selected)
        pairs.append(dict(setting=setting, total=len(selected), both_success=counts[True, True],
                          lab_only=counts[True, False], h0_only=counts[False, True], both_fail=counts[False, False],
                          net_extra_successes=counts[True, False] - counts[False, True]))

    task_rows = []
    for task in sorted({k[0] for k in keys}):
        row = dict(task=task)
        for spec in sources["runs"]:
            table = indexed[spec["id"]]
            for setting, suffix in zip(SETTINGS, ("clean", "randomized")):
                row[spec["id"] + "_" + suffix] = sum(r["success"] == "True" for k, r in table.items()
                                                                  if k[:2] == (task, setting))
        row["lab20k_total"] = row["lab-20000_clean"] + row["lab-20000_randomized"]
        row["h0_total"] = row["h0-15000_clean"] + row["h0-15000_randomized"]
        row["difference"] = row["lab20k_total"] - row["h0_total"]
        task_rows.append(row)
    summary = dict(passed=True, reference_sha256=REFERENCE_SHA, original_keys=500, groups=100,
                   runs=runs, paired_lab20k_h0=pairs,
                   task_comparison=dict(lab_higher=sum(r["difference"] > 0 for r in task_rows),
                                        tied=sum(r["difference"] == 0 for r in task_rows),
                                        h0_higher=sum(r["difference"] < 0 for r in task_rows)),
                   scope="Offline recount of archived small receipts and CSVs; no image/video/weight re-audit or new simulation.")
    flat = [{k: r[k] for k in ("id", "label", "step", "completed", "timeout_zero", "success", "total", "success_rate",
                               "clean_success", "clean_total", "randomized_success", "randomized_total")}
            for r in runs]
    return summary, {"summary.json": json_bytes(summary), "summary.csv": csv_bytes(flat),
                     "by-task.csv": csv_bytes(task_rows), "paired-lab20k-h0.csv": csv_bytes(paired_rows)}


def plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {row["id"]: row for row in summary["runs"]}
    steps = (8000, 14000, 16000, 18000, 20000)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3), layout="constrained")
    for label, field, denominator, color in [("All", "success", 500, "#244b75"),
                                             ("Clean", "clean_success", 250, "#218577"),
                                             ("Randomized", "randomized_success", 250, "#bd674b")]:
        axes[0].plot([s / 1000 for s in steps], [100 * runs[f"lab-{s}"][field] / denominator for s in steps],
                     marker="o", label=label, color=color)
    axes[0].set(xlabel="Lab optimizer steps (thousands)", ylabel="Success rate (%)",
                title="Lab q/v LoRA r256: completed checkpoints", ylim=(0, 80), xticks=[8, 14, 16, 18, 20])
    axes[0].legend(frameon=False)
    for offset, identity, color in [(-0.18, "h0-15000", "#9aa8b7"), (0.18, "lab-20000", "#244b75")]:
        row = runs[identity]
        values = [100 * row["success"] / 500, 100 * row["clean_success"] / 250,
                  100 * row["randomized_success"] / 250]
        bars = axes[1].bar([i + offset for i in range(3)], values, width=0.35,
                           label="H0 15k" if identity.startswith("h0") else "Lab r256 20k", color=color)
        axes[1].bar_label(bars, labels=[f"{x:.1f}" for x in values], padding=3, fontsize=9)
    axes[1].set(xticks=range(3), xticklabels=["All (500)", "Clean (250)", "Randomized (250)"],
                ylabel="Success rate (%)", title="Fixed original E0 keys; different training recipes", ylim=(0, 85))
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle("Development evaluation; cross-host RGB / hardware / RNG differences remain", fontsize=11)
    fig.savefig(ROOT / "results.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify committed derived tables without writing")
    parser.add_argument("--plot", action="store_true", help="Regenerate the figure (requires matplotlib)")
    args = parser.parse_args()
    result, outputs = recount()
    for name, raw in outputs.items():
        path = ROOT / name
        if args.check:
            require(path.read_bytes() == raw, f"Derived table changed: {name}")
        else:
            path.write_bytes(raw)
    if args.plot:
        plot(result)
    print(json.dumps(dict(passed=True, runs=len(result["runs"]), original_keys=500,
                          paired_lab20k_h0=result["paired_lab20k_h0"], tasks=result["task_comparison"]), ensure_ascii=False))
