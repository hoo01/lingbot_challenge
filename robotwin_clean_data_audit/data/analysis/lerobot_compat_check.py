"""Static compatibility check: HF RoboTwin LeRobot packs vs LingBot-VLA v2 data loader.

Inputs (no images): ../raw/RoboTwin_lerobot_v21_noimg, ../raw/RoboTwin_lerobot_v30_noimg,
../raw/instructions/{task}/instructions/episode{i}.json (raw aloha-agilex_clean_50), ../raw/traj_pkl.
Fetch them with joint_angle_check.py --download, fetch_meta_extra.py, fetch_traj_pkl.py.
"""
import glob, json, os
import numpy as np
import pandas as pd
import yaml

from joint_angle_check import TASKS, EPS_PER_TASK, ARM_IDX, load_episodes, load_planner_paths, lerobot_vs_planner

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "../raw")
V21 = os.path.join(RAW, "RoboTwin_lerobot_v21_noimg")
V30 = os.path.join(RAW, "RoboTwin_lerobot_v30_noimg")
LINGBOT_ROBOT_CFG = os.environ.get("LINGBOT_REPO", os.path.join(HERE, "../ref/lingbot-vla-v2")) + "/configs/robot_configs/robotwin.yaml"


def feature_keys_check():
    cfg = yaml.safe_load(open(LINGBOT_ROBOT_CFG))
    need = set()
    for group in ("states", "actions"):
        for item in cfg[group]:
            for spec in item.values():
                for ok in spec["origin_keys"]:
                    need.update(ok.keys())
    for item in cfg["images"]:
        need.update(v["origin_keys"] for v in item.values())
    rows = []
    for name, d in [("v2.1", V21), ("v3.0", V30)]:
        info = json.load(open(os.path.join(d, "meta/info.json")))
        feats = info["features"]
        for k in sorted(need):
            f = feats.get(k)
            rows.append(dict(pack=name, key=k, present=f is not None,
                             dtype=f and f["dtype"], shape=f and f["shape"],
                             codec=f and (f.get("info") or {}).get("video.codec")))
        rows.append(dict(pack=name, key="<fps>", present=True, dtype=info["fps"], shape=None, codec=None))
        rows.append(dict(pack=name, key="<codebase_version>", present=True, dtype=info["codebase_version"], shape=None, codec=None))
    return pd.DataFrame(rows)


def load_v30():
    files = sorted(glob.glob(os.path.join(V30, "data/chunk-*/file-*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=["observation.state", "action", "episode_index", "frame_index", "task_index"])
                    for f in files])
    tasks = pd.read_parquet(os.path.join(V30, "meta/tasks.parquet"))
    # v3.0 tasks.parquet: index = task string, column task_index
    idx2task = {int(r.task_index): t for t, r in tasks.iterrows()}
    eps = {}
    for ei, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        eps[int(ei)] = dict(state=np.stack(g["observation.state"].to_numpy()).astype(np.float64),
                            action=np.stack(g["action"].to_numpy()).astype(np.float64),
                            task=idx2task[int(g["task_index"].iloc[0])],
                            n_task=g["task_index"].nunique())
    return eps


def _key(e):
    return (len(e["state"]), np.round(e["state"][:, ARM_IDX].sum(0), 4).tobytes())


def v21_vs_v30(eps21, eps30):
    """Match episodes by content (episode order differs between packs)."""
    by_key = {}
    for e in eps21:
        by_key.setdefault(_key(e), []).append(e)
    n_same_index = n_matched = n_same_task = 0
    diff_task, unmatched, mapping = [], [], {}
    for ei, e30 in eps30.items():
        cands = [e for e in by_key.get(_key(e30), [])
                 if np.abs(e["state"] - e30["state"]).max() < 1e-6 and np.abs(e["action"] - e30["action"]).max() < 1e-6]
        if not cands:
            unmatched.append(ei)
            continue
        e21 = cands[0]
        n_matched += 1
        mapping[ei] = e21["episode_index"]
        n_same_index += e21["episode_index"] == ei
        if e21["instr"] == e30["task"]:
            n_same_task += 1
        else:
            diff_task.append((ei, e21["episode_index"], e21["instr"], e30["task"]))
    return dict(n_matched=n_matched, n_same_index=n_same_index, n_same_task=n_same_task,
                diff_task=diff_task, unmatched=unmatched, mapping=mapping)


def load_v21_tasks(eps21):
    eps_meta = [json.loads(l) for l in open(os.path.join(V21, "meta/episodes.jsonl"), encoding="utf8")]
    for e, m in zip(eps21, eps_meta):
        assert e["episode_index"] == m["episode_index"]
        e["instr"] = m["tasks"][0]
        e["n_task"] = len(m["tasks"])


def instruction_check(eps21, replaced):
    rows = []
    for e in eps21:
        t, i = e["task"], e["episode_index"] % EPS_PER_TASK
        d = json.load(open(os.path.join(RAW, "instructions", t, "instructions", f"episode{i}.json"), encoding="utf8"))
        s = e["instr"].strip()
        where = "seen" if s in [x.strip() for x in d["seen"]] else "unseen" if s in [x.strip() for x in d["unseen"]] else "neither"
        in_task_seen = where == "seen"
        if where == "neither":
            # does it appear in the seen list of ANY episode of the same task? (template drawn for another raw episode)
            for j in range(EPS_PER_TASK):
                dj = json.load(open(os.path.join(RAW, "instructions", t, "instructions", f"episode{j}.json"), encoding="utf8"))
                if s in [x.strip() for x in dj["seen"]]:
                    where = f"seen_of_ep{j}"
                    break
        rows.append(dict(task=t, ep=i, where=where, replaced=(t, i) in replaced,
                         n_seen=len(d["seen"]), n_unseen=len(d["unseen"])))
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 120)

    print("=== LingBot robotwin.yaml origin keys vs HF packs ===")
    print(feature_keys_check().to_string(index=False))

    eps21 = load_episodes()
    load_v21_tasks(eps21)
    print(f"\nv2.1: episodes={len(eps21)}, instructions per episode: {sorted({e['n_task'] for e in eps21})}, "
          f"unique instructions={len({e['instr'] for e in eps21})}")

    eps30 = load_v30()
    r = v21_vs_v30(eps21, eps30)
    print(f"v2.1 vs v3.0 (matched by content): matched {r['n_matched']}/{len(eps30)}, "
          f"same episode_index {r['n_same_index']}, same instruction {r['n_same_task']}, unmatched {len(r['unmatched'])}")
    blocks = sorted({(e30 // EPS_PER_TASK, e21 // EPS_PER_TASK) for e30, e21 in r["mapping"].items()})
    print("v3.0 block -> v2.1 block (task):", [(a, TASKS[b]) for a, b in blocks][:60])
    for ei, e21i, a, b in r["diff_task"][:5]:
        print(f"  v3.0 ep{ei} = v2.1 ep{e21i}: v2.1={a!r}\n      v3.0={b!r}")
    eps30_list = [dict(episode_index=r["mapping"][k], instr=v["task"],
                       task=TASKS[r["mapping"][k] // EPS_PER_TASK]) for k, v in eps30.items() if k in r["mapping"]]

    paths = load_planner_paths()
    lv = lerobot_vs_planner(eps21, paths)
    replaced = {(r.task, r.ep) for r in lv.itertuples()}
    ic = instruction_check(eps21, replaced)
    print("\n=== HF instruction vs raw instructions/episode{i}.json ===")
    print(ic.groupby(["replaced", ic["where"].str.replace(r"seen_of_ep\d+", "seen_of_other_ep", regex=True)])
          .size().rename("n_episodes").to_string())
    print("raw seen/unseen per episode:", sorted(set(zip(ic.n_seen, ic.n_unseen))))
    print("tasks with non-seen instructions:", ic[ic["where"] != "seen"].groupby("task").size().to_dict())
    ic.to_csv(os.path.join(HERE, "lerobot_instruction_check.csv"), index=False)
    print("examples of 'neither' (not replaced):")
    for row in ic[(ic["where"] == "neither") & ~ic.replaced].head(5).itertuples():
        print("  ", row.task, row.ep, repr(next(e["instr"] for e in eps21 if e["task"] == row.task and e["episode_index"] % EPS_PER_TASK == row.ep)))
    ic30 = instruction_check(eps30_list, replaced)
    print("\n=== v3.0 instruction vs raw ===")
    print(ic30.groupby(["replaced", ic30["where"].str.replace(r"seen_of_ep\d+", "seen_of_other_ep", regex=True)])
          .size().rename("n_episodes").to_string())


if __name__ == "__main__":
    main()
