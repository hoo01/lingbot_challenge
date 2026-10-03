"""Check whether RoboTwin 2.0 aloha-agilex clean_50 joint angles approach the +-pi boundary.

Data source: TianxingChen/RoboTwin2.0 -> lerobot_dataset/RoboTwin_lerobot_v21.zip
(= aloha-agilex_clean_50 of all 50 tasks, 2500 episodes). Only parquet + meta are extracted
from the remote zip via HTTP Range (videos are never downloaded).

Usage:
    python joint_angle_check.py --download   # first time: fetch parquet/meta (~60 MB)
    python joint_angle_check.py              # stats + histogram + cross-check

Read-only: does not touch any training code or config.
"""
import argparse, glob, json, os, zipfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LEROBOT_DIR = os.path.join(HERE, "../raw/RoboTwin_lerobot_v21_noimg")
HDF5_NPZ_DIR = os.path.join(HERE, "../raw/hdf5_joints")
PI = np.pi

# Episodes are stored in contiguous blocks of 50 per task. Block order was inferred from the
# instruction text in meta/episodes.jsonl and verified numerically against raw hdf5 for the
# tasks available in HDF5_NPZ_DIR (see cross_check()).
TASKS = [
    "adjust_bottle", "handover_block", "open_laptop", "place_burger_fries", "place_object_basket",
    "rotate_qrcode", "stamp_seal", "beat_block_hammer", "handover_mic", "open_microwave",
    "place_can_basket", "place_object_scale", "scan_object", "turn_switch", "blocks_ranking_rgb",
    "hanging_mug", "pick_diverse_bottles", "place_cans_plasticbox", "place_object_stand", "shake_bottle",
    "blocks_ranking_size", "lift_pot", "pick_dual_bottles", "place_container_plate", "place_phone_stand",
    "shake_bottle_horizontally", "click_alarmclock", "move_can_pot", "place_a2b_left", "place_dual_shoes",
    "place_shoe", "stack_blocks_three", "click_bell", "move_pillbottle_pad", "place_a2b_right",
    "place_empty_cup", "press_stapler", "stack_blocks_two", "dump_bin_bigbin", "move_playingcard_away",
    "place_bread_basket", "place_fan", "put_bottles_dustbin", "stack_bowls_three", "grab_roller",
    "move_stapler_pad", "place_bread_skillet", "place_mouse_pad", "put_object_cabinet", "stack_bowls_two",
]
EPS_PER_TASK = 50

# 14-dim layout from meta/info.json + LingBot configs/robot_configs/robotwin.yaml:
# [0:6] left arm joints, [6] left gripper, [7:13] right arm joints, [13] right gripper
ARM_IDX = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
ARM_NAMES = [f"L{j}" for j in range(1, 7)] + [f"R{j}" for j in range(1, 7)]
FIELDS = ["state", "action"]  # state <- observation.state


def download():
    from remote_zip import open_remote
    z = zipfile.ZipFile(open_remote("lerobot_dataset/RoboTwin_lerobot_v21.zip", block=4 << 20))
    infos = sorted([i for i in z.infolist() if "/videos/" not in i.filename and not i.is_dir()],
                   key=lambda i: i.header_offset)
    for i in infos:
        p = os.path.join(LEROBOT_DIR, i.filename.split("/", 1)[1])
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(z.read(i))
    print(f"extracted {len(infos)} files to {LEROBOT_DIR}")


def load_episodes():
    """Return list of dicts: episode_index, task, state (T,14), action (T,14)."""
    files = sorted(glob.glob(os.path.join(LEROBOT_DIR, "data/chunk-*/episode_*.parquet")))
    assert len(files) == len(TASKS) * EPS_PER_TASK, len(files)
    eps = []
    for fp in files:
        df = pd.read_parquet(fp, columns=["observation.state", "action", "episode_index", "frame_index"])
        df = df.sort_values("frame_index")
        ei = int(df["episode_index"].iloc[0])
        eps.append(dict(
            episode_index=ei,
            task=TASKS[ei // EPS_PER_TASK],
            state=np.stack(df["observation.state"].to_numpy()).astype(np.float64),
            action=np.stack(df["action"].to_numpy()).astype(np.float64),
        ))
    eps.sort(key=lambda e: e["episode_index"])
    return eps


def check_format(eps):
    e = eps[0]
    print(f"episodes={len(eps)} frames={sum(len(x['state']) for x in eps)} "
          f"state shape/ep0={e['state'].shape} action shape/ep0={e['action'].shape}")
    # action[t] should equal state[t+1] (RoboTwin pi0 process_data: state=v[:-1], action=v[1:])
    err = max(np.abs(x["action"][:-1] - x["state"][1:]).max() for x in eps)
    print(f"max |action[t] - state[t+1]| over all episodes = {err:.3g}")
    allv = np.concatenate([x["state"] for x in eps])
    print("gripper dims (6,13) unique-ish range:", allv[:, 6].min(), allv[:, 6].max(), allv[:, 13].min(), allv[:, 13].max())
    return err


def joint_stats(eps, field):
    rows = []
    for k, (d, name) in enumerate(zip(ARM_IDX, ARM_NAMES)):
        seqs = [x[field][:, d] for x in eps]
        v = np.concatenate(seqs)
        a = np.abs(v)
        dth = np.concatenate([np.abs(np.diff(s)) for s in seqs])
        cross = sum(1 for s in seqs if (s > 0.9 * PI).any() and (s < -0.9 * PI).any())
        rows.append(dict(
            joint=name, dim=d, min=v.min(), max=v.max(),
            p01=np.percentile(v, 1), p99=np.percentile(v, 99),
            max_abs=a.max(), margin_to_pi=PI - a.max(),
            frac_gt_0p9pi=(a > 0.9 * PI).mean(), frac_gt_0p95pi=(a > 0.95 * PI).mean(),
            n_gt_pi=int((a > PI).sum()),
            max_step=dth.max(), n_step_gt_pi=int((dth > PI).sum()),
            n_traj_cross_0p9pi=cross,
        ))
    return pd.DataFrame(rows)


def task_summary(eps, field):
    rows = []
    for t in TASKS:
        te = [x for x in eps if x["task"] == t]
        arm = np.concatenate([x[field][:, ARM_IDX] for x in te])
        steps = np.concatenate([np.abs(np.diff(x[field][:, ARM_IDX], axis=0)) for x in te])
        a = np.abs(arm)
        j = int(a.max(0).argmax())
        n_ep_gt_pi = sum(1 for x in te if (np.abs(x[field][:, ARM_IDX]) > PI).any())
        rows.append(dict(task=t, max_abs=a.max(), argmax_joint=ARM_NAMES[j],
                         n_gt_0p9pi=int((a > 0.9 * PI).sum()), n_gt_pi=int((a > PI).sum()),
                         n_ep_gt_pi=n_ep_gt_pi,
                         max_step=steps.max(), n_step_gt_pi=int((steps > PI).sum())))
    return pd.DataFrame(rows).sort_values("max_abs", ascending=False)


def wrap(x):
    return (x + PI) % (2 * PI) - PI


def wrap_simulation(eps, field, chunk=50):
    """What would naive wrapping to [-pi, pi] do? Count the 2pi jumps it creates."""
    rows = []
    for d, name in zip(ARM_IDX, ARM_NAMES):
        n_jump, ep_hit, chunk_hit, chunk_tot = 0, 0, 0, 0
        for x in eps:
            jump = np.abs(np.diff(wrap(x[field][:, d]))) > PI  # jump[t]: between t and t+1
            n_jump += int(jump.sum())
            ep_hit += int(jump.any())
            # chunks = windows [t, t+chunk) over the episode; a chunk is hit if any jump lies inside
            T = len(jump) + 1
            starts = max(T - chunk + 1, 1)
            c = np.concatenate([[0], np.cumsum(jump)])
            hits = c[np.minimum(np.arange(starts) + chunk - 1, T - 1)] - c[np.arange(starts)]
            chunk_hit += int((hits > 0).sum())
            chunk_tot += starts
        rows.append(dict(joint=name, jumps_created=n_jump, episodes_hit=ep_hit,
                         chunk50_hit=chunk_hit, chunk50_frac=chunk_hit / chunk_tot))
    return pd.DataFrame(rows)


def cross_check(eps):
    """Compare LeRobot parquet against raw hdf5 joint_action/vector (from fetch_hdf5_joints.py)."""
    out = []
    for fp in sorted(glob.glob(os.path.join(HDF5_NPZ_DIR, "*.npz"))):
        task = os.path.basename(fp)[:-4]
        npz = np.load(fp)
        base = TASKS.index(task) * EPS_PER_TASK
        n_ep = len([k for k in npz.files if k.endswith("_vector")])
        errs = []
        for i in range(n_ep):
            v = npz[f"ep{i}_vector"]
            e = eps[base + i]
            if len(v) - 1 != len(e["state"]):
                errs.append(np.inf)
                continue
            errs.append(max(np.abs(v[:-1] - e["state"]).max(), np.abs(v[1:] - e["action"]).max()))
        hv = np.concatenate([npz[f"ep{i}_vector"][:, ARM_IDX] for i in range(n_ep)])
        errs = np.array(errs)
        out.append(dict(task=task, n_ep=n_ep, n_identical=int((errs < 1e-5).sum()),
                        mismatched_eps=[int(i) for i in np.where(errs >= 1e-5)[0]],
                        max_abs_err_identical=float(errs[errs < 1e-5].max()),
                        hdf5_max_abs_angle=float(np.abs(hv).max()),
                        hdf5_n_step_gt_pi=int(sum((np.abs(np.diff(npz[f"ep{i}_vector"][:, ARM_IDX], axis=0)) > PI).sum()
                                                  for i in range(n_ep)))))
    return pd.DataFrame(out)


TRAJ_PKL_DIR = os.path.join(HERE, "../raw/traj_pkl")


def load_planner_paths():
    """Planner joint paths from _traj_data/episode*.pkl (fetch_traj_pkl.py). These are what the raw
    hdf5 joint_action is replayed from (denser sampling), so they stand in for the raw hdf5 of all 50
    tasks. Returns {(task, ep): (N, 12) array}; arms that never move are filled with their home 0."""
    import pickle
    out = {}
    for t in TASKS:
        for i in range(EPS_PER_TASK):
            fp = os.path.join(TRAJ_PKL_DIR, t, f"episode{i}.pkl")
            if not os.path.exists(fp):
                continue
            with open(fp, "rb") as f:
                pk = pickle.load(f)
            arms = []
            for side in ("left", "right"):
                segs = [np.asarray(s["position"], dtype=np.float64) for s in pk[f"{side}_joint_path"]
                        if s.get("status") == "Success" and np.ndim(s["position"]) == 2]
                arms.append(np.concatenate(segs) if segs else np.zeros((1, 6)))
            n = max(len(a) for a in arms)
            arms = [np.concatenate([a, np.repeat(a[-1:], n - len(a), 0)]) for a in arms]  # hold last pose
            out[(t, i)] = np.concatenate(arms, axis=1)
    return out


def planner_stats(paths):
    """Same metrics as joint_stats on the raw (planner/hdf5) source. Fractions are over planner points."""
    seqs = list(paths.values())
    rows = []
    for k, name in enumerate(ARM_NAMES):
        v = np.concatenate([s[:, k] for s in seqs])
        a = np.abs(v)
        dth = np.concatenate([np.abs(np.diff(s[:, k])) for s in seqs])
        rows.append(dict(
            joint=name, min=v.min(), max=v.max(), p01=np.percentile(v, 1), p99=np.percentile(v, 99),
            max_abs=a.max(), frac_gt_0p9pi=(a > 0.9 * PI).mean(), frac_gt_0p95pi=(a > 0.95 * PI).mean(),
            n_pts_gt_pi=int((a > PI).sum()),
            n_ep_gt_pi=sum(1 for s in seqs if (np.abs(s[:, k]) > PI).any()),
            max_step=dth.max(), n_step_gt_pi=int((dth > PI).sum()),
            n_traj_cross_0p9pi=sum(1 for s in seqs if (s[:, k] > 0.9 * PI).any() and (s[:, k] < -0.9 * PI).any()),
            wrap_jumps_created=int(sum((np.abs(np.diff(wrap(s[:, k]))) > PI).sum() for s in seqs)),
        ))
    return pd.DataFrame(rows)


def planner_task_summary(paths):
    rows = []
    for t in TASKS:
        te = [s for (tt, _), s in paths.items() if tt == t]
        a = np.abs(np.concatenate(te))
        j = int(a.max(0).argmax())
        rows.append(dict(task=t, max_abs=a.max(), argmax_joint=ARM_NAMES[j],
                         n_ep_gt_0p9pi=sum(1 for s in te if (np.abs(s) > 0.9 * PI).any()),
                         n_ep_gt_pi=sum(1 for s in te if (np.abs(s) > PI).any()),
                         n_ep_gt_1p5pi=sum(1 for s in te if (np.abs(s) > 1.5 * PI).any()),
                         n_step_gt_pi=int(sum((np.abs(np.diff(s, axis=0)) > PI).sum() for s in te)),
                         wrap_jumps_created=int(sum((np.abs(np.diff(wrap(s), axis=0)) > PI).sum() for s in te))))
    return pd.DataFrame(rows).sort_values("max_abs", ascending=False)


def lerobot_vs_planner(eps, paths, tol=0.05):
    """Episodes whose per-joint max|theta| differs between the LeRobot zip and the raw planner path,
    i.e. the LeRobot zip was not built from the same raw episode."""
    rows = []
    for e in eps:
        t, i = e["task"], e["episode_index"] % EPS_PER_TASK
        if (t, i) not in paths:
            continue
        lm = np.abs(e["state"][:, ARM_IDX]).max(0)
        pm = np.abs(paths[(t, i)]).max(0)
        if np.abs(lm - pm).max() > tol:
            rows.append(dict(task=t, ep=i, lerobot_max_abs=lm.max(), raw_max_abs=pm.max(),
                             raw_argmax_joint=ARM_NAMES[int(pm.argmax())]))
    return pd.DataFrame(rows)


def plot(eps, paths, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 6, figsize=(26, 8), sharex=True)
    bins = np.linspace(-7.6, 7.6, 305)
    raw = np.concatenate(list(paths.values())) if paths else None
    for k, (d, name) in enumerate(zip(ARM_IDX, ARM_NAMES)):
        ax = axes.flat[k]
        act = np.concatenate([x["action"][:, d] for x in eps])
        title = f"{name} (dim {d})  LeRobot |θ|max={np.abs(act).max():.2f}"
        if raw is not None:
            ax.hist(raw[:, k], bins=bins, density=True, color="#4C72B0", alpha=0.85, label="raw (hdf5/planner)")
            title += f"  raw={np.abs(raw[:, k]).max():.2f}"
        ax.hist(act, bins=bins, density=True, histtype="step", color="#222222", lw=0.9, label="LeRobot action")
        ax.set_yscale("log")
        for x, c, ls in [(PI, "#C44E52", "-"), (-PI, "#C44E52", "-"),
                         (0.9 * PI, "#DD8452", "--"), (-0.9 * PI, "#DD8452", "--")]:
            ax.axvline(x, color=c, ls=ls, lw=1.2)
        ax.set_title(title, fontsize=10)
        if k % 6 == 0:
            ax.set_ylabel("density (log)")
        if k >= 6:
            ax.set_xlabel("joint angle [rad]")
    axes.flat[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("RoboTwin 2.0 aloha-agilex clean_50, 50 tasks x 50 eps — arm joint angles "
                 "(red: ±π, orange dashed: ±0.9π)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    if args.download:
        download()

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    eps = load_episodes()
    check_format(eps)

    results = {}
    for field in FIELDS:
        df = joint_stats(eps, field)
        results[field] = df
        print(f"\n=== per-joint stats: {field} ===")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        ts = task_summary(eps, field)
        results[field + "_tasks"] = ts
        print(f"\n=== per-task summary: {field} (top 10 by max|θ|) ===")
        print(ts.head(10).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        ws = wrap_simulation(eps, field)
        results[field + "_wrap_sim"] = ws
        print(f"\n=== naive wrap to [-pi,pi] simulation: {field} ===")
        print(ws.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        hit = sum(1 for x in eps if (np.abs(np.diff(wrap(x[field][:, ARM_IDX]), axis=0)) > PI).any())
        print(f"episodes with >=1 wrap-created jump (any joint): {hit}/{len(eps)}")

    paths = load_planner_paths()
    if paths:
        print(f"\n=== raw source (planner paths, {len(paths)} episodes): per-joint ===")
        ps = planner_stats(paths)
        print(ps.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        pt = planner_task_summary(paths)
        print("\n=== raw source: tasks with any |θ| > 0.9π ===")
        print(pt[pt.n_ep_gt_0p9pi > 0].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        lv = lerobot_vs_planner(eps, paths)
        print(f"\n=== LeRobot zip episodes that differ from raw source: {len(lv)} ===")
        if len(lv):
            print(lv.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        results.update(raw_joint=ps, raw_tasks=pt, lerobot_vs_raw=lv)

    cc = cross_check(eps)
    if len(cc):
        print("\n=== cross-check LeRobot vs raw hdf5 ===")
        print(cc.to_string(index=False))
        results["cross_check"] = cc

    with open(os.path.join(HERE, "joint_angle_stats.json"), "w") as f:
        json.dump({k: v.to_dict(orient="records") for k, v in results.items()}, f, indent=1)
    plot(eps, paths, os.path.join(HERE, "joint_angle_hist.png"))


if __name__ == "__main__":
    main()
