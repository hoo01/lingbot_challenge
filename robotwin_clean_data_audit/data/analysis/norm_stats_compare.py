"""Recompute LingBot bounds_99_woclip norm stats on HF RoboTwin LeRobot clean_50 (50 tasks x 50) and compare
with the repo's assets/norm_stats/robotwin.json (computed on aloha-agilex_randomized_500).

Replicates scripts/compute_norm_stats.py semantics without torch/lerobot:
- one sample per frame t (len(dataset) = total frames)
- observation.state.*: value at t
- action.*: chunk of chunk_size actions starting at t, indices clamped to the episode's last frame
  (LeRobot padding), all chunk elements merged (norm_merge_chunk_dim=True)
- q01/q99/q02/q98 exact (np.quantile) instead of the 5000-bin running histogram (diff ~ range/5000)

Writes norm_stats_clean.json (same layout as robotwin.json) and prints comparison tables.
"""
import json, os
import numpy as np
import pandas as pd

from joint_angle_check import load_episodes

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_STATS = os.path.join(HERE, "../ref/lingbot-vla-v2/assets/norm_stats/robotwin.json")
CHUNK = 50
# configs/robot_configs/robotwin.yaml
KEYS = {
    "arm.position": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
    "effector.position": [6, 13],
}
NAMES = {"arm.position": [f"L{j}" for j in range(1, 7)] + [f"R{j}" for j in range(1, 7)],
         "effector.position": ["L_grip", "R_grip"]}


def gather(eps):
    state = np.concatenate([e["state"] for e in eps])
    chunks = []
    for e in eps:
        T = len(e["action"])
        idx = np.minimum(np.arange(T)[:, None] + np.arange(CHUNK)[None, :], T - 1)  # (T, CHUNK)
        chunks.append(e["action"][idx.reshape(-1)])
    return state, np.concatenate(chunks)


def stats(x):
    q = np.quantile(x, [0.01, 0.99, 0.02, 0.98], axis=0)
    return dict(mean=x.mean(0), std=x.std(0), q01=q[0], q99=q[1], q02=q[2], q98=q[3], min=x.min(0), max=x.max(0))


def hist_quantiles(x, qs, bins=5000):
    """LingBot RunningStats-style quantile: 5000-bin histogram over [min, max], first bin edge whose CDF >= q."""
    out = np.zeros((len(qs), x.shape[1]))
    for j in range(x.shape[1]):
        edges = np.linspace(x[:, j].min() - 1e-10, x[:, j].max() + 1e-10, bins + 1)
        h, _ = np.histogram(x[:, j], bins=edges)
        c = np.cumsum(h) / h.sum()
        for i, q in enumerate(qs):
            out[i, j] = edges[np.searchsorted(c, q)]
    return out


def norm(x, s):
    return (x - s["q01"]) / (s["q99"] - s["q01"] + 1e-6) * 2.0 - 1.0


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    eps = load_episodes()
    state, action = gather(eps)
    print(f"frames={len(state)}  action chunk elements={len(action)} (= frames x {CHUNK})")

    repo = json.load(open(REPO_STATS))["norm_stats"]
    out, rows = {}, []
    for src, arr in [("observation.state", state), ("action", action)]:
        for k, dims in KEYS.items():
            key = f"{src}.{k}"
            x = arr[:, dims]
            s = stats(x)
            out[key] = {n: v.tolist() for n, v in s.items() if n in ("mean", "std", "q01", "q99", "q02", "q98")}
            r = {n: np.asarray(v) for n, v in repo[key].items()}
            n_rand, n_own = norm(x, r), norm(x, s)
            hq = hist_quantiles(x, [0.01, 0.99])
            for j, name in enumerate(NAMES[k]):
                rows.append(dict(
                    key=key.replace("observation.state", "state"), dim=name,
                    rand_q01=r["q01"][j], clean_q01=s["q01"][j], rand_q99=r["q99"][j], clean_q99=s["q99"][j],
                    d_q01=s["q01"][j] - r["q01"][j], d_q99=s["q99"][j] - r["q99"][j],
                    range_ratio=(s["q99"][j] - s["q01"][j]) / (r["q99"][j] - r["q01"][j] + 1e-12),
                    rand_mean=r["mean"][j], clean_mean=s["mean"][j], rand_std=r["std"][j], clean_std=s["std"][j],
                    out_1p05_rand=(np.abs(n_rand[:, j]) > 1.05).mean(), out_1p05_clean=(np.abs(n_own[:, j]) > 1.05).mean(),
                    max_abs_norm_rand=np.abs(n_rand[:, j]).max(), max_abs_norm_clean=np.abs(n_own[:, j]).max(),
                    hist_vs_exact_q01=hq[0, j] - s["q01"][j], hist_vs_exact_q99=hq[1, j] - s["q99"][j],
                ))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    with open(os.path.join(HERE, "norm_stats_clean.json"), "w") as f:
        json.dump({"norm_stats": out, "count": int(len(state))}, f, indent=2)
    df.to_csv(os.path.join(HERE, "norm_stats_compare.csv"), index=False)
    print("saved norm_stats_clean.json, norm_stats_compare.csv")


if __name__ == "__main__":
    main()
