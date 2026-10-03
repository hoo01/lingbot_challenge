"""Stream aloha-agilex_clean_50 hdf5 episodes for a few tasks; keep only joint_action/endpose (images discarded in memory)."""
import zipfile, io, sys, os, h5py, numpy as np
from remote_zip import open_remote
out = "../raw/hdf5_joints"; os.makedirs(out, exist_ok=True)
for task in sys.argv[1:]:
    z = zipfile.ZipFile(open_remote(f"dataset/{task}/aloha-agilex_clean_50.zip", block=4 << 20))
    names = sorted([n for n in z.namelist() if "/data/" in n and n.endswith(".hdf5")],
                   key=lambda n: int(n.rsplit("episode", 1)[1].split(".")[0]))
    d = {}
    for n in names:
        ep = int(n.rsplit("episode", 1)[1].split(".")[0])
        with h5py.File(io.BytesIO(z.read(n)), "r") as h:
            d[f"ep{ep}_vector"] = h["joint_action/vector"][()]
            d[f"ep{ep}_left_endpose"] = h["endpose/left_endpose"][()]
            d[f"ep{ep}_right_endpose"] = h["endpose/right_endpose"][()]
        print(task, ep, flush=True)
    np.savez(f"{out}/{task}.npz", **d)
