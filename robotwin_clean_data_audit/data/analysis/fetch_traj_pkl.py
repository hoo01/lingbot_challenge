"""Fetch _traj_data/episode*.pkl (planner joint paths, no images) of aloha-agilex_clean_50 for all tasks."""
import zipfile, os, sys, concurrent.futures as cf
from remote_zip import open_remote
from joint_angle_check import TASKS
out = "../raw/traj_pkl"

def fetch(task):
    d = os.path.join(out, task)
    if os.path.exists(os.path.join(d, "done")):
        return task
    os.makedirs(d, exist_ok=True)
    z = zipfile.ZipFile(open_remote(f"dataset/{task}/aloha-agilex_clean_50.zip", block=1 << 20))
    infos = sorted([i for i in z.infolist() if "/_traj_data/" in i.filename and i.filename.endswith(".pkl")],
                   key=lambda i: i.header_offset)
    for i in infos:
        open(os.path.join(d, os.path.basename(i.filename)), "wb").write(z.read(i))
    open(os.path.join(d, "done"), "w").close()
    return task

with cf.ThreadPoolExecutor(6) as ex:
    for t in ex.map(fetch, TASKS):
        print(t, flush=True)
