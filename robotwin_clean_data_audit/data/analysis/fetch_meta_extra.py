"""Fetch (no images) HF LeRobot v3.0 parquet/meta and raw instructions/*.json of all 50 tasks."""
import zipfile, os, concurrent.futures as cf
from remote_zip import open_remote
from joint_angle_check import TASKS
RAW = "../raw"

def extract(zpath, keep, out_dir, strip_root=True):
    z = zipfile.ZipFile(open_remote(zpath, block=4 << 20))
    infos = sorted([i for i in z.infolist() if keep(i.filename) and not i.is_dir()], key=lambda i: i.header_offset)
    for i in infos:
        rel = i.filename.split("/", 1)[1] if strip_root else i.filename
        p = os.path.join(out_dir, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(z.read(i))
    return len(infos)

def instr(task):
    d = os.path.join(RAW, "instructions", task)
    if os.path.exists(os.path.join(d, "done")): return task
    for _ in range(4):
        try:
            extract(f"dataset/{task}/aloha-agilex_clean_50.zip", lambda n: "/instructions/" in n, d)
            open(os.path.join(d, "done"), "w").close(); return task
        except Exception as e: err = e
    raise err

if __name__ == "__main__":
    if not os.path.exists(os.path.join(RAW, "RoboTwin_lerobot_v30_noimg/meta/info.json")):
        print("v30", extract("lerobot_dataset/RoboTwin_lerobot_v30.zip", lambda n: "/videos/" not in n,
                             os.path.join(RAW, "RoboTwin_lerobot_v30_noimg")))
    with cf.ThreadPoolExecutor(6) as ex:
        print(len(list(ex.map(instr, TASKS))), "tasks instructions")
