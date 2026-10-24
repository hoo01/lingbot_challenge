"""Paths and atomic receipts for this independently authorized new stage."""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path('/mnt/raid1/lingbotvla_sft_aug6000_b192_20261022')
PARENT=Path('/mnt/raid1/lingbotvla_sft_amd8000_to10000_20261020')
ASSETS=Path('/mnt/raid1/lingbotvla_smoke_20261009')
PYTHON=ASSETS/'env/bin/python'
DETECTOR=Path('/mnt/raid1/lingbotvla_groundingdino_boxprobe_20261021')

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)
def ticks(pid):
    raw=Path(f'/proc/{pid}/stat').read_text()
    return raw[raw.rfind(')')+1:].split()[19]
def proof(pid,command):return dict(pid=pid,create_ticks=ticks(pid),command=command,time=time.time())
def stopped():return (ROOT/'user-stop.json').exists()
