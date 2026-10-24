"""Fixed AMD8 experiment identity; actual official frame sampler owns ordering."""
import bisect
import hashlib
import json
import math
from pathlib import Path

ROOT=Path('/mnt/raid1/lingbotvla_sft_amd8000_to10000_20261020')
ASSETS=Path('/mnt/raid1/lingbotvla_smoke_20261009')
UPSTREAM=ASSETS/'code/lingbot-vla-v2'
SOURCE=ROOT/'incoming/model8000'
ARCHIVE=Path('/data/lingbotvla/archives/lab20k-full-sft-amd8-20261017')
REVISION='84afa4f2fbef16ad6d41dbb1c3b27967f2d7a8e6'
NORM_SHA='b24e6a7ccf44fa3ecc9dbbe060cf9b0937cf3307bc2a68232bc98c4578c22f6d'
MANIFEST_SHA='911e7ea61baecd505368fe6d5eb70052827b989ba91f93b6d6d13d331df28e78'
WORLD,MICRO,ACCUM,STEPS,SAVE_EVERY,GLOBAL_BATCH,SEED=4,16,4,10000,1000,256,42
ORIGINAL_FRAMES=549787
FRAMES=549737
START=8000
LOCAL_BATCH=MICRO*ACCUM
UPDATES_PER_EPOCH=math.ceil(FRAMES/WORLD)//LOCAL_BATCH
SAMPLING='official_StatefulDistributedSampler_shuffle_seed42_padding_loader_drop_last'
PEAKS={'language':1e-5,'vision':5e-6,'action_expert':2e-5,'projections_alignment':2e-5}


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    temporary.replace(path)


def seed_for(draw,stream):
    value=f'lab20k-full-sft-v1:{SEED}:{int(draw)}:{stream}'.encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8],'little')%(2**63-1)


def category(name):
    if '.lora_' in name:raise ValueError('Dense SFT cannot contain LoRA factors: '+name)
    if '.qwenvl.' in name:return 'vision' if '.visual.' in name else 'language'
    if '.qwen_expert.' in name:return 'action_expert'
    return 'projections_alignment'


def lr_multiplier(update,total=STEPS):
    assert 1<=update<=total and total==STEPS
    if update<=500:return update/500
    return .2+.4*(1+math.cos(math.pi*(update-500)/(total-500)))


class FrameCatalog:
    def __init__(self,records):
        self.ends=[]
        for record in records:self.ends.append(int(record['frames'])+(self.ends[-1] if self.ends else 0))
        assert len(records)==2500 and self.ends[-1]==ORIGINAL_FRAMES

    def locate(self,index):
        assert 0<=index<ORIGINAL_FRAMES
        episode=bisect.bisect_right(self.ends,index)
        return episode,index-(self.ends[episode-1] if episode else 0)
