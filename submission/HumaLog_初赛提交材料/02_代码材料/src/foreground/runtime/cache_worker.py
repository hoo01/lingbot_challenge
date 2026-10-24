"""GPU proposal producer on original Clean RGB; does not approve semantic trust."""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import time
import traceback

from common import ROOT,ASSETS,PARENT,DETECTOR,atomic,read,sha,stopped
from task_prompts import BASE,phrases,prompt

def main(rank):
    os.environ['CUDA_VISIBLE_DEVICES']=str(rank)
    os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import h5py
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    from torchvision.transforms.v2 import Resize
    from aug6000.foreground import array_sha
    from aug6000.grounding import GroundingBoxDetector,BoxPolicy
    sys.path.insert(0,str(ASSETS/'code/src'))
    from lingbot_competition.data import decode_rgb
    torch.set_num_threads(2)
    records=read(ROOT/'artifacts/manifest.json')['records']
    assert set(r['task'] for r in records)==set(BASE) and len(records)==2500
    inv={r['path']:r for r in read(ROOT/'artifacts/input_inventory.json')['clean_episodes']}
    model_receipt=read(DETECTOR/'model-receipt.json')
    detector=GroundingBoxDetector(DETECTOR/'model',device='cuda:0',shortest_edge=800,
        policy=BoxPolicy(score_threshold=.25),expected_files=model_receipt['files'])
    detector.provenance['revision']=model_receipt['revision']
    resize=Resize((256,256),antialias=True)
    out=ROOT/'foreground/proposals';out.mkdir(parents=True,exist_ok=True)
    audit=ROOT/'foreground/audit';audit.mkdir(parents=True,exist_ok=True)
    owned=records[rank::4]
    def load_images(h,indices):
        originals=[];canonicals=[]
        for index in indices:
            rgb=decode_rgb(h['observation/head_camera/rgb'][index])
            raw=np.ascontiguousarray(rgb.transpose(2,0,1),dtype=np.float32)
            originals.append(raw);canonicals.append(resize(torch.from_numpy(raw)).numpy())
        return originals,canonicals
    def write_preview(record,frame,canonical,detections,kind):
        pixels=np.floor(canonical.transpose(1,2,0)+.5).clip(0,255).astype(np.uint8)
        image=Image.fromarray(pixels);draw=ImageDraw.Draw(image)
        for d in detections:
            if d['score']<.25:continue
            b=d['box'];draw.rectangle(b,outline=(255,40,0),width=1)
        name=f"{record['task']}__e{record['episode']:02d}__f{frame:04d}"
        Image.fromarray(pixels).save(audit/(name+'-raw.png'))
        image.save(audit/(name+'-boxes.png'))
        atomic(audit/(name+'.json'),dict(task=record['task'],episode=record['episode'],frame=frame,
            kind=kind,rgb_sha256=array_sha(canonical),detections=detections,phrases=phrases(record),
            reviewed=False,automatic_policy_approved=False))
    count=0;start=time.monotonic()
    # Prioritize all calibration/holdout views before the long full-data pass.
    for record in owned:
        if record['episode'] not in (0,17,39,48):continue
        if stopped():raise InterruptedError('Human stop')
        path=ASSETS/record['path'];source_inv=inv[str(path)]
        assert (path.stat().st_size,path.stat().st_mtime_ns)==(source_inv['bytes'],source_inv['mtime_ns'])
        indices=sorted(set([0,int(record['frames'])//3,2*int(record['frames'])//3,int(record['frames'])]))
        with h5py.File(path,'r') as h:originals,raws=load_images(h,indices)
        ds,_=detector.predict_batch(raws,[prompt(record)]*len(raws),detector_images=originals)
        for i,raw,pred in zip(indices,raws,ds):
            write_preview(record,i,raw,pred,'calibration' if record['episode'] in (0,17) else 'holdout')
        count+=len(indices)
    atomic(ROOT/f'artifacts/cache-audit-rank{rank}.json',dict(passed=True,rank=rank,images=count,time=time.time(),
        semantic_quality_passed=False,producer=detector.provenance))
    done=0;frames=0
    for record in owned:
        if stopped():raise InterruptedError('Human stop')
        name=f"{record['task']}__e{record['episode']:02d}"
        dest=out/(name+'.json.gz')
        assert not dest.exists(),'Never overwrite an existing proposal episode'
        path=ASSETS/record['path'];source_inv=inv[str(path)]
        assert (path.stat().st_size,path.stat().st_mtime_ns)==(source_inv['bytes'],source_inv['mtime_ns'])
        entries=[]
        with h5py.File(path,'r') as h:
            n=len(h['observation/head_camera/rgb']);assert n==record['frames']+1
            for first in range(0,n,4):
                if stopped():raise InterruptedError('Human stop')
                indices=list(range(first,min(n,first+4)));originals,raws=load_images(h,indices)
                ds,_=detector.predict_batch(raws,[prompt(record)]*len(raws),detector_images=originals)
                entries.extend(dict(frame=i,rgb_sha256=array_sha(raw),detections=pred)
                    for i,raw,pred in zip(indices,raws,ds))
                frames+=len(indices)
                if first%32==0:
                    atomic(ROOT/f'artifacts/cache-progress-rank{rank}.json',dict(phase='detecting',rank=rank,
                        episodes=done,images=frames,current_episode=name,frame=first,time=time.time(),
                        mean_images_per_second=frames/max(time.monotonic()-start,.001)))
        tmp=dest.with_name(dest.name+'.partial')
        with gzip.open(tmp,'wt',encoding='utf8',compresslevel=1) as f:
            json.dump(dict(task=record['task'],episode=record['episode'],source_sha256=record['sha256'],
                prompt=prompt(record),phrases=phrases(record),producer=detector.provenance,
                reviewed=False,automatic_policy_approved=False,frames=entries),f)
        tmp.replace(dest)
        atomic(out/(name+'.receipt.json'),dict(path=dest.name,bytes=dest.stat().st_size,sha256=sha(dest),frames=n))
        done+=1
    atomic(ROOT/f'artifacts/cache-result-rank{rank}.json',dict(passed=True,rank=rank,episodes=done,
        frames=frames,time=time.time(),seconds=time.monotonic()-start,semantic_quality_passed=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rank',required=True,type=int);args=p.parse_args()
    try:main(args.rank)
    except BaseException as e:
        atomic(ROOT/f'artifacts/cache-error-rank{args.rank}.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()))
        raise
