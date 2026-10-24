"""Immutable per-episode proposal union and packed mask production; no approval."""
import argparse,gzip,json,os,sys,time
from pathlib import Path
from common import ROOT,PARENT,ASSETS,DETECTOR,read,sha,atomic,stopped
from cache_policy import select
from correct_foreground import supplement_prompt,merge,record_input_binding

def main(rank):
    os.environ['CUDA_VISIBLE_DEVICES']=str(rank);os.environ['OMP_NUM_THREADS']='2'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import h5py,numpy as np,torch
    from torchvision.transforms.v2 import Resize
    from aug6000.foreground import array_sha,write_episode_cache
    from aug6000.grounding import GroundingBoxDetector
    sys.path.insert(0,str(ASSETS/'code/src'))
    from lingbot_competition.data import decode_rgb
    torch.set_num_threads(2)
    records=read(ROOT/'artifacts/manifest.json')['records'];assert len(records)==2500
    inventory={v['path']:v for v in read(ROOT/'artifacts/input_inventory.json')['clean_episodes']}
    output=ROOT/'foreground/packed';output.mkdir(exist_ok=True)
    supplements=ROOT/'foreground/supplements';supplements.mkdir(exist_ok=True)
    detector=None;resize=Resize((256,256),antialias=True)
    entries={};queries=0;frames=0;eligible=0;begin=time.monotonic()
    for record in records[rank::4]:
        if stopped():raise InterruptedError('Human stop')
        name=f"{record['task']}__e{record['episode']:02d}"
        path=ROOT/'foreground/proposals'/(name+'.json.gz')
        receipt=read(path.with_name(name+'.receipt.json'))
        assert path.stat().st_size==receipt['bytes'] and sha(path)==receipt['sha256']
        with gzip.open(path,'rt',encoding='utf8') as f:original=json.load(f)
        record_input_binding(original,record)
        source=ASSETS/record['path'];inv=inventory[str(source)]
        assert (source.stat().st_size,source.stat().st_mtime_ns)==(inv['bytes'],inv['mtime_ns'])
        assert inv['sha256']==record['sha256']
        requested=[(i,supplement_prompt(frame,original['phrases'])) for i,frame in enumerate(original['frames'])]
        requested=[(i,prompt) for i,prompt in requested if prompt]
        additions={}
        if requested:
            if detector is None:
                detector=GroundingBoxDetector(DETECTOR/'model',expected_files=read(DETECTOR/'model-receipt.json')['files'])
            with h5py.File(source,'r') as h:
                for first in range(0,len(requested),4):
                    if stopped():raise InterruptedError('Human stop')
                    requests=requested[first:first+4];natives=[];raws=[]
                    for index,prompt in requests:
                        x=torch.from_numpy(decode_rgb(h['observation/head_camera/rgb'][index])).permute(2,0,1).float()
                        natives.append(x.numpy());raws.append(resize(x).numpy())
                        assert array_sha(raws[-1])==original['frames'][index]['rgb_sha256']
                    ds,_=detector.predict_batch(raws,[x[1] for x in requests],detector_images=natives)
                    additions.update({str(index):dict(prompt=prompt,detections=d) for (index,prompt),d in zip(requests,ds)})
                    queries+=len(requests)
        masks=[];metadata=[]
        for index,frame in enumerate(original['frames']):
            corrected=merge(frame,additions[str(index)]['detections']) if str(index) in additions else frame
            mask,info=select(corrected,original['phrases'])
            masks.append(mask);metadata.append(info);eligible+=info['policy_eligible']
        packed=output/(name+'.npz');assert not packed.exists()
        checksum=write_episode_cache(packed,np.asarray(masks,bool),metadata)
        sup=supplements/(name+'.json.gz');assert not sup.exists()
        tmp=sup.with_name(sup.name+'.partial')
        with gzip.open(tmp,'wt',encoding='utf8',compresslevel=1) as f:
            json.dump(dict(original_proposal_sha256=receipt['sha256'],source_sha256=record['sha256'],
                object_only_additions=additions,reviewed=False,semantic_quality_passed=False),f)
        tmp.replace(sup)
        key=f"{record['task']}/episode{record['episode']}"
        entry=dict(path='packed/'+packed.name,sha256=checksum,bytes=packed.stat().st_size,
            frames=len(metadata),proposal_sha256=receipt['sha256'],supplement_sha256=sha(sup))
        atomic(packed.with_suffix('.receipt.json'),entry);entries[key]=entry;frames+=len(metadata)
        atomic(ROOT/f'artifacts/foreground-pack-progress-rank{rank}.json',dict(rank=rank,
            episodes=len(entries),frames=frames,object_queries=queries,eligible_frames=eligible,
            semantic_quality_passed=False,time=time.time()))
    atomic(ROOT/f'artifacts/foreground-pack-result-rank{rank}.json',dict(passed=True,
        machine_processing_only=True,semantic_quality_passed=False,rank=rank,episodes=entries,
        frames=frames,object_queries=queries,eligible_frames=eligible,
        seconds=time.monotonic()-begin,time=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);a=p.parse_args();main(a.rank)
