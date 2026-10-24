"""Small real-input audit of object-only supplementation while full cache runs."""
import json,os,sys,time
from pathlib import Path
from common import ROOT,ASSETS,DETECTOR,read,atomic,sha,stopped
from correct_foreground import supplement_prompt,merge
from cache_policy import select

def main():
    os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['OMP_NUM_THREADS']='2'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import h5py,numpy as np,torch
    from PIL import Image,ImageDraw
    from torchvision.transforms.v2 import Resize
    from aug6000.foreground import array_sha
    from aug6000.grounding import GroundingBoxDetector
    sys.path.insert(0,str(ASSETS/'code/src'))
    from lingbot_competition.data import decode_rgb
    torch.set_num_threads(2)
    dest=ROOT/'foreground/audit-supplement-001';assert not dest.exists();dest.mkdir()
    records={(r['task'],r['episode']):r for r in read(ROOT/'artifacts/manifest.json')['records']}
    files=sorted((ROOT/'foreground/audit').glob('*.json'));assert len(files)==800
    detector=None;resize=Resize((256,256),antialias=True)
    rows=[];todo=[];start=time.monotonic()
    for path in files:
        item=read(path);prompt=supplement_prompt(item,item['phrases'])
        if prompt:todo.append((path,item,prompt))
    for first in range(0,len(todo),4):
        if stopped():raise InterruptedError('Human stop')
        batch=todo[first:first+4];native=[];canonical=[]
        if detector is None:
            detector=GroundingBoxDetector(DETECTOR/'model',expected_files=read(DETECTOR/'model-receipt.json')['files'])
        for path,item,prompt in batch:
            rec=records[item['task'],item['episode']]
            with h5py.File(ASSETS/rec['path'],'r') as source:
                x=torch.from_numpy(decode_rgb(source['observation/head_camera/rgb'][item['frame']])).permute(2,0,1).float()
                native.append(x.numpy());canonical.append(resize(x).numpy())
            assert array_sha(canonical[-1])==item['rgb_sha256']
        predictions,_=detector.predict_batch(canonical,[x[2] for x in batch],detector_images=native)
        for (path,item,prompt),prediction in zip(batch,predictions):
            value=merge(item,prediction)
            value['supplement']=dict(prompt=prompt,original_audit_sha256=sha(path),
                unchanged_rgb_sha256=item['rgb_sha256'],object_only=True,reviewed=False)
            atomic(dest/path.name,value)
        atomic(ROOT/'artifacts/audit-supplement-progress.json',dict(completed=min(first+4,len(todo)),
            requested=len(todo),time=time.time(),formal_started=False))
    for path in files:
        original=read(path);item=read(dest/path.name) if (dest/path.name).exists() else original
        if not (dest/path.name).exists():atomic(dest/path.name,item)
        mask,info=select(item,item['phrases'])
        image=Image.open(path.with_name(path.stem+'-raw.png')).convert('RGB')
        draw=ImageDraw.Draw(image)
        for d in item['detections']:
            if d['score']>=.25:draw.rectangle(d['box'],outline=(255,40,0),width=1)
        image.save(dest/(path.stem+'-boxes.png'))
        # Preserve exact original preview bytes; never use rounded previews as detector input.
        (dest/(path.stem+'-raw.png')).write_bytes(path.with_name(path.stem+'-raw.png').read_bytes())
        rows.append(dict(task=item['task'],episode=item['episode'],frame=item['frame'],kind=item['kind'],
            supplement=bool(item.get('supplement')),**info))
    atomic(ROOT/'artifacts/audit-supplement-result.json',dict(passed=True,
        machine_processing_only=True,semantic_quality_passed=False,original_frames=800,
        extra_object_queries=len(todo),eligible_before=sum(select(read(p),read(p)['phrases'])[1]['policy_eligible'] for p in files),
        eligible_after=sum(r['policy_eligible'] for r in rows),rows=rows,
        seconds=time.monotonic()-start,time=time.time()))
    print(json.dumps(dict(processed=800,extra_object_queries=len(todo),eligible_after=sum(r['policy_eligible'] for r in rows))))

if __name__=='__main__':main()
