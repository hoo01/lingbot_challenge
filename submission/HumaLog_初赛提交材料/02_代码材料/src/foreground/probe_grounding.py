"""Bounded, read-only-original-Clean offline box/latency probe (no training)."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aug6000.foreground import array_sha, file_sha
from aug6000.grounding import BoxPolicy, GroundingBoxDetector, expanded_box_mask


TASKS = {
    'move_pillbottle_pad': ('pill bottle. bottle. blue square. blue pad.', ['pill_bottle', 'placement_pad']),
    'move_playingcard_away': ('playing card. playing cards.', ['playing_card']),
    'hanging_mug': ('mug. cup. mug rack. hook. stand.', ['mug', 'rack_hook']),
    'stamp_seal': ('rubber stamp. stamp. colored square. square pad.', ['stamp', 'stamp_target']),
    'turn_switch': ('switch. switch lever. handle.', ['switch_handle']),
    'place_phone_stand': ('phone. smartphone. phone stand. holder.', ['phone', 'phone_stand']),
    'place_shoe': ('shoe. shoe platform. blue block. rectangular pad.', ['shoe', 'placement_block']),
    'place_object_scale': ('rubik cube. toy car. stapler. mouse. bottle. electronic scale. weighing scale.', ['object', 'scale']),
    'put_object_cabinet': ('mouse. stapler. toy car. rubik cube. bread. phone. playing cards. soap. box. cabinet. cabinet handle.', ['object', 'cabinet']),
    'click_bell': ('bell. bell button. button.', ['bell_button']),
    'blocks_ranking_size': ('block. cube. blocks.', ['all_three_blocks']),
    'place_cans_plasticbox': ('can. soda can. plastic box. container.', ['both_cans', 'plastic_box']),
}
ROBOT_PROMPT = 'robot arm. robot gripper. robotic claw.'


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf8')


def overlay(raw, detections, threshold, label=True):
    arr = np.floor(raw.transpose(1,2,0) + .5).clip(0,255).astype(np.uint8)
    image = Image.fromarray(arr).resize((512,512))
    draw = ImageDraw.Draw(image)
    for i, d in enumerate(detections):
        if d['score'] < threshold: continue
        b = [v*2 for v in d['box']]
        draw.rectangle(b, outline=(0,255,40), width=2)
        if label:
            text = f"{i}:{d['label'][:24]} {d['score']:.2f}"
            draw.text((max(0,b[0]),max(0,b[1]-10)),text,fill=(255,20,20),stroke_width=1,stroke_fill=(255,255,255))
    return image


def main(a):
    import torch
    from torchvision.transforms.v2 import Resize

    torch.set_num_threads(2)
    assets, parent, output = map(Path, (a.assets,a.parent_root,a.output))
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(assets/'code/src'))
    from lingbot_competition.data import decode_rgb
    from lingbot_competition import robotwin_codec

    records = json.loads((parent/'artifacts/manifest.json').read_text())['records']
    inventory = {r['path']:r for r in json.loads((parent/'artifacts/input_inventory.json').read_text())['clean_episodes']}
    if len(records) != 2500 or any(r['split']!='train' for r in records):
        raise ValueError('Original 2500 Clean only')
    samples, raws = [], []
    raw_dir = output/'raw'; raw_dir.mkdir()
    start = time.perf_counter()
    for task, (phrases, required) in TASKS.items():
        for episode in (0,17):
            record = next(r for r in records if r['task']==task and r['episode']==episode)
            path = assets/record['path']; stat=path.stat(); inv=inventory[str(path)]
            if (inv['sha256']!=record['sha256'] or (stat.st_size, stat.st_mtime_ns)!=(inv['bytes'],inv['mtime_ns'])):
                raise ValueError('Original Clean input identity changed')
            with h5py.File(path,'r') as source:
                n=len(source['observation/head_camera/rgb'])
                for phase, frame in [('initial',0),('middle',(n-1)//2),('terminal',n-1)]:
                    t=time.perf_counter()
                    pixels=decode_rgb(source['observation/head_camera/rgb'][frame])
                    raw=Resize((256,256),antialias=True)(torch.from_numpy(pixels).permute(2,0,1).float()).numpy()
                    decode_s=time.perf_counter()-t
                    sample_id=f'{task}-e{episode}-f{frame:04d}'
                    raw_path=raw_dir/(sample_id+'.png')
                    Image.fromarray(np.floor(raw.transpose(1,2,0)+.5).clip(0,255).astype(np.uint8)).save(raw_path)
                    samples.append(dict(id=sample_id,task=task,episode=episode,frame=frame,frames=n,phase=phase,
                        source_sha256=record['sha256'],rgb_sha256=array_sha(raw),decode_resize_s=decode_s,
                        png=str(raw_path.relative_to(output)),prompt=ROBOT_PROMPT+' '+phrases,
                        required_roles=['visible_left_gripper','visible_right_gripper']+required,
                        instruction_examples=record['instructions'][:2]))
                    raws.append(raw)
    decode_wall=time.perf_counter()-start
    receipt=json.loads(Path(a.model_receipt).read_text())
    policy=BoxPolicy()
    load_start=time.perf_counter()
    detector=GroundingBoxDetector(a.model_dir,device='cuda:0',shortest_edge=800,
        policy=policy,expected_files=receipt['files'])
    detector.provenance['revision']=receipt['revision']
    model_load_s=time.perf_counter()-load_start
    # Explicitly test that detector scaling remains distinct from student normalization.
    normalization_checks=[]
    for value in (0.,.5,2.,127.5,255.):
        inp=np.full((3,256,256),value,np.float32)
        p=detector._prepare_inputs([inp],['robot.'])['pixel_values']
        expected=(value/255.-torch.tensor([.485,.456,.406]))/torch.tensor([.229,.224,.225])
        diff=float((p[0,:,400,400]-expected).abs().max())
        if diff>1e-5:raise ValueError('Detector preprocessing scaled raw255 incorrectly')
        normalization_checks.append(dict(raw255=value,max_abs_error=diff))
    # Warmup/first-kernel overhead is recorded and excluded from steady throughput.
    warmup=[]
    for _ in range(2):
        _, timing=detector.predict_batch(raws[:2],[s['prompt'] for s in samples[:2]])
        warmup.append(timing)
    for batch in (1,4,8):
        timings=[]; results=[]; wall_start=time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for i in range(0,len(samples),batch):
            ds,timing=detector.predict_batch(raws[i:i+batch],[s['prompt'] for s in samples[i:i+batch]])
            timings.append(timing);results.extend(ds)
        elapsed=time.perf_counter()-wall_start
        name=f'800-b{batch}'
        data=dict(variant=name,images=len(samples),elapsed_s=elapsed,images_per_second=len(samples)/elapsed,
            timings=timings,detections={s['id']:ds for s,ds in zip(samples,results)},
            max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            max_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        save_json(output/(name+'.json'),data)
        save_json(output/'progress.json',dict(phase='measuring',completed_variant=name,time=time.time()))
        if batch==4:
            primary_results=results
    # Same raw frames/prompts, cheaper detector resize. Not assumed equivalent quality.
    detector.shortest_edge=480
    warm480,_=detector.predict_batch(raws[:4],[s['prompt'] for s in samples[:4]])
    timings=[];results=[];wall_start=time.perf_counter()
    for i in range(0,len(samples),4):
        ds,timing=detector.predict_batch(raws[i:i+4],[s['prompt'] for s in samples[i:i+4]])
        timings.append(timing);results.extend(ds)
    elapsed=time.perf_counter()-wall_start
    save_json(output/'480-b4.json',dict(variant='480-b4',images=len(samples),elapsed_s=elapsed,
        images_per_second=len(samples)/elapsed,timings=timings,
        detections={s['id']:ds for s,ds in zip(samples,results)}))
    # Candidate bit masks remain diagnostic/unreviewed and cannot unlock training.
    proposals=output/'proposals';proposals.mkdir()
    visual=output/'visual';visual.mkdir()
    for s,raw,ds in zip(samples,raws,primary_results):
        s['protection']={}
        for threshold in (.20,.12):
            selected=[d['box'] for d in ds if d['score']>=threshold]
            mask,boxes=expanded_box_mask(selected,policy=replace(policy,score_threshold=threshold))
            np.savez_compressed(proposals/f"{s['id']}-t{int(threshold*100)}.npz",
                packed=np.packbits(mask.reshape(-1),bitorder='little'),rgb_sha256=np.array(s['rgb_sha256']),
                reviewed=np.array(False))
            s['protection'][str(threshold)]=dict(boxes=boxes,protected_fraction=float(mask.mean()),reviewed=False)
        overlay(raw,ds,.20).save(visual/(s['id']+'-boxes.png'))
        mask,_=expanded_box_mask([d['box'] for d in ds if d['score']>=.20])
        pixels=np.floor(raw.transpose(1,2,0)+.5).clip(0,255).astype(np.uint8)
        tint=pixels.copy();tint[mask]=np.clip(.55*tint[mask]+.45*np.array([255,40,0]),0,255)
        Image.fromarray(tint).save(visual/(s['id']+'-protected.png'))
    # Each task contact sheet has 2 rows (episodes) x 3 phases. Raw + boxes nearby.
    for task in TASKS:
        indices=[i for i,s in enumerate(samples) if s['task']==task]
        sheet=Image.new('RGB',(3*768,2*440),(255,255,255));draw=ImageDraw.Draw(sheet)
        for j,i in enumerate(indices):
            s=samples[i];x=(j%3)*768;y=(j//3)*440
            pix=np.floor(raws[i].transpose(1,2,0)+.5).clip(0,255).astype(np.uint8)
            sheet.paste(Image.fromarray(pix).resize((384,384)),(x,y+40))
            sheet.paste(overlay(raws[i],primary_results[i],.20,label=False).resize((384,384)),(x+384,y+40))
            draw.text((x+5,y+5),s['id']+' '+s['phase'],fill=(0,0,0))
            draw.text((x+5,y+20),'LEFT original | RIGHT detector boxes >=0.20',fill=(0,0,0))
        sheet.save(visual/(task+'-sheet.jpg'),quality=95)
    save_json(output/'samples.json',dict(samples=samples,producer=detector.provenance,
        policy=asdict(policy),all_proposals_reviewed=False,production_ready=False,
        source_manifest_sha256=file_sha(parent/'artifacts/manifest.json'),
        decoder_source_sha256=file_sha(robotwin_codec.__file__),
        decoder_resize_png_wall_s=decode_wall,model_load_and_sha_s=model_load_s,
        detector_normalization_checks=normalization_checks,warmup=warmup,
        cached_rgb_frames_full=sum(r['frames']+1 for r in records),
        batch1_and4_comparison_max_box_abs_diff=max(
            (float(np.max(np.abs(np.asarray(x['box'])-np.asarray(y['box']))))
             for ds1,ds4 in zip(json.loads((output/'800-b1.json').read_text())['detections'].values(),primary_results)
             for x,y in zip(ds1,ds4)),default=0),
        finished=time.time()))
    save_json(output/'progress.json',dict(phase='completed_proposals_need_visual_review',time=time.time()))
    print(json.dumps(dict(passed=True,samples=len(samples),output=str(output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('assets','parent-root','model-dir','model-receipt','output'):
        p.add_argument('--'+name,required=True)
    main(p.parse_args())
