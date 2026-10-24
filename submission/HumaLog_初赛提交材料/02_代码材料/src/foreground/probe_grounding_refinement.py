"""Compare precise prompts/source resolution and time a real small cache stream.

All outputs are UNREVIEWED detector proposals. Does not start a training job or a
full-data cache. The measured stream includes original HDF5 decoding and packing.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from PIL import Image,ImageDraw

from aug6000.foreground import array_sha,file_sha
from aug6000.grounding import BoxPolicy,GroundingBoxDetector,expanded_box_mask
from probe_grounding import save_json, overlay


PRECISE={
    'move_pillbottle_pad':'pill bottle. blue square pad.',
    'move_playingcard_away':'deck of playing cards.',
    'hanging_mug':'mug. gray mug rack.',
    'stamp_seal':'rubber stamp. colored square marker.',
    'turn_switch':'toggle switch.',
    'place_phone_stand':'smartphone. phone holder.',
    'place_shoe':'shoe. blue rectangular marker.',
    'place_object_scale':'electronic weighing scale.',
    'put_object_cabinet':'cabinet. cabinet door handle.',
    'click_bell':'desk bell.',
    'blocks_ranking_size':'colored cubes.',
    'place_cans_plasticbox':'soda cans. plastic storage box.',
}
NOUNS=[('rubik','rubik cube'),('stapler','stapler'),('mouse','computer mouse'),
       ('toy car','toy car'),('toycar','toy car'),('bread','bread'),('phone','smartphone'),
       ('playing card','deck of playing cards'),('soap','bar of soap'),
       ('tea box','tea box'),('coffee box','coffee box'),('bottle','bottle')]
NOUNS += [('bell','desk bell')]


def precise_prompt(sample):
    text=' '.join(sample['instruction_examples']).lower()
    extra=[]
    if sample['task'] in ('place_object_scale','put_object_cabinet'):
        extra=sorted({noun for token,noun in NOUNS if token in text})
        if not extra:raise ValueError('No concrete object class in original Clean instructions')
    return 'robot gripper. robot arm. '+PRECISE[sample['task']]+' '+'. '.join(extra)+('.' if extra else '')


def main(a):
    import torch
    from torchvision.transforms.v2 import Resize
    torch.set_num_threads(2)
    assets=Path(a.assets);baseline=Path(a.baseline);output=Path(a.output)
    output.mkdir(exist_ok=False)
    sys.path.insert(0,str(assets/'code/src'))
    from lingbot_competition.data import decode_rgb
    base=json.loads((baseline/'samples.json').read_text());samples=base['samples']
    receipt=json.loads(Path(a.model_receipt).read_text())
    policy=BoxPolicy(score_threshold=.25)
    detector=GroundingBoxDetector(a.model_dir,device='cuda:0',shortest_edge=800,
        policy=policy,expected_files=receipt['files'])
    detector.provenance['revision']=receipt['revision']
    detector.provenance['prompt_policy']='short task/robot nouns; mixed-object class from original Clean instructions'
    for sample in samples:sample['prompt']=precise_prompt(sample)
    def read_one(sample):
        source=assets/f"datasets/raw/{sample['task']}/aloha-agilex_clean_50/data/episode{sample['episode']}.hdf5"
        with h5py.File(source,'r') as h:
            rgb=decode_rgb(h['observation/head_camera/rgb'][sample['frame']])
        original=np.ascontiguousarray(rgb.transpose(2,0,1),dtype=np.float32)
        canonical=Resize((256,256),antialias=True)(torch.from_numpy(original)).numpy()
        if array_sha(canonical)!=sample['rgb_sha256']:raise ValueError('Original RGB/canonical SHA changed')
        sample['original_shape_chw']=list(original.shape)
        return canonical,original
    pairs=[read_one(s) for s in samples]
    # Full-image flags from coarse proposal masks are counted, never called quality.
    for name in ('canonical','native'):
        raws=[p[0] for p in pairs];originals=[p[1] for p in pairs]
        detector.predict_batch(raws[:4],[s['prompt'] for s in samples[:4]],
            detector_images=originals[:4] if name=='native' else None)
        wall=time.perf_counter();results=[];timings=[]
        for i in range(0,len(samples),4):
            ds,timing=detector.predict_batch(raws[i:i+4],[s['prompt'] for s in samples[i:i+4]],
                detector_images=originals[i:i+4] if name=='native' else None)
            timings.append(timing);results.extend(ds)
        elapsed=time.perf_counter()-wall
        save_json(output/(name+'.json'),dict(images=len(samples),elapsed_s=elapsed,
            images_per_second=len(samples)/elapsed,timings=timings,
            detections={s['id']:ds for s,ds in zip(samples,results)}))
        visual=output/name;visual.mkdir()
        for task in PRECISE:
            indices=[i for i,s in enumerate(samples) if s['task']==task]
            sheet=Image.new('RGB',(3*768,2*440),'white');draw=ImageDraw.Draw(sheet)
            for j,i in enumerate(indices):
                sample=samples[i];raw=raws[i];ds=results[i];x=(j%3)*768;y=(j//3)*440
                pixels=np.floor(raw.transpose(1,2,0)+.5).clip(0,255).astype(np.uint8)
                mask,_=expanded_box_mask([d['box'] for d in ds if d['score']>=.25],policy=policy)
                tint=pixels.copy();tint[mask]=np.clip(.55*tint[mask]+.45*np.array([255,40,0]),0,255)
                sheet.paste(Image.fromarray(pixels).resize((384,384)),(x,y+40))
                sheet.paste(Image.fromarray(tint).resize((384,384)),(x+384,y+40))
                draw.text((x+4,y+4),sample['id'],fill='black')
                draw.text((x+4,y+20),f'{name} detection t=.25 | protected={mask.mean():.1%}',fill='black')
                overlay(raw,ds,.25).save(visual/(sample['id']+'-boxes.png'))
            sheet.save(visual/(task+'-sheet.jpg'),quality=95)
        save_json(output/'progress.json',dict(phase='refinement',completed=name,time=time.time()))
    # A streaming HDF5 -> detector -> expanded box -> bitpack/JSON pass, no PNGs.
    # Uses exactly the same 72 frames; cache candidates cannot be used as reviewed FG.
    dest=output/'timed-native-proposals';dest.mkdir()
    phases=dict(read_decode_resize_sha_s=0.,detection_s=0.,raster_pack_write_s=0.)
    wall=time.perf_counter()
    for i in range(0,len(samples),4):
        before=time.perf_counter();batch=samples[i:i+4];pairs=[read_one(s) for s in batch]
        loaded=time.perf_counter();raws=[p[0] for p in pairs];originals=[p[1] for p in pairs]
        ds,timing=detector.predict_batch(raws,[s['prompt'] for s in batch],detector_images=originals)
        detected=time.perf_counter()
        for sample,raw,prediction in zip(batch,raws,ds):
            mask,boxes=expanded_box_mask([d['box'] for d in prediction if d['score']>=.25],policy=policy)
            np.savez_compressed(dest/(sample['id']+'.npz'),packed=np.packbits(mask,bitorder='little'),
                rgb_sha256=np.array(array_sha(raw)),reviewed=np.array(False))
            save_json(dest/(sample['id']+'.json'),dict(detections=prediction,expanded_boxes=boxes,
                rgb_sha256=array_sha(raw),reviewed=False))
        written=time.perf_counter()
        phases['read_decode_resize_sha_s']+=loaded-before
        phases['detection_s']+=detected-loaded
        phases['raster_pack_write_s']+=written-detected
    elapsed=time.perf_counter()-wall
    save_json(output/'end-to-end.json',dict(images=len(samples),elapsed_s=elapsed,
        images_per_second=len(samples)/elapsed,phases=phases,
        bytes_written=sum(p.stat().st_size for p in dest.iterdir()),
        batch_size=4,detector_short_edge=800,inference_dtype='float32',device='one RTX4090',
        includes='HDF5 open/read, original decoder, canonical resize/RGB SHA, native detector resize/ImageNet preprocessing, CUDA synchronized FP32 forward, postprocess, box raster, bitpack NPZ + JSON write',
        excludes='model download/load/warmup, semantic review, full-data cold disk cache effects',
        full_cached_rgb_frames=base['cached_rgb_frames_full'],
        full_single_gpu_hours=base['cached_rgb_frames_full']/(len(samples)/elapsed)/3600))
    save_json(output/'samples.json',dict(samples=samples,producer=detector.provenance,policy=policy.__dict__,
        reviewed=False,production_ready=False,finished=time.time()))
    save_json(output/'progress.json',dict(phase='completed_proposals_need_visual_review',time=time.time()))
    print(json.dumps(dict(completed=True,images=len(samples),output=str(output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('assets','baseline','model-dir','model-receipt','output'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
