"""Bind externally produced/reviewed masks to the ORIGINAL Clean RGB, offline.

This does not pretend RGB supplies actor labels. The input annotation bundle must
identify a fixed public producer or manual annotation, weights/source SHA and a
review report. A mask array alone cannot pass semantic foreground acceptance.

python build_foreground_cache.py --annotations bundle.json --parent-root ... --assets ... --output NEW_DIR

Bundle example is in IMPLEMENTATION.md. No simulator calls or GPU imports.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np

from aug6000.foreground import ForegroundCache, array_sha, file_sha, is_sha, write_episode_cache


def build(annotations, parent_root, assets, output):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from torchvision.transforms.v2 import Resize
    torch.set_num_threads(1)
    annotations,parent_root,assets,output=map(Path,(annotations,parent_root,assets,output))
    sys.path.insert(0,str(assets/'code/src'))
    from lingbot_competition.data import decode_rgb
    bundle=json.loads(annotations.read_text(encoding='utf8'))
    if bundle.get('schema')!='aug6000_reviewed_annotations_v1':raise ValueError('Unknown annotation schema')
    producer=bundle['producer']
    if not (all(producer.get(k) for k in ('name','revision','thresholds','review_protocol'))
            and is_sha(producer.get('weights_sha256'))):raise ValueError('Incomplete producer provenance')
    review_path=(annotations.parent/bundle['review_report']['path']).resolve()
    if file_sha(review_path)!=bundle['review_report']['sha256']:raise ValueError('Review report SHA mismatch')
    review=json.loads(review_path.read_text(encoding='utf8'))
    if not (review.get('passed') is True and review.get('all_required_instances_reviewed') is True
            and review.get('producer')==producer):raise ValueError('Semantic review not passed/bound to producer')
    source_manifest=parent_root/'artifacts/manifest.json'
    records=json.loads(source_manifest.read_text())['records']
    if len(records)!=2500 or not all(r['split']=='train' for r in records):
        raise ValueError('Only the inherited 2500 original Clean episodes are allowed')
    inventory={r['path']:r for r in json.loads((parent_root/'artifacts/input_inventory.json').read_text())['clean_episodes']}
    by_key={f"{r['task']}/episode{r['episode']}":r for r in records}
    if set(bundle['episodes'])!=set(by_key) or set(review.get('episodes',{}))!=set(by_key):
        raise ValueError('Full original Clean episode coverage is required; partial caches are diagnostic only')
    output.mkdir(parents=True,exist_ok=False)
    from lingbot_competition import robotwin_codec
    manifest=dict(schema='aug6000_foreground_v1',camera='head_camera',
        canonical_rgb='torchvision_resize256_antialias_float32_raw255',producer=producer,episodes={},
        source_manifest_sha256=file_sha(source_manifest),data_filter_sha256=file_sha(parent_root/'artifacts/data-filter-plan.json'),
        decoder_source_sha256=file_sha(robotwin_codec.__file__))
    total=0
    for key,record in sorted(by_key.items()):
        annotation=bundle['episodes'][key];reviewed=review['episodes'][key]
        mask_path=(annotations.parent/annotation['masks']).resolve()
        if (file_sha(mask_path)!=annotation['masks_sha256'] or annotation['masks_sha256']!=reviewed['masks_sha256']):
            raise ValueError('Mask file not bound to its semantic review: '+key)
        masks=np.load(mask_path,allow_pickle=False,mmap_mode='r')
        n=record['frames']+1
        if masks.shape!=(n,256,256) or masks.dtype!=np.bool_ or not masks.any(axis=(1,2)).all():
            raise ValueError('Need nonempty binary primary masks for every frame, including future terminal frame')
        required=annotation['required_instances'];covered=reviewed['covered_instances_per_frame']
        if (not required or len(covered)!=n or any(not set(required)<=set(labels) for labels in covered)
                or reviewed.get('all_frames_reviewed') is not True):
            raise ValueError('Required visible instance coverage/review missing: '+key)
        path=assets/record['path'];stat=path.stat();inv=inventory[str(path)]
        if (inv['sha256']!=record['sha256'] or (stat.st_size,stat.st_mtime_ns)!=(inv['bytes'],inv['mtime_ns'])):
            raise ValueError('Original HDF5 identity changed')
        metadata=[]
        with h5py.File(path,'r') as source:
            rgb=source['observation/head_camera/rgb']
            if len(rgb)!=n:raise ValueError('Source frame count differs')
            for i in range(n):
                # Exact inherited decoder, including RoboTwin legacy JPEG channel convention.
                pixels=decode_rgb(rgb[i])
                raw=Resize((256,256),antialias=True)(torch.from_numpy(pixels).permute(2,0,1).float()).numpy()
                metadata.append(dict(rgb_sha256=array_sha(raw),reviewed=True,required_labels=required,
                                     covered_labels=covered[i]))
        destination=output/(key+'.npz')
        digest=write_episode_cache(destination,np.asarray(masks),metadata)
        manifest['episodes'][key]=dict(path=str(destination.relative_to(output)).replace('\\','/'),sha256=digest,frames=n)
        total+=n
    review_copy=output/'semantic-review.json';review_copy.write_bytes(review_path.read_bytes())
    manifest['quality_gate']=dict(passed=True,review_report_path=review_copy.name,
        review_report_sha256=file_sha(review_copy),eligible_frames=549737,cached_rgb_frames=total,
        all_required_instances_reviewed=True)
    mp=output/'manifest.json';mp.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    digest=file_sha(mp)
    ForegroundCache(mp,digest).assert_formal_ready(549737)
    (output/'manifest.sha256').write_text(digest+'\n',encoding='ascii')
    return dict(passed=True,manifest_sha256=digest,cached_rgb_frames=total,episodes=len(records))


def main():
    p=argparse.ArgumentParser()
    for key in ('annotations','parent-root','assets','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.annotations,a.parent_root,a.assets,a.output)))


if __name__=='__main__':main()
