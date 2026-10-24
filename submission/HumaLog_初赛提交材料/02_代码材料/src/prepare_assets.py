"""Prepare pinned public assets and a new Clean2500 frame index, without training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

C = Path(__file__).resolve().parents[1]
CODE_REV = 'be969b8fd117fb70550c5d4bf4bc328211b5b1b6'
BASE_REV = '11c703bf6a5c1f45b3b69168482da11fdbba53d7'
DATA_REV = '3dc3b798668feb99ac61cc9086d84cbcc3d79186'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def public_file(repo,revision,name,repo_type=None):
    from huggingface_hub import hf_hub_download,get_hf_file_metadata,hf_hub_url
    url = hf_hub_url(repo,name,revision=revision,repo_type=repo_type)
    meta = get_hf_file_metadata(url)
    assert meta.commit_hash == revision
    path = Path(hf_hub_download(repo,name,revision=revision,repo_type=repo_type))
    assert meta.size == path.stat().st_size
    digest = sha(path)
    if len(meta.etag) == 64:
        assert digest == meta.etag
    return path,dict(source=name,revision=revision,sha256=digest,bytes=path.stat().st_size)


def sources(assets):
    repo = assets / 'code/lingbot-vla-v2'
    assert not repo.exists(), 'Choose a new assets directory; existing sources are not changed'
    repo.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['git','clone','--no-checkout','https://github.com/Robbyant/lingbot-vla-v2.git',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'checkout','--detach',CODE_REV],check=True)
    subprocess.run(['git','-C',str(repo),'submodule','update','--init','--recursive'],check=True)
    helpers = assets / 'code/src/lingbot_competition'
    shutil.copytree(C / 'src/data_helpers/lingbot_competition',helpers)
    save(assets / 'artifacts/code-source.json',dict(repository='https://github.com/Robbyant/lingbot-vla-v2',revision=CODE_REV))


def models(assets):
    from huggingface_hub import snapshot_download
    destination = assets / 'models/lingbot-vla-v2-6b'
    assert not destination.exists()
    source = Path(snapshot_download('robbyant/lingbot-vla-v2-6b',revision=BASE_REV,
                    allow_patterns=['*.safetensors','model.safetensors.index.json','config.json']))
    destination.mkdir(parents=True)
    inventory = []
    for path in sorted(source.iterdir()):
        if path.is_file() and (path.suffix=='.safetensors' or path.name in ['config.json','model.safetensors.index.json']):
            shutil.copyfile(path,destination / path.name)
            inventory.append(dict(path=str(destination/path.name),sha256=sha(destination/path.name),bytes=path.stat().st_size))
    assert len(list(destination.glob('*.safetensors'))) == 6
    for repo,revision,name,target in [
        ('robbyant/lingbot-vla-v2-6b',BASE_REV,'depth/model.pt','models/teachers/depth/model.pt'),
        ('robbyant/lingbot-vla-v2-6b',BASE_REV,'dino_video/teacher_step_10000.pth','models/teachers/dino_video/teacher_step_10000.pth'),
        ('robbyant/lingbot-vla-v2-6b',BASE_REV,'dino_video/config.yaml','models/teachers/dino_video/config.yaml'),
        ('Ruicheng/moge-2-vitb-normal','ca5f0e07ff01d3e5a364c1d954ed12ee1814b368','model.pt','models/moge-2-vitb-normal/model.pt')]:
        path,info = public_file(repo,revision,name)
        out = assets / target;out.parent.mkdir(parents=True,exist_ok=True)
        assert not out.exists();shutil.copyfile(path,out)
        inventory.append(dict(info,repository=repo,path=str(out)))
    processor = C.parent / '03_模型权重/checkpoint/processor/Qwen3-VL-4B-Instruct'
    shutil.copytree(processor,assets / 'models/Qwen3-VL-4B-Instruct')
    save(assets / 'artifacts/downloaded-models.json',dict(base_revision=BASE_REV,files=inventory))


def clean_data(assets):
    tasks = json.loads((C / 'configs/official-task-list.json').read_text(encoding='utf8'))['tasks']
    archives = []
    for task in tasks:
        target = assets / 'datasets/raw' / task
        assert not target.exists(), 'Do not overwrite existing data: ' + task
        name = 'dataset/' + task + '/aloha-agilex_clean_50.zip'
        path,info = public_file('TianxingChen/RoboTwin2.0',DATA_REV,name,'dataset')
        target.mkdir(parents=True)
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                # Reject traversal and symlink entries before extracting any archive member.
                resolved = (target / member.filename).resolve()
                assert resolved.is_relative_to(target.resolve())
                assert (member.external_attr >> 16) & 0o170000 != 0o120000
            archive.extractall(target)
        save(target / '.verified-source.json',info)
        archives.append(dict(info,path='dataset/' + task + '/aloha-agilex_clean_50.zip'))
        print(json.dumps(dict(task=task,clean_archive_verified=True)),flush=True)
    save(assets / 'artifacts/clean-source-manifest.json',dict(setting='clean',revision=DATA_REV,archives=archives))


def index(assets,runtime):
    import sys
    sys.path.insert(0,str(C / 'src/data_helpers'))
    from lingbot_competition.data import read_joints,decode_rgb,CAMERAS
    import h5py
    assert not (runtime / 'artifacts/input_inventory.json').exists(), 'Index is immutable once prepared'
    tasks = json.loads((C / 'configs/official-task-list.json').read_text(encoding='utf8'))['tasks']
    provenance = json.loads((assets / 'artifacts/clean-source-manifest.json').read_text(encoding='utf8'))
    assert provenance['setting']=='clean' and provenance['revision']==DATA_REV
    by_task = {row['source'].split('/')[1]:row for row in provenance['archives']}
    assert set(by_task)==set(tasks)
    records,files = [],[]
    for task in tasks:
        base = assets / 'datasets/raw' / task
        proof = json.loads((base / '.verified-source.json').read_text(encoding='utf8'))
        assert proof['sha256']==by_task[task]['sha256'] and proof['revision']==DATA_REV
        paths = list(base.rglob('episode*.hdf5'))
        ids = {int(path.stem.removeprefix('episode')):path for path in paths}
        assert len(paths)==len(ids)==50 and set(ids)==set(range(50))
        for episode,path in sorted(ids.items()):
            instruction = path.parent.parent / 'instructions' / ('episode'+str(episode)+'.json')
            texts = json.loads(instruction.read_text(encoding='utf8'))['seen']
            assert texts and all(isinstance(s,str) and s.strip() for s in texts)
            before = path.stat()
            with h5py.File(path,'r') as h:
                q = read_joints(h)
                for camera in CAMERAS:
                    rgb = h['observation/'+camera+'/rgb']
                    assert len(rgb)==len(q)
                    for frame in [0,len(q)//2,len(q)-1]:decode_rgb(rgb[frame])
            h = sha(path);after = path.stat()
            assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            records.append(dict(task=task,episode=episode,path=str(path.relative_to(assets)),
                                frames=len(q)-1,split='train',instructions=texts,sha256=h))
            files.append(dict(path=str(path),sha256=h,bytes=after.st_size,mtime_ns=after.st_mtime_ns))
    assert len(records)==2500 and sum(r['frames'] for r in records)==549787
    artifacts = runtime / 'artifacts';artifacts.mkdir(parents=True,exist_ok=True)
    for name in ['norm_stats.json','norm_stats.provenance.json','robotwin_relative.yaml']:
        shutil.copyfile(C / 'configs/assets' / name,artifacts / name)
    save(artifacts / 'manifest.json',dict(records=records,source_revision=DATA_REV,setting='clean'))
    save(artifacts / 'input_inventory.json',dict(status='verified',clean_episodes=files,
         valid_start_frames=549787,source_revision=DATA_REV,official_source_revision=CODE_REV,
         normalization='Preserved submitted statistics; not recomputed from resized/augmented frames'))
    print(json.dumps(dict(passed=True,episodes=2500,valid_start_frames=549787,
                         index_relocated=True,exact_historical_inventory_bytes=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['sources','models','clean-data','index'],required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--runtime-root',type=Path)
    a = p.parse_args();assets = a.assets.resolve()
    if a.mode == 'index':
        if a.runtime_root is None:p.error('--runtime-root required for index')
        index(assets,a.runtime_root.resolve())
    elif a.mode=='sources':sources(assets)
    elif a.mode=='models':models(assets)
    else:clean_data(assets)


if __name__ == '__main__':
    main()
