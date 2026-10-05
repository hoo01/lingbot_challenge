"""Create a separate raw-checkpoint run and pristine pinned upstream source."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import yaml
from eval_platform.contracts import source_lock

REVISION = 'be969b8fd117fb70550c5d4bf4bc328211b5b1b6'

def prepare(root, name):
    repository = root/'third_party/lingbot-vla-v2'
    upstream = root/'third_party/lingbot-vla-v2-official-be969b8f'
    if not upstream.exists():
        archive = subprocess.check_output(['git','-C',str(repository),'archive',REVISION])
        upstream.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(upstream, filter='data')
    run = root/'runs'/name
    run.mkdir(exist_ok=False)
    checkpoint = run/'checkpoints/global_step_0/hf_ckpt'
    checkpoint.mkdir(parents=True)
    source = root/'models/lingbot-vla-v2-6b'
    for file in source.iterdir():
        if file.suffix in ('.safetensors','.json'):
            (checkpoint/file.name).symlink_to(file)
    cli = yaml.safe_load((upstream/'configs/vla/robotwin/robotwin.yaml').read_text())
    cli['model'].update(model_path=str(source),tokenizer_path=str(root/'models/Qwen3-VL-4B-Instruct'))
    norm = upstream/'assets/norm_stats/robotwin.json'
    cli['data'].update(norm_stats_file=str(norm), img_size=256)
    # Data arguments normally stringify these dictionaries before saving the CLI.
    for name in ('joints','norm_type'):
        cli['data'][name] = [str(x) for x in cli['data'][name]]
    for name in ('lr','lr_min','sequence_wise_loss_coeff','router_z_loss_coeff'):
        cli['train'][name] = float(cli['train'][name])
    cli['train'].update(use_compile=True, enable_fp32=True)
    (run/'lingbotvla_cli.yaml').write_text(yaml.safe_dump(cli,sort_keys=False))
    tracked = ['deploy/lingbot_vla_v2_policy.py',
               'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py',
               'lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py']
    _, source_hashes = source_lock(root)
    assert source_hashes['official_revision']==REVISION
    manifest = {'kind':'original_pretrained_base_official_inference_defaults',
                'source_byte_layout':source_hashes['source_byte_layout'],
                'official_revision':REVISION, 'source':str(upstream), 'checkpoint':str(checkpoint),
                'normalization':str(norm),'normalization_sha256':hashlib.sha256(norm.read_bytes()).hexdigest(),
                'no_adapters':True,'weights_unchanged':True,'img_size':256,
                'use_compile':True,'precision':'fp32','moe_backend':'official',
                'vision_device':'cuda','embedding_device':'cuda',
                'sampling_seed_at_process_start':42,'per_episode_reseed':False,
                'use_length':50,'num_steps':10,'flow_noise_generation':'upstream_native',
                'pristine_source_sha256':{p:hashlib.sha256((upstream/p).read_bytes()).hexdigest() for p in tracked}}
    (run/'official-defaults-manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(os.environ.get('LINGBOT_ROOT',Path(__file__).resolve().parents[1])).resolve())
    p.add_argument('--name',default='base50_official_20261004')
    a=p.parse_args();prepare(a.root,a.name)
