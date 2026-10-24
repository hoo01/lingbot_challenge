"""Prepare a NEW local runtime for the frozen 16000 evaluator; optional explicit launch."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

C = Path(__file__).resolve().parents[1]
TEMPLATE = C / 'src/eval_runtime'


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(p, value):
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2) + '\n',encoding='utf8')


def verify_template():
    lock = json.loads((C / 'configs/submission-source-lock.json').read_text(encoding='utf8'))['files']
    count = 0
    for name, expected in lock.items():
        if name.startswith('src/eval_runtime/'):
            assert sha(C / name) == expected, 'Frozen evaluation source changed: ' + name
            if name.endswith('.py'):
                ast.parse((C / name).read_bytes())
            count += 1
    assert count > 15
    return count


def link_dir(src, dst):
    src = Path(src).resolve()
    assert src.is_dir(), str(src)
    dst.symlink_to(src, target_is_directory=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only',action='store_true',help='Validate frozen code/config only; no GPU imports')
    p.add_argument('--root',type=Path,help='NEW writable runtime directory')
    p.add_argument('--checkpoint-root',type=Path,default=C.parent / '03_模型权重/checkpoint')
    p.add_argument('--upstream',type=Path)
    p.add_argument('--robotwin',type=Path)
    p.add_argument('--oidn-root',type=Path)
    p.add_argument('--render-proof',type=Path,help='Actual renderer acceptance JSON from this host')
    p.add_argument('--train-python',type=Path)
    p.add_argument('--sim-python',type=Path)
    p.add_argument('--gpu',default='0')
    p.add_argument('--gpu-lock',type=Path,help='Use the same lease path as other work on this host')
    p.add_argument('--port',type=int,default=18160)
    p.add_argument('--setting',choices=['clean','randomized'],default='clean')
    p.add_argument('--tasks',nargs='+',help='Omit for all 50; split tasks across independent GPU runtimes')
    p.add_argument('--episodes',type=int,choices=[30],default=30)
    p.add_argument('--seed',type=int,choices=[0],default=0)
    p.add_argument('--vulkan-icd',type=Path,default=Path('/etc/vulkan/icd.d/nvidia_icd.json'))
    p.add_argument('--ld-library-path',help='Verified host simulator library path if different from default')
    p.add_argument('--launch',action='store_true',help='After preparation, actually occupy the chosen free GPU')
    a = p.parse_args()
    files = verify_template()
    contract_path = C / 'configs/eval-contract.json'
    contract = json.loads(contract_path.read_text(encoding='utf8'))
    tasks = a.tasks or contract['tasks']
    assert tasks and len(set(tasks)) == len(tasks) and set(tasks) <= set(contract['tasks'])
    assert 1024 <= a.port <= 65535
    if a.check_only:
        print(json.dumps(dict(passed=True,source_files=files,episodes_per_task=30,setting=a.setting,
                             official_seed=0,actual_seed_start=100000,model_or_gpu_loaded=False,
                             actual_gpu_or_renderer_validation=False)))
        return
    for key in ['root','upstream','robotwin','oidn_root','render_proof','train_python','sim_python','gpu_lock']:
        if getattr(a,key) is None:
            p.error('--' + key.replace('_','-') + ' is required to prepare a new runtime')
    root, package = a.root.resolve(), a.checkpoint_root.resolve()
    assert not root.exists(), 'Existing runtime is never modified or restarted by this entry'
    assert not root.is_relative_to(package) and not package.is_relative_to(root)
    proof = json.loads(a.render_proof.read_text(encoding='utf8'))
    assert proof.get('passed') is True, 'Use an actual host renderer acceptance; never synthesize it'
    model_manifest = json.loads((C.parent / '03_模型权重/model_manifest.json').read_text(encoding='utf8'))
    original = package / model_manifest['model_relative_path']
    # One full content verification for a newly prepared independent runtime.
    for row in model_manifest['weight_files']:
        src = package / row['path']
        assert src.stat().st_size == row['bytes'] and sha(src) == row['sha256'], src.name
    for rel,key in [('assets/norm_stats.json','norm_sha256'),('assets/robotwin_relative.yaml','robot_sha256')]:
        assert sha(package / rel) == contract[key]
    for path in [a.train_python,a.sim_python,a.vulkan_icd]:
        assert path.is_file(), str(path)
    audit = C / 'configs/model-audit/runtime-parameter-audit.json'
    assert sha(audit) == contract['runtime_parameter_audit_sha256']
    # Refuse a busy device. This only queries the selected GPU when explicitly invoked.
    gpu = subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert gpu.startswith('GPU-') and '\n' not in gpu
    apps = subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True)
    assert not apps.strip(), 'Selected GPU has existing applications'
    pci = subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=pci.bus_id','--format=csv,noheader'],text=True).strip()
    import socket
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1',a.port)) != 0, 'Port already in use'
    root.mkdir(parents=True)
    for name in ['artifacts','configs','runs','run','cache','checkpoints','envs','models','third_party']:
        (root / name).mkdir()
    for src in TEMPLATE.rglob('*'):
        if not src.is_file():
            continue
        dst = root / src.relative_to(TEMPLATE)
        dst.parent.mkdir(parents=True,exist_ok=True)
        raw = src.read_bytes().replace(b'__HUMALOG_EVAL_ROOT__',str(root).encode())
        dst.write_bytes(raw)
    for name in ['train','sim']:
        # Do not resolve a venv's bin/python symlink to the system interpreter.
        src = getattr(a,name + '_python').absolute().parent.parent
        assert (src / 'bin/python').is_file(), 'Use a venv directory with bin/python'
        link_dir(src,root / 'envs' / name)
    link_dir(a.upstream,root / 'third_party/lingbot-vla-v2-official-be969b8f')
    link_dir(a.robotwin,root / 'third_party/RoboTwin')
    link_dir(a.oidn_root,root / 'third_party/oidn-2.3.3.x86_64.linux')
    link_dir(package / 'assets',root / 'assets')
    link_dir(package / 'processor/Qwen3-VL-4B-Instruct',root / 'models/Qwen3-VL-4B-Instruct')
    shutil.copyfile(a.render_proof,root / 'artifacts/oidn-blackwell.json')
    folder = root / 'checkpoints/model16000'
    checkpoint = folder / 'checkpoints/global_step_16000/hf_ckpt_fp32'
    checkpoint.mkdir(parents=True)
    for src in original.glob('*.safetensors'):
        # Runtime links only; the actual submission weight directory contains real files.
        (checkpoint / src.name).symlink_to(src)
    for name in ['config.json','model.safetensors.index.json']:
        shutil.copyfile(original / name,checkpoint / name)
    for name in ['runtime-construction.json','runtime-parameter-audit.json']:
        shutil.copyfile(C / 'configs/model-audit' / name,checkpoint / name)
    for name in ['norm_stats.json','norm_stats.provenance.json']:
        shutil.copyfile(package / 'assets' / name,folder / name)
    import yaml
    cli = yaml.safe_load((C / 'configs/inference-cli.yaml').read_text(encoding='utf8'))
    cli['model'].update(model_path=str(checkpoint),tokenizer_path=str(root / 'models/Qwen3-VL-4B-Instruct'))
    cli['data'].update(robot_config_path=str(root / 'assets/robotwin_relative.yaml'),
                       norm_stats_file=str(folder / 'norm_stats.json'),image_augment=False)
    cli['train'].update(use_compile=True,enable_fp32=True,enable_gradient_checkpointing=False,
                        vit_attn_implementation='flash_attention_2',precompute_grid_thw=True)
    (folder / 'lingbotvla_cli.yaml').write_text(yaml.safe_dump(cli,sort_keys=False),encoding='utf8')
    scope = {k:v for k,v in contract.items() if k not in ['shards']}
    scope.update(shard_id=root.name,tasks=tasks,settings=['demo_' + a.setting],planned=len(tasks)*30,
                 physical_gpu=gpu,physical_pci_bus_id=pci,policy_port=a.port,checkpoint=str(checkpoint),
                 model_revision=contract['revision'],protocol_id=contract['id'],
                 protocol_sha256=sha(contract_path),seeds=contract['candidate_seeds'],
                 render_device='cuda:0',vulkan_icd=str(a.vulkan_icd.resolve()),
                 simulation_ld_library_path=a.ld_library_path or str(root / 'third_party/oidn-2.3.3.x86_64.linux/lib') + ':/usr/lib/x86_64-linux-gnu',
                 simulation_extra_env={})
    save(root / 'artifacts/eval2000-scope.json',scope)
    save(root / 'artifacts/model-registration.json',dict(checkpoint=str(checkpoint),model_revision=contract['revision']))
    save(root / 'configs/runtime.json',dict(gpu_lock=str(a.gpu_lock.absolute())))
    os.environ['LINGBOT_ROOT'] = str(root)
    sys.path[:0] = [str(root / 'src'),str(root / 'scripts')]
    from eval_platform.contracts import write_inference_manifest
    from eval_platform.fork2000_plan import build,validate
    manifest = write_inference_manifest(root,checkpoint,folder / 'norm_stats.json',root / 'run','published_aug16000_FP32_exact_values')
    manifest.update(frozen_protocol_sha256=scope['protocol_sha256'],model_revision=contract['revision'])
    save(root / 'run/official-defaults-manifest.json',manifest)
    scenes = build(tasks)
    save(root / 'artifacts/fixed-scenes.json',scenes)
    scope['fixed_scenes_sha256'] = sha(root / 'artifacts/fixed-scenes.json')
    save(root / 'artifacts/eval2000-scope.json',scope)
    validate(dict(tasks=tasks,settings=scope['settings'],seeds=scope['seeds'],fixed_scenes=scenes,expert_check=False))
    save(root / 'artifacts/prepared.json',dict(passed=True,weight_sha_verified=True,planned=len(tasks)*30,
         gpu_started=False,actual_startup_acceptance_pending=True))
    print(json.dumps(dict(prepared=str(root),planned=len(tasks)*30,launch=a.launch)))
    if a.launch:
        subprocess.run([str(a.train_python.absolute()),'-B',str(root / 'scripts/run_shard16000.py')],check=True)


if __name__ == '__main__':
    main()
