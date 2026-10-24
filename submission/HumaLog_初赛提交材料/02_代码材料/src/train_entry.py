"""Launch preserved numerical training sources; never invent a restore acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

C = Path(__file__).resolve().parents[1]
NORM = 'b24e6a7ccf44fa3ecc9dbbe060cf9b0937cf3307bc2a68232bc98c4578c22f6d'
FINAL_ROOT = Path('/mnt/raid1/lingbotvla_sft_aug6000_b192_20261022')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_final_source():
    lock = json.loads((C / 'configs/frozen-training-identity.json').read_text(encoding='utf8'))['code']
    assert len(lock) == 23
    for name, expected in lock.items():
        assert sha(C / 'src/final_runtime' / name) == expected, 'Final source changed: ' + name
    return lock


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['base20k', 'aug16000'], required=True)
    p.add_argument('--runtime-root', type=Path)
    p.add_argument('--assets', type=Path)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--run', default='reproduce')
    p.add_argument('--resume', type=Path)
    p.add_argument('--data-only', action='store_true')
    p.add_argument('--check-only', action='store_true', help='Static source/config check, no model or training imports')
    a = p.parse_args()
    verify_final_source()
    assert Path(a.run).name == a.run and a.run not in ('', '.', '..')
    if a.check_only:
        print(json.dumps(dict(passed=True,check='source/config only',final_exact_sources=23,
                             stage=a.stage,actual_gpu_or_restore_validation=False)))
        return
    if not a.runtime_root:
        p.error('--runtime-root is required to launch')
    runtime = a.runtime_root.resolve()
    assert not (runtime / 'user-stop.json').exists(), 'Stopped runtime is not automatically restarted'
    if a.stage == 'base20k':
        if not a.assets:
            p.error('--assets is required for base20k')
        assets = a.assets.resolve()
        for name in ['manifest.json','norm_stats.json','norm_stats.provenance.json',
                     'robotwin_relative.yaml','input_inventory.json']:
            assert (runtime / 'artifacts' / name).is_file(), 'Prepare training assets/index first: ' + name
        assert sha(runtime / 'artifacts/norm_stats.json') == NORM
        env = dict(os.environ, HUMALOG_ASSETS=str(assets), HUMALOG_TRAIN_ROOT=str(runtime),
                   PYTHONPATH=os.pathsep.join([str(C / 'src/base20k'),str(C / 'src/data_helpers'),
                                              str(assets / 'code/lingbot-vla-v2')]),
                   QWEN3VL_PATH=str(assets / 'models/Qwen3-VL-4B-Instruct'),
                   TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
        command = [a.python, '-B', '-m', 'torch.distributed.run','--standalone','--nproc_per_node=4',
                   str(C / 'src/base20k/train_fresh.py'),'--run',a.run,'--steps','20000',
                   '--schedule-steps','20000','--batch','32','--accum','1','--save-every','2000']
        if a.data_only:
            command = [a.python,'-B',str(C / 'src/base20k/train_fresh.py'),'--run',a.run,'--phase','data']
        if a.resume:
            command += ['--resume',str(a.resume.resolve())]
        cwd = assets / 'code/lingbot-vla-v2'
    else:
        # Frozen restoration checks bind both exact paths and file content. Keep
        # this path inside a dedicated container/mount namespace for reproduction.
        assert runtime == FINAL_ROOT, 'Use the documented frozen layout inside an isolated container'
        if not a.resume:
            p.error('aug16000 requires a complete 11000-boundary training state via --resume')
        gate = runtime / 'artifacts/geometryfix11000_requested/ready-for-formal.json'
        assert gate.is_file() and json.loads(gate.read_text()).get('passed') is True, 'Actual restore gate required'
        root_code = runtime / 'train_geometryfix11000_requested'
        for name, expected in verify_final_source().items():
            assert sha(root_code / name) == expected, 'Runtime differs: ' + name
        assert sha(runtime / 'artifacts/norm_stats.json') == NORM
        assert (a.resume / 'complete.json').is_file(), 'Inference weights cannot substitute Adam/RNG/loader state'
        assert not a.data_only, 'Use base20k --data-only for the raw reader check'
        env = dict(os.environ,PYTHONPATH=os.pathsep.join([str(root_code),
                   '/mnt/raid1/lingbotvla_smoke_20261009/code/src',
                   '/mnt/raid1/lingbotvla_smoke_20261009/code/lingbot-vla-v2']),
                   TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
        command = [a.python,'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                   str(root_code / 'worker_bootstrap.py'),'--run',a.run,'--role','formal',
                   '--batch','16','--accum','3','--end-step','16000','--resume',str(a.resume.resolve())]
        cwd = runtime
    assert not (runtime / 'runs' / a.run).exists(), 'Choose a new run name; this entry never overwrites an existing run'
    subprocess.run(command, cwd=cwd, env=env, check=True)


if __name__ == '__main__':
    main()
