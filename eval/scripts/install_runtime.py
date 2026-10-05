"""Install into a NEW runtime, reusing an already validated Linux environment."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def install(package, source, target, check=False, catalog=None, server_id=None):
    package, source, target = package.resolve(), source.resolve(), target.resolve()
    if source == target or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError('Use a separate sibling directory; never install over the active research runtime')
    if target.exists() and any(target.iterdir()):
        raise FileExistsError('Target must be new/empty. Existing results are never overwritten')
    required = ['envs/train/bin/python', 'envs/sim/bin/python', 'third_party/RoboTwin',
        'third_party/lingbot-vla-v2', 'third_party/lingbot-vla-v2-official-be969b8f',
        'third_party/oidn-2.3.3.x86_64.linux', 'models/Qwen3-VL-4B-Instruct',
        'models/lingbot-vla-v2-6b', 'artifacts/oidn-blackwell.json']
    missing = [name for name in required if not (source/name).exists()]
    if missing:
        raise FileNotFoundError('Incomplete source runtime: ' + ', '.join(missing))
    sys.path.insert(0, str(package/'src'))
    from eval_platform.contracts import validate_source
    pinned = json.loads((package/'configs/upstream-lock.json').read_text())
    validated=validate_source(source/'third_party/lingbot-vla-v2-official-be969b8f', pinned)
    plan = dict(source=str(source), target=str(target), shared_gpu_lock=str(source/'runs/gpu.lock'),
                resources='symlinks, do not move or patch existing model/sim environments',
                web='new CPU-only virtual environment', gpu_work_started=False,
                source_byte_layout=validated['source_byte_layout'])
    if catalog is not None:
        server = catalog['servers'][server_id]
        plan.update(server_id=server_id, server_label=server['label'], web_port=server['web_port'])
    if check:
        return plan
    target.mkdir(parents=True, exist_ok=True)
    for name in ('scripts', 'src', 'configs', 'results'):
        shutil.copytree(package/name, target/name, ignore=shutil.ignore_patterns('__pycache__', 'servers.local.json'))
    for name in ('README.md', 'run_eval.sh', 'eval.sh', 'start_web.sh', 'connect.py', 'requirements-web.txt'):
        shutil.copy2(package/name, target/name)
    for name in ('models', 'third_party', 'datasets'):
        if (source/name).exists():
            (target/name).symlink_to(source/name, target_is_directory=True)
    (target/'envs').mkdir()
    for name in ('train', 'sim'):
        (target/'envs'/name).symlink_to(source/'envs'/name, target_is_directory=True)
    for name in ('artifacts', 'runs', 'logs', 'cache', 'incoming', 'checkpoints', 'platform-state'):
        (target/name).mkdir()
    shutil.copy2(source/'artifacts/oidn-blackwell.json', target/'artifacts/oidn-blackwell.json')
    runtime = dict(gpu_lock=plan['shared_gpu_lock'])
    if catalog is not None:
        runtime['server_id'] = server_id
        # Bind this installation's default; client-side private key paths stay local.
        deployed = json.loads(json.dumps(catalog))
        deployed['default_server'] = server_id
        for server in deployed['servers'].values():
            server['ssh'].pop('identity_file', None)
        (target/'configs/servers.json').write_text(json.dumps(deployed, ensure_ascii=False, indent=2), encoding='utf-8')
    (target/'configs/runtime.json').write_text(json.dumps(runtime, indent=2))
    (target/'artifacts/install-plan.json').write_text(json.dumps(plan, indent=2))
    subprocess.run([str(source/'envs/train/bin/python'), '-m', 'venv', str(target/'envs/web')], check=True)
    subprocess.run([str(target/'envs/web/bin/python'), '-m', 'pip', 'install',
                    '-r', str(target/'requirements-web.txt')], check=True)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', help='Server profile; default rented-5090')
    parser.add_argument('--config', help='Server catalog JSON')
    parser.add_argument('--reuse-runtime', type=Path, help='Must match the selected profile')
    parser.add_argument('--root', type=Path, help='Must match the selected profile')
    parser.add_argument('--check', action='store_true', help='Read-only; no installation or GPU allocation')
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(package/'src'))
    from eval_platform.servers import load_catalog, select_server
    catalog = load_catalog(args.config)
    name, server = select_server(catalog, args.server)
    source, target = Path(server['reuse_runtime']), Path(server['runtime_root'])
    if (args.reuse_runtime and args.reuse_runtime != source) or (args.root and args.root != target):
        parser.error('Path overrides must match the profile; edit your server catalog first')
    print(json.dumps(install(package, source, target, args.check, catalog, name), ensure_ascii=False, indent=2))
