"""Resolve a server profile before starting local Linux UI or evaluation processes."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from eval_platform.servers import load_catalog, select_server


def resolve(catalog, name, action, arguments):
    name, server = select_server(catalog, name)
    root = Path(server['runtime_root'])
    override = os.environ.get('LINGBOT_ROOT')
    if override and Path(override) != root:
        raise ValueError('LINGBOT_ROOT differs from the selected server. Update its catalog entry first')
    command = [str(root/'envs/web/bin/python')]
    if action == 'web':
        command += [str(root/'scripts/start_eval_platform.py'), '--port', str(server['web_port']),
                    '--gpu', str(server['gpu'])]
    elif action == 'eval':
        command += [str(root/'scripts/team_eval.py'), '--gpu', str(server['gpu'])]
    elif action != 'check':
        raise ValueError('Unknown local action')
    return name, root, command + list(arguments)


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('action', choices=['web', 'eval', 'check'])
    parser.add_argument('--config')
    parser.add_argument('--server')
    parser.add_argument('--dry-run', action='store_true')
    args, rest = parser.parse_known_args()
    try:
        catalog = load_catalog(args.config)
        name, root, command = resolve(catalog, args.server, args.action, rest)
        if args.dry_run:
            print(json.dumps(dict(server=name, root=str(root), argv=command)))
            return 0
        binding_path = root/'configs/runtime.json'
        if not binding_path.exists():
            raise ValueError(f'Team runtime not installed: {root}. Run setup_eval.sh --server {name} --check first')
        binding = json.loads(binding_path.read_text())
        if binding.get('server_id') != name:
            raise ValueError('Selected server differs from the installed runtime binding')
        required = ['envs/web/bin/python', 'envs/train/bin/python', 'envs/sim/bin/python',
                    'scripts/team_eval.py', 'scripts/start_eval_platform.py']
        missing = [p for p in required if not (root/p).exists()]
        if missing:
            raise ValueError('Incomplete runtime: ' + ', '.join(missing))
        if args.action == 'check':
            if rest:
                raise ValueError('check does not accept extra arguments')
            print(json.dumps(dict(server=name, host=socket.gethostname(), root=str(root),
                                  installed=True, gpu_work_started=False), ensure_ascii=False))
            return 0
        env = dict(os.environ, LINGBOT_ROOT=str(root), LINGBOT_SERVER=name,
                   PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1')
        if args.config:
            env['LINGBOT_SERVERS_CONFIG'] = str(Path(args.config).resolve())
        # exec keeps existing signal/termination behavior of the local scripts.
        os.execve(command[0], command, env)
    except (ValueError, OSError) as error:
        parser.exit(2, str(error) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
