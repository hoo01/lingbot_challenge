"""Connect to a configured evaluation server using the user's own OpenSSH login."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from eval_platform.servers import load_catalog, select_server, public_server, remote_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Server catalog JSON; default configs/servers[.local].json')
    parser.add_argument('--server', help='Server ID; defaults to rented-5090 in the supplied catalog')
    parser.add_argument('--dry-run', action='store_true', help='Show argv without connecting or starting anything')
    parser.add_argument('action', choices=['show', 'check', 'tunnel', 'web', 'eval'])
    parser.add_argument('arguments', nargs=argparse.REMAINDER, help='Arguments forwarded to web/eval')
    args = parser.parse_args()
    try:
        catalog = load_catalog(args.config)
        name, server = select_server(catalog, args.server)
        arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
        if args.action in ('show', 'check', 'tunnel') and arguments:
            parser.error(f'{args.action} does not accept additional arguments')
        if args.action == 'show':
            print(json.dumps(dict(selected=name, **public_server(name, server),
                                 runtime_root=server['runtime_root'], reuse_runtime=server['reuse_runtime']),
                             ensure_ascii=False, indent=2))
            return 0
        command = remote_command(name, server, args.action, arguments)
        if args.dry_run:
            print(json.dumps(dict(server=name, argv=command), ensure_ascii=False, indent=2))
            return 0
        print(f'{server["label"]} [{name}] -> {server["runtime_root"]}', flush=True)
        if args.action == 'tunnel':
            print('Keep this process running. Browser: ' + public_server(name, server)['url'], flush=True)
        return subprocess.call(command)
    except (ValueError, OSError) as error:
        parser.exit(2, str(error) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
