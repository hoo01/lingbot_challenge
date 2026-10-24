"""One server catalog for deployment, command-line access and UI navigation."""
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex

PACKAGE = Path(__file__).resolve().parents[2]


def load_catalog(path=None):
    path = path or os.environ.get('LINGBOT_SERVERS_CONFIG')
    if path is None:
        local = PACKAGE / 'configs/servers.local.json'
        path = local if local.exists() else PACKAGE / 'configs/servers.json'
    catalog = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if catalog.get('version') != 1 or not isinstance(catalog.get('servers'), dict):
        raise ValueError('Expected server catalog version 1 with a servers object')
    if catalog.get('default_server') not in catalog['servers']:
        raise ValueError('default_server must name a configured server')
    local_ports = set()
    for name, server in catalog['servers'].items():
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', name):
            raise ValueError('Invalid server ID')
        if set(server) - {'label', 'ssh', 'runtime_root', 'reuse_runtime', 'web_port', 'local_port', 'gpu', 'gpus', 'note'}:
            raise ValueError(f'{name}: unknown server fields; credentials belong in OpenSSH')
        if not isinstance(server.get('label'), str) or not server['label'].strip():
            raise ValueError(f'{name}: label is required')
        for key in ('runtime_root', 'reuse_runtime'):
            value = server.get(key, '')
            if not isinstance(value, str) or not PurePosixPath(value).is_absolute() or '..' in PurePosixPath(value).parts or any(c in value for c in '\n\r\0'):
                raise ValueError(f'{name}: {key} must be an absolute Linux path')
        if server['runtime_root'] == server['reuse_runtime']:
            raise ValueError(f'{name}: use a separate team runtime')
        for key in ('web_port', 'local_port'):
            if type(server.get(key)) is not int or not 1 <= server[key] <= 65535:
                raise ValueError(f'{name}: invalid {key}')
        if server['local_port'] in local_ports:
            raise ValueError('Each server needs a different local_port')
        local_ports.add(server['local_port'])
        if not re.fullmatch(r'\d+', str(server.get('gpu', ''))):
            raise ValueError(f'{name}: gpu must be one numeric index')
        values = server.get('gpus', [server['gpu']])
        if not isinstance(values, list) or not 1 <= len(values) <= 2 or any(
                not re.fullmatch(r'\d+', str(v)) for v in values) or len(set(map(str, values))) != len(values):
            raise ValueError(f'{name}: gpus must contain one or two distinct indices')
        ssh = server.get('ssh', {})
        if set(ssh) - {'host', 'user', 'port', 'identity_file', 'preferred_authentications', 'address_family'}:
            raise ValueError('Unknown SSH field; never store passwords')
        if 'address_family' in ssh and ssh['address_family'] not in ('any', 'inet', 'inet6'):
            raise ValueError(f'{name}: invalid SSH address family')
        if 'preferred_authentications' in ssh and ssh['preferred_authentications'] not in (
                'password', 'publickey', 'keyboard-interactive'):
            raise ValueError(f'{name}: invalid SSH authentication method')
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._:-]*', ssh.get('host', '')):
            raise ValueError(f'{name}: invalid SSH host or alias')
        if 'port' in ssh and (type(ssh['port']) is not int or not 1 <= ssh['port'] <= 65535):
            raise ValueError(f'{name}: invalid SSH port')
        for key in ('user', 'identity_file'):
            if key in ssh and (not isinstance(ssh[key], str) or not ssh[key] or any(c in ssh[key] for c in '\n\r\0')):
                raise ValueError(f'{name}: invalid SSH {key}')
    return catalog


def select_server(catalog, name=None):
    name = name or os.environ.get('LINGBOT_SERVER') or catalog['default_server']
    if name not in catalog['servers']:
        raise ValueError(f'Unknown server {name!r}; configured: {", ".join(catalog["servers"])}')
    return name, catalog['servers'][name]


def public_server(name, server):
    # Never expose SSH account, key path, or authentication material to the page.
    return dict(id=name, label=server['label'], note=server.get('note', ''),
                parallelism=len(server.get('gpus', [server['gpu']])),
                url=f'http://127.0.0.1:{server["local_port"]}/?server={name}')


def server_state(root):
    catalog = load_catalog()
    binding = Path(root) / 'configs/runtime.json'
    runtime = json.loads(binding.read_text()) if binding.exists() else {}
    name = os.environ.get('LINGBOT_SERVER') or runtime.get('server_id')
    if runtime.get('server_id') and name != runtime['server_id']:
        raise ValueError('Server selection differs from installed runtime binding')
    if name:
        name, selected = select_server(catalog, name)
        current = public_server(name, selected)
    else:
        current = dict(id=None, label='尚未绑定服务器', note='使用配置入口安装并启动团队版服务。')
    return dict(current=current, default_server=catalog['default_server'],
                servers=[public_server(n, s) for n, s in catalog['servers'].items()])


def ssh_command(server):
    ssh = server['ssh']
    command = ['ssh', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3']
    if 'port' in ssh:
        command += ['-p', str(ssh['port'])]
    if 'user' in ssh:
        command += ['-l', ssh['user']]
    if 'identity_file' in ssh:
        command += ['-i', str(Path(ssh['identity_file']).expanduser())]
    if 'preferred_authentications' in ssh:
        command += ['-o', 'PreferredAuthentications=' + ssh['preferred_authentications']]
    if 'address_family' in ssh:
        command += ['-o', 'AddressFamily=' + ssh['address_family']]
    return command


def remote_command(name, server, action, arguments=()):
    if action == 'tunnel':
        return ssh_command(server) + ['-o', 'ExitOnForwardFailure=yes', '-N', '-L',
            f'127.0.0.1:{server["local_port"]}:127.0.0.1:{server["web_port"]}', server['ssh']['host']]
    if action not in ('web', 'eval', 'check'):
        raise ValueError('Unknown remote action')
    root = server['runtime_root']
    command = ['env', f'LINGBOT_ROOT={root}', f'LINGBOT_SERVER={name}',
               'python3', root + '/scripts/runtime_entry.py', action,
               '--server', name, *arguments]
    # One quoted POSIX command is interpreted by the remote login shell.
    return ssh_command(server) + [server['ssh']['host'], shlex.join(command)]
