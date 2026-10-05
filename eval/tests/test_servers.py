"""Server routing checks use local temporary files only; never connect to a GPU."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

EVAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL/'src'))
from eval_platform.servers import load_catalog, select_server, remote_command, server_state


@pytest.fixture
def catalog(monkeypatch):
    monkeypatch.delenv('LINGBOT_SERVER', raising=False)
    monkeypatch.delenv('LINGBOT_ROOT', raising=False)
    monkeypatch.delenv('LINGBOT_SERVERS_CONFIG', raising=False)
    result = load_catalog(EVAL/'configs/servers.json')
    assert list(result['servers']) == ['rented-5090']
    result['servers']['test-server'] = dict(
        label='Test server', ssh={'host': 'test-server'},
        reuse_runtime='/srv/test-runtime', runtime_root='/srv/test-eval',
        web_port=8788, local_port=18789, gpu='0')
    return result


def test_cli_default_remote_and_tunnel_use_same_profile(catalog):
    name, server = select_server(catalog)
    assert name == 'rented-5090'
    command = remote_command(name, server, 'eval', ['--base', '--title', "a'; touch /tmp/nope"])
    assert command[command.index('-l')+1] == 'root@ackcs-00gjhs2c'
    assert command[command.index('-p')+1] == '2233'
    remote = shlex.split(command[-1])
    assert remote[-1] == "a'; touch /tmp/nope"
    assert 'LINGBOT_ROOT=/root/shared-nvme/lingbot-eval-team' in remote
    tunnel = remote_command(name, server, 'tunnel')
    assert tunnel[tunnel.index('-L')+1] == '127.0.0.1:18788:127.0.0.1:8788'
    assert 'ExitOnForwardFailure=yes' in tunnel
    assert 'PreferredAuthentications=password' in tunnel
    assert 'PreferredAuthentications=password' in command
    assert 'AddressFamily=inet' in command
    assert 'AddressFamily=inet' in tunnel


def test_selecting_backup_changes_ssh_root_and_local_port(catalog):
    name, server = select_server(catalog, 'test-server')
    assert remote_command(name, server, 'tunnel')[-1] == 'test-server'
    assert '127.0.0.1:18789:127.0.0.1:8788' in remote_command(name, server, 'tunnel')
    assert 'LINGBOT_ROOT=/srv/test-eval' in shlex.split(remote_command(name, server, 'web')[-1])
    assert not any(x.startswith('PreferredAuthentications=') for x in remote_command(name, server, 'check'))
    with pytest.raises(ValueError, match='Unknown server'):
        select_server(catalog, 'typo')


@pytest.mark.parametrize('change', ['password', 'duplicate_port', 'unsafe_host', 'relative_root', 'unsafe_auth', 'unsafe_address_family'])
def test_bad_catalog_does_not_fall_back_to_default(catalog, tmp_path, change):
    server = catalog['servers']['test-server']
    if change == 'password':
        server['ssh']['password'] = 'do-not-store'
    elif change == 'duplicate_port':
        server['local_port'] = catalog['servers']['rented-5090']['local_port']
    elif change == 'unsafe_host':
        server['ssh']['host'] = '-oProxyCommand=anything'
    elif change == 'unsafe_auth':
        server['ssh']['preferred_authentications'] = 'password -oStrictHostKeyChecking=no'
    elif change == 'unsafe_address_family':
        server['ssh']['address_family'] = 'inet -oProxyCommand=anything'
    else:
        server['runtime_root'] = 'relative/path'
    path = tmp_path/'servers.json'
    path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError):
        load_catalog(path)


def test_cli_profile_precedes_environment_and_config_remains_editable(catalog, tmp_path, monkeypatch):
    catalog['servers']['test-server']['local_port'] = 28789
    path = tmp_path/'servers.json'
    path.write_text(json.dumps(catalog))
    monkeypatch.setenv('LINGBOT_SERVERS_CONFIG', str(path))
    monkeypatch.setenv('LINGBOT_SERVER', 'test-server')
    assert select_server(load_catalog())[1]['local_port'] == 28789
    assert select_server(load_catalog(), 'rented-5090')[0] == 'rented-5090'


def test_ui_identity_is_installed_binding_not_default(catalog, tmp_path, monkeypatch):
    path = tmp_path/'servers.json'
    path.write_text(json.dumps(catalog))
    monkeypatch.setenv('LINGBOT_SERVERS_CONFIG', str(path))
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/runtime.json').write_text(json.dumps(dict(server_id='test-server')))
    state = server_state(tmp_path)
    assert state['default_server'] == 'rented-5090'
    assert state['current']['id'] == 'test-server'
    assert state['current']['url'] == 'http://127.0.0.1:18789/?server=test-server'
    assert all('ssh' not in item for item in state['servers'])
    monkeypatch.setenv('LINGBOT_SERVER', 'rented-5090')
    with pytest.raises(ValueError, match='binding'):
        server_state(tmp_path)


def test_local_entry_rejects_root_profile_mismatch(catalog, monkeypatch):
    spec = importlib.util.spec_from_file_location('runtime_entry', EVAL/'scripts/runtime_entry.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    name, root, command = entry.resolve(catalog, None, 'web', [])
    assert name == 'rented-5090' and command[-4:] == ['--port', '8788', '--gpu', '0']
    monkeypatch.setenv('LINGBOT_ROOT', '/srv/unrelated-runtime')
    with pytest.raises(ValueError, match='differs'):
        entry.resolve(catalog, None, 'eval', ['--base'])


def test_wrong_tunnel_blocks_api_write_before_queue_change(catalog, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from eval_platform.store import Store
    import importlib
    path = tmp_path/'servers.json'
    path.write_text(json.dumps(catalog))
    monkeypatch.setenv('LINGBOT_SERVERS_CONFIG', str(path))
    monkeypatch.setenv('LINGBOT_ROOT', str(tmp_path))
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/runtime.json').write_text(json.dumps(dict(server_id='rented-5090')))
    module = importlib.import_module('eval_platform.app')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'store', Store(tmp_path))
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: '')
    client = TestClient(module.app)
    state = client.get('/api/state').json()
    assert state['server']['current']['id'] == 'rented-5090'
    assert client.post('/api/uploads', headers={'X-Lingbot-Server': 'test-server'}).status_code == 409
    assert not (tmp_path/'incoming').exists()
    assert client.post('/api/uploads', headers={'X-Lingbot-Server': 'rented-5090'}).status_code == 200


def test_connect_dry_run_never_starts_ssh(catalog):
    result = subprocess.run([sys.executable, str(EVAL/'connect.py'), '--dry-run', 'web'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data['server'] == 'rented-5090'
    assert '--with-worker' not in data['argv'][-1]
