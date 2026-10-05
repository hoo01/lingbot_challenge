import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import os
import subprocess
import pytest

EVAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL/'src'))
from eval_platform.store import Store, digest
from eval_platform import resources
from eval_platform.contracts import source_lock, validate_source


@pytest.fixture
def bundle(tmp_path):
    store = Store(tmp_path)
    task = tmp_path/'third_party/RoboTwin/description/task_instruction/click_bell.json'
    task.parent.mkdir(parents=True)
    task.write_text('{}')
    path = tmp_path/'incoming/test'
    path.mkdir(parents=True)
    cli = path/'lingbotvla_cli.yaml'
    cli.write_text('model: {chunk_size: 50}\ntrain: {num_steps: 10}\ndata: {img_size: 256}\n')
    norm = path/'norm.json'
    norm.write_text('{"norm_stats":{"a":1},"count":2}')
    provenance = norm.with_suffix('.provenance.json')
    provenance.write_text(json.dumps(dict(clean_only=True,norm_sha256=digest(norm))))
    manifest = path/'clean_manifest.json'
    manifest.write_text(json.dumps(dict(clean_only=True,datasets=[dict(task='click_bell')])) )
    receipt = dict(kind='clean_sft_export',cli_sha256=digest(cli),norm_sha256=digest(norm),
                   provenance_sha256=digest(provenance),manifest_sha256=digest(manifest))
    (path/'eval_handoff.json').write_text(json.dumps(receipt))
    header = json.dumps({'model.embed_tokens.weight':dict(dtype='F32',shape=[1],data_offsets=[0,4])}).encode()
    (path/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+struct.pack('<f',0))
    return store, path, cli, norm


@pytest.mark.parametrize('filename', ['lingbotvla_cli.yaml','norm.json','norm.provenance.json','clean_manifest.json'])
def test_export_binding_rejects_tampering(bundle, filename):
    store,path,cli,norm = bundle
    with (path/filename).open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='SHA|sha|match'):
        store.register('changed', str(path), str(cli), str(norm))
    assert store.checkpoints() == []


def test_missing_handoff_has_actionable_error(bundle):
    store,path,cli,norm = bundle
    (path/'eval_handoff.json').unlink()
    with pytest.raises(ValueError, match='Missing training handoff'):
        store.register('missing', str(path), str(cli), str(norm))


def test_factorized_model_cannot_enter_official_loader(bundle):
    store,path,cli,norm = bundle
    (path/'factorized-adapter.json').write_text('{}')
    with pytest.raises(ValueError, match='merged HF model'):
        store.register('factorized', str(path), str(cli), str(norm))


@pytest.mark.skipif(os.name=='nt', reason='Production symlinks are tested on Linux')
def test_profile_and_frozen_metadata(bundle):
    store,path,cli,norm = bundle
    ck = store.register('test', str(path), str(cli), str(norm))
    for record in ck['metadata_files']:
        assert digest(record['path']) == record['sha256']
    config = dict(checkpoint_id=ck['id'],tasks=['click_bell'],settings=['demo_clean'],seeds=[1])
    job = store.submit('test',config)
    assert job['config']['inference_profile']=='official_defaults_v1'
    assert job['config']['vision_device']==job['config']['embedding_device']=='cuda'
    for change in [dict(precision='bf16'),dict(execute=25)]:
        with pytest.raises(ValueError):
            store.submit('unsupported',{**config,**change})


def test_shared_lock_path_survives_runtime_relocation(tmp_path,monkeypatch):
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/runtime.json').write_text(json.dumps(dict(gpu_lock='/source/runs/gpu.lock')))
    monkeypatch.delenv('LINGBOT_GPU_LOCK',raising=False)
    assert resources.lock_path(tmp_path)==Path('/source/runs/gpu.lock')


@pytest.mark.parametrize('memory,allowed',[('32607, 28496',False),('24564, 0',False),('32607, 600',True)])
def test_gpu_gate_rejects_training_and_small_card(monkeypatch,memory,allowed):
    monkeypatch.setattr(resources.subprocess,'check_output',lambda *a,**k:memory)
    if allowed:
        assert resources.check_gpu('0')['used_mib']==600
    else:
        with pytest.raises(RuntimeError,match='GPU unavailable'):
            resources.check_gpu('0')


def test_source_tamper_fails_closed(tmp_path):
    (tmp_path/'configs').mkdir()
    source=tmp_path/'third_party/lingbot-vla-v2-official-be969b8f'
    source.mkdir(parents=True)
    (source/'x.py').write_text('changed')
    (tmp_path/'configs/upstream-lock.json').write_text(json.dumps(dict(pristine_source_sha256={'x.py':'0'*64})))
    with pytest.raises(ValueError,match='source differs'):
        source_lock(tmp_path)


def test_baseline_export_is_consistent_with_frozen_scenes():
    data=json.loads((EVAL/'results/baseline_official.json').read_text())
    rows=[json.loads(x) for x in (EVAL/'results/baseline_episodes.jsonl').read_text().splitlines()]
    groups=data['groups']
    assert len(groups)==100 and len(rows)==581
    assert sum(g['valid'] for g in groups)==500
    assert sum(g['successes'] for g in groups)==52
    assert sum(g['initialization_failures'] for g in groups)==81
    assert data['completion']['attempted']==581+3 and data['official_benchmark'] is False
    for g in groups:
        valid=[r for r in rows if r['task']==g['task'] and r['setting']==g['setting'] and r['status']=='policy_completed']
        assert sorted(r['seed'] for r in valid)==sorted(g['valid_seeds'])
    assert digest(EVAL/'results/baseline_valid_scenes.json')==data['evidence']['frozen_manifest_sha256']


def test_installer_refuses_active_or_nonempty_runtime(tmp_path):
    spec=importlib.util.spec_from_file_location('installer',EVAL/'scripts/install_runtime.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source=tmp_path/'research';source.mkdir()
    with pytest.raises(ValueError,match='separate sibling'):
        module.install(EVAL,source,source,check=True)
    target=tmp_path/'team';target.mkdir();(target/'keep.txt').write_text('existing results')
    with pytest.raises(FileExistsError):
        module.install(EVAL,source,target,check=True)
    assert (target/'keep.txt').read_text()=='existing results'


@pytest.mark.skipif(os.name=='nt', reason='Runtime installation uses Linux resource symlinks')
def test_installer_deploys_minimal_package_without_obsolete_documents(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('installer', EVAL/'scripts/install_runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, target = tmp_path/'source', tmp_path/'target'
    for name in ('envs/train', 'envs/sim', 'third_party/RoboTwin',
                 'third_party/lingbot-vla-v2', 'third_party/lingbot-vla-v2-official-be969b8f',
                 'third_party/oidn-2.3.3.x86_64.linux', 'models/Qwen3-VL-4B-Instruct',
                 'models/lingbot-vla-v2-6b', 'artifacts'):
        (source/name).mkdir(parents=True)
    for name in ('envs/train/bin/python', 'envs/sim/bin/python'):
        path = source/name
        path.parent.mkdir()
        path.write_text('fake interpreter')
    (source/'artifacts/oidn-blackwell.json').write_text('{}')
    from eval_platform import contracts
    monkeypatch.setattr(contracts, 'validate_source', lambda *args: {'source_byte_layout': 'pristine'})
    commands = []
    monkeypatch.setattr(module.subprocess, 'run', lambda command, **kwargs: commands.append(command))
    module.install(EVAL, source, target)
    assert {str(p.relative_to(target)) for p in target.rglob('*.md')} == {'README.md', 'results/baseline.md'}
    assert (target/'README.md').read_bytes() == (EVAL/'README.md').read_bytes()
    assert (target/'results/baseline_official.json').read_bytes() == (EVAL/'results/baseline_official.json').read_bytes()
    assert (target/'envs/train').resolve() == source/'envs/train'
    assert (source/'envs/train/bin/python').read_text() == 'fake interpreter'
    assert len(commands) == 2 and commands[0][1:3] == ['-m', 'venv']


@pytest.mark.skipif(os.name=='nt', reason='Linux flock contract')
def test_linux_lock_blocks_second_process_and_releases(tmp_path,monkeypatch):
    monkeypatch.delenv('LINGBOT_GPU_LOCK',raising=False)
    env={**os.environ,'PYTHONPATH':str(EVAL/'src')}
    env.pop('LINGBOT_GPU_LOCK',None)
    code='from pathlib import Path; from eval_platform.resources import gpu_lock; import sys\nwith gpu_lock(Path(sys.argv[1])): print("owned")'
    with resources.gpu_lock(tmp_path):
        blocked=subprocess.run([sys.executable,'-c',code,str(tmp_path)],env=env,capture_output=True,text=True)
        assert blocked.returncode!=0 and 'BlockingIOError' in blocked.stderr
    released=subprocess.run([sys.executable,'-c',code,str(tmp_path)],env=env,capture_output=True,text=True)
    assert released.returncode==0 and released.stdout.strip()=='owned'


def test_pinned_newline_variants_do_not_accept_code_changes(tmp_path):
    (tmp_path/'assets/norm_stats').mkdir(parents=True)
    code=tmp_path/'code.py'; norm=tmp_path/'assets/norm_stats/robotwin.json'
    code.write_bytes(b'x=1\n');norm.write_bytes(b'{}\n')
    lock=dict(pristine_source_sha256={'code.py':digest(code)},normalization_sha256=digest(norm))
    code.write_bytes(b'x=1\r\n');norm.write_bytes(b'{}\r\n')
    lock['historical_crlf']=dict(pristine_source_sha256={'code.py':digest(code)},normalization_sha256=digest(norm))
    assert validate_source(tmp_path,lock)['source_byte_layout']=='historical_crlf'
    code.write_bytes(b'x=2\r\n')
    with pytest.raises(ValueError,match='source differs'):
        validate_source(tmp_path,lock)
