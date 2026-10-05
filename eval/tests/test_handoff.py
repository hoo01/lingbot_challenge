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
from eval_platform.contracts import source_lock, validate_source, model_construction


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
    header = json.dumps({'model.embed_tokens.weight':dict(dtype='F32',shape=[1],data_offsets=[0,4])}).encode()
    (path/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+struct.pack('<f',0))
    return store, path, cli, norm


@pytest.mark.skipif(os.name=='nt', reason='Production symlinks are tested on Linux')
def test_registration_ignores_obsolete_export_receipts(bundle):
    store,path,cli,norm = bundle
    for filename in ('eval_handoff.json', 'norm.provenance.json', 'clean_manifest.json'):
        (path/filename).write_text('{"obsolete_sha256":"outdated"}')
    ck = store.register('old bundle', str(path), str(cli), str(norm))
    assert ck['kind'] == 'team_merged_checkpoint'
    assert ck['norm_sha256'] == digest(norm)
    assert all(Path(record['path']).name not in
               {'eval_handoff.json', 'norm.provenance.json', 'clean_manifest.json'}
               for record in ck['metadata_files'])
    assert store.jobs() == []


@pytest.mark.skipif(os.name=='nt', reason='Production symlinks are tested on Linux')
def test_upload_and_web_registration_need_only_model_cli_and_norm(bundle, monkeypatch):
    from fastapi.testclient import TestClient
    import importlib
    store,path,cli,norm = bundle
    monkeypatch.setenv('LINGBOT_ROOT', str(store.root))
    from eval_platform import app
    importlib.reload(app)
    client = TestClient(app.app)
    upload = client.post('/api/uploads').json()
    for file in (path/'model.safetensors', cli, norm):
        response = client.put(f"/api/uploads/{upload['id']}/{file.name}", content=file.read_bytes())
        assert response.status_code == 200
    uploaded = Path(upload['path'])
    response = client.post('/api/checkpoints', json=dict(name='uploaded model',
        checkpoint=str(uploaded), cli=str(uploaded/cli.name), norm=str(uploaded/norm.name)))
    assert response.status_code == 200, response.text
    ck = response.json()
    assert ck['files'][0]['sha256'] == digest(uploaded/'model.safetensors')
    assert ck['norm_sha256'] == digest(norm)
    assert len(store.checkpoints()) == 1 and store.jobs() == []


@pytest.mark.parametrize('section,key,value', [
    ('data', 'img_size', 224), ('model', 'chunk_size', 25), ('train', 'num_steps', 5)])
def test_incompatible_inference_profile_still_rejected(bundle, section, key, value):
    import yaml
    store,path,cli,norm = bundle
    data = yaml.safe_load(cli.read_text())
    data[section][key] = value
    cli.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='profile requires'):
        store.register('incompatible', str(path), str(cli), str(norm))
    assert store.checkpoints() == []


@pytest.mark.parametrize('filename', ['lingbotvla_cli.yaml', 'norm.json'])
def test_missing_registration_file_returns_http_400_without_creating_records(bundle, monkeypatch, filename):
    from fastapi.testclient import TestClient
    import importlib
    store, path, cli, norm = bundle
    (path/filename).unlink()
    monkeypatch.setenv('LINGBOT_ROOT', str(store.root))
    from eval_platform import app
    importlib.reload(app)
    response = TestClient(app.app).post('/api/checkpoints', json=dict(
        name='missing', checkpoint=str(path), cli=str(cli), norm=str(norm)))
    assert response.status_code == 400
    assert 'file not found' in response.json()['detail']
    assert filename in response.json()['detail']
    assert store.checkpoints() == [] and store.jobs() == []
    assert not (store.root/'checkpoints').exists()


@pytest.mark.parametrize('content', ['model: [broken', 'model: null\ntrain: {}\ndata: {}\n'])
def test_invalid_cli_is_an_input_error(bundle, content):
    store, path, cli, norm = bundle
    cli.write_text(content)
    with pytest.raises(ValueError, match='CLI YAML'):
        store.register('bad config', str(path), str(cli), str(norm))
    assert store.checkpoints() == []


@pytest.mark.skipif(os.name=='nt', reason='Production checkpoint symlinks require Linux')
def test_cli_register_only_persists_model_without_job_or_gpu(bundle):
    store, path, cli, norm = bundle
    # No NVIDIA tool or GPU lock is available in this subprocess.
    env = {**os.environ, 'LINGBOT_ROOT': str(store.root), 'PYTHONPATH': str(EVAL/'src'),
           'PATH': '', 'LINGBOT_GPU_LOCK': '/unwritable/nonexistent/gpu.lock'}
    result = subprocess.run([sys.executable, str(EVAL/'scripts/team_eval.py'),
        '--checkpoint', str(path), '--cli', str(cli), '--norm', str(norm),
        '--title', 'registered-only', '--register-only'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output['status'] == 'registered'
    assert store.checkpoint(output['checkpoint_id'])['name'] == 'registered-only'
    assert store.jobs() == []


def test_web_api_preserves_plain_text_and_json_errors():
    import shutil
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed for the browser response regression test')
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8');
vm.runInThisContext(source.slice(source.indexOf('async function api('),source.indexOf('const post=')));
(async()=>{
  for(const [body,message] of [['Internal Server Error','Internal Server Error'],['{"detail":"missing CLI"}','missing CLI'],['','HTTP 500']]){
    global.fetch=async()=>new Response(body,{status:500});
    await assert.rejects(()=>api('/check'),error=>error.message===message);
  }
  global.fetch=async()=>new Response('{"id":"new-model"}',{status:200});
  assert.deepStrictEqual(await api('/check'),{id:'new-model'});
})().catch(e=>{console.error(e);process.exit(1)});
'''
    result = subprocess.run([node, '-e', script, str(EVAL/'src/eval_platform/web/app.js')],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_factorized_model_cannot_enter_official_loader(bundle):
    store,path,cli,norm = bundle
    (path/'factorized-adapter.json').write_text('{}')
    with pytest.raises(ValueError, match='merged HF model'):
        store.register('factorized', str(path), str(cli), str(norm))


def test_model_construction_must_be_supported(bundle):
    store,path,cli,norm = bundle
    assert model_construction(path) == 'native'
    construction = path/'runtime-construction.json'
    construction.write_text(json.dumps(dict(nonpersistent_buffer_initialization='bf16_then_fp32',vision_device='cpu')))
    with pytest.raises(ValueError, match='Unsupported runtime-construction'):
        store.register('unsupported',str(path),str(cli),str(norm))
    assert store.checkpoints() == []


@pytest.mark.skipif(os.name=='nt', reason='Production symlinks are tested on Linux')
def test_profile_and_frozen_metadata(bundle):
    store,path,cli,norm = bundle
    construction=path/'runtime-construction.json'
    construction.write_text(json.dumps(dict(nonpersistent_buffer_initialization='bf16_then_fp32')))
    audit=path/'action-audit.json';audit.write_text('{"example":true}')
    ck = store.register('test', str(path), str(cli), str(norm))
    assert model_construction(Path(ck['checkpoint']))=='bf16_then_fp32'
    import yaml
    runtime=yaml.safe_load(Path(ck['cli']).read_text())
    assert runtime['train']['vit_attn_implementation']=='flash_attention_2'
    assert runtime['train']['enable_fp32'] and runtime['train']['use_compile']
    assert (Path(ck['cli']).parent/'original_training_cli.yaml').read_bytes()==cli.read_bytes()
    assert (Path(ck['cli']).parent/'action-audit.json').read_bytes()==audit.read_bytes()
    for record in ck['metadata_files']:
        assert digest(record['path']) == record['sha256']
    config = dict(checkpoint_id=ck['id'],tasks=['click_bell'],settings=['demo_clean'],seeds=[1])
    job = store.submit('test',config)
    assert job['config']['inference_profile']=='official_defaults_v1'
    assert job['config']['vision_device']==job['config']['embedding_device']=='cuda'
    for change in [dict(precision='bf16'),dict(execute=25)]:
        with pytest.raises(ValueError):
            store.submit('unsupported',{**config,**change})


@pytest.mark.skipif(os.name=='nt', reason='Production symlinks are tested on Linux')
@pytest.mark.parametrize('changed', ['weight', 'norm', 'cli', 'construction', 'audit'])
def test_automatic_hashes_block_modified_files_before_model_start(bundle, monkeypatch, changed):
    from eval_platform import runner
    store,path,cli,norm = bundle
    (path/'runtime-construction.json').write_text(
        '{"nonpersistent_buffer_initialization":"bf16_then_fp32"}')
    (path/'action-audit.json').write_text('{"example":true}')
    ck = store.register('test', str(path), str(cli), str(norm))
    job = store.submit('test', dict(checkpoint_id=ck['id'],tasks=['click_bell'],
                                 settings=['demo_clean'],seeds=[1]))
    changed_path = {
        'weight': path/'model.safetensors', 'norm': Path(ck['norm']), 'cli': Path(ck['cli']),
        'construction': Path(ck['checkpoint'])/'runtime-construction.json',
        'audit': Path(ck['cli']).parent/'action-audit.json',
    }[changed]
    content = changed_path.read_bytes()
    if changed == 'weight':
        changed_path.write_bytes(content[:-4] + struct.pack('<f', 1))
    else:
        changed_path.write_bytes(content + b' ')
    monkeypatch.setattr(runner, 'check_gpu', lambda gpu: None)
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **k: pytest.fail('Model must not start'))
    with pytest.raises(RuntimeError, match='changed|Changed'):
        runner.run(store.root, job['id'], 'unused', 18006)


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
