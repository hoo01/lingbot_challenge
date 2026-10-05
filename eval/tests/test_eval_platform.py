import concurrent.futures
import json
from pathlib import Path
import sys
import os
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from eval_platform.store import Store
from eval_platform.reports import report,csv_report


@pytest.fixture
def platform(tmp_path):
    task=tmp_path/'third_party/RoboTwin/description/task_instruction'
    task.mkdir(parents=True);(task/'adjust_bottle.json').write_text('{}')
    s=Store(tmp_path)
    with s.connect() as db:
        db.execute('INSERT INTO checkpoints VALUES (?,?)',('base',json.dumps({'id':'base'})))
    return s


def config():
    return dict(checkpoint_id='base',tasks=['adjust_bottle'],settings=['demo_clean','demo_randomized'],seeds=[100010,100011])


def test_recording_is_optional_and_new_jobs_use_ten_fps(platform):
    off=platform.submit('no recording',config())['config']
    assert off['video'] is False and off['video_fps']==10
    request={**config(),'video':True}
    on=platform.submit('record',request)['config']
    assert on['video'] is True and on['video_fps']==10
    legacy=platform.submit('explicit legacy fps',{**request,'video_fps':25})['config']
    assert legacy['video_fps']==25
    with pytest.raises(ValueError):platform.submit('bad',{**config(),'video':'false'})
    with pytest.raises(ValueError):platform.submit('bad',{**config(),'video_fps':0})


def test_workers_claim_different_jobs_atomically(platform):
    for i in range(6):platform.submit(str(i),config())
    with concurrent.futures.ThreadPoolExecutor(6) as ex:
        claimed=list(ex.map(lambda i:platform.claim(f'GPU-{i}',100+i),range(6)))
    assert len({j['id'] for j in claimed})==6
    assert platform.claim('GPU-0',123) is None


def test_one_gpu_cannot_claim_twice_and_queued_cancel_never_runs(platform):
    a=platform.submit('first',config());b=platform.submit('cancelled',config())
    platform.cancel(b['id'])
    assert platform.claim('GPU-a',1)['id']==a['id']
    assert platform.claim('GPU-b',2) is None
    assert platform.cancel(a['id'])['status']=='cancel_requested'
    assert platform.claim('GPU-a',3) is None


def test_denominators_and_incomplete_lines(platform):
    job=platform.submit('report',config());out=platform.run_folder(job['id'])/'evaluation';out.mkdir()
    rows=[dict(task='adjust_bottle',setting='demo_clean',seed=100010,status='policy_completed',success=True,seconds=60),
          dict(task='adjust_bottle',setting='demo_clean',seed=100011,status='setup_error',success=False),
          dict(task='adjust_bottle',setting='demo_randomized',seed=100010,status='policy_error',success=False)]
    (out/'episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n{"unfinished":')
    r=report(platform.run_folder(job['id']),job['config'])
    assert (r['planned'],r['attempted'],r['valid'],r['successes'],r['setup_errors'],r['execution_errors'])==(4,3,1,1,1,1)
    assert r['success_rate']==1
    assert 'demo_randomized,2,1,0,0,0,1' in csv_report(r)


def test_invalid_paths_and_duplicate_seeds(platform):
    with pytest.raises(ValueError):platform.path('../outside')
    bad=config();bad['seeds']=[1,1]
    with pytest.raises(ValueError):platform.submit('bad',bad)
    bad=config();bad['tasks']=['not_a_task']
    with pytest.raises(ValueError):platform.submit('bad',bad)


def test_corrupt_completed_record_is_visible(platform):
    job=platform.submit('corrupt',config());out=platform.run_folder(job['id'])/'evaluation';out.mkdir()
    (out/'episodes.jsonl').write_text('{broken}\n')
    assert report(platform.run_folder(job['id']),job['config'])['record_error'] is not None


@pytest.mark.skipif(os.name=='nt', reason='Production symlinks are tested on Linux')
def test_checkpoint_requires_full_model_and_binds_norm(platform):
    import struct
    folder=platform.root/'incoming/model';folder.mkdir(parents=True)
    cli=folder/'lingbotvla_cli.yaml';cli.write_text('model: {}\ntrain: {}\ndata: {img_size: 256}\n')
    norm=folder/'norm.json';norm.write_text('{"state":{"mean":[0],"std":[1]}}')
    def weights(name):
        header=json.dumps({name:{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
        (folder/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+struct.pack('<f',0))
    weights('adapter.lora_A')
    with pytest.raises(ValueError):platform.register('bad',str(folder),str(cli),str(norm))
    weights('model.language_model.embed_tokens.weight')
    from eval_platform.store import digest
    provenance=norm.with_suffix('.provenance.json')
    provenance.write_text(json.dumps(dict(clean_only=True,norm_sha256=digest(norm))))
    manifest=folder/'clean_manifest.json'
    manifest.write_text(json.dumps(dict(clean_only=True,datasets=[dict(task='adjust_bottle')])) )
    (folder/'eval_handoff.json').write_text(json.dumps(dict(kind='clean_sft_export',
        cli_sha256=digest(cli),norm_sha256=digest(norm),provenance_sha256=digest(provenance),
        manifest_sha256=digest(manifest))))
    result=platform.register('base',str(folder),str(cli),str(norm))
    original=Path(result['norm']).read_bytes();norm.write_text('{"changed":true}')
    assert Path(result['norm']).read_bytes()==original
    assert len(result['files'][0]['sha256'])==64


def test_api_traversal_csrf_and_upload(platform,monkeypatch):
    monkeypatch.setenv('LINGBOT_ROOT',str(platform.root))
    from fastapi.testclient import TestClient
    import importlib
    app=importlib.import_module('eval_platform.app');app.store=platform
    client=TestClient(app.app)
    assert client.post('/api/uploads',headers={'Origin':'https://evil.invalid'}).status_code==403
    bundle=client.post('/api/uploads').json()
    assert client.put(f"/api/uploads/{bundle['id']}/norm.json",content=b'{"a":1}').status_code==200
    assert client.put(f"/api/uploads/{bundle['id']}/norm.json",content=b'{}').status_code==409
    assert client.put(f"/api/uploads/{bundle['id']}/script.py",content=b'print(1)').status_code==400
    job=platform.submit('api',config())
    assert client.get(f"/api/jobs/{job['id']}/files/%2e%2e/secret.json").status_code==404
