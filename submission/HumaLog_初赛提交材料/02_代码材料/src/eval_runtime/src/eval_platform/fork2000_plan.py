"""Round-robin blocks of five, no score-dependent task allocation."""
from pathlib import Path
import hashlib,json,os

MODE='unfiltered_wall900_stall240_zero_continue_v1'
TARGET=30
SEEDS=list(range(100000,100300))
SCORED={'policy_completed','policy_timeout'}

def active(config):return config.get('timeout_policy',{}).get('mode')==MODE
def is_scored(row):return row.get('status') in SCORED
def retained_error(row):
    return (row.get('status')=='setup_error' and row.get('native_initialization_error') is True
            and row.get('policy_entered') is False and row.get('expert_attempts')==0
            and row.get('success') is False and bool(row.get('error')))

def build(tasks):
    scope=json.loads((Path(os.environ['LINGBOT_ROOT'])/'artifacts/eval2000-scope.json').read_text())
    return [dict(task=t,setting=s,seed=n) for t in tasks for s in scope['settings'] for n in SEEDS]

def validate(config):
    scope=json.loads((Path(os.environ['LINGBOT_ROOT'])/'artifacts/eval2000-scope.json').read_text())
    assert config['tasks']==scope['tasks'] and config['settings']==scope['settings']
    assert config['seeds']==SEEDS and config['fixed_scenes']==build(config['tasks']) and config.get('expert_check') is False
    assert not any(config.get(k) for k in ('paired_reference','paired_entry','scene_replay','expert_quota','valid_per_group'))
    return True

def trailing_init(group):
    count=0
    for r in reversed(group):
        if is_scored(r):break
        count+=1
    return count

def select(config,groups):
    for level in range(5,31,5):
        for t in config['tasks']:
            for s in config['settings']:
                group=groups[t,s]
                if sum(is_scored(r) for r in group)<level and len(group)<300 and trailing_init(group)<20:
                    return dict(task=t,setting=s,seed=SEEDS[len(group)])
    return None

def validate_history(config,records):
    groups={(t,s):[] for t in config['tasks'] for s in config['settings']}
    for r in records:
        expected=select(config,groups)
        assert expected and all(r.get(k)==v for k,v in expected.items()),'Unexpected, skipped or repeated scene / block order'
        assert r.get('expert_attempts')==0 and (is_scored(r) or retained_error(r)),'Unresolved engineering interruption; do not change seed'
        assert r.get('actual_environment_seed',r['seed'])==r['seed']
        groups[r['task'],r['setting']].append(r)
    return groups

def next_scene(config,records):return select(config,validate_history(config,records))
def candidates(config,records,task,setting):
    while True:
        item=next_scene(config,records)
        if item is None or (item['task'],item['setting'])!=(task,setting):return
        yield SEEDS.index(item['seed']),item['seed']

def write_scene_file(folder,scenes):
    path=Path(folder)/'fixed-scenes.input.json';raw=json.dumps(scenes,indent=2).encode()
    if path.exists():assert path.read_bytes()==raw
    else:path.write_bytes(raw)
    return str(path),hashlib.sha256(raw).hexdigest()

def read_scene_file(path,expected_sha256,root):
    p=Path(path).resolve();assert p.is_relative_to(Path(root).resolve()) and p.stat().st_size<5000000
    raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected_sha256
    scenes=json.loads(raw);assert 0<len(scenes)<=9000
    return scenes
