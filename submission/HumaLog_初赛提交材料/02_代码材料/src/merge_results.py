"""Build official count JSON exclusively from hash-verified, completed shard receipts."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

from validate_results import validate

C = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def combine(directories, contract, tasks):
    groups = {(task, setting): [] for setting in ('demo_clean', 'demo_randomized') for task in tasks}
    seen, allocation = set(), set()
    expected_protocol = hashlib.sha256((C / 'configs/eval-contract.json').read_bytes()).hexdigest()
    for directory in directories:
        directory = Path(directory)
        manifest = read_json(directory / 'receipt-manifest.json')
        assert manifest.get('passed') is True and manifest['revision'] == contract['revision']
        assert manifest['protocol_sha256'] == expected_protocol
        for name, binding in manifest['files'].items():
            assert Path(name).name == name and name not in ('.', '..')
            raw = (directory / name).read_bytes()
            assert len(raw) == binding['bytes'] and hashlib.sha256(raw).hexdigest() == binding['sha256'], name
        assert {'verification.json','outcomes.csv','by-group.csv','non-scored-errors.csv'} <= set(manifest['files'])
        verification = read_json(directory / 'verification.json')
        assert verification.get('passed') is True and verification['revision'] == contract['revision']
        assert verification['protocol_sha256'] == expected_protocol and verification['per_group'] == 30
        assert verification['video_disabled_verified'] is True
        assert verification['source_complete_sha256'] == contract['source_complete_sha256']
        assert verification['runtime_parameter_audit_sha256'] == contract['runtime_parameter_audit_sha256']
        rows = list(csv.DictReader(io.StringIO((directory / 'outcomes.csv').read_text(encoding='utf8'))))
        assert len(rows) == verification['scored']
        successes = timeouts = 0
        per_file = Counter()
        for row in rows:
            assert row['model_revision'] == contract['revision'] and row['protocol_sha256'] == expected_protocol
            key = (row['task'], row['setting'], int(row['seed']))
            group = key[:2]
            assert group in groups and key not in seen, 'Duplicate or unknown task/setting/seed'
            assert 100000 <= key[2] < 100300 and row['success'] in ('0', '1')
            assert row['status'] in ('policy_completed', 'policy_timeout')
            assert row['status'] != 'policy_timeout' or row['success'] == '0'
            seen.add(key)
            groups[group].append(int(row['success']))
            per_file[group] += 1
            successes += int(row['success'])
            timeouts += row['status'] == 'policy_timeout'
        assert successes == verification['success'] and timeouts == verification['timeout']
        assert len(per_file) == verification['groups'] and all(n == 30 for n in per_file.values())
        assert not allocation.intersection(per_file), 'A task group was assigned to multiple receipts'
        allocation.update(per_file)
    assert len(allocation) == 100 and all(len(rows) == 30 for rows in groups.values()), 'Incomplete 100-group coverage'
    result = dict(schema_version=1,team_id='1380507',results={})
    for short, setting in [('clean','demo_clean'),('randomized','demo_randomized')]:
        result['results'][short] = {task:dict(attempts=len(groups[task,setting]),successes=sum(groups[task,setting])) for task in tasks}
    validate(result, tasks, expected_attempts=30)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt-dirs', nargs='+', type=Path, required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    contract = read_json(C / 'configs/eval-contract.json')
    tasks = read_json(C / 'configs/official-task-list.json')['tasks']
    result = combine(a.receipt_dirs,contract,tasks)
    # A completed result is immutable through this entry. Never replace a real result silently.
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf8') as stream:
        stream.write(json.dumps(result,ensure_ascii=False,indent=2) + '\n')
    print(json.dumps(dict(passed=True,output=str(a.output),scored=3000)))


if __name__ == '__main__':
    main()
