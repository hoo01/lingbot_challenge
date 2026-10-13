"""Recompute the E1 report tables from archived small files; no remote/GPU access."""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
F1 = HERE.parent / 'f1-15000-completed-20261012'
TRAIN = REPO / 'train/evidence/e1_shuweilin_20261011_080143'
REPORT = HERE.with_suffix('.md')
NORM = '7e6ac279d147cb8422108d36216768111c19bb2113ca9eee1a23df22df238df0'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity(path):
    content = path.read_bytes()
    # Previously tracked training files are checked out as CRLF on Windows;
    # their archive and provenance hashes describe the LF Git blob bytes.
    policy = 'git_archived_lf' if path.is_relative_to(TRAIN) else 'original_bytes'
    if policy == 'git_archived_lf':
        content = content.replace(b'\r\n', b'\n')
    return dict(path=path.relative_to(REPO).as_posix(), byte_policy=policy,
                bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def key(row):
    return row['task'], row['setting'], int(row['seed'])


def truth(value):
    assert value in ('True', 'False', 'true', 'false', '1', '0'), value
    return value.lower() in ('true', '1')


def csv_text(items):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(items[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(items)
    return stream.getvalue()


def compute():
    for entry in load(HERE / 'manifest.json'):
        path = HERE / entry['file']
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], path
    verification = load(HERE / 'verification.json')
    assert verification['passed'] and verification['status'] == 'completed'
    assert not verification['issues']
    assert sha(HERE / 'verification.json') == 'ce0f9b52cbc96de04d0e781d9340e9857b5da9d74b3ede7b2960290639dc909f'
    assert sha(HERE / 'official-baseline-verification.json') == '37f332f9c95a0f6be1f3f8ce44152e761062874b98d2ae6bfb2d4a4ddc8c9ec6'
    assert source_identity(TRAIN / 'assets/norm_stats.json')['sha256'] == NORM
    mapping = load(HERE / 'recipient-action-mapping.json')
    assert mapping['passed'] and mapping['norm_sha256'] == NORM
    assert mapping['state_added_once_per_horizon'] and mapping['absolute_grippers']
    assert mapping['physical_max_error'] < 1e-6
    assert mapping['export_vs_training_action_audit_available'] is False
    for runtime in verification['runtime']:
        contract = runtime['contract']
        assert runtime['dtype'] == 'fp32'
        assert runtime['parameter_numel'] == {'torch.float32': 6375906359}
        assert runtime['learned_precision']['exact_saved_fp32_values']
        assert contract['normalization_sha256'] == NORM
        assert contract['action_subtract_state'] == {
            'action.arm.position': True, 'action.effector.position': False}
        assert contract['action_relative_type']['action.arm.position'] == 'joint_delta'
    current = rows(HERE / 'paired-outcomes.csv')
    baseline_rows = rows(F1 / 'paired-outcomes.csv')
    baseline = {key(row): row for row in baseline_rows}
    assert len(current) == len({key(row) for row in current}) == len(baseline_rows) == len(baseline) == 500
    assert {key(row) for row in current} == set(baseline)
    groups = Counter(key(row)[:2] for row in current)
    assert len(groups) == 100 and set(groups.values()) == {5}
    assert verification['completed'] == 500 and verification['timeout_zero'] == 0
    pair_counts = Counter()
    tasks = defaultdict(lambda: dict(scored=0, e0_success=0, e1_success=0, f1_success=0))
    setting_totals = defaultdict(lambda: dict(scored=0, e0_success=0, e1_success=0, f1_success=0))
    pairs = []
    for row in sorted(current, key=key):
        old = baseline[key(row)]
        assert row['status'] == 'policy_completed' and row['expert_attempts'] == '0'
        assert all(truth(row[field]) for field in (
            'instruction_equal', 'parameters_equal', 'index_equal',
            'initial_joints_equal', 'three_initial_camera_sha_equal'))
        e0, e1, f1 = (truth(old['e0_success']), truth(row['success']), truth(old['f1_success']))
        category = ('both_success' if e0 and e1 else 'e0_only' if e0 else
                    'e1_only' if e1 else 'both_failure')
        pair_counts[category] += 1
        for aggregate in (tasks[row['task']], setting_totals[row['setting']]):
            aggregate['scored'] += 1
            for label, success in (('e0', e0), ('e1', e1), ('f1', f1)):
                aggregate[label + '_success'] += success
        pairs.append(dict(task=row['task'], setting=row['setting'], seed=int(row['seed']),
                          e0_success=e0, e1_success=e1, category=category,
                          e0_episode_sha256=old['e0_episode_sha256']))
    assert sum(t['e1_success'] for t in tasks.values()) == verification['success'] == 51
    assert sum(t['e0_success'] for t in tasks.values()) == 54
    assert sum(t['f1_success'] for t in tasks.values()) == 49
    group_rows = rows(HERE / 'by-group.csv')
    assert len(group_rows) == len({(r['task'], r['setting']) for r in group_rows}) == 100
    actual_groups = Counter(key(row)[:2] for row in current if truth(row['success']))
    for row in group_rows:
        group = (row['task'], row['setting'])
        assert group in groups and int(row['scored']) == int(row['completed']) == 5
        assert int(row['timeout_zero']) == 0 and int(row['success']) == actual_groups[group]
    by_task = [dict(task=task, **value, e1_minus_e0=value['e1_success']-value['e0_success'])
               for task, value in sorted(tasks.items())]
    delta_counts = Counter('increase' if r['e1_minus_e0'] > 0 else
                           'decrease' if r['e1_minus_e0'] < 0 else 'equal' for r in by_task)
    official = load(HERE / 'official-baseline-verification.json')
    assert official['passed'] and official['completed'] == 500 and official['success'] == 449
    summary = dict(job_id=verification['job_id'], scored=500, completed=500,
                   timeout_zero=0, success=51, success_rate=0.102,
                   settings=dict(sorted(setting_totals.items())),
                   paired_e0_e1=dict(sorted(pair_counts.items())),
                   task_delta_counts=dict(sorted(delta_counts.items())),
                   E0_success=54, F1_success=49, official_success=449,
                   E1_minus_E0_percentage_points=-0.6,
                   E1_minus_F1_percentage_points=0.4,
                   total_actions=verification['total_actions'],
                   finite_prediction_chunks=verification['finite_prediction_chunks'],
                   initial_rgb_pngs_checked=verification['initial_rgb_pngs_checked'],
                   all_expert_attempts=verification['all_expert_attempts'])
    return {'report-summary.json': json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
            'e0-e1-paired-outcomes.csv': csv_text(pairs),
            'by-task-comparison.csv': csv_text(by_task)}, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Save derived tables and the local check receipt')
    args = parser.parse_args()
    outputs, summary = compute()
    for name, content in outputs.items():
        path = HERE / name
        if args.write:
            path.write_bytes(content.encode('utf-8'))
        else:
            assert path.read_bytes() == content.encode('utf-8'), path
    assert REPORT.is_file()
    sources = [HERE / e['file'] for e in load(HERE / 'manifest.json')]
    sources += [HERE / 'manifest.json', HERE / 'recipient-action-mapping.json',
                HERE / 'official-baseline-verification.json', F1 / 'paired-outcomes.csv',
                TRAIN / 'assets/norm_stats.json', TRAIN / 'run_metadata.json',
                TRAIN / 'metrics_summary.json', TRAIN / 'training_config.yaml']
    receipt = dict(passed=True, verified_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                   scope='Local archived metadata, SHA and CSV checks only; no remote, model, simulation or video access',
                   summary=summary,
                   source_files=[source_identity(p) for p in sources],
                   output_files=[dict(path=p.relative_to(REPO).as_posix(), bytes=p.stat().st_size,
                                      sha256=sha(p)) for p in [REPORT, Path(__file__), *[HERE / name for name in outputs]]])
    if args.write:
        (HERE / 'report-verification.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    else:
        saved = load(HERE / 'report-verification.json')
        assert saved['passed'] and saved['summary'] == summary
        assert saved['source_files'] == receipt['source_files'] and saved['output_files'] == receipt['output_files']
    print(json.dumps(dict(passed=True, **summary), ensure_ascii=False))


if __name__ == '__main__':
    main()
