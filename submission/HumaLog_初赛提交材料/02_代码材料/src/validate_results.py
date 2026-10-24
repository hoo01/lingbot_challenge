"""Validate the organizer's count-only JSON. This does not certify provenance."""
import argparse
import json
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(value, tasks, expected_attempts=None):
    require(isinstance(value, dict), 'Result must be an object')
    require(set(value) == {'schema_version', 'team_id', 'results'}, 'Unexpected top-level fields')
    require(type(value['schema_version']) is int and value['schema_version'] == 1, 'schema_version must be integer1')
    require(value['team_id'] == '1380507', 'team_id must be string1380507')
    require(isinstance(value['results'], dict) and set(value['results']) == {'clean', 'randomized'}, 'Need both settings')
    require(len(tasks) == len(set(tasks)) == 50, 'Official task list must contain50 unique tasks')
    summary = {}
    for setting in ['clean', 'randomized']:
        group = value['results'][setting]
        require(isinstance(group, dict) and set(group) == set(tasks), f'{setting}: must cover exactly50 official tasks')
        attempts = successes = 0
        for task in tasks:
            row = group[task]
            require(isinstance(row, dict) and set(row) == {'attempts', 'successes'}, f'{setting}/{task}: unexpected fields')
            n, k = row['attempts'], row['successes']
            require(type(n) is int and type(k) is int, f'{setting}/{task}: counts must be completed integer measurements')
            require(n > 0 and 0 <= k <= n, f'{setting}/{task}: require attempts>0 and0<=successes<=attempts')
            if expected_attempts is not None:
                require(n == expected_attempts, f'{setting}/{task}: attempts differ from expected{expected_attempts}')
            attempts += n
            successes += k
        summary[setting] = dict(attempts=attempts, successes=successes)
    return dict(schema_and_count_bounds_valid=True, episode_provenance_verified=False, settings=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--expected-attempts', type=int)
    args = parser.parse_args()
    if args.expected_attempts is not None and args.expected_attempts <= 0:
        parser.error('--expected-attempts must be positive')
    tasks = json.loads((Path(__file__).resolve().parents[1] / 'configs/official-task-list.json').read_text(encoding='utf8'))['tasks']
    try:
        result = validate(json.loads(args.results.read_text(encoding='utf8')), tasks, args.expected_attempts)
    except (ValueError, OSError, TypeError) as error:
        print(json.dumps(dict(valid=False, error=str(error)), ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
