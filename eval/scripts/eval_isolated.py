"""Run each remaining scene in a fresh simulator process with bounded waits.

The policy server, scene list, observations and scoring remain unchanged. All
completed records, including initialization failures, are retained. A stalled
worker is recorded as an infrastructure interruption and stops the suite; it is
never silently omitted, converted into a policy failure, or replaced by a seed.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import os


ROOT = Path(os.environ.get('LINGBOT_ROOT', Path(__file__).resolve().parents[1])).resolve()


def run_worker(command, progress_path, episode_timeout, stall_timeout,
               poll_seconds=2, terminate_grace=10):
    def signature():
        try:
            return progress_path.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    previous = signature()
    started = last_progress = time.monotonic()
    process = subprocess.Popen(command)
    try:
        while process.poll() is None:
            now = time.monotonic()
            current = signature()
            if current != previous:
                previous, last_progress = current, now
            if now - started >= episode_timeout:
                raise TimeoutError(f'Episode exceeded {episode_timeout}s; PID {process.pid}')
            if now - last_progress >= stall_timeout:
                raise TimeoutError(f'No completed action chunk for {stall_timeout}s; PID {process.pid}')
            time.sleep(poll_seconds)
        return process.returncode
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=terminate_grace)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def records(output):
    path = output / 'episodes.jsonl'
    result = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    keys = [(r['task'], r['setting'], r['seed']) for r in result]
    assert len(keys) == len(set(keys)), 'Duplicate episode records'
    assert all(r['status'] in ('policy_completed', 'setup_error') for r in result), \
        'Recorded execution errors must be investigated before resuming'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-completed', action='store_true')
    parser.add_argument('--episode-timeout', type=float, default=900)
    parser.add_argument('--stall-timeout', type=float, default=240)
    args, worker_args = parser.parse_known_args()
    assert args.episode_timeout > 0 and args.stall_timeout > 0
    if args.output.exists() and any(args.output.iterdir()) and not args.resume_completed:
        raise FileExistsError(args.output)
    resume = args.resume_completed

    def event(kind, **fields):
        args.output.mkdir(parents=True, exist_ok=True)
        row = {'time': datetime.now(timezone.utc).isoformat(), 'event': kind, **fields}
        with (args.output / 'execution-events.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    try:
        while True:
            before = records(args.output)
            protocol_path = args.output / 'protocol.json'
            if protocol_path.exists():
                protocol = json.loads(protocol_path.read_text())
                expected = {(t, s, n) for t in protocol['tasks'] for s in protocol['settings']
                            for n in protocol['seeds']}
                actual = {(r['task'], r['setting'], r['seed']) for r in before}
                assert actual <= expected, 'Unexpected episode records'
                if actual == expected:
                    summary = json.loads((args.output / 'summary.json').read_text())
                    assert summary['attempted_episodes'] == summary['planned_episodes'] == len(expected)
                    assert summary['policy_errors'] == summary['infrastructure_errors'] == 0
                    event('isolated_suite_completed', episodes=len(before))
                    return 1 if summary['setup_errors'] else 0
            command = [sys.executable, str(ROOT / 'scripts/eval_dev_suite.py'),
                       '--output', str(args.output), '--one-episode', *worker_args]
            if resume:
                command.append('--resume-completed')
            code = run_worker(command, args.output / 'worker-progress.json',
                              args.episode_timeout, args.stall_timeout)
            if code != 0:
                raise RuntimeError(f'Isolated simulator worker exited {code}; preserve its records')
            after = records(args.output)
            assert len(after) == len(before) + 1, 'Worker must record exactly one remaining episode'
            event('isolated_episode_completed', count=len(after), episode=after[-1],
                  episode_timeout=args.episode_timeout, stall_timeout=args.stall_timeout)
            resume = True
    except BaseException as error:
        progress_path = args.output / 'worker-progress.json'
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else None
        event('infrastructure_interruption', error=repr(error), last_progress=progress,
              note='Stopped without selecting another seed or changing completed outcomes')
        if isinstance(error, SystemExit):
            raise
        # Exit 1 is reserved by the queue for a fully completed suite with
        # initialization failures. Execution errors must never use that code.
        raise SystemExit(2) from error


if __name__ == '__main__':
    raise SystemExit(main())
