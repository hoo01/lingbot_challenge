"""Run each remaining scene in a fresh simulator process with bounded waits.

The policy server, scene list, observations and scoring remain unchanged. All
completed records, including initialization failures, are retained. A stalled
worker stops the suite unless its request explicitly authorizes timeout failures.
Authorized timeouts consume their exact fixed scene and continue without retry.
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
sys.path.insert(0, str(ROOT / 'src'))
from eval_platform.planning import next_candidate, planned_count
from eval_platform.fork2000_plan import active, is_scored, retained_error
from eval_platform.expert_outcomes import acceptable_record
from eval_platform.timeout_outcomes import (PAIRED_TIMEOUT_MODE, UNFILTERED_TIMEOUT_MODE, validate_timeout_policy,
                                           build_timeout_record, write_timeout_result)


class EpisodeWallTimeout(TimeoutError):
    kind = 'episode_wall_clock'
    def __init__(self, seconds, pid, elapsed):
        super().__init__(f'Episode exceeded {seconds}s; PID {pid}')
        self.elapsed = elapsed


class EpisodeStallTimeout(TimeoutError):
    kind = 'action_chunk_stall'

    def __init__(self, seconds, pid, elapsed):
        super().__init__(f'No completed action chunk for {seconds}s; PID {pid}')
        self.elapsed = elapsed


def run_worker(command, progress_path, episode_timeout, stall_timeout,
               poll_seconds=2, terminate_grace=10):
    def signature():
        try:
            return progress_path.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    previous = signature()
    started = last_progress = time.monotonic()
    leases = tuple(int(x) for x in os.environ.get('LINGBOT_LEASE_FDS', '').split(',') if x)
    process = subprocess.Popen(command, pass_fds=leases)
    try:
        while process.poll() is None:
            now = time.monotonic()
            current = signature()
            if current != previous:
                previous, last_progress = current, now
            if now - started >= episode_timeout:
                raise EpisodeWallTimeout(episode_timeout, process.pid, now-started)
            if now - last_progress >= stall_timeout:
                raise EpisodeStallTimeout(stall_timeout, process.pid, now-started)
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
    config_path = output / 'protocol.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    assert all(acceptable_record(config,r) for r in result), \
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

    def retain_crash(before, reason):
        protocol=json.loads((args.output/'protocol.json').read_text())
        raise RuntimeError(reason+'; exact scene preserved; engineering errors never substitute seeds')
        scene=next_candidate(protocol,before)
        if scene is None:raise RuntimeError('No remaining scene for the crash')
        progress_path=args.output/'worker-progress.json'
        progress=json.loads(progress_path.read_text()) if progress_path.exists() else None
        if not progress or any(progress.get(k)!=v for k,v in scene.items()):
            raise RuntimeError('Crash has no matching actual scene; diagnose the runtime')
        folder=args.output/scene['task']/scene['setting']/str(scene['seed'])
        if not folder.exists():raise RuntimeError('No original scene artifacts for crash')
        row=dict(**scene,status='infrastructure_error',success=False,expert_attempts=0,
            error=reason,non_scored=True,disposition='preserved_continue_next_seed',
            policy_episode_index=protocol['seeds'].index(scene['seed']),
            execution_segment=os.environ.get('LINGBOT_EVAL_SEGMENT','original'),
            crash_progress=progress,seconds=0)
        terminal=folder/'episode.json'
        if terminal.exists():
            row=json.loads(terminal.read_text())
            if not retained_error(row):raise ValueError('Do not overwrite any existing terminal result')
        else:terminal.write_text(json.dumps(row,indent=2))
        with (args.output/'episodes.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
        from eval_dev_suite import summarize
        summary={**summarize([*before,row],protocol),'attempted_episodes':len(before)+1,
            'planned_episodes':planned_count(protocol)}
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
        event('simulator_error_retained_continue',episode=row,scored=False)

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    try:
        while True:
            before = records(args.output)
            protocol_path = args.output / 'protocol.json'
            if protocol_path.exists():
                protocol = json.loads(protocol_path.read_text())
                if next_candidate(protocol, before) is None:
                    summary = json.loads((args.output / 'summary.json').read_text())
                    planned = planned_count(protocol)
                    basis = 'scored' if active(protocol) else ('policy_completed' if protocol.get('valid_per_group') else 'attempted_episodes')
                    assert summary[basis] == summary['planned_episodes'] == planned, 'Native initialization guard or candidate exhaustion: incomplete, do not claim completion'
                    assert summary['attempted_episodes'] == len(before)
                    if not active(protocol):assert summary['policy_errors'] == summary['infrastructure_errors'] == 0
                    event('isolated_suite_completed', episodes=len(before))
                    return 0 if active(protocol) else (1 if summary['setup_errors'] else 0)
            command = [sys.executable, str(ROOT / 'scripts/eval_dev_suite.py'),
                       '--output', str(args.output), '--one-episode', *worker_args]
            if resume:
                command.append('--resume-completed')
            try:
                code = run_worker(command, args.output / 'worker-progress.json',
                                  args.episode_timeout, args.stall_timeout)
            except (EpisodeWallTimeout, EpisodeStallTimeout) as error:
                protocol = json.loads((args.output/'protocol.json').read_text())
                if not validate_timeout_policy(protocol):
                    raise
                if (isinstance(error, EpisodeStallTimeout) and
                        protocol['timeout_policy']['mode'] not in (PAIRED_TIMEOUT_MODE, UNFILTERED_TIMEOUT_MODE)):
                    raise
                if (args.episode_timeout,args.stall_timeout) != (900,240):
                    raise ValueError('The approved wall-time policy fixes both guards')
                after = records(args.output)
                if len(after) == len(before):
                    scene = next_candidate(protocol,before)
                    progress = json.loads((args.output/'worker-progress.json').read_text())
                    folder = args.output/scene['task']/scene['setting']/str(scene['seed'])
                    try:
                        row = build_timeout_record(protocol,scene,folder,error=repr(error),elapsed=error.elapsed,
                            progress=progress,segment=os.environ.get('LINGBOT_EVAL_SEGMENT','original'),kind=error.kind)
                    except ValueError as diagnosis:
                        if not active(protocol):raise
                        retain_crash(before,repr(error)+'; non-policy interruption: '+repr(diagnosis))
                        resume=True
                        continue
                    after = write_timeout_result(args.output,protocol,before,row)
                    event('episode_timeout_counted',episode=row,note='User-authorized zero score; no retry or replacement seed')
                elif len(after) == len(before)+1:
                    event('deadline_after_terminal_record',episode=after[-1],note='Preserve the completed result; never replace it with timeout')
                else:
                    raise ValueError('Unexpected episode records at wall-time boundary')
                resume = True
                continue
            if code != 0:
                protocol=json.loads((args.output/'protocol.json').read_text())
                after=records(args.output)
                if not active(protocol):raise RuntimeError(f'Isolated simulator worker exited {code}')
                if len(after)==len(before):retain_crash(before,f'Isolated simulator worker exited {code}; actual partial artifacts retained')
                elif len(after)!=len(before)+1:raise ValueError('Unexpected row count after failed worker')
                raise RuntimeError(f'Worker {code}: preserved exact key; no automatic next seed')
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
