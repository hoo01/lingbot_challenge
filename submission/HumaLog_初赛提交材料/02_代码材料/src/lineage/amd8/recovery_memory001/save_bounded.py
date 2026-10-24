"""Bounded rank-at-a-time local states; broker acknowledgements never gate updates."""
import ctypes
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import time

import torch
import torch.distributed as dist
import checkpoint as ckpt
from async_checkpoint import freeze_training_state, tensor_bytes
from policy import ROOT, WORLD, atomic_json, sha

STORAGE = Path('/var/tmp/lingbotvla-sft-amd8-20261017-checkpoints')
SLOT_BYTES = 10 * 2**30
HEADER_BYTES = 64 * 2**20
RUN_NAME = 'formal_sft_10000'
CONTROL = ROOT / 'artifacts/checkpoint-memory001'
MODE = 'serial_rank_local_full_fp32_adam_async_upload_v1'
META = ('norm_stats.json', 'norm_stats.provenance.json', 'robotwin_relative.yaml',
        'sampling-binding.json', 'source-binding.json', 'asset-binding.json')


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def trim_process_heap():
    gc.collect()
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError, OSError):
        pass


def available_memory():
    cg = Path('/sys/fs/cgroup')
    limit = (cg/'memory.max').read_text().strip()
    return None if limit == 'max' else int(limit)-int((cg/'memory.current').read_text())


def prepare_control():
    CONTROL.mkdir(parents=True, exist_ok=True)
    plan_path = CONTROL/'slots.json'
    if not plan_path.exists():
        source = ROOT/'runs'/RUN_NAME/'checkpoint_sft_002000'
        complete, ack = read(source/'complete.json'), read(source/'archive-complete.json')
        assert complete['step'] == 2000 and complete['world'] == WORLD == 8
        assert complete['local_retained'] and ack['passed'] and ack['readbacks'] == 16
        atomic_json(plan_path, {'version': 1, 'steps': {'2000': 0}})


def owned_directory(step):
    path = STORAGE/RUN_NAME/f'checkpoint_sft_{step:06d}'
    assert path.resolve() == path and path.parent == STORAGE/RUN_NAME
    return path


def select_slot(step):
    """Never reclaim the latest complete state or an unacknowledged old state."""
    prepare_control()
    plan = read(CONTROL/'slots.json')
    if str(step) in plan['steps']:
        return None, 'existing step reservation retained for diagnosis'
    busy = set()
    for key, slot in plan['steps'].items():
        directory = owned_directory(int(key))
        folder = ROOT/'runs'/RUN_NAME/directory.name
        has_state = directory.exists() and any(directory.iterdir())
        if not (folder/'local-transfer-pruned.json').exists() and ((folder/'complete.json').exists() or has_state):
            busy.add(slot)
    free = sorted({0, 1}-busy)
    if not free:
        return None, 'both private slots retained; uploads pending or previous state protected'
    slot = free[0]
    plan['steps'][str(step)] = slot
    atomic_json(CONTROL/'slots.json', plan)
    return slot, None


def allocate_slot(slot, rank, release=False):
    assert slot in (0, 1) and 0 <= rank < WORLD == 8
    path = STORAGE/'two-checkpoint-capacity.reserve'
    assert path.is_file() and not path.is_symlink()
    assert path.resolve() == path and path.stat().st_size == 2*WORLD*SLOT_BYTES
    offset = (slot*WORLD+rank)*SLOT_BYTES
    assert offset >= 0 and offset+SLOT_BYTES <= path.stat().st_size
    with path.open('r+b') as stream:
        if release:
            libc = ctypes.CDLL(None, use_errno=True)
            libc.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong]
            if libc.fallocate(stream.fileno(), 3, offset, SLOT_BYTES):
                number = ctypes.get_errno()
                raise OSError(number, os.strerror(number))
        else:
            os.posix_fallocate(stream.fileno(), offset, SLOT_BYTES)
        os.fsync(stream.fileno())


def enqueue(operation, relative, expected=None, **extra):
    request_id = operation+'__'+relative.replace('/', '_').replace('.', '_')
    value = dict(id=request_id, operation=operation, relative=relative, expected=expected, **extra)
    target = ROOT/'artifacts/archive-requests'/(request_id+'.json')
    if target.exists():
        assert read(target) == value
    else:
        atomic_json(target, value)
    return request_id


def enqueue_uploads(folder, complete):
    for name, proof in complete['receipts'].items():
        enqueue('upload', str(folder.relative_to(ROOT))+'/'+name+'.pt', proof)


def advise_checkpoint_cache():
    """Only advise clean pages of already verified, archived, owned files."""
    if not hasattr(os, 'posix_fadvise'):
        return
    for folder in (ROOT/'runs'/RUN_NAME).glob('checkpoint_sft_*'):
        complete, ack = read(folder/'complete.json'), read(folder/'archive-complete.json')
        if not complete or not ack or not ack.get('passed'):
            continue
        directory = owned_directory(complete['step'])
        for name, proof in complete['receipts'].items():
            path = directory/(name+'.pt')
            if not path.exists():
                continue
            assert not path.is_symlink() and path.resolve() == path
            assert proof['readback_equal'] and path.stat().st_size == proof['bytes']
            with path.open('rb') as stream:
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def save_rank(payload, data, folder, rank, slot):
    directory = owned_directory(payload['step'])
    directory.mkdir(parents=True, exist_ok=True)
    spool = ROOT/'spool'/folder.relative_to(ROOT)
    assert spool.resolve() == directory
    assert tensor_bytes(payload)+tensor_bytes(data)+HEADER_BYTES < SLOT_BYTES
    names = (f'data_rank{rank}', f'rank{rank}')
    for name in names:
        path = directory/(name+'.pt')
        assert not path.exists() and not path.with_name(path.name+'.partial').exists()
    allocate_slot(slot, rank, release=True)
    for name, value in zip(names, (data, payload)):
        path = directory/(name+'.pt')
        ckpt.write_tensor_file(path, value)
        loaded = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        assert ckpt.equal(value, loaded), 'Actual full local state readback differs'
        del loaded
        proof = dict(step=payload['step'], bytes=path.stat().st_size,
                     sha256=sha(path), readback_equal=True, local_retained=True,
                     format='torch_save_full_fp32_adam_v1')
        atomic_json(folder/(name+'.json'), proof)
        if hasattr(os, 'posix_fadvise'):
            with path.open('rb') as stream:
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    atomic_json(folder/f'local-ready-rank{rank}.json', dict(passed=True, rank=rank,
        step=payload['step'], both_actual_readbacks_equal=True, slot=slot, time=time.time()))


def publish_local(folder, run, identity, step, slot):
    receipts = {f'{prefix}{rank}': read(folder/f'{prefix}{rank}.json')
                for prefix in ('rank', 'data_rank') for rank in range(WORLD)}
    assert all(p and p['readback_equal'] and p['step'] == step for p in receipts.values())
    assert all(read(folder/f'local-ready-rank{r}.json')['passed'] for r in range(WORLD))
    for name in META:
        shutil.copy2(ROOT/'artifacts'/name, folder/name)
    for name in ('config.json', 'model_config.json', 'lingbotvla_cli.yaml'):
        shutil.copy2(run/name, folder/name)
    atomic_json(folder/'checkpoint-execution.json', dict(mode=MODE,
        inherited_numerical_identity=identity, runtime_binding_sha256=sha(CONTROL/'runtime-binding.json'),
        bounded_one_rank_cpu_snapshot=True, uploads_independent_of_training=True, slot=slot))
    complete = dict(step=step, world=WORLD, identity=identity, receipts=receipts, retained=True,
        format='dense_full_fp32_adam_official_prefetch_local_raw_v2', snapshot_mode=MODE,
        local_retained=True, local_storage=str(owned_directory(step)), local_slot=slot)
    atomic_json(folder/'complete.json', complete)
    atomic_json(ROOT/'artifacts/latest-local-checkpoint.json', dict(passed=True, step=step,
        checkpoint=str(folder.relative_to(ROOT)), complete_sha256=sha(folder/'complete.json'),
        local_actual_readbacks=16, archive_pending=True, time=time.time()))
    enqueue_uploads(folder, complete)


def prune_old(folder):
    """Same safety contract: new complete plus actual new archive ack required."""
    latest = read(ROOT/'artifacts/latest-local-checkpoint.json')
    complete, ack = read(folder/'complete.json'), read(folder/'archive-complete.json')
    assert latest['step'] == complete['step'] and latest['complete_sha256'] == sha(folder/'complete.json')
    assert ack['passed'] and ack['world'] == WORLD and ack['readbacks'] == 16
    plan = read(CONTROL/'slots.json')
    for previous in folder.parent.glob('checkpoint_sft_*'):
        old = read(previous/'complete.json')
        if not old or old['step'] >= complete['step'] or (previous/'local-transfer-pruned.json').exists():
            continue
        old_ack = read(previous/'archive-complete.json')
        assert old_ack and old_ack['passed'] and len(old['receipts']) == 16
        directory = owned_directory(old['step'])
        expected = {name+'.pt' for name in old['receipts']}
        assert {p.name for p in directory.iterdir()} <= expected
        rows = []
        for name, proof in old['receipts'].items():
            path = directory/(name+'.pt')
            assert path.is_file() and not path.is_symlink() and path.stat().st_size == proof['bytes']
            rows.append(dict(name=path.name, bytes=proof['bytes'], archived_sha256=proof['sha256']))
        for row in rows:
            (directory/row['name']).unlink()
        for rank in range(WORLD):
            allocate_slot(plan['steps'][str(old['step'])], rank)
        atomic_json(previous/'local-transfer-pruned.json', dict(passed=True, step=old['step'],
            replaced_by=complete['step'], files=rows, all_metadata_retained=True,
            latest_complete_preserved=True, actual_archive_ack_sha256=sha(previous/'archive-complete.json'),
            time=time.time()))


def service_archives():
    """CPU owner calls this; no waiting, GPU state or tensors are involved."""
    status = []
    for folder in sorted((ROOT/'runs'/RUN_NAME).glob('checkpoint_sft_*')):
        complete = read(folder/'complete.json')
        if not complete or (folder/'archive-complete.json').exists():
            continue
        enqueue_uploads(folder, complete)
        acknowledgements = []
        for name, proof in complete['receipts'].items():
            key = enqueue('upload', str(folder.relative_to(ROOT))+'/'+name+'.pt', proof)
            ack = read(ROOT/'artifacts/archive-acks'/(key+'.json'))
            if ack and (ack.get('id') != key or not ack.get('passed')):
                atomic_json(folder/'upload-error.json', dict(step=complete['step'], ack=ack,
                    training_may_continue=True, checkpoint_retained=True))
            acknowledgements.append(ack and ack.get('id') == key and ack.get('passed'))
        if not all(acknowledgements):
            status.append(dict(step=complete['step'], uploaded_files=sum(bool(x) for x in acknowledgements),
                               expected_files=16, nonblocking=True))
            continue
        for rank in range(WORLD):
            marker = folder/f'async-archived-rank{rank}.json'
            if not marker.exists():
                progress = read(folder.parent/f'progress_rank{rank}.json') or {}
                atomic_json(marker, dict(passed=True, step=complete['step'], rank=rank,
                    updates_advanced_during_archive=progress.get('step', 0) > complete['step'],
                    local_full_state_retained=True, readback_equal=True, time=time.time()))
        keep = [*META, 'config.json', 'model_config.json', 'lingbotvla_cli.yaml', 'checkpoint-execution.json']
        keep += [f'{prefix}{rank}.json' for prefix in ('rank', 'data_rank', 'local-ready-rank', 'async-snapshot-rank', 'async-archived-rank') for rank in range(WORLD)]
        key = enqueue('finalize', str(folder.relative_to(ROOT)), small_files=sorted(keep))
        ack = read(ROOT/'artifacts/archive-acks'/(key+'.json'))
        if not ack:
            status.append(dict(step=complete['step'], uploaded_files=16, waiting_finalize=True, nonblocking=True))
            continue
        if not (ack.get('passed') and ack.get('id') == key and ack.get('world') == 8 and ack.get('readbacks') == 16):
            atomic_json(folder/'upload-error.json', dict(step=complete['step'], ack=ack,
                training_may_continue=True, checkpoint_retained=True))
            continue
        atomic_json(folder/'archive-complete.json', ack)
        latest = read(ROOT/'artifacts/latest-local-checkpoint.json')
        if latest['step'] == complete['step']:
            latest.update(archive_pending=False, actual_archive_ack_sha256=sha(folder/'archive-complete.json'))
            atomic_json(ROOT/'artifacts/latest-local-checkpoint.json', latest)
            prune_old(folder)
    atomic_json(CONTROL/'upload-status.json', dict(time=time.time(), pending=status,
        uploads_independent_of_training=True, saved_checkpoints_never_reported_as_uploaded_early=True))
    return status


class BoundedLocalArchive:
    def __init__(self, run, identity, args):
        self.run, self.identity, self.args = Path(run), identity, args
        self.rank = dist.get_rank()
        assert self.run.name == RUN_NAME and dist.get_world_size() == WORLD == 8

    def health(self):
        if (CONTROL/'internal-stop.json').exists():
            raise RuntimeError('Actual recovery/input fault; retain source and diagnostics')

    def finish(self):
        # All local tensors are already persisted and freed. CPU owner/broker handles upload.
        return None

    def capture(self, model, optimizer, step, loader, generator):
        device = 'cuda' if dist.get_backend() == 'nccl' else 'cpu'
        started = time.time()
        while True:
            choice = [-1]
            reason = None
            if self.rank == 0:
                try:
                    if (ROOT/'user-stop.json').exists():
                        choice[0] = -2
                        reason = 'explicit human stop'
                    else:
                        advise_checkpoint_cache()
                        slot, reason = select_slot(step)
                        choice[0] = -1 if slot is None else slot
                except Exception as error:
                    reason = repr(error)
                if choice[0] < 0:
                    atomic_json(CONTROL/f'skipped-{step:06d}.json', dict(step=step,
                        saved=False, uploaded=False, reason=reason, training_may_continue=step < 10000,
                        final_state_retained_live=step == 10000, time=time.time()))
            flag = torch.tensor(choice, device=device)
            dist.broadcast(flag, src=0)
            slot = int(flag.item())
            if slot == -2:
                return dict(step=step, saved=False, user_stopped=True)
            if slot >= 0:
                break
            if step < 10000:
                return dict(step=step, saved=False, skipped=True, archive_pending=False)
            if (ROOT/'user-stop.json').exists():
                return dict(step=step, saved=False, user_stopped=True)
            # No more updates remain. Retain the final live state until safe local storage is available.
            time.sleep(15)
        folder = self.run/f'checkpoint_sft_{step:06d}'
        folder.mkdir(exist_ok=True)
        failed = False
        for active in range(WORLD):
            dist.barrier()
            outcome = [0]
            if self.rank == active:
                payload = data = None
                try:
                    if (ROOT/'user-stop.json').exists():
                        raise RuntimeError('Explicit human stop during local save')
                    trim_process_heap()
                    required = sum(ckpt.local(p).numel()*p.element_size() for p in model.parameters())*3+8*2**30
                    available = available_memory()
                    if available is not None and available < required:
                        raise MemoryError(f'Bounded one-rank save needs {required}, actual headroom {available}')
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    payload, data = freeze_training_state(model, optimizer, step, self.identity,
                        self.args, loader, generator, WORLD)
                    snapshot_bytes = tensor_bytes(payload)+tensor_bytes(data)
                    save_rank(payload, data, folder, self.rank, slot)
                    atomic_json(folder/f'async-snapshot-rank{self.rank}.json', dict(passed=True,
                        step=step, rank=self.rank, cpu_tensor_bytes=snapshot_bytes,
                        captured_before_training_continues=True, all_tensor_storage_independent=True,
                        snapshot_mode=MODE, time=time.time()))
                except Exception as error:
                    outcome[0] = 1
                    atomic_json(CONTROL/f'save-error-{step:06d}-rank{self.rank}.json', dict(
                        step=step, error=repr(error), partial_preserved=True, time=time.time()))
                finally:
                    del payload, data
                    trim_process_heap()
            flag = torch.tensor(outcome, device=device)
            dist.broadcast(flag, src=active)
            if int(flag.item()):
                failed = True
                break
        dist.barrier()
        result = dict(step=step, saved=False, skipped=failed)
        if self.rank == 0:
            try:
                if failed:
                    raise RuntimeError('Actual bounded local save failed; partial retained and updates may continue')
                publish_local(folder, self.run, self.identity, step, slot)
                result = dict(step=step, path=str(folder), saved=True, local_readbacks=16,
                              archive_pending=True, upload_blocks_training=False, seconds=time.time()-started)
            except Exception as error:
                atomic_json(CONTROL/f'skipped-{step:06d}.json', dict(step=step, saved=False,
                    uploaded=False, error=repr(error), partial_preserved=True,
                    training_may_continue=step < 10000, time=time.time()))
        dist.barrier()
        if (folder/'complete.json').exists():
            result = dict(step=step, path=str(folder), saved=True, local_readbacks=16,
                          archive_pending=True, upload_blocks_training=False,
                          seconds=time.time()-started)
        if step == 10000 and not (folder/'complete.json').exists():
            if self.rank == 0:
                atomic_json(CONTROL/'final-state-pending.json', dict(step=step,
                    saved=False, final_model_retained_in_live_workers=True,
                    partial_files_preserved=True, needs_action=True, time=time.time()))
            while not (ROOT/'user-stop.json').exists():
                time.sleep(15)
        return result
