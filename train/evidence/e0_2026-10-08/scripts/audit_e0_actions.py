#!/usr/bin/env python3
"""Fresh-process E0 export audit under the local FP32 SDPA/reference-MoE backend.

Reconstructs a reference independently from original base + training.pt, then
loads the FP32 export in a different Python process. Saves real held-out inputs
and explicit flow noise, compares every physical 50x14 action and normalized
50x55 output with atol=rtol=1e-6. Does not train or copy full model weights.

IMPORTANT: Both sides use lrvla.training_runtime construction. This is NOT a
test of the platform's native/compatibility constructor, FlashAttention, or
compiled inference. It never selects or writes runtime-construction.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile

ATOL = RTOL = 1e-6
ROOT = Path('/data/lingbot-vla-robotwin')
RUN = ROOT / 'runs/e0_jiali_20261008_004327'
REVIEWED_SOURCE = {
    "src/lrvla/training_checkpoint.py": "5e06d1dbcd9f08cf82f710ee06bbfbfa4e7f81873d1c11bd67ad59c1b5caa4ea",
    "src/lrvla/training_runtime.py": "91e3f5d900dc3069927684d2167cc5b745c21c1a00925fedf9ec9b5adc869889",
    "src/lrvla/training_models.py": "98f805f25a69a043a59e8c82c6a1ba4894281ab5695a766079ac411f7ec24043",
    "src/lrvla/training_data.py": "86e6f6587d598712154898f04d2d127414bf16d7107d76da5f339f088ce2d925",
    "src/lrvla/training_identity.py": "53f8f372f0f97a21b2197f19774c4f55ed286e2666f99b46a7863470e476df57",
    "vendor/lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py": "9b3c9a2ad6823f9d58da51b732477785a1e4db0d29db80e030e094c7b605e40c",
    "vendor/lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/configuration_lingbot_vla.py": "37987af05ca8980debf5bfe81211cc2a9db1b2fb0e14e15076fa6497c5baf023",
    "vendor/lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py": "2221459c676c27eb70e13edd29d6e25c2c6008a1fc3f0b7c820b8abbbf11aff6",
    "vendor/lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py": "eb2202bd01916c891b366c90f9bf11a2782f9387031cc24ed08851b3d8f01e96",
    "vendor/lingbot-vla-v2/lingbotvla/data/vla_data/utils.py": "cfbd6e08dda8f7a3a462c50f6c230c372189f8148b3ffe6a962e240dccd86a16",
    "vendor/lingbot-vla-v2/lingbotvla/data/vla_data/base_dataset.py": "3d1550e644778ab4ec15c22fae4a9ed799ff5b0f27d7d81b83b218d2c777da65"
}


def log(message):
    print(f'[{datetime.now():%H:%M:%S}] {message}', flush=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def resolve(parent, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def configuration(args):
    import yaml
    cfg = yaml.safe_load((args.run / 'training_config.yaml').read_text())
    cli = yaml.safe_load((args.run / 'lingbotvla_cli.yaml').read_text())
    meta = json.loads((args.run / 'run_metadata.json').read_text())
    data = dict(cli['data'])
    for key in ('norm_stats_file', 'robot_config_root', 'train_path'):
        data[key] = str(resolve(args.run, data[key]))
    return cfg, cli, meta, data


def compare_arrays(reference, reloaded):
    """Fail closed for a wrong shape or nonfinite output; reference is tolerance anchor."""
    import numpy as np
    reference, reloaded = np.asarray(reference), np.asarray(reloaded)
    same_shape = reference.shape == reloaded.shape
    finite = bool(np.isfinite(reference).all() and np.isfinite(reloaded).all())
    result = {'reference_shape': list(reference.shape), 'reloaded_shape': list(reloaded.shape),
              'finite': finite, 'passed': False, 'max_abs_error': None,
              'max_tolerance_ratio': None, 'mismatched_elements': None}
    if same_shape and finite:
        difference = np.abs(reference.astype(np.float64) - reloaded.astype(np.float64))
        tolerance = ATOL + RTOL * np.abs(reference.astype(np.float64))
        result.update(passed=bool((difference <= tolerance).all()),
                      max_abs_error=float(difference.max()),
                      max_tolerance_ratio=float((difference / tolerance).max()),
                      mismatched_elements=int((difference > tolerance).sum()))
    return result


def buffer_fingerprints(policy):
    import torch
    result = {}
    for prefix, module in policy.named_modules():
        for name in sorted(module._non_persistent_buffers_set):
            tensor = module._buffers.get(name)
            if tensor is None:
                continue
            value = tensor.detach().cpu().contiguous()
            raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
            result[f'{prefix}.{name}'] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                                         'sha256': hashlib.sha256(raw).hexdigest()}
    return result


def make_fixture(args, cfg, data, policy, processor):
    import torch
    from lrvla.training_data import LocalCleanDataset, episode_split
    from lrvla.training_identity import manifest_content_sha256
    manifest = json.loads(Path(data['train_path']).read_text())
    meta = json.loads((args.run / 'run_metadata.json').read_text())
    require(manifest.get('clean_only') is True, 'Expected the clean-only training manifest.')
    require(manifest_content_sha256(manifest) == meta['manifest_content_sha256'],
            'Saved clean manifest content differs from the training run.')
    entries = {entry['task']: entry for entry in manifest['datasets']}
    tasks = [name for name in ('click_bell', 'open_microwave') if name in entries]
    tasks += [name for name in sorted(entries) if name not in tasks]
    require(len(tasks) >= 2, 'Need at least two tasks in the saved clean manifest.')
    generator = torch.Generator(device='cpu').manual_seed(42001)
    fixture = []
    keys = ('images', 'img_masks', 'lang_tokens', 'lang_masks', 'state', 'image_grid_thw',
            'state_joint_mask', 'action_joint_mask')
    for task in tasks[:2]:
        entry = dict(entries[task])
        entry['path'] = str(resolve(args.root, entry['path']))
        ids = entry.get('episode_ids', list(range(int(entry['episodes']))))
        _, heldout = episode_split(ids, int(cfg.get('val_episodes', 5)))
        require(bool(heldout), f'No validation episodes for {task}.')
        episode = heldout[0]
        log(f'Fixture: {task}, held-out episode {episode}, first frame.')
        dataset = LocalCleanDataset(entry, [episode], data, policy.config, processor,
                                    Path(data['robot_config_root']) / 'robotwin.yaml', training=False)
        raw = dataset.dataset[0]
        require(raw['observation.state'].shape[-1] == 14, 'Expected 14D robot state.')
        observation = dataset.transform.apply(raw, policy_eval=True)
        require(all(key in observation for key in keys), 'Incomplete preprocessed observation.')
        fixture.append({'task': task, 'episode': int(episode), 'frame': 0,
                        'observation': {key: observation[key].detach().cpu().clone() for key in keys},
                        'noise': torch.randn((1, 50, 55), generator=generator, dtype=torch.float32)})
        del dataset, raw, observation
        gc.collect()
    torch.save(fixture, args.work / 'fixture.pt')


def worker(args):
    import numpy as np
    import torch
    import transformers
    from types import SimpleNamespace
    from lrvla.training_runtime import build_policy
    from lrvla.training_models import configure_trainable, sdpa_attention
    cfg, cli, meta, data = configuration(args)
    torch.cuda.set_device(args.device)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    qwen = resolve(args.run, cli['model']['tokenizer_path'])
    export = args.run / 'checkpoints' / f'global_step_{args.step}' / 'hf_ckpt_fp32'
    checkpoint = export.parent / 'training.pt'
    is_reference = args.worker == 'reference'
    weights = resolve(args.root, cfg['base_checkpoint']) if is_reference else export
    log(f'{args.worker.upper()} step {args.step}: constructing model in process {os.getpid()}.')
    policy = build_policy(args.root, cli, weights, qwen, device=args.device,
                          weight_dtype=torch.bfloat16 if is_reference else torch.float32)
    if is_reference:
        selection = configure_trainable(policy, cfg)
        require(selection == meta['selection'], 'Trainable selection mismatch.')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        require(payload['state']['global_step'] == args.step, 'Training checkpoint step mismatch.')
        for key in ('base_identity', 'selection', 'norm_sha256', 'qwen_config_sha256',
                    'model_geometry', 'manifest_content_sha256', 'loss_normalization',
                    'data_sampling', 'distributed'):
            require(payload['metadata'].get(key) == meta.get(key), f'Checkpoint metadata mismatch: {key}')
        trainable = payload['trainable_state']
        require(set(trainable) == set(selection['trainable_names']), 'Trained tensor inventory mismatch.')
        parameters = dict(policy.named_parameters())
        with torch.no_grad():
            for name, value in trainable.items():
                require(value.dtype == torch.float32 and value.shape == parameters[name].shape,
                        f'Trained tensor dtype/shape mismatch: {name}')
                require(bool(torch.isfinite(value).all()), f'Nonfinite trained tensor: {name}')
                parameters[name].copy_(value)
        del payload, trainable, parameters
        gc.collect()
    policy.float().eval().requires_grad_(False)
    policy.config.use_cache = True
    policy.model.config.use_cache = True
    policy.model.qwenvl_with_expert.attention_interface = partial(sdpa_attention, compute_dtype=torch.float32)
    policy.model._use_compile_predict_velocity = False
    policy.model._compiled_predict_velocity = None
    require(all(p.dtype == torch.float32 and p.device.type == 'cuda' for p in policy.parameters()),
            'Audit requires all parameters in FP32 on CUDA.')
    require(policy.config.chunk_size == policy.config.n_action_steps == 50
            and policy.config.max_action_dim == policy.config.max_state_dim == 55
            and policy.config.num_steps == 10 and data['img_size'] == 256,
            'Unexpected model/input geometry.')
    processor = transformers.AutoProcessor.from_pretrained(qwen, padding_side='right', local_files_only=True)
    if args.make_fixture:
        require(is_reference and not (args.work / 'fixture.pt').exists(), 'Fixture already exists.')
        make_fixture(args, cfg, data, policy, processor)
    from lingbotvla.data.vla_data.utils import FeatureTransform
    transform = FeatureTransform(str(Path(data['robot_config_root']) / 'robotwin.yaml'),
                                 SimpleNamespace(**data), policy.config, processor,
                                 chunk_size=50, norm_stats_path=data['norm_stats_file'],
                                 image_augment=False, use_depth_align=False, use_future_image=False)
    require(transform.org_features['actions'] == ['action'], 'Expected a single 14D action output.')
    fixture = torch.load(args.work / 'fixture.pt', map_location='cpu', weights_only=True)
    require(len(fixture) == 2, 'Fixture must contain exactly two observations.')
    buffers = buffer_fingerprints(policy)
    outputs = {}
    with torch.inference_mode():
        for index, case in enumerate(fixture):
            log(f'{args.worker.upper()} step {args.step}: full action chunk for {case["task"]}.')
            observation = case['observation']
            inputs = {}
            for key in ('images', 'img_masks', 'lang_tokens', 'lang_masks', 'state', 'image_grid_thw'):
                value = observation[key].unsqueeze(0).to(args.device)
                inputs[key] = value.float() if value.is_floating_point() else value
            # sample_actions updates noise in-place; always clone the saved fixture.
            normalized = policy.model.sample_actions(**inputs, noise=case['noise'].to(args.device).clone())
            torch.cuda.synchronize()
            normalized = normalized.squeeze(0).float().cpu()
            require(tuple(normalized.shape) == (50, 55), 'Expected a full normalized 50x55 action chunk.')
            inverse_input = {key: value.clone() for key, value in observation.items()}
            inverse_input['actions'] = normalized.clone()
            physical = torch.as_tensor(transform.unapply(inverse_input)['action']).float().cpu()
            require(tuple(physical.shape) == (50, 14), 'Expected a full physical 50x14 action chunk.')
            require(bool(torch.isfinite(normalized).all() and torch.isfinite(physical).all()),
                    'Nonfinite action output; audit stopped.')
            outputs[f'case_{index}_normalized'] = normalized.numpy()
            outputs[f'case_{index}_physical'] = physical.numpy()
    prefix = args.work / f'{args.step}-{args.worker}'
    np.savez(str(prefix) + '.npz', **outputs)
    write_json(str(prefix) + '.json', {
        'pid': os.getpid(), 'fixture_sha256': sha256(args.work / 'fixture.pt'),
        'torch': str(torch.__version__), 'transformers': transformers.__version__,
        'cuda': torch.version.cuda, 'device': args.device, 'gpu': torch.cuda.get_device_name(),
        'nonpersistent_buffers_before_inference': buffers,
        'cases': [{key: case[key] for key in ('task', 'episode', 'frame')} for case in fixture],
    })


def run_worker(args, step, kind, make_fixture=False):
    command = [sys.executable, str(Path(__file__).resolve()), '--root', str(args.root),
               '--run', str(args.run), '--work', str(args.work), '--device', args.device,
               '--step', str(step), '--worker', kind]
    if make_fixture:
        command.append('--make-fixture')
    env = dict(os.environ, PYTHONHASHSEED='42', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false')
    subprocess.run(command, check=True, env=env, cwd=args.root)


def main(args):
    import fcntl
    import numpy as np
    import torch
    from lrvla.training_identity import base_checkpoint_identity
    lock = (args.run / 'e0_action_audit.lock').open('a')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another action audit is running for this run.')
    require(args.steps and all(step in (5000, 10000, 15000) for step in args.steps),
            'This helper is for milestones 5000, 10000, 15000.')
    for relative, digest in REVIEWED_SOURCE.items():
        require(sha256(args.root / relative) == digest, f'Reviewed source changed: {relative}')
    cfg, cli, meta, data = configuration(args)
    require(cfg.get('mode') == 'expert', 'Expected expert-mode E0 training.')
    require(sha256(data['norm_stats_file']) == meta['norm_sha256'], 'Training norm hash mismatch.')
    require(sha256(resolve(args.run, cli['model']['tokenizer_path']) / 'config.json')
            == meta['qwen_config_sha256'], 'Qwen configuration hash mismatch.')
    require(torch.cuda.is_available(), 'CUDA unavailable. Activate the lingbot-vla environment.')
    if args.device == 'auto':
        free, index = max((torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count()))
        args.device = f'cuda:{index}'
    else:
        free = torch.cuda.mem_get_info(torch.device(args.device))[0]
    require(free >= 30 * 2**30, 'Need a GPU with at least 30 GiB free for this FP32 audit.')
    log(f'Audit device: {args.device}. No training or model export will run.')
    log('Checking original base hashes; this reads the base model once.')
    require(base_checkpoint_identity(resolve(args.root, cfg['base_checkpoint'])) == meta['base_identity'],
            'Original base weights differ from this training run.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    args.work = Path(tempfile.mkdtemp(prefix=f'action-audit-{stamp}-', dir=args.run))
    log(f'Inputs, outputs and reports: {args.work}')
    scope = {
        'profile': 'local_fp32_sdpa_reference_moe_export_reload', 'parameter_dtype': 'float32',
        'vision_attention': 'sdpa', 'coupled_attention': 'sdpa_float32',
        'moe': 'torch_reference_fused_storage', 'compile': False, 'matmul_precision': 'high',
        'constructor': 'lrvla.training_runtime.build_policy_then_float_on_both_sides',
        'noise': 'saved_explicit_fp32_tensor_cloned_per_forward',
        'chunk_size': 50, 'physical_action_dim': 14, 'flow_steps': 10,
        'norm_sha256': meta['norm_sha256'], 'atol': ATOL, 'rtol': RTOL,
    }
    summaries = []
    for number, step in enumerate(dict.fromkeys(args.steps)):
        export = args.run / 'checkpoints' / f'global_step_{step}' / 'hf_ckpt_fp32'
        receipt = json.loads((export / 'export-receipt.json').read_text())
        checkpoint_hash = sha256(export.parent / 'training.pt')
        require(receipt['step'] == step and receipt['dtype'] == 'torch.float32'
                and receipt['trainable_values_exact'] is True
                and receipt['checkpoint_sha256'] == checkpoint_hash
                and receipt['base_identity_sha256'] == meta['base_identity']['sha256'],
                f'Step {step}: export receipt differs from the training checkpoint.')
        log(f'Step {step}: fingerprinting FP32 shards; weights stay on this server.')
        index = json.loads((export / 'model.safetensors.index.json').read_text())
        names = sorted(set(index['weight_map'].values()))
        require(set(names) == {path.name for path in export.glob('*.safetensors')}, 'Shard/index file mismatch.')
        require(all(Path(name).name == name for name in names), 'Shard names must be local basenames.')
        files = names + ['model.safetensors.index.json', 'config.json']
        hashes = {name: sha256(export / name) for name in files}
        stats = {name: ((export / name).stat().st_size, (export / name).stat().st_mtime_ns) for name in files}
        report = {'status': 'running', 'scope': scope, 'step': step, 'created_at': stamp,
                  'script_sha256': sha256(__file__), 'reviewed_source_sha256': REVIEWED_SOURCE,
                  'cli_sha256': sha256(args.run / 'lingbotvla_cli.yaml'),
                  'training_checkpoint_sha256': checkpoint_hash,
                  'base_identity_sha256': meta['base_identity']['sha256'], 'export_file_sha256': hashes,
                  'official_runtime': {'status': 'not_tested', 'native_loader_verified': False,
                                       'bf16_then_fp32_loader_verified': False,
                                       'runtime_construction_selected': None},
                  'limitation': 'Checks saved-checkpoint reconstruction against export reload on the local '
                                'reference backend only. Not a closed-loop score or official loader audit.'}
        report_path = args.work / f'action-audit-{step}.json'
        if (export / 'action-audit.json').exists():
            shutil.copy2(export / 'action-audit.json', args.work / f'previous-action-audit-{step}.json')
        write_json(report_path, report)
        try:
            run_worker(args, step, 'reference', make_fixture=(number == 0))
            run_worker(args, step, 'reload')
            reference_meta = json.loads((args.work / f'{step}-reference.json').read_text())
            reload_meta = json.loads((args.work / f'{step}-reload.json').read_text())
            require(reference_meta['pid'] != reload_meta['pid'], 'Reference and reload must be different processes.')
            require(reference_meta['fixture_sha256'] == reload_meta['fixture_sha256'], 'Fixture hash mismatch.')
            for key in ('torch', 'transformers', 'cuda', 'device', 'gpu', 'cases'):
                require(reference_meta[key] == reload_meta[key], f'Worker settings differ: {key}')
            require(reference_meta['nonpersistent_buffers_before_inference']
                    == reload_meta['nonpersistent_buffers_before_inference'], 'Local construction buffers differ.')
            results = {}
            with np.load(args.work / f'{step}-reference.npz', allow_pickle=False) as before, \
                 np.load(args.work / f'{step}-reload.npz', allow_pickle=False) as after:
                expected = {f'case_{i}_{kind}' for i in range(2) for kind in ('normalized', 'physical')}
                require(set(before.files) == set(after.files) == expected, 'Output inventory mismatch.')
                for key in sorted(expected):
                    shape = (50, 14) if key.endswith('physical') else (50, 55)
                    require(before[key].shape == after[key].shape == shape, f'Incomplete action chunk: {key}')
                    results[key] = compare_arrays(before[key], after[key])
            require(all(stats[name] == ((export / name).stat().st_size, (export / name).stat().st_mtime_ns)
                        for name in files), 'Export files changed during the audit.')
            report.update(status='passed' if all(row['passed'] for row in results.values()) else 'failed',
                          reference=reference_meta, reload=reload_meta, comparisons=results,
                          fixture_sha256=sha256(args.work / 'fixture.pt'))
            write_json(report_path, report)
            # A prior audit stays available in its timestamped work directory.
            write_json(export / 'action-audit.json', report)
            require(report['status'] == 'passed', f'Step {step}: action differences exceed tolerance; see {report_path}')
            maximum = max(row['max_abs_error'] for row in results.values())
            log(f'STEP {step} REFERENCE AUDIT PASSED. Maximum absolute action error: {maximum:.9g}')
            summaries.append({'step': step, 'max_abs_error': maximum, 'report': str(report_path)})
        except BaseException as error:
            if report['status'] == 'running':
                report.update(status='error', error=f'{type(error).__name__}: {error}')
            write_json(report_path, report)
            write_json(export / 'action-audit.json', report)
            raise
    write_json(args.work / 'summary.json', {'status': 'passed', 'scope': scope, 'results': summaries,
                                           'official_runtime_status': 'not_tested'})
    log(f'ALL {len(summaries)} REFERENCE ACTION AUDITS PASSED.')
    log('OFFICIAL RUNTIME NOT TESTED. No runtime-construction.json was written or selected.')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--steps', type=int, nargs='+', default=[5000, 10000, 15000])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--worker', choices=['reference', 'reload'], help=argparse.SUPPRESS)
    parser.add_argument('--step', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--work', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--make-fixture', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.root, args.run = args.root.resolve(), args.run.resolve()
    sys.path.insert(0, str(args.root / 'src'))
    return args


if __name__ == '__main__':
    try:
        arguments = parse_args()
        worker(arguments) if arguments.worker else main(arguments)
    except Exception as error:
        log(f'AUDIT STOPPED: {type(error).__name__}: {error}')
        raise
