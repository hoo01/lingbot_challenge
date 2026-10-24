"""RoboTwin clean-only HDF5 reader, following the official pi0 time alignment.

At image/state index t, the first action is joint_action[t + 1]. Images are
decoded directly rather than decoded and JPEG-encoded a second time.
"""
import bisect
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from .robotwin_codec import decode_image_bit

CAMERAS = {
    'head_camera': 'observation.images.cam_high',
    'left_camera': 'observation.images.cam_left_wrist',
    'right_camera': 'observation.images.cam_right_wrist',
}
ARM_INDICES = list(range(6)) + list(range(7, 13))
GRIPPER_INDICES = [6, 13]

def read_joints(h5):
    parts = [np.asarray(h5[f'joint_action/{key}'], dtype=np.float32).reshape(-1, dim)
             for key, dim in [('left_arm', 6), ('left_gripper', 1),
                              ('right_arm', 6), ('right_gripper', 1)]]
    q = np.concatenate(parts, axis=-1)
    if len(q) < 2 or q.shape[1] != 14 or not np.isfinite(q).all():
        raise ValueError('Invalid joint trajectory')
    return q

def decode_rgb(value):
    return decode_image_bit(value)

def episode_split(task, episode, seed=42, validation_count=5):
    order = sorted(range(50), key=lambda i: hashlib.sha256(f'{seed}:{task}:{i}'.encode()).digest())
    return 'validation' if episode in order[:validation_count] else 'train'

def action_chunk(q, frame, chunk_size=50):
    if not 0 <= frame < len(q) - 1:
        raise IndexError(frame)
    indices = frame + 1 + np.arange(chunk_size)
    return q[np.minimum(indices, len(q) - 1)].copy(), indices >= len(q)

def build_manifest(root, tasks, output, validation_count=5):
    root, output = Path(root), Path(output)
    source = json.loads((root / 'artifacts/clean-source-manifest.json').read_text())
    allowed = {x['path'].split('/')[1]: x for x in source['archives']}
    assert len(allowed) == 50 and source['setting'] == 'clean'
    records, states, actions = [], [], []
    for task in tasks:
        if task not in allowed:
            raise ValueError(f'Task not in official clean manifest: {task}')
        base = root / 'datasets/raw' / task
        provenance = json.loads((base / '.verified-source.json').read_text())
        if (provenance['source'] != f'dataset/{task}/aloha-agilex_clean_50.zip'
                or provenance['sha256'] != allowed[task]['sha256']
                or provenance['revision'] != source['revision']):
            raise ValueError(f'Invalid provenance for {task}')
        files = list(base.rglob('episode*.hdf5'))
        if len(files) != 50:
            raise ValueError(f'{task}: expected exactly 50 episodes, found {len(files)}')
        by_id = {int(f.stem.removeprefix('episode')): f for f in files}
        if set(by_id) != set(range(50)):
            raise ValueError(f'{task}: expected episode IDs 0..49')
        for ep, f in sorted(by_id.items()):
            instruction_file = f.parent.parent / 'instructions' / f'episode{ep}.json'
            instructions = json.loads(instruction_file.read_text())['seen']
            if not instructions or not all(isinstance(x, str) and x.strip() for x in instructions):
                raise ValueError(f'Missing seen instructions in {instruction_file}')
            with h5py.File(f, 'r') as h5:
                q = read_joints(h5)
                for camera in CAMERAS:
                    frames = h5[f'observation/{camera}/rgb']
                    assert len(frames) == len(q), f'Misaligned camera {f}: {camera}'
                    for t in [0, len(q) // 2, len(q) - 1]:
                        decode_rgb(frames[t])
            split = episode_split(task, ep, validation_count=validation_count)
            record = {'task': task, 'episode': ep, 'path': str(f.relative_to(root)),
                      'frames': len(q) - 1, 'split': split, 'instructions': instructions,
                      'sha256': hashlib.sha256(f.read_bytes()).hexdigest()}
            records.append(record)
            if split == 'train':
                states.append(q[:-1])
                actions.append(q[1:])
        print('AUDITED', task, flush=True)
    stats = {}
    for prefix, arrays in [('observation.state', states), ('action', actions)]:
        values = np.concatenate(arrays)
        for field, cols in [('arm.position', ARM_INDICES), ('effector.position', GRIPPER_INDICES)]:
            x = values[:, cols].astype(np.float64)
            stats[f'{prefix}.{field}'] = {
                'mean': x.mean(0).tolist(), 'std': x.std(0).tolist(),
                'min': x.min(0).tolist(), 'max': x.max(0).tolist(),
                'q01': np.quantile(x, 0.01, axis=0).tolist(),
                'q99': np.quantile(x, 0.99, axis=0).tolist(),
            }
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'root': str(root), 'source_revision': source['revision'],
                'source_manifest_sha256': hashlib.sha256((root / 'artifacts/clean-source-manifest.json').read_bytes()).hexdigest(),
                'validation_episodes_per_task': validation_count,
                'action_alignment': 'image[t], state[t] -> action[t+1:t+51]',
                'records': records}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (output / 'norm_stats.json').write_text(json.dumps({'norm_stats': stats,
        'num_samples': sum(len(x) for x in states), 'split': 'train'}, indent=2))
    return manifest

class CleanDataset(Dataset):
    def __init__(self, manifest_path, transform, split='train', color_aug=0.0, state_noise=0.0, image_size=224):
        self.manifest = json.loads(Path(manifest_path).read_text())
        self.root = Path(self.manifest['root'])
        self.records = [r for r in self.manifest['records'] if r['split'] == split]
        if not self.records:
            raise ValueError(f'Empty split: {split}')
        self.ends = np.cumsum([r['frames'] for r in self.records]).tolist()
        self.transform = transform
        self.color_aug = color_aug if split == 'train' else 0.0
        self.state_noise = state_noise if split == 'train' else 0.0
        self.training = split == 'train'
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError('image_size must be positive')

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ep = bisect.bisect_right(self.ends, index)
        t = index - (self.ends[ep - 1] if ep else 0)
        record = self.records[ep]
        instruction_id = int(torch.randint(len(record['instructions']), ()).item()) if self.training else 0
        with h5py.File(self.root / record['path'], 'r') as h5:
            q = read_joints(h5)
            action, pad = action_chunk(q, t, self.transform.chunk_size)
            item = {'observation.state': torch.from_numpy(q[t].copy()),
                    'action': torch.from_numpy(action), 'action_is_pad': torch.from_numpy(pad),
                    'task': record['instructions'][instruction_id]}
            # A shared color transform preserves synchronization across the three cameras.
            brightness = 1.0 + (float(torch.rand(())) * 2 - 1) * self.color_aug
            contrast = 1.0 + (float(torch.rand(())) * 2 - 1) * self.color_aug
            for camera, key in CAMERAS.items():
                rgb = decode_rgb(h5[f'observation/{camera}/rgb'][t])
                if self.color_aug:
                    rgb = np.clip(((rgb.astype(np.float32) - 127.5) * contrast + 127.5) * brightness, 0, 255).astype(np.uint8)
                # Match the official deployment wrapper: resize float32 [0,255].
                image = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32)
                from torchvision.transforms.v2 import Resize
                item[key] = Resize((self.image_size, self.image_size), antialias=True)(image)
        batch = self.transform.apply(item)
        # The upstream transform exposes tail padding but does not apply it to loss.
        batch['joint_mask'] &= ~batch['action_is_pad'].bool().unsqueeze(-1)
        if self.state_noise:
            active = batch['state_joint_mask'].clone()
            active[28:] = False  # Never perturb grippers or unused embodiment fields.
            batch['state'] += torch.randn_like(batch['state']) * self.state_noise * active
        return {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
