"""Wrap the already verified parent dataset without changing its shared source.

    dataset = AugmentedOfficialClean(parent_dataset, cache, recipe_sha256)

The inherited official sampler receives identical indices and dataset length.
Only RGB construction and auxiliary validity change. The parent dataset remains
responsible for source/norm/filter identity checks and spawned transform setup.
"""
import json
import sys

import h5py
import numpy as np
import torch

from .foreground import is_sha
from .images import augment_pair
from .integration import apply_transform
from .policy import plan_for
from .geometry import geometry_plan
from .allocation import allocate,pair_eligible


class AugmentedOfficialClean:
    def __init__(self, parent, foreground_cache, recipe_sha256, *, diagnostic=False,
                 allow_parent_boundary_restore=False):
        if not is_sha(recipe_sha256):
            raise ValueError('Recipe SHA must be bound to checkpoint data state')
        if not diagnostic:
            if foreground_cache is None:
                raise ValueError('Formal recipe needs a reviewed foreground cache')
            foreground_cache.assert_formal_ready(len(parent))
            parent_module = sys.modules[type(parent).__module__]
            if (foreground_cache.manifest.get('source_manifest_sha256') != parent_module.MANIFEST_SHA
                    or foreground_cache.manifest.get('data_filter_sha256') != parent.filters.sha):
                raise ValueError('Foreground cache is bound to a different Clean source/filter')
        self.parent, self.cache, self.recipe_sha = parent, foreground_cache, recipe_sha256
        self.diagnostic = diagnostic
        self.allow_parent_boundary_restore = allow_parent_boundary_restore

    def __len__(self): return len(self.parent)

    def state_dict(self):
        return dict(contract='aug6000_rgb_and_mask_v5_geometry45_b192', parent=self.parent.state_dict(),
                    foreground_manifest_sha256=None if self.cache is None else self.cache.sha256,
                    recipe_sha256=self.recipe_sha, diagnostic=self.diagnostic)

    def load_state_dict(self, state):
        if state == self.state_dict(): return
        if self.allow_parent_boundary_restore:
            from policy import ROOT
            proof=json.loads((ROOT/'artifacts/geometryfix11000_requested/source-binding.json').read_text())
            expected=dict(contract='aug6000_rgb_and_mask_v4a_b192_allocated',parent=self.parent.state_dict(),
                foreground_manifest_sha256=self.cache.sha256,recipe_sha256=proof['old_recipe_sha256'],diagnostic=False)
            if state==expected:
                return  # Preserve official cursor and prefetch position; only future RGB is reconstructed differently.
        raise ValueError('Resume dataset/filter/recipe/foreground identity differs')

    def __getitem__(self, key):
        from lingbot_competition.data import CAMERAS, action_chunk, decode_rgb, read_joints
        from torchvision.transforms.v2 import Resize
        p = self.parent
        parent_module = sys.modules[type(p).__module__]
        frame_index, draw, epoch = map(int, key)
        if not 0 <= frame_index < len(self):
            raise IndexError(frame_index)
        plan = geometry_plan(plan_for(draw))
        original_index = int(p.filters.original_indices[frame_index])
        episode, frame = p.catalog.locate(original_index)
        record = p.records[episode]
        path = parent_module.ASSETS / record['path']
        inventory, stat = p.file_inventory[str(path)], path.stat()
        if ((stat.st_size, stat.st_mtime_ns) != (inventory['bytes'], inventory['mtime_ns'])
                or inventory['sha256'] != record['sha256']):
            raise ValueError('Original Clean input identity changed')
        r = np.random.default_rng(parent_module.seed_for(draw, 'instruction'))
        kept = p.filters.instruction_indices(record)
        instruction_index = kept[int(r.integers(len(kept)))]
        with h5py.File(path, 'r') as source:
            joints = read_joints(source)
            action, padding = action_chunk(joints, frame, 50)
            future = min(frame+49, len(joints)-1)
            item = {'observation.state': torch.from_numpy(joints[frame].copy()),
                    'action': torch.from_numpy(action.copy()), 'action_is_pad': torch.from_numpy(padding),
                    'task': record['instructions'][instruction_index]}
            raw = []
            if tuple(CAMERAS) != ('head_camera', 'left_camera', 'right_camera'):
                raise ValueError('Actual primary camera order changed')
            for camera in CAMERAS:
                pair = []
                for t in (frame, future):
                    decoded = torch.from_numpy(decode_rgb(source[f'observation/{camera}/rgb'][t]))
                    pair.append(Resize((256,256), antialias=True)(decoded.permute(2,0,1).float()).numpy())
                raw.append(pair)
        raw = np.asarray(raw, dtype=np.float32)
        foregrounds = None if self.cache is None else [self.cache.get(
            record['task'], record['episode'], t, raw[0,i]) for i,t in enumerate((frame,future))]
        if foregrounds is not None and any(f is None for f in foregrounds): foregrounds = None
        plan,allocation=allocate(plan,pair_eligible(foregrounds,raw))
        augmented = augment_pair(raw, plan, foregrounds,
                                 require_protection_for_geometry=not self.diagnostic)
        batch = apply_transform(p.transform, item, augmented)
        batch['joint_mask'] &= ~batch['action_is_pad'].bool().unsqueeze(-1)
        restored = p.transform.unapply({k:batch[k].clone() for k in (
            'actions','state','action_joint_mask','state_joint_mask')})['action']
        if not torch.allclose(restored.float(), torch.from_numpy(action), atol=2e-5, rtol=1e-5):
            raise ValueError('Physical action normalization roundtrip failed')
        result = {k:v for k,v in batch.items() if isinstance(v, torch.Tensor)}
        result['_meta'] = json.dumps(dict(draw=draw, epoch=epoch, official_frame_index=frame_index,
            original_frame_index=original_index, task=record['task'], episode=record['episode'],
            frame=frame, future_frame=future, instruction_index=instruction_index,
            augmentation=augmented.metadata, allocation=allocation,
            foreground_manifest_sha256=None if self.cache is None else self.cache.sha256,
            diagnostic=self.diagnostic))
        return result
