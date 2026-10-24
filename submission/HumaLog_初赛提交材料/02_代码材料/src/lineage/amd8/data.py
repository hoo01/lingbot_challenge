"""Original raw-HDF5 windows, ordered by the actual installed official sampler."""
import json
import ast
import inspect
from pathlib import Path
import textwrap
from types import SimpleNamespace
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from images import augment, plan_for, tensor_sha, validate_raw
from policy import (ASSETS, GLOBAL_BATCH, ROOT, STEPS, NORM_SHA, MANIFEST_SHA,
                    FrameCatalog, FRAMES, MICRO, ACCUM, LOCAL_BATCH, seed_for, sha)
from official_sampler import OfficialTaggedSampler


class OfficialClean(Dataset):
    def __init__(self, config, cli, audit_images=False):
        asset_binding=json.loads((ROOT/'artifacts/asset-binding.json').read_text())
        assert asset_binding['passed'] and all(row['equal'] for key in ('tokenizer','teachers','official_source') for row in asset_binding[key])
        assert sha(ROOT / 'artifacts/norm_stats.json') == NORM_SHA
        assert sha(ROOT / 'artifacts/manifest.json') == MANIFEST_SHA
        self.records = json.loads((ROOT / 'artifacts/manifest.json').read_text())['records']
        assert all(record['split'] == 'train' for record in self.records)
        self.catalog = FrameCatalog(self.records)
        inventory = json.loads((ROOT / 'artifacts/input_inventory.json').read_text())
        self.file_inventory = {r['path']: r for r in inventory['clean_episodes']}
        self.audit_images = audit_images
        self.model_config, self.cli = config, cli
        self._build_transform()

    def _build_transform(self):
        from lingbotvla.models import build_processor
        from lingbotvla.data.vla_data.utils import FeatureTransform
        config, cli = self.model_config, self.cli
        # The shared adapter also guesses unit RGB from max<=2 in pad_and_concat.
        # Our reader declares raw255, including dark images. Override only that
        # local helper in a NEW subclass; leave shared/historical code untouched.
        transform_class = raw_feature_transform_class(FeatureTransform)
        self.transform = transform_class(str(ROOT / 'artifacts/robotwin_relative.yaml'),
            SimpleNamespace(**cli['data']), config, build_processor(config.tokenizer_path),
            norm_stats_path=str(ROOT / 'artifacts/norm_stats.json'), chunk_size=50,
            image_augment=False, use_depth_align=True, use_future_image=True)
        processor = self.transform.processor.image_processor
        assert processor.do_rescale and abs(processor.rescale_factor - 1 / 255) < 1e-12
        assert list(processor.image_mean) == list(processor.image_std) == [.5, .5, .5]
        assert processor.do_normalize and not self.transform.image_augment

    def __getstate__(self):
        state = dict(self.__dict__)
        state['transform'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build_transform()  # Reapply the explicit contract in each spawned worker.

    def __len__(self):
        return FRAMES

    def state_dict(self):
        return dict(contract='official_frame_index_tagged_draw_v1')

    def load_state_dict(self, value):
        assert value == self.state_dict()

    def __getitem__(self, key):
        index,draw,epoch=map(int,key)
        return self.read_frame(index,draw,epoch)

    def read_draw(self, draw, force=None, synthetic=None):
        # Isolated CPU input tests only. Actual training always supplies the
        # official sampled frame index via __getitem__.
        return self.read_frame(int(draw)%FRAMES,int(draw),0,force,synthetic)

    def read_frame(self, frame_index, draw, epoch, force=None, synthetic=None):
        from lingbot_competition.data import read_joints, action_chunk, CAMERAS, decode_rgb
        from torchvision.transforms.v2 import Resize
        episode, frame = self.catalog.locate(frame_index)
        record = self.records[episode]
        path = ASSETS / record['path']
        verified = self.file_inventory[str(path)]
        stat = path.stat()
        assert (stat.st_size, stat.st_mtime_ns) == (verified['bytes'], verified['mtime_ns'])
        assert verified['sha256'] == record['sha256']
        plan = plan_for(draw, force)
        instruction_rng = np.random.default_rng(seed_for(draw, 'instruction'))
        rgb_hashes = {}
        audited = self.audit_images or draw < GLOBAL_BATCH
        with h5py.File(path, 'r') as source:
            joints = read_joints(source)
            assert 0 <= frame < len(joints) - 1
            action, padding = action_chunk(joints, frame, 50)
            future = min(frame + 49, len(joints) - 1)
            instruction = record['instructions'][int(instruction_rng.integers(len(record['instructions'])))]
            item = {'observation.state': torch.from_numpy(joints[frame].copy()),
                    'action': torch.from_numpy(action), 'action_is_pad': torch.from_numpy(padding),
                    'task': instruction}
            for view, key in CAMERAS.items():
                pair = []
                for index, t in enumerate((frame, future)):
                    image = torch.from_numpy(decode_rgb(source[f'observation/{view}/rgb'][t]))
                    raw = Resize((256, 256), antialias=True)(image.permute(2, 0, 1).float())
                    if synthetic is not None:  # Only the isolated CPU input gate supplies this.
                        raw = torch.full_like(raw, float(synthetic))
                    pixels = augment(raw, plan, draw, view, index)
                    pair.append(pixels)
                    if audited:
                        rgb_hashes[f'{view}:{index}'] = tensor_sha(pixels)
                item[key] = torch.stack(pair)
        # FeatureTransform mutates action/state in-place: keep the physical reference independent.
        reference = torch.from_numpy(action.copy())
        batch = self.transform.apply(item)
        batch['joint_mask'] &= ~batch['action_is_pad'].bool().unsqueeze(-1)
        restored = self.transform.unapply({k: batch[k].clone() for k in (
            'actions', 'state', 'action_joint_mask', 'state_joint_mask')})['action']
        assert torch.allclose(restored.float(), reference, atol=2e-5, rtol=1e-5)
        for key in ('pil_images', 'future_pil_images'):
            validate_raw(batch[key])
        result = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        result['_meta'] = json.dumps(dict(draw=draw, official_frame_index=frame_index, epoch=epoch,
            task=record['task'], episode=record['episode'],
            frame=frame, future_frame=future, instruction=instruction, augmentation=plan,
            image_contract='raw255_external_augmentation_v1',
            internal_augmentation=False, rgb_sha256=rgb_hashes,
            teacher_current_unit_mean=float((batch['pil_images'] / 255).mean()),
            teacher_future_unit_mean=float((batch['future_pil_images'] / 255).mean())))
        return result


def build_loader(dataset, rank, microbatch, accum, workers=2):
    from torchdata.stateful_dataloader import StatefulDataLoader
    assert (microbatch,accum)==(MICRO,ACCUM)
    sampler = OfficialTaggedSampler(dataset,rank)
    generator = torch.Generator().manual_seed(42 + rank)
    loader = StatefulDataLoader(dataset, batch_size=LOCAL_BATCH, sampler=sampler,
        num_workers=workers, multiprocessing_context='spawn', persistent_workers=False,
        prefetch_factor=2, drop_last=True, pin_memory=True, generator=generator)
    return loader, sampler, generator


def raw_feature_transform_class(original):
    cached = globals().get('RawFeatureTransform')
    if cached is not None:
        return cached
    tree = ast.parse(textwrap.dedent(inspect.getsource(original.pad_and_concat)))
    method = tree.body[0]
    helpers = [node for node in method.body if isinstance(node, ast.FunctionDef)
               and node.name == '_to_uint8']
    if helpers:
        assert len(helpers) == 1
        helper = helpers[0]
        assert [arg.arg for arg in helper.args.args] == ['img']
        helper.body = [ast.Return(value=ast.Name(id='img', ctx=ast.Load()))]
        ast.fix_missing_locations(tree)
        namespace = {}
        exec(compile(tree, '<raw255-only-pad-and-concat>', 'exec'),
             original.pad_and_concat.__globals__, namespace)
        replacement = namespace['pad_and_concat']
    else:
        replacement = original.pad_and_concat
    # Global registration makes the subclass pickleable in spawned workers.
    cls = type('RawFeatureTransform', (original,),
               {'__module__': __name__, 'pad_and_concat': replacement})
    globals()['RawFeatureTransform'] = cls
    return cls
