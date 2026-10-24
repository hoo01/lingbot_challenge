"""Fail-closed foreground cache; no detector or simulator is run by a worker.

Trust is an explicit reviewed provenance assertion, not inferred from confidence
or nonzero pixels. This verifies asset identity, not semantic completeness.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def array_sha(raw):
    value = np.asarray(raw, dtype='<f4', order='C')
    return hashlib.sha256(value.tobytes()).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


@dataclass(frozen=True)
class Foreground:
    mask: np.ndarray
    rgb_sha256: str
    producer_sha256: str
    reviewed: bool
    required_labels: tuple
    covered_labels: tuple
    policy_approved: bool = False
    policy_sha256: str = ''
    entity_mask: object = None

    def validated(self, raw):
        m = np.asarray(self.mask)
        trusted = self.reviewed or (self.policy_approved and is_sha(self.policy_sha256))
        if (not trusted or not is_sha(self.producer_sha256) or not self.required_labels
                or not set(self.required_labels) <= set(self.covered_labels)):
            raise ValueError('Foreground is unreviewed or missing required instances')
        if m.shape != (256, 256) or m.dtype != np.bool_ or not m.any():
            raise ValueError('Empty/invalid foreground is not a trusted background mask')
        if array_sha(raw) != self.rgb_sha256:
            raise ValueError('Foreground RGB SHA does not match decoded/resized raw255')
        return m.copy()


def mask_from_instance_labels(labels, protected_ids):
    """Only raw integer labels with an explicit instance mapping are accepted.

    RoboTwin get_segmentation returns palette RGB and casts IDs to uint8; its
    colored output MUST NOT be interpreted as raw actor IDs using this helper.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer) or not protected_ids:
        raise ValueError('Need dense integer labels and explicit protected instance IDs')
    if not set(protected_ids) <= set(np.unique(labels).tolist()):
        raise ValueError('A required protected actor is missing')
    return np.isin(labels, protected_ids)


class ForegroundCache:
    """Immutable packed episode files, bounded per-worker LRU, SHA-bound manifest.

    The manifest must itself be bound in the data/checkpoint state. Without an
    approved manifest, diagnostic use can fall back; formal readiness stays false.
    """
    def __init__(self, manifest_path, expected_sha256, max_episodes=4):
        self.path = Path(manifest_path)
        if not is_sha(expected_sha256) or file_sha(self.path) != expected_sha256:
            raise ValueError('Foreground manifest SHA mismatch')
        self.sha256 = expected_sha256
        self.manifest = json.loads(self.path.read_text(encoding='utf8'))
        self.policy_schema = self.manifest.get('schema') == 'aug6000_foreground_policy_v2'
        self.policy_gate_verified = False
        if self.manifest.get('schema') not in ('aug6000_foreground_v1','aug6000_foreground_policy_v2'):
            raise ValueError('Unknown foreground cache schema')
        if (self.manifest.get('camera') != 'head_camera'
                or self.manifest.get('canonical_rgb') != 'torchvision_resize256_antialias_float32_raw255'):
            raise ValueError('Foreground camera/decode/resize contract mismatch')
        producer = self.manifest.get('producer', {})
        if not all(producer.get(k) for k in ('name', 'revision', 'thresholds', 'review_protocol')):
            raise ValueError('Missing foreground producer provenance')
        if not is_sha(producer.get('weights_sha256')):
            raise ValueError('Missing foreground weights/source SHA')
        self.producer_sha = hashlib.sha256(json.dumps(producer, sort_keys=True).encode()).hexdigest()
        self.max_episodes = max_episodes
        self._episodes = OrderedDict()
        if not 1 <= max_episodes <= 16:
            raise ValueError('Cache bound must be 1..16 episodes')

    def __getstate__(self):
        return {**self.__dict__, '_episodes': OrderedDict()}

    def assert_formal_ready(self, expected_frame_count):
        q = self.manifest.get('quality_gate', {})
        if self.policy_schema:
            return self._assert_policy_ready(expected_frame_count)
        if not (q.get('passed') is True and q.get('review_report_sha256')
                and is_sha(q['review_report_sha256']) and q.get('eligible_frames') == expected_frame_count
                and q.get('all_required_instances_reviewed') is True):
            raise ValueError('Foreground production/semantic quality gate has not passed')
        if not isinstance(q.get('review_report_path'), str):
            raise ValueError('Foreground semantic review path is missing')
        path = (self.path.parent / q['review_report_path']).resolve()
        if not path.is_relative_to(self.path.parent.resolve()) or file_sha(path) != q['review_report_sha256']:
            raise ValueError('Foreground semantic review file SHA/path mismatch')
        report = json.loads(path.read_text(encoding='utf8'))
        if (report.get('passed') is not True or report.get('all_required_instances_reviewed') is not True
                or report.get('producer') != self.manifest['producer']):
            raise ValueError('Foreground review is not bound to this producer')
        episodes = self.manifest.get('episodes', {})
        if len(episodes) != 2500 or set(episodes) != set(report.get('episodes', {})):
            raise ValueError('Semantic review lacks full original Clean episode coverage')

    def _assert_policy_ready(self, expected_frame_count):
        q=self.manifest.get('quality_gate',{})
        if not(q.get('passed') is True and q.get('eligible_frames')==expected_frame_count
               and is_sha(q.get('review_report_sha256')) and is_sha(q.get('policy_sha256'))
               and q.get('all_frames_manually_reviewed') is False):
            raise ValueError('Audited automatic foreground policy has not passed')
        path=(self.path.parent/q['review_report_path']).resolve()
        if not path.is_relative_to(self.path.parent.resolve()) or file_sha(path)!=q['review_report_sha256']:
            raise ValueError('Automatic-policy review SHA/path mismatch')
        report=json.loads(path.read_text(encoding='utf8'))
        if not(report.get('passed') is True and report.get('policy_sha256')==q['policy_sha256']
               and report.get('calibration_frames')==400 and report.get('holdout_frames')==400
               and report.get('tasks_reviewed')==50 and report.get('all_frames_manually_reviewed') is False
               and report.get('producer')==self.manifest['producer']
               and report.get('disabled_rectangle_tasks',[])==q.get('disabled_rectangle_tasks',[])):
            raise ValueError('Incomplete calibration/holdout visual policy review')
        episodes=self.manifest.get('episodes',{})
        if len(episodes)!=2500 or set(episodes)!=set(report.get('episodes',{})):
            raise ValueError('Automatic cache lacks the original2500 episode coverage')
        self.policy_gate_verified=True

    def get(self, task, episode, frame, raw):
        key = f'{task}/episode{int(episode)}'
        record = self.manifest.get('episodes', {}).get(key)
        if record is None:
            return None
        if key not in self._episodes:
            path = (self.path.parent / record['path']).resolve()
            if not path.is_relative_to(self.path.parent.resolve()):
                raise ValueError('Cache path escapes the manifest directory')
            if path.stat().st_size > 32 * 1024**2 or file_sha(path) != record['sha256']:
                raise ValueError('Foreground episode size/SHA mismatch')
            with np.load(path, allow_pickle=False) as z:
                packed = z['packed']
                metadata = json.loads(str(z['metadata'].item()))
            if packed.dtype != np.uint8 or packed.shape != (len(metadata), 8192):
                raise ValueError('Invalid bit-packed foreground shape')
            self._episodes[key] = (packed, metadata)
            while len(self._episodes) > self.max_episodes:
                self._episodes.popitem(last=False)
        self._episodes.move_to_end(key)
        packed, metadata = self._episodes[key]
        if not 0 <= frame < len(metadata):
            return None
        info = metadata[frame]
        entity=None
        if self.policy_schema:
            entity=np.zeros((256,256),bool)
            for box in info.get('entity_boxes',[]):
                x1,y1,x2,y2=box
                entity[max(0,int(np.floor(y1))):min(256,int(np.ceil(y2))),
                       max(0,int(np.floor(x1))):min(256,int(np.ceil(x2)))]=True
        fg = Foreground(np.unpackbits(packed[frame], bitorder='little').reshape(256, 256).astype(bool),
                        info['rgb_sha256'], self.producer_sha, info.get('reviewed') is True,
                        tuple(info.get('required_labels', [])), tuple(info.get('covered_labels', [])),
                        policy_approved=(self.policy_gate_verified and info.get('policy_eligible') is True
                            and task not in self.manifest.get('quality_gate',{}).get('disabled_rectangle_tasks',[])),
                        policy_sha256=self.manifest.get('quality_gate',{}).get('policy_sha256',''),
                        entity_mask=entity)
        if array_sha(raw) != fg.rgb_sha256:
            raise ValueError('Foreground RGB SHA mismatch')
        return fg


def write_episode_cache(path, masks, metadata):
    """Pack supplied reviewed masks; does not manufacture detector/review evidence."""
    path = Path(path)
    m = np.asarray(masks)
    if path.exists() or m.dtype != np.bool_ or m.ndim != 3 or m.shape[1:] != (256, 256):
        raise ValueError('New path and boolean [frames,256,256] masks required')
    if len(metadata) != len(m):
        raise ValueError('Foreground metadata/frame count mismatch')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('xb') as stream:
        np.savez_compressed(stream, packed=np.packbits(m.reshape(len(m), -1), axis=1, bitorder='little'),
                            metadata=np.array(json.dumps(metadata, sort_keys=True)))
    temporary.replace(path)
    return file_sha(path)
