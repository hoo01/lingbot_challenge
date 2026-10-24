"""Replay E0's scene-entry side effects without expert simulation or new text.

The upstream clutter function mutates default bounds on every setup. Each
isolated E0 episode had either one policy setup or an expert plus policy setup.
An immutable per-key receipt selects that historical bound explicitly. New
episodes never inherit mutable defaults from a preceding call.
"""
import hashlib
import json
from pathlib import Path

MODE = 'e0_scene_entry_replay_v1'


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_scene_replay(binding, paired_binding, fixed_scenes, repo):
    if not isinstance(binding, dict) or set(binding) != {'path', 'sha256'}:
        raise ValueError('Scene replay requires a path and SHA256')
    path = Path(binding['path'])
    if file_sha(path) != binding['sha256']:
        raise ValueError('Scene replay receipt changed')
    data = json.loads(path.read_text())
    if data.get('mode') != MODE or data.get('paired_reference_sha256') != paired_binding['sha256']:
        raise ValueError('Scene replay is not bound to this E0 reference')
    for name, expected in data['robotwin_source_sha256'].items():
        if file_sha(Path(repo) / name) != expected:
            raise ValueError('RoboTwin scene source changed: ' + name)
    by_key = {}
    for row in data['scenes']:
        key = (row['task'], row['setting'], row['seed'])
        if key in by_key or type(row['prior_clutter_setups']) is not int or row['prior_clutter_setups'] not in (0, 1):
            raise ValueError('Ambiguous E0 clutter setup history')
        if row['instruction_choice_count'] != 5:
            raise ValueError('E0 instruction choice cardinality must remain five')
        record = row['e0_episode_record']
        from .host_migration import original_record_path
        if file_sha(original_record_path(record['path'])) != record['sha256']:
            raise ValueError('Historical E0 episode changed')
        by_key[key] = row
    requested = {(r['task'], r['setting'], r['seed']) for r in fixed_scenes}
    if not requested <= set(by_key):
        raise ValueError('Missing fixed E0 scene replay key')
    return {key: by_key[key] for key in requested}


def install_clutter_copy(base_class, prior_setups):
    """Copy argument lists and reconstruct the exact historical additions.

    No setup, scene step, expert or planner is executed here. Use the wrapper
    only within the isolated simulator process, after validating the receipt.
    """
    if type(prior_setups) is not int or prior_setups not in (0, 1):
        raise ValueError('Unsupported historical setup count')
    original = base_class.get_cluttered_table
    if getattr(original, '_e0_clutter_copy', False):
        raise ValueError('Scene wrapper must be installed once per process')
    receipt = dict(mode=MODE, prior_clutter_setups=prior_setups, clutter_calls=0,
                   expert_called=False, simulation_steps_added=0)

    def copied(self, cluttered_numbers=10, xlim=None, ylim=None, zlim=None):
        x = list((-0.59, 0.59) if xlim is None else xlim)
        y = list((-0.34, 0.34) if ylim is None else ylim)
        z = list((0.741,) if zlim is None else zlim)
        # Preserve sequential floating-point additions used by legacy E0.
        for _ in range(prior_setups):
            for i in (0, 1):
                x[i] += self.table_xy_bias[0]
                y[i] += self.table_xy_bias[1]
        receipt['clutter_calls'] += 1
        if receipt['clutter_calls'] != 1:
            raise ValueError('Unexpected repeated clutter setup in one policy entry')
        receipt['clutter_bounds_before_upstream'] = dict(x=x[:], y=y[:], z=z[:])
        result = original(self, cluttered_numbers=cluttered_numbers, xlim=x, ylim=y, zlim=z)
        receipt['clutter_bounds_after_upstream'] = dict(x=x[:], y=y[:], z=z[:])
        return result

    copied._e0_clutter_copy = True
    base_class.get_cluttered_table = copied
    return receipt


def rng_sha():
    import numpy as np

    name, values, index, gaussian, cached = np.random.get_state()
    return hashlib.sha256(name.encode() + values.tobytes() +
        json.dumps([index, gaussian, cached], separators=(',', ':')).encode()).hexdigest()


def replay_instruction_choice(count):
    """Advance NumPy exactly as choice over five strings; keep frozen text.

    The upstream generator uses only Python random for templates and leaves
    NumPy unchanged. Python's template RNG was isolated in E0. No instruction
    generator is called and no sampled index is used as model input.
    """
    import numpy as np

    if count != 5:
        raise ValueError('Unsupported historical instruction selection')
    before = rng_sha()
    index = int(np.random.choice(count))
    return dict(instruction_choice_count=count, discarded_choice_index=index,
                numpy_rng_before=before, numpy_rng_after=rng_sha(),
                instruction_source='immutable_E0_reference', instruction_regenerated=False)
