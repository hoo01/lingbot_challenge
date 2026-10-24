"""Fixed E0 input references, never scene replacement or expert retry budgets."""
import hashlib
import json
import math
from pathlib import Path

VERSION = 'e0_fixed_inputs_expert_provenance_v1'
FROZEN_ENTRY = 'frozen_e0_no_recheck_v1'


def validate_entry(config):
    mode = config.get('paired_entry')
    if mode is None:
        return False
    if (mode != FROZEN_ENTRY or config.get('expert_check') or not config.get('paired_reference')
            or not config.get('fixed_scenes') or config.get('expert_quota') or config.get('valid_per_group')):
        raise ValueError('Frozen E0 entry requires its exact fixed reference and no new expert screening')
    return True


def prepare_policy_entry(task, env, repo, reference):
    """Initialize pinned score fields, verify objects, and reuse E0 substitutions.

    No expert, planner, motion or simulation step is called. Arm substitutions
    and the instruction come from the immutable E0 reference, never resampling.
    """
    from eval_policy_entry import policy_entry
    objects, receipt = policy_entry(task, env, repo)
    if any(reference['parameters'].get(k) != v for k, v in objects.items()):
        raise ValueError('Official scene objects differ from frozen E0 substitutions')
    return dict(reference['parameters']), {**receipt, 'flow': FROZEN_ENTRY,
        'instruction_source': 'frozen_E0_final_episode',
        'reference_episode': reference['e0_episode_record']}


def read_reference(binding, fixed_scenes):
    if not isinstance(binding,dict) or set(binding) != {'path','sha256'}:
        raise ValueError('Paired reference needs a path and SHA256')
    path = Path(binding['path'])
    value = path.read_bytes()
    if hashlib.sha256(value).hexdigest() != binding['sha256']:
        raise ValueError('Paired reference changed')
    manifest = json.loads(value)
    if manifest.get('version') != VERSION or manifest.get('source_job') != '843758599804':
        raise ValueError('Unknown E0 paired reference')
    by_key = {}
    for row in manifest['scenes']:
        key = (row['task'],row['setting'],row['seed'])
        if key in by_key or type(key[2]) is not int:
            raise ValueError('Duplicate paired reference key')
        if (type(row['expert_passed']) is not bool or type(row['expert_budget_exhausted']) is not bool
                or row['expert_passed'] == row['expert_budget_exhausted']):
            raise ValueError('Reference must preserve expert pass or exhausted entry')
        if not isinstance(row['instruction'],str) or not row['instruction'].strip():
            raise ValueError('Empty paired instruction')
        if (len(row['initial_state']) != 14 or
                any(not isinstance(x,(float,int)) or not math.isfinite(x) for x in row['initial_state'])):
            raise ValueError('Invalid paired initial joints')
        by_key[key] = row
    if not fixed_scenes:
        raise ValueError('Paired inputs require fixed scenes')
    requested = {(r['task'],r['setting'],r['seed']) for r in fixed_scenes}
    if not requested <= set(by_key):
        raise ValueError('Fixed scene absent from the E0 reference')
    return {key:by_key[key] for key in requested}


def compare_initial(reference, observation, instruction):
    import numpy as np
    cameras = {name:hashlib.sha256(np.asarray(observation['observation'][name]['rgb']).tobytes()).hexdigest()
               for name in ('head_camera','left_camera','right_camera')}
    state_equal = np.array_equal(np.asarray(observation['joint_action']['vector']),
                                 np.asarray(reference['initial_state']))
    if instruction != reference['instruction'] or not state_equal:
        raise ValueError('Paired instruction or initial joints differ from E0')
    return dict(source_job='843758599804', instruction_equal=True, initial_state_equal=True,
        camera_sha256=cameras, camera_equal={name:cameras[name]==reference['initial_camera_sha256'][name]
                                            for name in cameras},
        reference_episode=reference['e0_episode_record'])
