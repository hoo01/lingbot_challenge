"""Explicit, bound consent for a published model's cross-host RGB comparison."""
import hashlib
import json
from pathlib import Path

RECORD_DIFFERENCES = 'e0_cross_host_rgb_recorded_v1'


def read_camera_policy(binding, *, root, checkpoint, norm_sha256,
                       paired_reference, scene_replay):
    if not isinstance(binding, dict) or set(binding) != {'path', 'sha256'}:
        raise ValueError('Camera policy requires a bound authorization receipt')
    path = Path(binding['path']).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Camera policy receipt must stay inside this evaluator')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding['sha256']:
        raise ValueError('Camera policy authorization receipt changed')
    proof = json.loads(raw)
    if (proof.get('mode') != RECORD_DIFFERENCES
            or Path(proof.get('root', '')).resolve() != Path(root).resolve()
            or Path(proof.get('checkpoint', '')).resolve() != Path(checkpoint).resolve()
            or proof.get('norm_sha256') != norm_sha256
            or not paired_reference or not scene_replay
            or proof.get('paired_reference_sha256') != paired_reference['sha256']
            or proof.get('scene_replay_sha256') != scene_replay['sha256']
            or proof.get('source_job') != '843758599804'
            or proof.get('authorization_kind') != 'new_soarm_eval_known_cross_host_rgb_difference'
            or proof.get('fixed_instruction_seed_parameters_index_joints') is not True
            or proof.get('e0_pixel_identity_claimed') is not False
            or not proof.get('user_authorization', {}).get('quote')
            or proof.get('user_authorization', {}).get('informed_of_rgb_difference') is not True):
        raise ValueError('Camera policy must preserve this model, fixed E0 inputs and informed authorization')
    return {**proof, 'receipt_sha256': binding['sha256']}


def enforce_camera_policy(check, policy=None):
    """Keep false equality fields visible; an absent policy remains strict."""
    equal = all(check['camera_equal'].values())
    if policy is None:
        if not equal:
            raise ValueError('Replayed E0 initial camera pixels differ; preserve scene evidence')
        return
    if policy.get('mode') != RECORD_DIFFERENCES or not policy.get('receipt_sha256'):
        raise ValueError('Unvalidated cross-host camera policy')
    check.update(camera_policy=RECORD_DIFFERENCES,
                 camera_policy_receipt_sha256=policy['receipt_sha256'],
                 cross_host_rgb_difference=not equal,
                 e0_pixel_identity_required=False,
                 e0_pixel_identity_observed=equal)
