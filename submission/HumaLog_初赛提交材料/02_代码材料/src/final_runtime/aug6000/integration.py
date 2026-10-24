"""Explicit FeatureTransform and teacher boundary; no global monkey patch."""
import copy

import numpy as np
import torch

from .images import validate_raw


CAMERA_KEYS = ('observation.images.cam_high', 'observation.images.cam_left_wrist',
               'observation.images.cam_right_wrist')


def validate_transform(transform):
    p = transform.processor.image_processor
    if (transform.image_augment or not p.do_rescale or abs(p.rescale_factor-1/255) > 1e-12
            or not p.do_normalize or list(p.image_mean) != [.5]*3 or list(p.image_std) != [.5]*3):
        raise ValueError('Must use explicit raw255 FeatureTransform with single rescale and mean/std .5')


def apply_transform(transform, item, augmented):
    """Student processor sees strong RGB; raw teacher fields are replaced by weak RGB.

    item has original physical state/action/task. Teacher tensors have exactly the
    parent slot0/current+future layout. Parent relative-action norm is untouched.
    """
    validate_transform(transform)
    item = copy.deepcopy(item)  # Parent normalizer mutates action/state in place.
    for index, key in enumerate(CAMERA_KEYS):
        item[key] = torch.from_numpy(augmented.student[index].copy())
    batch = transform.apply(item)
    current = torch.from_numpy(augmented.teacher[:, 0].copy())
    future = torch.from_numpy(augmented.teacher[:, 1].copy())
    if batch['pil_images'].shape != current.shape or batch['future_pil_images'].shape != future.shape:
        raise ValueError('Actual FeatureTransform camera/future layout differs from audited mapping')
    # Verify the raw255 pad helper was actually overridden, including dark images.
    expected_student_current = torch.from_numpy(augmented.student[:, 0].copy())
    expected_student_future = torch.from_numpy(augmented.student[:, 1].copy())
    if not torch.equal(batch['pil_images'], expected_student_current):
        raise ValueError('Raw255 student image changed by unexpected padding/unit conversion')
    if not torch.equal(batch['future_pil_images'], expected_student_future):
        raise ValueError('Future raw255 image mapping changed')
    batch['pil_images'], batch['future_pil_images'] = current, future
    batch['_aux_valid'] = torch.from_numpy(augmented.auxiliary_grid.copy())
    for value in (current, future): validate_raw(value.numpy())
    return batch


def split_auxiliary_fields(batch):
    """Pop ONLY adapter metadata before frozen-teacher and ordinary model forwards."""
    valid = batch.pop('_aux_valid')
    if valid.dtype != torch.bool or valid.ndim != 3 or valid.shape[1:] != (2, 256):
        raise ValueError('Collated auxiliary validity has the wrong shape')
    return valid


def assert_frozen_teachers(teachers):
    if any(teacher.training or any(p.requires_grad for p in teacher.parameters()) for teacher in teachers):
        raise ValueError('Teachers must be eval, frozen, and called under no_grad')


def forward_batch(model, batch, align, teacher_forward, teachers=None):
    """Trainer integration: keep full microbatch/FSDP forwards on every rank.

    teacher_forward is the inherited targets_for_batch(align,batch,teachers)
    helper. Its raw current/future fields were already replaced by weak RGB.
    The caller owns engineering gates, optimizer, accumulation and save/resume.
    """
    from .losses import replace_auxiliary_losses
    batch = dict(batch)
    validity = split_auxiliary_fields(batch)
    with torch.no_grad():
        teachers, targets = teacher_forward(align, batch, teachers)
    assert_frozen_teachers(teachers)
    if any(v is not None and (v.requires_grad or not torch.isfinite(v).all()) for v in targets.values()):
        raise ValueError('Teacher targets must be detached and finite')
    outputs = model(**batch, **targets)
    return replace_auxiliary_losses(outputs, targets, validity, align), teachers


def normalize_for_contract_test(raw, network):
    """Reference values only; production keeps the actual inherited processors."""
    validate_raw(np.asarray(raw))
    unit = np.asarray(raw) / np.float32(255.)
    if network == 'qwen': return (unit-.5)/.5
    if network == 'depth': return unit
    if network == 'video':
        return (unit-np.array([.485,.456,.406], np.float32)[:,None,None]) / np.array(
            [.229,.224,.225], np.float32)[:,None,None]
    raise ValueError('Unknown network contract')
