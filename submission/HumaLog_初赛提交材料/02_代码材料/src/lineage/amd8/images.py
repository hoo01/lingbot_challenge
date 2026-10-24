"""Explicit raw255 augmentation, called inside every real reader/worker.

Augmentation is performed once here. The official FeatureTransform's internal
augmentation is disabled in the new reader, while its processor and targets
remain unchanged. No global patch to shared or historical code is needed.
"""
import hashlib
import numpy as np
import torch
from policy import GLOBAL_BATCH, STEPS, seed_for


def plan_for(draw, force=None):
    rng = np.random.default_rng(seed_for(draw, 'augmentation'))
    update = int(draw) // GLOBAL_BATCH + 1
    warmup = int(STEPS * .05)
    extra = .25 * min(1., max(0., (update - warmup) / 200))
    value = float(rng.random())
    branch = 'plain' if value < .25 else 'color' if value < 1 - extra else 'extra'
    if force is not None:
        assert force in ('plain', 'color', 'gamma', 'noise', 'neutral')
        branch = force
    result = dict(branch=branch, update=update, brightness=1., contrast=1.,
                  saturation=1., gamma=1., noise_sigma=0.)
    if branch not in ('plain', 'neutral'):
        result.update(brightness=float(rng.uniform(.7, 1.3)),
                      contrast=float(rng.uniform(.6, 1.4)),
                      saturation=float(rng.uniform(.5, 1.5)))
    if branch == 'extra':
        branch = str(rng.choice(['gamma', 'noise']))
        result['branch'] = branch
    if branch == 'gamma':
        result['gamma'] = float(rng.uniform(.85, 1.15))
    if branch == 'noise':
        result['noise_sigma'] = float(rng.uniform(.003, .01))
    return result


def validate_raw(raw):
    if not raw.is_floating_point() or not torch.isfinite(raw).all():
        raise ValueError('Reader RGB must be finite floating-point raw255')
    if raw.min() < 0 or raw.max() > 255:
        raise ValueError('Reader RGB is outside the explicit raw255 contract')


def augment(raw, plan, draw, view, frame, color_operation=None):
    validate_raw(raw)
    if plan['branch'] == 'plain':
        return raw.clone()
    if color_operation is None:
        from lingbotvla.data.vla_data.transform import apply_visual_augmentation
        color_operation = apply_visual_augmentation
    unit = raw / 255.0  # Even almost-black raw images have declared raw255 units.
    parameters = {k: torch.tensor(plan[k], dtype=raw.dtype, device=raw.device)
                  for k in ('brightness', 'contrast', 'saturation')}
    unit = color_operation(unit, parameters)
    if plan['branch'] == 'gamma':
        unit = unit.pow(plan['gamma'])
    elif plan['branch'] == 'noise':
        generator = torch.Generator(device='cpu').manual_seed(seed_for(draw, f'noise:{view}:{frame}'))
        unit = unit + torch.randn(unit.shape, generator=generator, dtype=unit.dtype) * plan['noise_sigma']
    result = unit.clamp(0, 1) * 255.0
    validate_raw(result)
    return result


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
