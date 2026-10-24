"""Replayable, per-global-draw randomness, independent of model/data RNG."""
from dataclasses import dataclass
import hashlib
import math

import numpy as np

RECIPE = 'amd10000_aug6000_full_sft_v4_b192_language1e5_mask7'
WORLD, MICRO, ACCUM = 4, 16, 3
FIRST_DRAW, GLOBAL_BATCH, UPDATES = 2560000, WORLD * MICRO * ACCUM, 6000
BRANCHES = ('plain', 'ordinary', 'protected_background_rectangle', 'whole_primary_mask')
INITIAL_LRS = dict(language=2e-6, vision=1e-6, action_expert=4e-6, projections_alignment=4e-6)
PEAK_LRS = dict(language=1e-5, vision=1e-5, action_expert=2e-5, projections_alignment=2e-5)


def seed_for(draw, stream):
    payload = f'{RECIPE}:42:{int(draw)}:{stream}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2**63 - 1)


def rng_for(draw, stream):
    return np.random.default_rng(seed_for(draw, stream))


def stage_step(draw):
    if not isinstance(draw, (int, np.integer)) or isinstance(draw, bool):
        raise ValueError('Global draw must be an integer')
    # The parent's draw anchor was produced with batch256. Dividing that
    # absolute counter by the new batch192 would incorrectly start at3334.
    step = (int(draw) - FIRST_DRAW) // GLOBAL_BATCH + 1
    if not 1 <= step <= UPDATES:
        raise ValueError('Draw outside the new 6000-update stage')
    return step


def draw_for_local_position(position, rank, *, microbatch=MICRO):
    """Tag committed new-stage samples, separately from official frame indices.

    Position starts at zero at the parent boundary and counts only new-stage
    per-rank samples across epochs. The caller must migrate the actual official
    sampler/loader cursor; this helper does not restore or shuffle that cursor.
    """
    values = (position, rank, microbatch)
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) for v in values):
        raise ValueError('Position, rank and microbatch must be integers')
    if not 0 <= rank < WORLD or microbatch not in (8, MICRO):
        raise ValueError('Expected one of four ranks and accepted microbatch16 or8')
    local_batch = GLOBAL_BATCH // WORLD
    if not 0 <= position < UPDATES * local_batch:
        raise ValueError('Local position outside the new stage')
    update, within = divmod(int(position), local_batch)
    micro, offset = divmod(within, int(microbatch))
    return FIRST_DRAW + update * GLOBAL_BATCH + micro * WORLD * microbatch + rank * microbatch + offset


def ramp(step):
    if not isinstance(step, (int, np.integer)) or isinstance(step, bool) or not 1 <= step <= UPDATES:
        raise ValueError('Expected successful stage update 1..6000')
    return min((step - 1) / 300, 1.)


def learning_rates(step):
    ramp(step)
    if step <= 300:
        return {k: v + (PEAK_LRS[k] - v) * (step - 1) / 299 for k, v in INITIAL_LRS.items()}
    mult = .3 + .35 * (1 + math.cos(math.pi * (step - 300) / 5700))
    return {k: v * mult for k, v in PEAK_LRS.items()}


def set_learning_rates(optimizer, step):
    """Update LR scalars only; never recreate Adam or touch moments/step counters."""
    rates = learning_rates(step)
    labels = [group.get('label') for group in optimizer.param_groups]
    if len(labels) != len(rates) or set(labels) != set(rates):
        raise ValueError('Optimizer groups differ from the inherited four-group identity')
    for group in optimizer.param_groups:
        group['lr'] = rates[group['label']]
    return rates


def branch_probabilities(step):
    a = ramp(step)
    return (1 - a) * np.array([.25, .75, 0., 0.]) + a * np.array([.10, .33, .50, .07])


@dataclass(frozen=True)
class Plan:
    draw: int
    step: int
    branch: str
    brightness: float = 1.
    contrast: float = 1.
    saturation: float = 1.
    gamma: float = 1.
    noise_sigma: float = 0.
    scale: float = 1.
    tx: int = 0
    ty: int = 0
    degradation: str = 'none'
    jpeg_quality: int = 95
    blur_kernel: int = 3
    blur_sigma: float = .5
    motion_direction: int = 0


def plan_for(draw):
    step = stage_step(draw)
    a = ramp(step)
    r = rng_for(draw, 'plan')
    branch = str(r.choice(BRANCHES, p=branch_probabilities(step)))
    if branch == 'plain':
        return Plan(int(draw), step, branch)
    photo_p = (1-a) * np.array([2/3, 1/6, 1/6]) + a * np.array([.5, .25, .25])
    photo = r.choice(('color', 'gamma', 'noise'), p=photo_p)
    saturation = float(r.uniform(.5 + .2*a, 1.5 - .2*a))
    gamma = float(r.uniform(.85 - .05*a, 1.15 + .05*a)) if photo == 'gamma' else 1.
    noise = float(r.uniform(.003, .01)) if photo == 'noise' else 0.
    visible = branch in BRANCHES[1:3]
    affine = visible and r.random() < .5*a
    scale = float(r.uniform(.9, 1.1)) if affine else 1.
    # np.rint: ties-to-even; the rounding contract is included in the recipe.
    tx, ty = (np.rint(r.uniform(-.06, .06, 2)*256).astype(int) if affine else (0, 0))
    degradation = str(r.choice(('none', 'jpeg', 'gaussian_blur', 'motion_blur'),
                             p=[1-.4*a, .25*a, .10*a, .05*a])) if visible else 'none'
    return Plan(int(draw), step, branch, float(r.uniform(.7, 1.3)), float(r.uniform(.6, 1.4)),
                saturation, gamma, noise, scale, int(tx), int(ty), degradation,
                int(r.integers(55, 96)), int(r.choice([3, 5])), float(r.uniform(.1, 1.)),
                int(r.integers(0, 4)))
