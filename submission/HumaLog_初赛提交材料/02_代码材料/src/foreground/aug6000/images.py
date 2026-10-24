"""CPU raw255 image operations, with separate weak teacher and strong student.

Input [views, current/future=2, RGB=3, 256,256]; primary slot is explicitly 0.
No scene rendering, object insertion, frame resampling or global RNG mutation.
"""
from dataclasses import asdict, dataclass
from io import BytesIO

import numpy as np
from PIL import Image
from scipy.ndimage import convolve, maximum_filter

from .foreground import Foreground
from .policy import BRANCHES, Plan, rng_for, stage_step


def validate_raw(raw):
    a = np.asarray(raw)
    if a.dtype != np.float32 or not np.isfinite(a).all() or a.min() < 0 or a.max() > 255:
        raise ValueError('Declared finite float32 raw255 is required; no brightness-based unit guess')


def dilate(mask, radius=7):
    return maximum_filter(np.asarray(mask, bool), size=2*radius+1, mode='constant', cval=False)


def target_grid_valid(pixel_valid, grid=(16, 16), halo=7):
    """A token is valid only if ALL of its pixel support is valid.

    Mapping is for full square resize without crop. Runtime head/teacher checks
    are mandatory; Qwen's grid must never be passed here.
    """
    m = np.asarray(pixel_valid)
    if m.dtype != np.bool_ or m.ndim != 2 or m.shape != (256, 256):
        raise ValueError('Expected boolean validity in canonical 256x256 coordinates')
    if grid != (16, 16):
        raise ValueError('Only verified 16x16 full-image auxiliary grids are supported')
    valid = ~dilate(~m, halo)
    return valid.reshape(16, 16, 16, 16).all(axis=(1, 3)).reshape(-1)


def _geometry_maps(scale, tx, ty):
    if not np.isfinite(scale) or not .9 <= scale <= 1.1 or abs(tx) > 15 or abs(ty) > 15:
        raise ValueError('Affine parameters outside the designed envelope')
    yy, xx = np.indices((256, 256), dtype=np.float32)
    x = (xx - 127.5 - tx)/scale + 127.5
    y = (yy - 127.5 - ty)/scale + 127.5
    valid = (x >= 0) & (x <= 255) & (y >= 0) & (y <= 255)
    return x, y, valid


def affine(raw_pair, scale, tx, ty, foregrounds=None, entity_foregrounds=None, prevent_crop=True):
    """Inverse bilinear RGB warp. Foreground covers every contributing source pixel."""
    x, y, valid = _geometry_maps(scale, tx, ty)
    if foregrounds is not None and prevent_crop:
        # Optional legacy crop guard. Formal geometry is independent of erasure protection.
        guard=foregrounds if entity_foregrounds is None else entity_foregrounds
        fy, fx = np.where(np.logical_or.reduce(guard))
        if len(fx) and (np.any(127.5+scale*(fx-127.5)+tx < 1)
                        or np.any(127.5+scale*(fx-127.5)+tx > 254)
                        or np.any(127.5+scale*(fy-127.5)+ty < 1)
                        or np.any(127.5+scale*(fy-127.5)+ty > 254)):
            return raw_pair.copy(), np.ones((256, 256), bool), foregrounds, True
    if scale == 1. and tx == ty == 0:
        return raw_pair.copy(), valid, foregrounds, False
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    dx, dy = x-x0, y-y0
    result = np.zeros_like(raw_pair)
    warped_fg = [np.zeros((256, 256), bool) for _ in range(2)] if foregrounds is not None else None
    for ox, oy, weight in ((0, 0, (1-dx)*(1-dy)), (1, 0, dx*(1-dy)),
                           (0, 1, (1-dx)*dy), (1, 1, dx*dy)):
        ix, iy = np.clip(x0+ox, 0, 255), np.clip(y0+oy, 0, 255)
        result += raw_pair[:, :, iy, ix] * weight.astype(np.float32)[None, None]
        if warped_fg is not None:
            for t in range(2):
                warped_fg[t] |= foregrounds[t][iy, ix] & (weight > 0) & valid
    result[:, :, ~valid] = 0.
    return result.clip(0, 255), valid, warped_fg, False


def _photo(raw, p):
    if p.brightness == p.contrast == p.saturation == p.gamma == 1.:
        return raw.copy()
    unit = raw / np.float32(255.)
    unit = unit * p.brightness
    # Match pinned LingBot color arithmetic, per frame, common sampled factors.
    mean = unit.mean(axis=(-3, -2, -1), keepdims=True)
    unit = (unit-mean)*p.contrast + mean
    gray = unit.mean(axis=-3, keepdims=True)
    unit = (gray+(unit-gray)*p.saturation).clip(0, 1)
    return (unit ** p.gamma * 255.).astype(np.float32)


def _degrade(raw_pair, p):
    result = raw_pair.copy()
    if p.noise_sigma:
        for t in range(2):
            result[t] += rng_for(p.draw, f'primary_noise:{t}').normal(
                0, p.noise_sigma*255, result[t].shape).astype(np.float32)
        result = result.clip(0, 255)
    if p.degradation == 'none':
        return result
    for t in range(2):
        if p.degradation == 'jpeg':
            if not 55 <= p.jpeg_quality <= 95:
                raise ValueError('JPEG quality outside the designed range')
            pixels = np.floor(result[t].transpose(1, 2, 0).clip(0, 255)+.5).astype(np.uint8)
            stream = BytesIO()
            Image.fromarray(pixels).save(stream, format='JPEG', quality=p.jpeg_quality, subsampling=0)
            stream.seek(0)
            with Image.open(stream) as image:
                result[t] = np.asarray(image.convert('RGB'), dtype=np.float32).transpose(2, 0, 1)
        else:
            if p.degradation == 'gaussian_blur':
                if p.blur_kernel not in (3, 5) or not .1 <= p.blur_sigma <= 1.:
                    raise ValueError('Gaussian parameters outside the designed range')
                coord = np.arange(p.blur_kernel)-p.blur_kernel//2
                g = np.exp(-coord**2/(2*p.blur_sigma**2))
                kernel = np.outer(g, g)
            elif p.degradation == 'motion_blur':
                if p.motion_direction not in range(4):
                    raise ValueError('Invalid motion direction')
                kernel = np.zeros((3, 3))
                if p.motion_direction == 0: kernel[1, :] = 1
                elif p.motion_direction == 1: kernel[:, 1] = 1
                elif p.motion_direction == 2: np.fill_diagonal(kernel, 1)
                else: np.fill_diagonal(kernel[:, ::-1], 1)
            else:
                raise ValueError('Unknown degradation')
            kernel = (kernel/kernel.sum()).astype(np.float32)
            for c in range(3):
                result[t, c] = convolve(result[t, c], kernel, mode='mirror')
    return result.clip(0, 255).astype(np.float32)


@dataclass
class AugmentedPair:
    student: np.ndarray
    teacher: np.ndarray
    pixel_valid: np.ndarray
    erased: np.ndarray
    protected: np.ndarray
    auxiliary_grid: np.ndarray
    metadata: dict


def augment_pair(raw, plan: Plan, foregrounds=None, wrists_present=True, require_protection_for_geometry=False):
    validate_raw(raw)
    if raw.shape != (3, 2, 3, 256, 256) or plan.branch not in BRANCHES:
        raise ValueError('Exactly primary/left/right and current/future raw255 are required')
    if stage_step(plan.draw) != plan.step:
        raise ValueError('Draw/stage update mismatch')
    branch, fallback = plan.branch, None
    fg = None
    if foregrounds is not None:
        if len(foregrounds) != 2 or not all(isinstance(f, Foreground) for f in foregrounds):
            raise ValueError('Need current and future foreground records')
        try:
            fg = [foregrounds[t].validated(raw[0, t]) for t in range(2)]
        except ValueError as exc:
            # Asset identity failure is corruption, not an ordinary mask miss.
            if 'SHA' in str(exc): raise
            fallback = str(exc)
    if fg is not None:
        union = dilate(fg[0] | fg[1])
        union[:12,:] = True; union[-12:,:] = True
        union[:,:12] = True; union[:,-12:] = True
        if float(union.mean()) >= .8:
            fg, fallback = None, 'current_future_protection_union_at_least_80percent'
    if branch == 'protected_background_rectangle' and fg is None:
        branch, fallback = 'ordinary', fallback or 'missing_foreground'
    if branch == 'whole_primary_mask' and not wrists_present:
        branch, fallback = 'ordinary', 'missing_wrist_view'
    teacher = raw.copy() if branch == 'plain' else _photo(raw, plan)
    support = np.ones((256, 256), bool)
    affine_rejected = False
    geometry_requested = branch in ('ordinary','protected_background_rectangle') and (
        plan.scale != 1. or plan.tx != 0 or plan.ty != 0)
    if geometry_requested:
        # Erasure still needs trusted foreground; affine does not. Move both
        # current/future RGB and available masks identically; mark black borders invalid.
        teacher[0], support, fg, affine_rejected = affine(
            teacher[0], plan.scale, plan.tx, plan.ty, fg, prevent_crop=False)
    student = teacher.copy()
    if branch in ('ordinary', 'protected_background_rectangle'):
        student[0] = _degrade(student[0], plan)
    protected = np.zeros_like(support) if fg is None else dilate(fg[0] | fg[1])
    if fg is not None:
        # Added in output coordinates after semantic dilation, never crop-guarded.
        protected[:12,:]=True;protected[-12:,:]=True
        protected[:,:12]=True;protected[:,-12:]=True
    erased = np.zeros_like(support)
    attempted = plan.branch == 'protected_background_rectangle'
    attempts = 0
    if branch == 'protected_background_rectangle':
        background = support & ~protected
        r = rng_for(plan.draw, 'rectangle')
        for attempts in range(1, 11):
            area = r.uniform(.05, .20)*256**2
            aspect = np.exp(r.uniform(np.log(.5), np.log(2.)))
            h, w = int(np.rint(np.sqrt(area/aspect))), int(np.rint(np.sqrt(area*aspect)))
            if not 1 <= h <= 256 or not 1 <= w <= 256: continue
            y, x = int(r.integers(257-h)), int(r.integers(257-w))
            candidate = np.zeros_like(support)
            candidate[y:y+h, x:x+w] = True
            candidate &= background
            fraction = float(candidate.mean())
            if .05 <= fraction <= .20 and target_grid_valid(support & ~candidate).mean() >= .5:
                erased = candidate
                break
        if erased.any() and background.any():
            fill = teacher[0, 0][:, background].mean(axis=1)
            for t in range(2): student[0, t][:, erased] = fill[:, None]
        else:
            branch, fallback = 'ordinary', 'no_safe_rectangle_after_10_attempts'
    pixel_valid = np.stack([support & ~erased]*2)
    if branch == 'whole_primary_mask':
        student[0].fill(127.5)
        pixel_valid[:] = False
    grid = np.stack([target_grid_valid(m) for m in pixel_valid])
    if (erased & protected).any():
        raise RuntimeError('Foreground protection invariant failed')
    validate_raw(student)
    validate_raw(teacher)
    metadata = dict(plan=asdict(plan), applied_branch=branch, rectangle_attempted=attempted,
                    rectangle_applied=bool(erased.any()), rectangle_attempts=attempts,
                    erased_area_fraction=float(erased.mean()), aux_valid_fraction=grid.mean(axis=1).tolist(),
                    fallback=fallback, affine_rejected_to_protect_foreground=affine_rejected,
                    geometry_requested=bool(geometry_requested), geometry_applied=bool(geometry_requested and not affine_rejected),
                    geometry_protection_policy='independent_affine_erasure_protection_only_v1',
                    foreground_semantic_completeness_asserted=False,
                    image_contract='float32_raw255_student_strong_teacher_weak_v2')
    return AugmentedPair(student, teacher, pixel_valid, erased, protected, grid, metadata)
