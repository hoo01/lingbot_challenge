"""Replace only auxiliary reductions; preserve heads, parameter names and Adam.

Called on the normal complete model output. No module hooks, mutable forward
contexts or rank-dependent skipped forwards are used, including during backward.
"""
import torch
import torch.nn.functional as F


def masked_mean(token_errors, valid):
    if token_errors.ndim != 2 or valid.shape != token_errors.shape or valid.dtype != torch.bool:
        raise ValueError('Need [batch,tokens] errors and boolean validity')
    if token_errors.shape[0] == 0 or not torch.isfinite(token_errors).all():
        raise ValueError('Empty or nonfinite auxiliary errors')
    valid = valid.to(token_errors.device)
    weights = valid.to(token_errors.dtype)
    # Original batch denominator, including all-invalid samples. Connected zero.
    return ((token_errors * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)).mean()


def _aligned(prediction, target, valid):
    if (prediction.ndim != 3 or prediction.shape != target.shape
            or prediction.shape[1] != 256 or valid.shape != prediction.shape[:2]):
        raise ValueError('Auxiliary target/prediction must match [B,256,C]; no silent reshape/broadcast')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite auxiliary features')
    return prediction.float(), target.detach().to(device=prediction.device).float()


def depth_loss(prediction, target, valid):
    p, t = _aligned(prediction, target, valid)
    return masked_mean(F.smooth_l1_loss(p, t, reduction='none').mean(dim=-1), valid)


def video_loss(prediction, target, valid):
    target = target.to(dtype=prediction.dtype)
    p, t = _aligned(prediction, target, valid)
    mse = F.mse_loss(p, t, reduction='none').mean(dim=-1)
    # Match the pinned LingBot helper's TWO normalizations, including eps.
    pnorm, tnorm = F.normalize(p, dim=-1, eps=1e-6), F.normalize(t, dim=-1, eps=1e-6)
    cosine = 1 - F.cosine_similarity(pnorm, tnorm, dim=-1, eps=1e-6)
    return masked_mean(mse, valid) + .2 * masked_mean(cosine, valid)


def validate_alignment_config(align):
    video, depth = align.get('video', {}), align.get('depth', {})
    expected = dict(use_patch_loss=True, use_current_patch_loss=True, use_cls_loss=False,
                    use_smooth_l1_loss=False, use_mse_loss=True, use_cosine_loss=True,
                    mse_loss_weight=1., cosine_loss_weight=.2, future_video_loss_weight=.004,
                    num_backbone_tokens=256, num_future_frames=1, input_size=256)
    if any(video.get(k) != v for k, v in expected.items()):
        raise ValueError('Video/CLS/teacher configuration differs from the verified recipe')
    if (align.get('mode') != 'query' or align.get('depth_loss_weight') != .004
            or align.get('future_depth_loss_weight') != .004
            or depth.get('token_size') != 16 or depth.get('num_backbone_tokens') != 256
            or depth.get('model_type') != 'MoRGBD' or depth.get('use_future_depth') is not True):
        raise ValueError('Depth auxiliary layout/coefficients differ from the verified recipe')


def replace_auxiliary_losses(outputs, targets, validity, align):
    """Use after the normal full forward, before backward; model object unchanged.

    validity is [B,current/future=2,256]. Grid layout must separately pass the
    actual teacher preprocessing gate. This function cannot infer that geometry.
    """
    validate_alignment_config(align)
    if len(outputs) != 11 or validity.ndim != 3 or validity.shape[1:] != (2, 256):
        raise ValueError('Unexpected LingBot forward output or auxiliary validity layout')
    if validity.dtype != torch.bool:
        raise ValueError('Auxiliary validity must be boolean')
    current, future = validity[:, 0], validity[:, 1]
    new_depth = .004 * depth_loss(outputs[7], targets['depth_targets'], current)
    new_future = .004 * depth_loss(outputs[8], targets['future_depth_targets'], future)
    new_video = .004 * (video_loss(outputs[9], targets['future_video_targets'], future)
                         + video_loss(outputs[10], targets['future_video_current_patch'], current))
    new = (new_depth, new_future, new_video)
    result = list(outputs)
    # Parent total still carries differentiable action/router/sequence terms.
    # The old auxiliary terms cancel in autograd; new reductions carry their gradients.
    result[0] = outputs[0].float() + sum(n - old.float() for n, old in zip(new, outputs[2:5]))
    result[2:5] = new
    result[6] = dict(outputs[6])
    # Replace stale upstream auxiliary metrics rather than presenting unmasked ones.
    for key in list(result[6]):
        if key.startswith('align/'):
            del result[6][key]
    result[6].update(masked_depth=new_depth.detach(), masked_future_depth=new_future.detach(),
                     masked_video=new_video.detach(),
                     auxiliary_valid_fraction=validity.float().mean(dim=-1).detach())
    return tuple(result)
