"""Official full-microbatch frozen teachers; no per-example serialization."""
import torch


def targets_for_batch(align,batch,teachers=None):
    from lingbotvla.models.vla.vision_models.module_utils import (
        build_depth_model,build_video_model,get_depth_target,get_video_target)
    if teachers is None:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            moge,depth=build_depth_model(align)
            video=build_video_model(align['video'])
        teachers=(moge,depth,video)
    assert all(not p.requires_grad for teacher in teachers for p in teacher.parameters())
    current=batch.pop('pil_images');future=batch.pop('future_pil_images')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        depth,_=get_depth_target('MoRGBD',teachers[:2],current)
        future_depth,_=get_depth_target('MoRGBD',teachers[:2],future)
        video=get_video_target(teachers[2],current,future,align['video'])
    targets=dict(depth_targets=depth,future_depth_targets=future_depth,
        future_video_targets=video['patch'],future_video_cls_targets=video.get('cls'),
        future_video_current_patch=video['current_patch'])
    assert all(x is None or (not x.requires_grad and torch.isfinite(x).all()) for x in targets.values())
    return teachers,targets
