"""Fixed-input action audit shared by G2 export and independent reload."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch


FIXTURE_KEYS = ("images", "img_masks", "lang_tokens", "lang_masks", "image_grid_thw", "state")


def load_action_fixture(path, device):
    examples = torch.load(path, map_location="cpu", weights_only=True)
    observation, noise = examples[0]["observation"], examples[0]["noise"]
    if tuple(noise.shape) != (1, 50, 55):
        raise ValueError(f"Expected fixed [1,50,55] audit noise, got {tuple(noise.shape)}")
    batch = {key: observation[key].unsqueeze(0).to(device) for key in FIXTURE_KEYS}
    batch["noise"] = noise.to(device).clone()
    return observation, batch


def physical_action(normalized, observation, processor, data_config, robot_config, model_config):
    """Use the deployment FeatureTransform inverse, not a hand-written norm formula."""
    from lingbotvla.data.vla_data.utils import FeatureTransform

    transform = FeatureTransform(
        str(robot_config), SimpleNamespace(**data_config), model_config, processor,
        chunk_size=model_config.chunk_size, norm_stats_path=data_config["norm_stats_file"],
        image_augment=False, use_depth_align=False, use_future_image=False,
    )
    result = transform.unapply({
        "state": observation["state"].float().clone(),
        "actions": normalized.float().cpu().clone(),
        "state_joint_mask": observation["state_joint_mask"].bool().clone(),
        "action_joint_mask": observation["action_joint_mask"].bool().clone(),
    })
    # reverse_features maps the two robot features back to the simulator's
    # original 14-D `action` layout (grippers at indices 6 and 13).
    physical = torch.as_tensor(result["action"], dtype=torch.float32)
    if tuple(physical.shape) != (50, 14) or not torch.isfinite(physical).all():
        raise ValueError(f"Invalid physical action audit shape or values: {tuple(physical.shape)}")
    return physical


def save_action_reference(model, processor, data_config, robot_config, validation_loader, output, context):
    """All ranks run the same fixed inference through FSDP; rank 0 saves both action spaces."""
    sample = next(iter(validation_loader))
    observation = {key: sample[key][0].detach().cpu() for key in FIXTURE_KEYS}
    action_mask = sample["joint_mask"][0].bool().any(dim=0).cpu()
    if int(action_mask.sum()) != 14:
        raise ValueError("Expected 14 physical action dimensions in G2 validation fixture")
    observation["action_joint_mask"] = action_mask
    observation["state_joint_mask"] = action_mask.clone()
    noise = torch.randn((1, 50, 55), generator=torch.Generator().manual_seed(20261012))
    fixture_path = output.with_name("action_audit_fixture.pt")
    if context.is_main:
        torch.save([{"observation": observation, "noise": noise}], fixture_path)
    context.barrier()
    observation, batch = load_action_fixture(fixture_path, context.device)
    policy = model.policy
    policy.config.use_cache = True
    policy.model.config.use_cache = True
    policy.model._use_compile_predict_velocity = False
    # Upstream precomputes the image grid on the validation batch (8 samples).
    # The fixed audit fixture contains one sample, so invalidate that cache
    # before the next forward; upstream will rebuild it for the fixture grid.
    vision = policy.model.qwenvl_with_expert
    for name in ("pos_embeds", "position_embeddings", "cu_seqlens",
                 "visual_split_sizes", "visual_max_seqlen"):
        setattr(vision, name, None)
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        normalized = model(batch, reduction="sample")
    model.train()
    if tuple(normalized.shape) != (1, 50, 55) or not torch.isfinite(normalized).all():
        raise ValueError("Invalid normalized G2 action before export")
    if context.is_main:
        physical = physical_action(normalized[0].detach().cpu(), observation, processor,
                                   data_config, robot_config, policy.config)
        torch.save({"normalized": normalized[0].detach().float().cpu(),
                    "physical": physical, "fixture": str(fixture_path),
                    "dtype": "bf16 autocast", "backend": "torch reference MoE / SDPA"}, output)
        print(json.dumps({"action_audit_reference": str(output),
                          "normalized_shape": [50, 55], "physical_shape": [50, 14]}), flush=True)
    context.barrier()
