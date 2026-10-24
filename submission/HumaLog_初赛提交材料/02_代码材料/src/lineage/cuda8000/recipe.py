"""Declare the accepted dense full-SFT stage without changing shared source."""
from policy import ASSETS, ROOT, SOURCE, STEPS, SAVE_EVERY, UPSTREAM, GLOBAL_BATCH, PEAKS


def make_recipe(run, microbatch, accum):
    from base_recipe import make_recipe as original_recipe
    assert microbatch * accum * 4 == GLOBAL_BATCH
    cli = original_recipe(UPSTREAM, ASSETS, ROOT, run, microbatch, accum, STEPS)
    train = cli['train']
    train.update(optimizer='adamw', use_moe_expert_lr=False, lr=2e-5, lr_min=4e-6,
        lr_start=0., lr_warmup_ratio=.05, lr_decay_style='cosine',
        weight_decay=0., save_steps=SAVE_EVERY, max_steps=STEPS,
        adam_betas=[.9, .95], adam_epsilon=1e-8, foreach=False, fused=False,
        freeze_vision_encoder=False, train_expert_only=False)
    cli['full_sft'] = dict(warm_start=str(SOURCE), steps=STEPS, global_batch=GLOBAL_BATCH,
        peak_learning_rates=PEAKS, no_lora=True, frozen_teachers=True,
        image_contract='raw255_external_augmentation_v1',
        sampling='original_frame_probability_times_target_exposure_v4')
    # Reader owns all augmentation. This disables only the second/internal pass.
    cli['data']['image_augment'] = False
    train['action_fp32'] = False
    cli['full_sft'].update(source_stage_step=8000, sampling='filtered_official_frame_shuffle_v1', optimizer_state='inherited_AdamW')
    return cli
