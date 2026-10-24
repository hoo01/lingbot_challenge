"""Continuation recipe: official sampling/BF16, own original relative actions."""
from policy import ASSETS,ROOT,SOURCE,STEPS,SAVE_EVERY,UPSTREAM,GLOBAL_BATCH,PEAKS,WORLD,MICRO,ACCUM,SAMPLING


def make_recipe(run,microbatch,accum):
    from base_recipe import make_recipe as original_recipe
    assert (microbatch,accum)==(MICRO,ACCUM)
    cli=original_recipe(UPSTREAM,ASSETS,ROOT,run,microbatch,accum,STEPS)
    train=cli['train']
    train.update(optimizer='adamw',use_moe_expert_lr=False,lr=2e-5,lr_min=4e-6,
        lr_start=0.,lr_warmup_ratio=.05,lr_decay_style='cosine',
        weight_decay=0.,save_steps=SAVE_EVERY,max_steps=STEPS,
        adam_betas=[.9,.95],adam_epsilon=1e-8,foreach=False,fused=False,
        freeze_vision_encoder=False,train_expert_only=False,
        enable_fp32=False,action_fp32=False,global_batch_size=GLOBAL_BATCH,
        micro_batch_size=microbatch,gradient_accumulation_steps=accum)
    cli['full_sft']=dict(warm_start=str(SOURCE),steps=STEPS,global_batch=GLOBAL_BATCH,
        world=WORLD,peak_learning_rates=PEAKS,no_lora=True,frozen_teachers=True,
        image_contract='raw255_external_augmentation_v1',sampling=SAMPLING,
        master_dtype='float32',fsdp_compute_dtype='bfloat16',fsdp_reduce_dtype='float32')
    cli['data'].update(image_augment=False,num_workers=2,prefetch_factor=2,pin_memory=True)
    return cli
