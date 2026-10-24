"""Declared recipe around the pinned official YAML and optimizer defaults."""
import ast
import bisect
from pathlib import Path

UPSTREAM_REVISION = 'be969b8fd117fb70550c5d4bf4bc328211b5b1b6'
BASE_REVISION = '11c703bf6a5c1f45b3b69168482da11fdbba53d7'
NORM_SHA256 = 'b24e6a7ccf44fa3ecc9dbbe060cf9b0937cf3307bc2a68232bc98c4578c22f6d'
LORA_RANK = 256
LORA_ALPHA = 512
LANGUAGE_SUFFIXES = {'q_proj', 'v_proj'}


def frame_location(ends, index):
    if not isinstance(index, int) or index < 0 or index >= ends[-1]:
        raise IndexError(index)
    episode = bisect.bisect_right(ends, index)
    return episode, index - (ends[episode - 1] if episode else 0)


def is_language_target(name):
    return (name.startswith('model.language_model.layers.')
            and name.rsplit('.', 1)[-1] in LANGUAGE_SUFFIXES)


def optimizer_defaults(repo):
    """Read literal defaults without initializing training/distributed state."""
    source = Path(repo) / 'lingbotvla/utils/arguments.py'
    cls = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'TrainingArguments')
    selected = {'weight_decay', 'lr_start', 'lr_warmup_ratio', 'lr_decay_ratio',
                'muon_momentum', 'muon_nesterov', 'muon_ns_steps',
                'muon_adjust_lr_fn', 'muon_exclude_name_patterns', 'max_grad_norm'}
    defaults = {}
    for node in cls.body:
        if not isinstance(node, ast.AnnAssign) or node.target.id not in selected:
            continue
        kw = {x.arg: x.value for x in node.value.keywords}
        defaults[node.target.id] = ([] if 'default_factory' in kw
                                   else ast.literal_eval(kw['default']))
    assert set(defaults) == selected
    return defaults


def make_recipe(repo, assets, root, run, batch, accum, steps):
    import yaml
    cli = yaml.safe_load((Path(repo) / 'configs/vla/robotwin/robotwin.yaml').read_text())
    train = {**optimizer_defaults(repo), **cli['train']}
    cli['train'] = train
    cli['model'].update(model_path=str(assets / 'models/lingbot-vla-v2-6b'),
                        tokenizer_path=str(assets / 'models/Qwen3-VL-4B-Instruct'))
    # Hardware execution settings are explicit; optimization groups remain official.
    train.update(freeze_vision_encoder=False, train_expert_only=False,
        skip_auxiliary_loss=False, enable_full_shard=True,
        enable_gradient_checkpointing=True, enable_fp32=True, action_fp32=True,
        use_compile=False, attention_implementation='flex_cached',
        vit_attn_implementation='sdpa', precompute_grid_thw=False,
        micro_batch_size=batch, global_batch_size=4 * batch * accum,
        gradient_accumulation_steps=accum, output_dir=str(run),
        max_steps=steps, save_steps=2000, loss_type='fm')
    for key in ('sequence_wise_loss_coeff', 'router_z_loss_coeff', 'lr', 'lr_min'):
        train[key] = float(train[key])
    assert train['optimizer'] == 'muon' and train['use_moe_expert_lr']
    assert train['bias_update_speed'] == 0
    assert train['sequence_wise_loss_coeff'] == 1e-3
    assert train['router_z_loss_coeff'] == 1e-4
    align = train['align_params']
    align['depth'].update(moge_path=str(assets / 'models/moge-2-vitb-normal/model.pt'),
        morgbd_path=str(assets / 'models/teachers/depth/model.pt'))
    align['video'].update(ckpt_path=str(assets / 'models/teachers/dino_video/teacher_step_10000.pth'),
        config_path=str(assets / 'models/teachers/dino_video/config.yaml'))
    assert align['depth']['use_future_depth'] and align['use_future_video']
    assert align['video']['use_current_patch_loss']
    cli['data'].update(use_future_image=True, image_augment=True, img_size=256,
        dataset_format='official_raw_hdf5', train_path=str(root / 'artifacts/manifest.json'),
        norm_stats_file=str(root / 'artifacts/norm_stats.json'),
        robot_config_root=str(root / 'artifacts'),
        robot_config_path=str(root / 'artifacts/robotwin_relative.yaml'),
        joints=[str(x) for x in cli['data']['joints']],
        norm_type=[str({'arm.position': 'meanstd'}), str({'end.position': 'meanstd'}),
                   str({'effector.position': 'minmax'})], drop_last=True)
    return cli
