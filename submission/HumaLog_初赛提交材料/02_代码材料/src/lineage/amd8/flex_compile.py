"""Compile only the pinned Flex attention operator; preserve its math contract."""
import importlib
import os
import torch

MODULE = 'lingbotvla.models.vla.lingbot_vla.flex_attention'
_original = None
_compiled = None
_choices_context = None
TARGET=os.environ.get('FLEX_COMPILE_TARGET','operator')
assert TARGET in ('operator','wrapper')


def setup():
    global _original, _compiled, _choices_context
    module = importlib.import_module(MODULE)
    if _original is None:
        if os.environ.get('FLEX_AMD_KPACK1')=='1':
            from dataclasses import replace
            from torch._inductor.choices import InductorChoices
            from torch._inductor.virtualized import V
            class LocalChoices(InductorChoices):
                def get_flex_attention_bwd_configs(self,head_dim,dtype,device_type='cuda'):
                    configs=super().get_flex_attention_bwd_configs(head_dim,dtype,device_type)
                    if dtype==torch.float32 and torch.version.hip:
                        return [replace(c,kpack=1) for c in configs]
                    return configs
            _choices_context=V.set_choices_handler(LocalChoices())
            _choices_context.__enter__()
        _original = module.flex_attention if TARGET=='operator' else module.flex_attention_with_block_mask
        _compiled = torch.compile(_original, backend='inductor', fullgraph=True,
                                  dynamic=False, mode='default')
    return module, _original, _compiled


def select(compiled):
    module, original, optimized = setup()
    selected=optimized if compiled else original
    if TARGET=='operator':module.flex_attention=selected
    else:
        module.flex_attention_with_block_mask=selected
        modeling=importlib.import_module('lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2')
        modeling.flex_attention_with_block_mask=selected


def implementation():
    return dict(scope='only_flex_attention_'+TARGET, backend='inductor',
                fullgraph=True, dynamic=False, mode='default',
                original_fp32_qkv=True, original_kernel_options=True,
                original_block_mask_and_scale=True, outer_model_compile=False,
                rocm_fp32_backward_kpack=1 if os.environ.get('FLEX_AMD_KPACK1')=='1' else 'installed_default',
                eager_fallback_allowed=False)
