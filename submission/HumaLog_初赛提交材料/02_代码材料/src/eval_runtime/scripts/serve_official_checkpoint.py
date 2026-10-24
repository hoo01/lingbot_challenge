"""Serve pinned upstream inference with fixed defaults and declared construction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

p=argparse.ArgumentParser()
p.add_argument('--run',type=Path,required=True)
p.add_argument('--port',type=int,default=18006)
a=p.parse_args()
manifest=json.loads((a.run/'official-defaults-manifest.json').read_text())
source=Path(manifest['source'])
for name,digest in manifest['pristine_source_sha256'].items():
    assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest, name
sys.path.insert(0,str(source))
import torch
from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
from deploy.websocket_policy_server import WebsocketPolicyServer
from eval_platform.contracts import model_construction, checkpoint_load_dtype
from eval_platform.robot_contract import BoundRobotConfigMixin

construction=model_construction(manifest['checkpoint'])
load_dtype=checkpoint_load_dtype(manifest['checkpoint'])
if manifest['kind']=='original_pretrained_base_official_inference_defaults' and construction!='native':
    raise ValueError('Original base must use native construction')


class ExportServer(BoundRobotConfigMixin,LingbotVLAv2Server):
    def _prepare_model_input(self, observation):
        import numpy as np
        capture=not (a.run/'first-preprocessing.json').exists()
        raw={k:np.asarray(v).copy() for k,v in observation.items() if isinstance(v,np.ndarray)} if capture else {}
        value=super()._prepare_model_input(observation)
        if capture:
            for key in ('observation.images.cam_high','observation.images.cam_left_wrist','observation.images.cam_right_wrist'):
                assert raw[key].dtype==np.uint8 and raw[key].ndim==3 and raw[key].shape[-1]==3
            assert raw['observation.state'].shape==(14,) and np.isfinite(raw['observation.state']).all()
            np.savez_compressed(a.run/'first-raw-input.npz',**raw,task=np.asarray(observation['task']))
            from normalization_check import check_roundtrip
            roundtrip=check_roundtrip(self.vla.feature_transform,raw)
            summary={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()) for k,v in value.items() if isinstance(v,torch.Tensor)}
            (a.run/'first-preprocessing.json').write_text(json.dumps(dict(tensors=summary,roundtrip=roundtrip,raw_rgb_uint8=True,single_native_processor=True,image_augmentation=False),indent=2))
        return value

    def load_model_weights(self, path_to_pi_model, strict=True):
        if construction=='bf16_then_fp32':
            # Restore construction-time rounding of nonpersistent buffers BEFORE
            # loading saved FP32 tensors, so learned weights are never rounded.
            self.vla.bfloat16().float()
        elif load_dtype=='float32':
            # HF Qwen child modules otherwise round FP32 weights into their
            # initial BF16 parameters before the upstream constructor promotes.
            self.vla.float()
        return super().load_model_weights(path_to_pi_model, strict=strict)


base=ExportServer(path_to_pi_model=manifest['checkpoint'],use_length=50,
    robot_norm_path=manifest['normalization'],
    chunk_ret=True,use_bf16=False,use_fp32=True,use_compile=True)
assert base.data_config.img_size==256 and base.config.num_steps==10
assert base.config.vit_attn_implementation=='flash_attention_2'
assert all(param.device.type=='cuda' for param in base.vla.parameters())
assert hashlib.sha256(Path(base.robot_norm_path).read_bytes()).hexdigest()==manifest['normalization_sha256']
if getattr(base.data_config,'robot_config_path',None):
    base.reset('robotwin')
from collections import Counter
from eval_platform.weight_precision_audit import verify_loaded_parameters
learned_precision_check = verify_loaded_parameters(base.vla,manifest['checkpoint'])
parameter_dtypes = Counter()
parameter_tensors = Counter()
for parameter in base.vla.parameters():
    parameter_dtypes[str(parameter.dtype)] += parameter.numel()
    parameter_tensors[str(parameter.dtype)] += 1
buffer_inventory = {}
for name, buffer in base.vla.named_buffers():
    if name.endswith('rotary_emb.inv_freq') or name.endswith('rotary_pos_emb.inv_freq'):
        value = buffer.detach().cpu().contiguous()
        buffer_inventory[name.replace('_orig_mod.','')] = dict(dtype=str(value.dtype),shape=list(value.shape),
            sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())
metadata={'checkpoint':manifest['checkpoint'],'dtype':'fp32','execute':25,'prediction_horizon':50,
          'execution_protocol':'H50_first25_reobserve_v1',
          'actual_parameter_dtypes_numel':dict(parameter_dtypes),
          'actual_parameter_dtypes_tensors':dict(parameter_tensors),
          'learned_parameter_precision_check':learned_precision_check,
          'rotary_buffers':buffer_inventory,
          'matmul_precision':torch.get_float32_matmul_precision(),'moe_backend':'official',
          'vision_device':'cuda','embedding_device':'cuda','flow_noise_generation':'upstream_native',
          'sampling_seed_at_process_start':42,'sampling_seed_per_episode':None,
          'use_compile':True,'image_size':256,'num_steps':10,
          'vit_attn_implementation':base.config.vit_attn_implementation,
          'precompute_grid_thw':base.config.precompute_grid_thw,
          'inference_profile':'aug16000_h50_execute25_no_video_v1','upstream_revision':manifest['official_revision'],
          'nonpersistent_buffer_initialization':construction,
          'checkpoint_parameter_load_dtype':load_dtype or 'upstream_constructor',
          'cuda_physical_uuid':str(torch.cuda.get_device_properties(0).uuid),
          'frozen_protocol_sha256':manifest['frozen_protocol_sha256'],
          'model_revision':manifest['model_revision'],
          'normalization_sha256':manifest['normalization_sha256']}
if hasattr(base,'robot_action_contract'):
    metadata['robot_action_contract'] = base.robot_action_contract
# Optional allocator housekeeping for a model and simulator sharing one GPU.
# It does not change weights, inference, returned actions or model RNG. Live
# tensors and compiled graph buffers stay allocated; only unused cache is freed.
release_cache=os.environ.get('LINGBOT_RELEASE_UNUSED_CUDA_CACHE')=='1'
if release_cache:
    metadata['unused_cuda_cache_release']='after_policy_request'
    upstream_infer=base.infer
    def infer_with_cache_release(*args,**kwargs):
        response=upstream_infer(*args,**kwargs)
        before=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
        torch.cuda.empty_cache()
        after=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
        with (a.run/'allocator-cache.jsonl').open('a') as log:
            log.write(json.dumps(dict(before=before,after=after))+'\n')
        return response
    base.infer=infer_with_cache_release
(a.run/'server-runtime.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps({'event':'official_server_ready','metadata':metadata}),flush=True)
# No EpisodeSeededPolicy, sampler replacement, CPU placement or MoE monkeypatch.
WebsocketPolicyServer(base,host='127.0.0.1',port=a.port,metadata=metadata).serve_forever()
