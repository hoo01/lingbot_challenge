"""Serve untouched upstream inference with the RoboTwin release defaults."""
import argparse
import hashlib
import json
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

base=LingbotVLAv2Server(path_to_pi_model=manifest['checkpoint'],use_length=50,
    robot_norm_path=manifest['normalization'],
    chunk_ret=True,use_bf16=False,use_fp32=True,use_compile=True)
assert base.data_config.img_size==256 and base.config.num_steps==10
assert base.config.vit_attn_implementation=='flash_attention_2'
assert all(param.device.type=='cuda' for param in base.vla.parameters())
assert hashlib.sha256(Path(base.robot_norm_path).read_bytes()).hexdigest()==manifest['normalization_sha256']
metadata={'checkpoint':manifest['checkpoint'],'dtype':'fp32','execute':50,
          'matmul_precision':torch.get_float32_matmul_precision(),'moe_backend':'official',
          'vision_device':'cuda','embedding_device':'cuda','flow_noise_generation':'upstream_native',
          'sampling_seed_at_process_start':42,'sampling_seed_per_episode':None,
          'use_compile':True,'image_size':256,'num_steps':10,
          'vit_attn_implementation':base.config.vit_attn_implementation,
          'precompute_grid_thw':base.config.precompute_grid_thw,
          'inference_profile':'official_defaults_v1','upstream_revision':manifest['official_revision'],
          'normalization_sha256':manifest['normalization_sha256']}
(a.run/'server-runtime.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps({'event':'official_server_ready','metadata':metadata}),flush=True)
# No EpisodeSeededPolicy, sampler replacement, CPU placement or MoE monkeypatch.
WebsocketPolicyServer(base,host='127.0.0.1',port=a.port,metadata=metadata).serve_forever()
