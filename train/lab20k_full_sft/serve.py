"""Serve H50 predictions with an explicit execution prefix (default 25)."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from contract import (COMPLETE_SHA, FP32BeforeLoadMixin, MODEL, NORM_SHA, ROBOT_SHA,
                      UPSTREAM, validate_bundle, validate_observation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--execute', type=int, choices=range(1, 51), default=25)
    parser.add_argument('--port', type=int, default=18025)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--no-compile', action='store_true', help='An explicitly labeled separate inference variant')
    parser.add_argument('--runtime-receipt', type=Path, default=Path('sft6000-server-runtime.json'))
    args = parser.parse_args()
    bundle, audit = validate_bundle(args.bundle)
    upstream = args.upstream.resolve()
    actual = subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'], text=True).strip()
    if actual != UPSTREAM:
        raise ValueError('Use the pinned upstream revision '+UPSTREAM)
    protected = ['deploy/lingbot_vla_v2_policy.py', 'deploy/websocket_policy_server.py', 'lingbotvla']
    dirty = subprocess.check_output(['git','-C',str(upstream),'status','--porcelain','--',*protected], text=True)
    if dirty.strip():
        raise ValueError('The upstream inference source has local modifications; use a clean checkout')
    os.environ['QWEN3VL_PATH'] = str(bundle/'processor/Qwen3-VL-4B-Instruct')
    sys.path.insert(0, str(upstream))
    import numpy as np
    import torch
    import yaml
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    from deploy.websocket_policy_server import WebsocketPolicyServer
    from lingbotvla.data.vla_data.utils import FeatureTransform
    torch.set_float32_matmul_precision('high')

    class DeliveryServer(FP32BeforeLoadMixin, LingbotVLAv2Server):
        delivery_audit = audit

        def reset(self, robo_name, path_to_pi_model=None):
            if robo_name != 'robotwin' or path_to_pi_model is not None:
                raise ValueError('Reset this fixed robotwin model; changing checkpoints requires a new server')
            self.robot_config = yaml.safe_load((bundle/'assets/robotwin_relative.yaml').read_text())
            self.global_step = 0
            self.last_action_chunk = None
            self.last_normalized_action_chunk = None
            self.vla.feature_transform = FeatureTransform(str(bundle/'assets/robotwin_relative.yaml'),
                self.data_config, self.config, self.processor, chunk_size=50,
                norm_stats_path=str(bundle/'assets/norm_stats.json'))
            if self.vla.feature_transform.action_subtract_state != {
                    'action.arm.position': True, 'action.effector.position': False}:
                raise ValueError('Wrong relative-arm/absolute-gripper transform')
            self.action_key = self.vla.feature_transform.org_features['actions']

        def infer(self, observation, **kwargs):
            if not observation.get('reset'):
                for obs in observation.get('batch', [observation]):
                    validate_observation(obs)
            result = super().infer(observation, **kwargs)
            if not observation.get('reset'):
                shape = (len(observation['batch']), args.execute, 14) if 'batch' in observation else (args.execute, 14)
                action = np.asarray(result['action'])
                if action.shape != shape or not np.isfinite(action).all():
                    raise ValueError('Wrong finite physical execution prefix')
            return result

    # Keep downloaded bytes immutable. Rebind machine paths in a disposable
    # loader layout, whose weights symlink back to the unchanged HF snapshot.
    with tempfile.TemporaryDirectory(prefix='lab20k-sft6000-') as scratch:
        root = Path(scratch)
        model = root/MODEL
        model.parent.mkdir(parents=True)
        model.symlink_to(bundle/MODEL, target_is_directory=True)
        cli = yaml.safe_load((bundle/'lingbotvla_cli.yaml').read_text())
        cli['data']['robot_config_path'] = str(bundle/'assets/robotwin_relative.yaml')
        cli['data']['norm_stats_file'] = str(bundle/'assets/norm_stats.json')
        cli['train']['use_compile'] = not args.no_compile
        (root/'lingbotvla_cli.yaml').write_text(yaml.safe_dump(cli, sort_keys=False))
        base = DeliveryServer(path_to_pi_model=str(model), use_length=args.execute,
            robot_norm_path=str(bundle/'assets/norm_stats.json'), chunk_ret=True,
            use_bf16=False, use_fp32=True, use_compile=not args.no_compile)
        if (base.config.chunk_size, base.config.num_steps, base.data_config.img_size) != (50,10,256):
            raise ValueError('Expected H50 / 10 flow steps / image size 256')
        if base.config.vit_attn_implementation != 'flash_attention_2':
            raise ValueError('Expected the official FlashAttention2 vision backend')
        if not all(p.dtype == torch.float32 and p.device.type == 'cuda' for p in base.vla.parameters()):
            raise ValueError('Expected the actual complete FP32 CUDA model')
        base.reset('robotwin')
        metadata = dict(model='Lab20k full-SFT +6000', source_complete_sha256=COMPLETE_SHA,
            predicted_horizon=50, execute=args.execute, flow_steps=10, image_size=256,
            checkpoint_dtype='float32', official_moe=True, use_compile=not args.no_compile,
            matmul_precision=torch.get_float32_matmul_precision(), upstream_revision=UPSTREAM,
            norm_sha256=NORM_SHA, robot_sha256=ROBOT_SHA,
            relative_arms=True, absolute_grippers=True, rgb='uint8 HWC RGB 0..255',
            process_seed=42, episode_reseed=False, **base.loaded_value_verification)
        args.runtime_receipt.parent.mkdir(parents=True, exist_ok=True)
        args.runtime_receipt.write_text(json.dumps(metadata, indent=2)+'\n')
        print(json.dumps(dict(event='ready', **metadata)), flush=True)
        WebsocketPolicyServer(base, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == '__main__':
    main()
