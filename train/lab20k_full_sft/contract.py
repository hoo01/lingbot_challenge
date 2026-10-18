"""Step-bound inference contract; deliberately independent of training/eval services."""
import hashlib
import json
from pathlib import Path

NORM_SHA = 'b24e6a7ccf44fa3ecc9dbbe060cf9b0937cf3307bc2a68232bc98c4578c22f6d'
ROBOT_SHA = '1ef96ff0d213561174813345f8674fc693f4831cde144e00e5b9781d7147876b'
COMPLETE_SHA = '56554a1a222c3a83295bd8a5f3733205c496fb1c8d34d4ba13346555cf505cff'
UPSTREAM = 'be969b8fd117fb70550c5d4bf4bc328211b5b1b6'
MODEL = Path('checkpoints/global_step_6000/hf_ckpt_fp32')
CAMERAS = ('observation.images.cam_high', 'observation.images.cam_left_wrist',
           'observation.images.cam_right_wrist')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_bundle(bundle):
    bundle = Path(bundle).resolve()
    release = json.loads(Path(__file__).with_name('release.json').read_text())
    if sha(bundle/'delivery_manifest.json') != release['delivery_manifest_sha256']:
        raise ValueError('Manifest differs from this fixed HF release')
    manifest = json.loads((bundle/'delivery_manifest.json').read_text())
    if manifest['step'] != 6000 or manifest['source_complete_sha256'] != COMPLETE_SHA:
        raise ValueError('This helper is bound to the published full-SFT +6000 checkpoint')
    for relative, proof in manifest['files'].items():
        path = bundle/relative
        if not path.resolve().is_relative_to(bundle) or path.stat().st_size != proof['bytes']:
            raise ValueError('Missing or wrong-size bundle file: '+relative)
        # Large weights are checked by HF's immutable LFS download and again as
        # exact tensors when loaded. All configuration bytes are checked here.
        if not relative.endswith('.safetensors') and sha(path) != proof['sha256']:
            raise ValueError('Bundle metadata SHA mismatch: '+relative)
    if sha(bundle/'assets/norm_stats.json') != NORM_SHA or sha(bundle/'assets/robotwin_relative.yaml') != ROBOT_SHA:
        raise ValueError('Wrong model-specific normalization or robot mapping')
    audit = json.loads((bundle/MODEL/'runtime-parameter-audit.json').read_text())
    if not (audit['source_checkpoint_step'] == 6000 and audit['source_complete_sha256'] == COMPLETE_SHA
            and audit['parameter_elements'] == 6375906359 and audit['all_export_readback_equal']):
        raise ValueError('Wrong source/export audit')
    return bundle, audit


def validate_observation(observation):
    import numpy as np
    for key in CAMERAS:
        image = np.asarray(observation[key])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(key+' must be uint8 HWC RGB in original 0..255 units')
    state = np.asarray(observation['observation.state'])
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError('Expected the finite original 14-joint physical state')
    if not isinstance(observation.get('task'), str) or not observation['task'].strip():
        raise ValueError('Expected the original nonempty task instruction')


def verify_loaded(model, audit):
    """Run once before serving, on actual loaded parameters and persistent buffers."""
    import torch
    expected = {**audit['trained_parameter_sha256'], **audit['persistent_buffer_sha256']}
    state = model.state_dict()
    parameters = list(model.parameters())
    if len(parameters) != 1672 or sum(p.numel() for p in parameters) != 6375906359:
        raise ValueError('Wrong actual model parameter count')
    if len(audit['trained_parameter_sha256']) != 1672 or len(audit['persistent_buffer_sha256']) != 36:
        raise ValueError('Wrong tensor inventory')
    if set(state) != set(expected):
        raise ValueError('Loaded persistent tensor keys differ from the exported checkpoint')
    for name, tensor in state.items():
        value = tensor.detach().cpu().contiguous()
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError('Expected finite FP32 saved tensor: '+name)
        if hashlib.sha256(value.numpy().tobytes()).hexdigest() != expected[name]:
            raise ValueError('Loaded saved value differs: '+name)
    return dict(parameter_tensors=1672, persistent_buffers=36,
                parameter_elements=6375906359, dtype='float32', all_saved_values_equal=True)


class FP32BeforeLoadMixin:
    def load_model_weights(self, path_to_pi_model, strict=True):
        # Upstream Qwen children initially use BF16. Promote BEFORE copying saved
        # FP32 values, so FP32 -> BF16 -> FP32 rounding cannot lose learned bits.
        self.vla.float()
        result = super().load_model_weights(path_to_pi_model, strict=strict)
        self.loaded_value_verification = verify_loaded(self.vla, self.delivery_audit)
        return result
