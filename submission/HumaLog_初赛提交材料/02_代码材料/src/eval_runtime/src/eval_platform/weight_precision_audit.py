"""Only the hash-bound dense AMD8 SFT+8000 checkpoint may run in this new root."""
import hashlib
import json
import os
from pathlib import Path


def verify_loaded_parameters(model, checkpoint):
    import torch
    root=Path(os.environ['LINGBOT_ROOT']).resolve()
    scope=json.loads((root/'artifacts/eval2000-scope.json').read_text())
    checkpoint=Path(checkpoint).resolve()
    assert checkpoint.is_relative_to(root/'checkpoints')
    registration=json.loads((root/'artifacts/model-registration.json').read_text())
    assert Path(registration['checkpoint']).resolve()==checkpoint
    path=checkpoint/'runtime-parameter-audit.json'
    binding=json.loads(path.read_text())
    assert binding['kind']=='augmented16000_dense_full_sft_FP32_values_all_1672_parameter_tensors'
    assert binding['source_checkpoint_step']==16000 and binding['parent_initial_step']==20000
    assert binding['source_complete_sha256']==scope['source_complete_sha256']
    assert binding['source_checkpoint_identity']==scope['source_checkpoint_identity']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==scope['runtime_parameter_audit_sha256']
    assert binding['source_checkpoint_identity']['dense_full_sft'] is True
    assert scope['source_world']==4
    assert binding['source_checkpoint_identity']['source_step']==11000
    assert binding['source_checkpoint_identity']['source_complete_sha256']=='4e23fe8c407b1a47dc932541df92d7067c2c64a2a9e512cbe03059fd63ce1a41'
    assert binding['source_checkpoint_identity']['geometry_fix_boundary']==11000
    assert binding['source_checkpoint_identity']['source_binding']=='6648f6b4be7ad62925cd47923adcb13d7ed473600b739fc08d14379b7fdd0d23'
    assert binding['source_checkpoint_identity']['phase_start']==10000
    assert binding['source_checkpoint_identity']['norm_stats.json']==scope['norm_sha256']
    params={n.replace('_orig_mod.',''):p for n,p in model.named_parameters()}
    assert len(params)==len(binding['trained_parameter_sha256'])==1672
    assert sum(p.numel() for p in params.values())==6375906359
    assert set(params)==set(binding['trained_parameter_sha256'])
    modules=dict(model.named_modules())
    buffers={n.replace('_orig_mod.',''):v for n,v in model.named_buffers()
      if n.rsplit('.',1)[-1] not in modules[n.rsplit('.',1)[0] if '.' in n else '']._non_persistent_buffers_set}
    assert len(buffers)==36 and set(buffers)==set(binding['persistent_buffer_sha256'])
    for values,expected in [(params,binding['trained_parameter_sha256']),(buffers,binding['persistent_buffer_sha256'])]:
        for name,value in values.items():
            value=value.detach().cpu().contiguous()
            assert value.dtype==torch.float32 and torch.isfinite(value).all(),name
            assert hashlib.sha256(value.numpy().tobytes()).hexdigest()==expected[name],name
    return dict(learned_tensors=1672,persistent_buffers=36,exact_saved_fp32_values=True,
       exact_saved_persistent_buffers=True,binding_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
       source_checkpoint_step=16000,parent_initial_step=20000,audit_scope='all_saved_model_parameters_and_persistent_buffers')
