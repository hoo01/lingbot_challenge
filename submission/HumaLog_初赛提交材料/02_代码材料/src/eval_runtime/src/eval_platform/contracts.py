"""CPU-only checkpoint and source checks; no model code is executed."""
import json
from pathlib import Path
from .store import digest


def validate_source(source, lock):
    """Accept only two explicitly pinned byte layouts, not arbitrary normalization."""
    candidates = [('git_blob_lf', lock)]
    if 'historical_crlf' in lock:
        candidates.append(('historical_crlf', {**lock, **lock['historical_crlf']}))
    for label, candidate in candidates:
        if all(digest(source/name)==value for name,value in candidate['pristine_source_sha256'].items()):
            norm = source/'assets/norm_stats/robotwin.json'
            if digest(norm) == candidate['normalization_sha256']:
                return {**candidate,'source_byte_layout':label}
    raise ValueError('Pinned upstream source differs (or upstream normalization differs)')


def source_lock(root):
    lock = json.loads((root / 'configs/upstream-lock.json').read_text())
    source = root / 'third_party/lingbot-vla-v2-official-be969b8f'
    return source, validate_source(source, lock)


def model_construction(source):
    """Only this declared construction rule is supported; execution stays FP32."""
    path = Path(source)/'runtime-construction.json'
    if not path.exists():
        return 'native'
    data = json.loads(path.read_text(encoding='utf-8'))
    supported = [
        {'nonpersistent_buffer_initialization': 'bf16_then_fp32'},
        {'nonpersistent_buffer_initialization': 'native', 'checkpoint_parameter_dtype': 'float32'},
    ]
    if data not in supported:
        raise ValueError('Unsupported runtime-construction.json')
    return data['nonpersistent_buffer_initialization']


def checkpoint_load_dtype(source):
    """A declared FP32 master checkpoint must be loaded into FP32 parameters."""
    model_construction(source)  # Reject undeclared extra runtime overrides.
    path = Path(source)/'runtime-construction.json'
    return json.loads(path.read_text()).get('checkpoint_parameter_dtype') if path.exists() else None


def validate_checkpoint(source, data, tensors):
    """Check loader compatibility without requiring export receipts or hashes."""
    if (source / 'factorized-adapter.json').exists() or any(
        'lora_' in name or 'parametrizations.' in name for name in tensors
    ):
        raise ValueError('Export a complete merged HF model; factorized research weights are unsupported')
    if data['data'].get('img_size') != 256:
        raise ValueError('This profile requires img_size=256; preserve the training CLI and review differences')
    combined = {**data['model'], **data['train']}
    if combined.get('chunk_size', 50) != 50 or combined.get('num_steps', 10) != 10:
        raise ValueError('This profile requires chunk_size=50 and num_steps=10')
    model_construction(source)


def write_inference_manifest(root, checkpoint, norm, run, kind):
    source, lock = source_lock(root)
    manifest = dict(kind=kind, official_revision=lock['official_revision'],
                    source_byte_layout=lock['source_byte_layout'],
                    source=str(source), checkpoint=str(checkpoint),
                    normalization=str(norm), normalization_sha256=digest(norm),
                    pristine_source_sha256=lock['pristine_source_sha256'])
    (run / 'official-defaults-manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest
