"""CPU-only checkpoint handoff and source checks; no model code is executed."""
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


def check_handoff(source, cli_path, norm_path, data, tensors):
    """Require the export receipt to bind the exact training inputs."""
    if (source / 'factorized-adapter.json').exists() or any(
        'lora_' in name or 'parametrizations.' in name for name in tensors
    ):
        raise ValueError('Export a complete merged HF model; factorized research weights are unsupported')
    if data['data'].get('img_size') != 256:
        raise ValueError('This profile requires img_size=256; preserve the training CLI and review differences')
    combined = {**data['model'], **data['train']}
    if combined.get('chunk_size', 50) != 50 or combined.get('num_steps', 10) != 10:
        raise ValueError('This profile requires chunk_size=50 and num_steps=10')
    receipt_path = source / 'eval_handoff.json'
    required = [receipt_path, norm_path.with_suffix('.provenance.json'), source/'clean_manifest.json']
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        raise ValueError('Missing training handoff files: ' + ', '.join(missing))
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt.get('kind') != 'clean_sft_export':
        raise ValueError('Expected clean_sft_export receipt from the training export')
    if receipt.get('cli_sha256') != digest(cli_path) or receipt.get('norm_sha256') != digest(norm_path):
        raise ValueError('Export receipt does not match CLI / normalization bytes')
    provenance_path = norm_path.with_suffix('.provenance.json')
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    if provenance.get('clean_only') is not True or provenance.get('norm_sha256') != digest(norm_path):
        raise ValueError('Clean normalization provenance is missing or inconsistent')
    if receipt.get('provenance_sha256') != digest(provenance_path):
        raise ValueError('Export provenance SHA mismatch')
    manifest_path = source / 'clean_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('clean_only') is not True or not manifest.get('datasets'):
        raise ValueError('Expected the actual clean training manifest')
    if receipt.get('manifest_sha256') != digest(manifest_path):
        raise ValueError('Export training-manifest SHA mismatch')
    # This checks bindings, not the honesty/completeness of the training pipeline.
    return {p.name: p.read_bytes() for p in (receipt_path, provenance_path, manifest_path)}


def write_inference_manifest(root, checkpoint, norm, run, kind):
    source, lock = source_lock(root)
    manifest = dict(kind=kind, official_revision=lock['official_revision'],
                    source_byte_layout=lock['source_byte_layout'],
                    source=str(source), checkpoint=str(checkpoint),
                    normalization=str(norm), normalization_sha256=digest(norm),
                    pristine_source_sha256=lock['pristine_source_sha256'])
    (run / 'official-defaults-manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest
