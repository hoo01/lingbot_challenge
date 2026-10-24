"""CPU package checks; full model content hashing is an explicit optional operation."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--weights',action='store_true',help='Require all 14 actual model files')
    p.add_argument('--hash-weights',action='store_true',help='Reread every actual model file and compare content SHA')
    a = p.parse_args()
    package = a.package.resolve();code = package / '02_代码材料'
    lock = json.loads((code / 'configs/submission-source-lock.json').read_text(encoding='utf8'))['files']
    for name, expected in lock.items():
        assert sha(code / name) == expected, name
        if name.endswith('.py'):
            ast.parse((code / name).read_bytes(),filename=name)
    parsed = 0
    for path in package.rglob('*.json'):
        if '.cache' not in path.parts:
            json.loads(path.read_text(encoding='utf8'));parsed += 1
    manifest = json.loads((package / '03_模型权重/model_manifest.json').read_text(encoding='utf8'))
    missing = []
    checkpoint = package / '03_模型权重/checkpoint'
    for row in manifest['weight_files']:
        path = checkpoint / row['path']
        if not path.is_file():missing.append(row['path']);continue
        assert path.stat().st_size == row['bytes'], path.name
        if a.hash_weights:
            assert sha(path) == row['sha256'],path.name
    if a.weights or a.hash_weights:
        assert not missing, 'Missing actual weight files: ' + str(len(missing))
    for name in ['norm_stats.json','norm_stats.provenance.json','robotwin_relative.yaml']:
        assert (checkpoint / 'assets' / name).is_file()
    assert sha(checkpoint / 'assets/norm_stats.json') == manifest['norm_sha256']
    print(json.dumps(dict(passed=True,python_sources=len(lock),parsed_json_files=parsed,
                         actual_weights_present=14-len(missing),weight_hashes_rechecked=a.hash_weights,
                         results_present=(package / '01_评测结果/results.json').is_file(),
                         independent_gpu_reproduction_certified=False)))


if __name__ == '__main__':
    main()
