"""Private POSIX compiler artifacts before importing Torch/Triton in each rank."""
import json
import os
from pathlib import Path
import runpy

rank = int(os.environ['LOCAL_RANK'])
assert 0 <= rank < 4
root = Path('/mnt/raid1/lingbotvla_sft_amd8000_to10000_20261020')
temporary = Path('/tmp/lingbotvla-amd8000-continue-20261020-compiler') / f'rank{rank}'
temporary.mkdir(parents=True, exist_ok=True)
os.environ['TRITON_CACHE_DIR'] = str(temporary / 'triton')
os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(temporary / 'inductor')
print(json.dumps(dict(event='private_compiler_cache', rank=rank,
    triton=os.environ['TRITON_CACHE_DIR'], inductor=os.environ['TORCHINDUCTOR_CACHE_DIR'],
    large_training_files=str(root), inference_cache_setting_unchanged=True)), flush=True)
runpy.run_path(str(root / 'code/train_full.py'), run_name='__main__')
