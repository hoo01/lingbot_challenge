"""Private rank-specific POSIX compiler caches, before any Torch import."""
import json
import os
from pathlib import Path
import runpy

rank=int(os.environ['LOCAL_RANK']);assert 0<=rank<8
root=Path('/workspace/lingbotvla_sft_from_lab20k_amd8_20261017')
temporary=Path('/var/tmp/lingbotvla-sft-amd8-20261017-compiler')/f'rank{rank}'
temporary.mkdir(parents=True,exist_ok=True)
(temporary/'tmp').mkdir(exist_ok=True)
os.environ['TMPDIR']=str(temporary/'tmp')
os.environ['TRITON_CACHE_DIR']=str(temporary/'triton')
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(temporary/'inductor')
os.environ['AITER_JIT_DIR']='/var/tmp/lingbotvla-sft-amd8-20261017-compiler/aiter'
os.environ['AITER_TRITON_ONLY']='1'
os.environ.setdefault('OMP_NUM_THREADS','4')
print(json.dumps(dict(event='private_compiler_cache',rank=rank,path=str(temporary),
    temporary=os.environ['TMPDIR'],large_training_files=str(root))),flush=True)
runpy.run_path(str(root/'code/train_full.py'),run_name='__main__')
