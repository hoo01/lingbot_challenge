"""Execute the original bootstrap, changing only the checkpoint implementation."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=Path('/workspace/lingbotvla_sft_from_lab20k_amd8_20261017')
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT/'code')]
binding=json.loads((ROOT/'artifacts/checkpoint-memory001/runtime-binding.json').read_text())
for name,expected in binding['runtime_files'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
original=runpy.run_path
called=False

def intercept(path,*args,**kwargs):
    global called
    if Path(path)==ROOT/'code/train_full.py':
        assert not called
        called=True
        runpy.run_path=original
        # The unchanged original bootstrap has already selected all private compiler caches.
        from save_bounded import BoundedLocalArchive
        from optimizer_audit import audited_factory
        from entry_observer import run_observed
        def install(globals_):
            globals_['AsyncArchive']=BoundedLocalArchive
            def sink(step,changes):
                globals_['atomic_json'](ROOT/'artifacts/checkpoint-memory001/update-audits'/
                    f'rank{globals_["RANK"]}-step{step}.json',dict(passed=True,step=step,
                    changes=changes,read_only_observation=True,original_optimizer_step_unchanged=True))
            globals_['make_optimizer']=audited_factory(globals_['make_optimizer'],
                globals_['update_sentinels'],globals_['ckpt'].local,sink)
        return run_observed(original,path,install)
    return original(path,*args,**kwargs)

runpy.run_path=intercept
original(str(ROOT/'code/worker_bootstrap.py'),run_name='__main__')
assert called
