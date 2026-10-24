import hashlib,json,os
from pathlib import Path
ROOT=Path('__HUMALOG_EVAL_ROOT__')
SOURCE=Path('/root/shared-nvme/lingbot-eval-team')
ARCHIVE=ROOT
PROOF=ROOT/'artifacts/paired-e0/archive-binding.json'
def original_record_path(path):
    p=Path(path)
    if not p.is_absolute() or not p.is_relative_to(SOURCE):return p
    if Path(os.environ.get('LINGBOT_ROOT','')).resolve()!=ROOT:return p
    proof=json.loads(PROOF.read_text());rel=p.relative_to(SOURCE)
    if (proof['root']!=str(ROOT) or proof['source_job']!='843758599804'
        or tuple(rel.parts[:2])!=('runs','eval-843758599804')):
        raise ValueError('Only the bound original E0 evidence can use snapshot translation')
    target=ROOT/rel
    if not target.resolve().is_relative_to(ROOT):raise ValueError('E0 snapshot path escaped')
    return target
def receipt(*args,**kwargs):raise ValueError('This is a new checkpoint evaluation, not an old host migration')
def verify_transition(*args,**kwargs):raise ValueError('Old host-transition recovery is not authorized in this new root')
