"""Supplement missing object proposals; never relax protection or semantic gates.

Robot and object phrases can interfere. Re-query only missing object roles, then
UNION original and supplementary boxes. Complete proposal files are immutable.
"""
import hashlib,json
import numpy as np
from cache_policy import select


def supplement_prompt(frame,phrases):
    _,info=select(frame,phrases)
    missing=set(info['missing_roles'])
    if not missing or 'robot' in missing or info['protected_fraction']>=.8:return None
    # Same original object names; no task policy/instruction rewriting.
    objects=[p for p in phrases if not p.lower().startswith('robot')]
    if not objects:return None
    return '. '.join(objects)+'.'


def merge(original,supplement):
    for d in supplement:
        if not 0<=d['score']<=1 or not np.isfinite(d['box']).all():
            raise ValueError('Invalid supplementary detector result')
    return dict(original,detections=[*original['detections'],*supplement])


def record_input_binding(proposal,record):
    assert proposal['source_sha256']==record['sha256']
    assert len(proposal['frames'])==int(record['frames'])+1
    assert [f['frame'] for f in proposal['frames']]==list(range(int(record['frames'])+1))

