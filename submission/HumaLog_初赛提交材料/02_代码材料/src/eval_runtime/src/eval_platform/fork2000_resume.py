"""CPU verification of an append-only continuation of this fork only."""
import hashlib
import json
from pathlib import Path


def verify(root,folder,config,manifest,diagnostic,records,shard):
    from .fork2000_plan import validate,validate_history
    from .planning import next_candidate
    from .resume import inside,sha256
    validate(config);validate_history(config,records)
    parent=inside(root,manifest['parent_folder'])
    source_config=json.loads((parent/'request.json').read_text())
    prior=json.loads(inside(root,manifest['source_config']).read_text())
    assert source_config==prior,'Parent request changed'
    expected={**prior,'seeds':config['seeds'],'fixed_scenes':config['fixed_scenes']}
    assert config==expected,'Only authorized candidate-pool extension is allowed'
    original=inside(root,manifest['source_ledger'])
    copied=inside(folder,shard['output'])/'evaluation/episodes.jsonl'
    assert original.read_bytes()==copied.read_bytes(),'Original five results must remain byte-identical'
    assert sha256(original)==diagnostic['parent_ledger_sha256']
    assert diagnostic['expert_or_planner_calls']==0 and diagnostic['preserved_results']==len(records)
    assert shard['first_remaining']==next_candidate(config,records)
    return manifest
