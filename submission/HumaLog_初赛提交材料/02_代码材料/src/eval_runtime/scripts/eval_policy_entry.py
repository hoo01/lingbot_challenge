"""Audited policy-only entry after expert-budget exhaustion.

Read object substitutions directly from pinned task source expressions. Use
official arm-neutral templates, since an unsuccessful expert may not establish
arm choices. Three score initializers are verbatim pre-motion source slices.
This is an explicit local adapter, not an upstream quota implementation.
"""
import ast
import hashlib
import json
from pathlib import Path

ENTRY_FLOW = 'pinned_object_templates_and_score_prelude_v1'


def _sha(text):
    return hashlib.sha256(text.replace('\r\n', '\n').encode()).hexdigest()


def policy_entry(task, env, repo):
    spec = json.loads(Path(__file__).with_name('eval_policy_entry_sources.json').read_text())
    source = Path(repo, 'envs', task + '.py').read_text()
    expected = spec[task]
    if _sha(source) != expected['source_sha256']:
        raise ValueError('Policy entry task source changed: ' + task)
    # These expressions allow only constants, setup-created self fields, and
    # indexed setup arrays; never call an expert, planner, motion or simulator.
    namespace = {'self': env}
    parameters = {key: eval(compile(ast.parse(expression, mode='eval'), '<pinned-info>', 'eval'),
                            {'__builtins__': {}}, namespace)
                  for key, expression in expected['object_expressions'].items()}
    if task == 'place_bread_basket' and len(env.bread) == 2:
        parameters['{C}'] = eval(compile(ast.parse(expected['second_bread'], mode='eval'),
                                        '<pinned-info>', 'eval'), {'__builtins__': {}}, namespace)
    if expected['score_prelude']:
        from envs.utils import ArmTag, get_face_prod
        exec(compile(expected['score_prelude'], '<pinned-score-prelude>', 'exec'),
             {'__builtins__': {}, 'ArmTag': ArmTag, 'get_face_prod': get_face_prod}, namespace)
    if any(not isinstance(value, str) for value in parameters.values()):
        raise ValueError('Official object substitution must be a string')
    # The unchanged official generator accepts exactly these object-only keys.
    return parameters, dict(flow=ENTRY_FLOW, task_source_sha256=expected['source_sha256'],
        instruction_source='official_object_only_templates',
        score_state={name: str(getattr(env, name)) for name in expected['score_attributes']},
        expert_called=False, simulation_steps_added=0)
