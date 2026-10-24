"""Replay the bound E0 model-data file order in this official eval process."""
import glob
import os
import hashlib
import importlib
import json
from pathlib import Path

ROOT = Path('__HUMALOG_EVAL_ROOT__')
JOB = '7cc771a96d5c'
BINDING = ROOT / 'artifacts/paired-e0/e0-enumeration.json'
BINDING_SHA = 'a8007f917479f259ac1191595dce5761ab5b4125f295ca8e051ac119deeca3d9'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(repo, data):
    repo = Path(repo).resolve()
    if data['version'] != 'e0_model_data_enumeration_v1' or data['source_job'] != '843758599804':
        raise ValueError('Unbound E0 enumeration')
    for name, digest in data['task_sources'].items():
        if sha(repo / name) != digest:
            raise ValueError('Enumeration task source changed: ' + name)
    for pattern, entries in data['patterns'].items():
        actual = glob.glob(str(repo / pattern))
        files = {Path(p).name: Path(p) for p in actual}
        expected = [e['name'] for e in entries]
        if len(expected) != len(set(expected)) or set(files) != set(expected):
            raise ValueError('Enumeration file set differs: ' + pattern)
        for entry in entries:
            if sha(files[entry['name']]) != entry['sha256']:
                raise ValueError('Enumeration asset bytes changed: ' + entry['name'])
    return repo


class GlobProxy:
    def __init__(self, repo, data):
        self.repo, self.data = Path(repo).resolve(), data
    def __getattr__(self, name):
        return getattr(glob, name)
    def glob(self, pattern, *args, **kwargs):
        result = glob.glob(pattern, *args, **kwargs)
        # Match the nominal in-repository glob, retaining asset-directory symlinks.
        path = Path(os.path.abspath(pattern))
        if not path.is_relative_to(self.repo):
            return result
        relative = path.relative_to(self.repo).as_posix()
        if relative not in self.data['patterns']:
            if relative.startswith('assets/objects/') and path.name == 'model_data*.json':
                raise ValueError('Unbound model-data enumeration: ' + relative)
            return result
        names = {Path(p).name: p for p in result}
        expected = self.data['patterns'][relative]
        if set(names) != {e['name'] for e in expected}:
            raise ValueError('Enumeration file set changed during setup')
        return [names[e['name']] for e in expected]


def install_for_scene(repo, task, output, paired_binding):
    # This repair is scoped to the original authorized formal job. Other
    # tasks/roots, including the held Lab job, keep their current behavior.
    if (Path(repo).resolve() != (ROOT / 'third_party/RoboTwin').resolve() or
            not Path(output).resolve().is_relative_to((ROOT / 'runs').resolve())):
        return None
    from .store import Store
    job_folder=next((p for p in Path(output).resolve().parents if p.parent==ROOT/'runs'),None)
    if job_folder is None or not job_folder.name.startswith('eval-'):raise ValueError('Unbound new evaluation folder')
    registered=json.loads((ROOT/'incoming/Lab-qv256-step-20000-20261014/model-registration.json').read_text())
    store=Store.__new__(Store);store.root=ROOT;store.state=ROOT/'platform-state';store.db=store.state/'queue.sqlite3'
    config=store.job(job_folder.name.removeprefix('eval-'))['config']
    if config['checkpoint_id']!=registered['model_id'] or registered['step']!=20000:raise ValueError('Enumeration belongs to another checkpoint')
    if sha(BINDING) != BINDING_SHA:
        raise ValueError('E0 enumeration receipt changed')
    data = json.loads(BINDING.read_text())
    if paired_binding['sha256'] != data['paired_reference_sha256']:
        raise ValueError('Enumeration paired input binding differs')
    repo = validate(repo, data)
    source = 'envs/' + task + '.py'
    if source not in data['task_sources']:
        return None
    module = importlib.import_module('envs.' + task)
    if isinstance(module.glob, GlobProxy):
        raise ValueError('Enumeration proxy installed more than once')
    module.glob = GlobProxy(repo, data)
    return dict(mode=data['version'], source_job=data['source_job'],
        binding_sha256=BINDING_SHA, task_source_sha256=data['task_sources'][source],
        rng_draws_added=0, assets_modified=False, simulation_steps_added=0)
