import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import uuid
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while data := f.read(8 * 1024 * 1024): h.update(data)
    return h.hexdigest()


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.state = self.root / 'platform-state'
        self.state.mkdir(parents=True, exist_ok=True)
        self.db = self.state / 'queue.sqlite3'
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS checkpoints(id TEXT PRIMARY KEY, spec TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, title TEXT NOT NULL,
              config TEXT NOT NULL, status TEXT NOT NULL, created TEXT NOT NULL,
              updated TEXT NOT NULL, gpu TEXT, pid INTEGER, error TEXT);
            CREATE TABLE IF NOT EXISTS workers(gpu TEXT PRIMARY KEY, pid INTEGER,
              updated TEXT NOT NULL, job TEXT);
            ''')

    def connect(self):
        db = sqlite3.connect(self.db, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def path(self, value, areas=None):
        p = (self.root / value).resolve()
        if not p.is_relative_to(self.root): raise ValueError('Path must be inside the evaluation project')
        if areas and not any(p.is_relative_to(self.root / x) for x in areas):
            raise ValueError('Checkpoint must be inside incoming, checkpoints, models or runs')
        return p

    def tasks(self):
        return sorted(p.stem for p in (self.root/'third_party/RoboTwin/description/task_instruction').glob('*.json'))

    def register(self, name, checkpoint, cli, norm, notes=''):
        import yaml
        if not 1 <= len(name.strip()) <= 120: raise ValueError('Checkpoint name must have 1-120 characters')
        source = self.path(checkpoint, ['incoming','checkpoints','models','runs'])
        cli_path, norm_path = self.path(cli), self.path(norm)
        if not source.is_dir():
            raise ValueError('Checkpoint directory not found: ' + str(checkpoint))
        for label, path in [('CLI YAML', cli_path), ('Normalization JSON', norm_path)]:
            if not path.is_file():
                raise ValueError(label + ' file not found: ' + str(path.relative_to(self.root)))
        shards = sorted(source.glob('*.safetensors'))
        if not shards: raise ValueError('A full safetensors checkpoint is required')
        try:
            data = yaml.safe_load(cli_path.read_text())
        except yaml.YAMLError as error:
            raise ValueError('Invalid CLI YAML: ' + str(error)) from error
        if not isinstance(data, dict) or not all(isinstance(data.get(k), dict) for k in ('model','train','data')):
            raise ValueError('CLI YAML must contain model, train and data mappings')
        stats = json.loads(norm_path.read_text())
        if not isinstance(stats,dict) or not stats: raise ValueError('Normalization JSON is empty')
        # Read safetensors headers only. No pickle or user Python is executed.
        tensors = set()
        for f in shards:
            with f.open('rb') as stream:
                size = int.from_bytes(stream.read(8), 'little')
                if not 0 < size < 100_000_000: raise ValueError('Invalid safetensors header')
                header = json.loads(stream.read(size))
                tensors.update(set(header) - {'__metadata__'})
        if not any('embed_tokens.weight' in k for k in tensors):
            raise ValueError('Adapter-only checkpoint detected; export one complete model first')
        from .contracts import validate_checkpoint
        validate_checkpoint(source, data, tensors)
        identifier = uuid.uuid4().hex[:12]
        folder = self.root / 'checkpoints' / identifier
        target = folder/'checkpoints/global_step_0/hf_ckpt'
        target.mkdir(parents=True)
        manifest = []
        for f in shards:
            f = self.path(str(f))
            manifest.append({'path':str(f), 'bytes':f.stat().st_size, 'mtime_ns':f.stat().st_mtime_ns, 'sha256':digest(f)})
            (target/f.name).symlink_to(f)
        for filename in ('config.json','model.safetensors.index.json','runtime-construction.json'):
            if (source/filename).exists(): (target/filename).write_bytes((source/filename).read_bytes())
        frozen_norm = folder/'norm_stats.json'
        frozen_norm.write_bytes(norm_path.read_bytes())
        (folder/'original_training_cli.yaml').write_bytes(cli_path.read_bytes())
        optional_files = []
        audit = source/'action-audit.json'
        if audit.is_file():
            archived_audit = folder/audit.name
            archived_audit.write_bytes(audit.read_bytes())
            optional_files.append(archived_audit)
        data['model']['tokenizer_path'] = str(self.root/'models/Qwen3-VL-4B-Instruct')
        data['data']['norm_stats_file'] = str(frozen_norm)
        # Keep the original training CLI above; use the same inference defaults
        # for SFT and base models, regardless of the training memory settings.
        inference_overrides = dict(use_compile=True, enable_fp32=True,
            enable_gradient_checkpointing=False, vit_attn_implementation='flash_attention_2',
            precompute_grid_thw=True)
        data['train'].update(inference_overrides)
        (folder/'lingbotvla_cli.yaml').write_text(yaml.safe_dump(data,sort_keys=False))
        spec = dict(id=identifier, name=name.strip(), checkpoint=str(target), cli=str(folder/'lingbotvla_cli.yaml'),
                    norm=str(frozen_norm), norm_sha256=digest(frozen_norm), files=manifest,
                    notes=notes[:2000], created=now(), original_source=str(source))
        spec['kind'] = 'team_merged_checkpoint'
        spec['inference_overrides'] = inference_overrides
        spec['metadata_files'] = [dict(path=str(f), sha256=digest(f)) for f in
            [folder/'lingbotvla_cli.yaml', folder/'original_training_cli.yaml', frozen_norm, *optional_files,
             *[f for f in target.iterdir() if f.suffix=='.json']]]
        (folder/'manifest.json').write_text(json.dumps(spec,indent=2))
        with self.connect() as db: db.execute('INSERT INTO checkpoints VALUES (?,?)',(identifier,json.dumps(spec)))
        return spec

    def checkpoints(self):
        with self.connect() as db: return [json.loads(r[0]) for r in db.execute('SELECT spec FROM checkpoints')]

    def checkpoint(self, identifier):
        with self.connect() as db: row=db.execute('SELECT spec FROM checkpoints WHERE id=?',(identifier,)).fetchone()
        if not row: raise ValueError('Unknown checkpoint')
        return json.loads(row[0])

    def submit(self, title, config):
        checkpoint = self.checkpoint(config['checkpoint_id'])
        if checkpoint.get('read_only'):
            raise ValueError('Historical registration is read-only; select a model registered for the current inference profile')
        tasks, settings, seeds = config['tasks'], config['settings'], config['seeds']
        if not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= set(self.tasks()): raise ValueError('Invalid tasks')
        if not settings or len(set(settings)) != len(settings) or not set(settings) <= {'demo_clean','demo_randomized'}: raise ValueError('Invalid settings')
        if not seeds or len(set(seeds)) != len(seeds) or any(type(x)!=int or not 0<=x<2**31 for x in seeds): raise ValueError('Invalid seeds')
        if len(tasks)*len(settings)*len(seeds)>10000: raise ValueError('At most 10000 episodes per job')
        if config.get('execute',50) != 50: raise ValueError('This profile fixes execute=50')
        if config.get('precision','fp32') != 'fp32': raise ValueError('This profile fixes precision=fp32')
        if type(config.get('video',False)) is not bool: raise ValueError('Video must be a boolean')
        if type(config.get('video_fps',10)) is not int or config.get('video_fps',10) not in (10,25): raise ValueError('Video fps must be 10 or 25')
        clean = {k:config[k] for k in ('checkpoint_id','tasks','settings','seeds')}
        clean.update(execute=config.get('execute',50), precision=config.get('precision','fp32'),
                     video=config.get('video',False), video_fps=config.get('video_fps',10), protocol='team_official_defaults_fixed_unfiltered_v1',
                     inference_profile='official_defaults_v1',
                     moe_backend='official',vision_device='cuda',embedding_device='cuda')
        from .servers import server_state
        clean['server_id'] = server_state(self.root)['current']['id']
        identifier = uuid.uuid4().hex[:12]
        folder = self.root/'runs'/f'eval-{identifier}'
        folder.mkdir(parents=True)
        (folder/'request.json').write_text(json.dumps(clean,indent=2))
        timestamp=now()
        with self.connect() as db:
            db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)',
                (identifier,title.strip()[:120] or identifier,json.dumps(clean),'queued',timestamp,timestamp,None,None,None))
        return self.job(identifier)

    @staticmethod
    def decode(row):
        r=dict(row);r['config']=json.loads(r['config']);return r

    def jobs(self):
        with self.connect() as db: return [self.decode(r) for r in db.execute('SELECT * FROM jobs ORDER BY created DESC')]

    def job(self, identifier):
        with self.connect() as db: r=db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone()
        if not r: raise ValueError('Unknown job')
        return self.decode(r)

    def claim(self,gpu,pid):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT 1 FROM jobs WHERE gpu=? AND status IN ('running','cancel_requested')",(gpu,)).fetchone(): return None
            row=db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
            if not row:return None
            db.execute("UPDATE jobs SET status='running',gpu=?,pid=?,updated=? WHERE id=?",(gpu,pid,now(),row[0]))
        return self.job(row[0])

    def update(self, identifier, status, error=None):
        with self.connect() as db:
            db.execute('UPDATE jobs SET status=?,error=?,updated=? WHERE id=?',(status,error,now(),identifier))

    def cancel(self, identifier):
        if self.job(identifier)['config'].get('history_import'):
            raise ValueError('Imported historical jobs are read-only')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT status FROM jobs WHERE id=?',(identifier,)).fetchone()
            if not row: raise ValueError('Unknown job')
            status = 'cancelled' if row[0]=='queued' else 'cancel_requested' if row[0]=='running' else row[0]
            db.execute('UPDATE jobs SET status=?,updated=? WHERE id=?',(status,now(),identifier))
        return self.job(identifier)

    def heartbeat(self,gpu,pid,job=None):
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO workers VALUES (?,?,?,?)',(gpu,pid,now(),job))

    def workers(self):
        with self.connect() as db:return [dict(r) for r in db.execute('SELECT * FROM workers')]

    def run_folder(self, identifier):
        if not re.fullmatch('[a-f0-9]{12}',identifier): raise ValueError('Invalid job ID')
        return self.root/'runs'/f'eval-{identifier}'
