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
            CREATE TABLE IF NOT EXISTS job_shards(job_id TEXT NOT NULL, shard INTEGER NOT NULL,
              lane INTEGER, config TEXT NOT NULL, status TEXT NOT NULL, gpu TEXT,
              pid INTEGER, error TEXT, updated TEXT NOT NULL, PRIMARY KEY(job_id,shard));
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
        robot_path = provenance_path = None
        if data['data'].get('robot_config_path'):
            supplied = Path(data['data']['robot_config_path'])
            candidate = supplied if supplied.is_absolute() else cli_path.parent / supplied
            # Export bundles move between hosts; resolve the bundled basename.
            if not candidate.is_file() or not candidate.resolve().is_relative_to(self.root):
                candidate = cli_path.parent / supplied.name
            robot_path = self.path(str(candidate))
            provenance_path = norm_path.with_suffix('.provenance.json')
            if not robot_path.is_file() or not provenance_path.is_file():
                raise ValueError('Model-specific robot config requires its YAML and norm provenance sidecar')
            proof = json.loads(provenance_path.read_text())
            if proof.get('robot_config_sha256') != digest(robot_path) or proof.get('norm_sha256') != digest(norm_path):
                raise ValueError('Model-specific robot config or normalization SHA mismatch')
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
        for filename in ('config.json','model.safetensors.index.json','runtime-construction.json',
                         'runtime-parameter-audit.json','precision-provenance.json'):
            if (source/filename).exists(): (target/filename).write_bytes((source/filename).read_bytes())
        frozen_norm = folder/'norm_stats.json'
        frozen_norm.write_bytes(norm_path.read_bytes())
        (folder/'original_training_cli.yaml').write_bytes(cli_path.read_bytes())
        optional_files = []
        if robot_path is not None:
            frozen_robot = folder/'robot_config.yaml'
            frozen_robot.write_bytes(robot_path.read_bytes())
            frozen_provenance = frozen_norm.with_suffix('.provenance.json')
            frozen_provenance.write_bytes(provenance_path.read_bytes())
            data['data']['robot_config_path'] = str(frozen_robot)
            optional_files.extend([frozen_robot,frozen_provenance])
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
        if len(tasks)*len(settings)*len(seeds)>15000: raise ValueError('At most 15000 candidate scenes; private scope still scores500')
        if config.get('execute',25) != 25: raise ValueError('This new profile fixes H50 execute25')
        if config.get('precision','fp32') != 'fp32': raise ValueError('This profile fixes precision=fp32')
        if type(config.get('video',False)) is not bool: raise ValueError('Video must be a boolean')
        if type(config.get('observer_video',True)) is not bool: raise ValueError('Observer video must be a boolean')
        if type(config.get('expert_check',False)) is not bool: raise ValueError('Expert check must be a boolean')
        if type(config.get('video_fps',10)) is not int or config.get('video_fps',10) not in (10,25): raise ValueError('Video fps must be 10 or 25')
        clean = {k:config[k] for k in ('checkpoint_id','tasks','settings','seeds')}
        clean.update(execute=config.get('execute',25), precision=config.get('precision','fp32'),
                     video=config.get('video',False), video_fps=config.get('video_fps',10), protocol='team_h50_execute25_fixed_unfiltered_v1',
                     inference_profile='aug16000_h50_execute25_no_video_v1',
                     moe_backend='official',vision_device='cuda',embedding_device='cuda')
        clean['expert_check'] = config.get('expert_check', False)
        clean['observer_video'] = config.get('observer_video', True)
        runtime_path=self.root/'configs/runtime.json'
        runtime=json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
        clean['release_unused_cuda_cache']=bool(runtime.get('release_unused_cuda_cache',False))
        if config.get('fixed_scenes') is not None:
            from .planning import validate_fixed_scenes
            validate_fixed_scenes(config)
            clean['fixed_scenes'] = config['fixed_scenes']
            clean['protocol'] = 'team_h50_execute25_explicit_scenes_v1'
        if clean['expert_check']:
            clean['protocol'] = 'team_official_defaults_fixed_expert_screened_v1'
        if config.get('paired_reference'):
            from .paired_inputs import read_reference, validate_entry
            binding = config['paired_reference']
            self.path(binding['path'])
            if config.get('paired_entry'):
                clean['paired_entry'] = config['paired_entry']
            if not clean['expert_check'] and not validate_entry({**clean,'paired_reference':binding}):
                raise ValueError('E0 paired entry requires the declared expert flow')
            read_reference(binding,clean.get('fixed_scenes'))
            clean['paired_reference'] = binding
            clean['protocol'] = 'e0_fixed500_paired_inputs_expert_entry_v1'
            if clean.get('paired_entry'):
                validate_entry(clean)
                clean['protocol'] = 'e0_fixed500_frozen_expert_policy_only_v3'
        elif config.get('paired_entry'):
            raise ValueError('Frozen E0 entry requires its paired reference')
        if config.get('scene_replay'):
            if not clean.get('paired_entry') or clean['expert_check']:
                raise ValueError('Scene replay requires the frozen entry with experts off')
            from .scene_replay import read_scene_replay
            binding = config['scene_replay']
            self.path(binding['path'])
            read_scene_replay(binding, clean['paired_reference'], clean['fixed_scenes'],
                              self.root/'third_party/RoboTwin')
            clean['scene_replay'] = binding
            clean['protocol'] = 'e0_fixed500_scene_replay_policy_only_v4'
        if config.get('timeout_policy'):
            from .timeout_outcomes import validate_timeout_policy
            clean['timeout_policy'] = config['timeout_policy']
            validate_timeout_policy(clean)
        from .servers import server_state
        server = server_state(self.root)['current']
        clean['server_id'] = server['id']
        count = config.get('parallelism') or server.get('parallelism', 1)
        if config.get('parallelism') is not None and (type(config['parallelism']) is not int or config['parallelism'] not in (1, 2)):
            raise ValueError('parallelism must be 1 or 2')
        if count > server.get('parallelism', 1):
            raise ValueError('Requested parallelism exceeds the configured GPU count')
        from .scheduling import partition
        clean['shards'] = partition(clean, count)
        if 'fixed_scenes' in clean and any(not any(r['task'] in s['tasks'] and r['setting'] in s['settings']
                for r in clean['fixed_scenes']) for s in clean['shards']):
            raise ValueError('Every assigned shard needs at least one fixed scene')
        clean['parallelism'] = len(clean['shards'])
        clean['partition'] = 'round_robin_groups_v1'
        if config.get('gpu_uuids'):
            ids=config['gpu_uuids']
            if (not isinstance(ids,list) or len(ids)<clean['parallelism'] or len(set(ids))!=len(ids)
                    or any(not isinstance(g,str) or not re.fullmatch(r'GPU-[a-fA-F0-9-]+',g) for g in ids)):
                raise ValueError('Invalid physical GPU assignment')
            clean['gpu_uuids']=ids
        if config.get('paired_camera_policy'):
            from .camera_pairing import read_camera_policy
            policy = config['paired_camera_policy']
            self.path(policy['path'])
            read_camera_policy(policy, root=self.root, checkpoint=checkpoint['checkpoint'],
                norm_sha256=checkpoint['norm_sha256'], paired_reference=clean.get('paired_reference'),
                scene_replay=clean.get('scene_replay'))
            clean['paired_camera_policy'] = policy
        from .fork2000_scope import validate_request
        validate_request(self.root, clean, checkpoint)
        identifier = uuid.uuid4().hex[:12]
        folder = self.root/'runs'/f'eval-{identifier}'
        folder.mkdir(parents=True)
        (folder/'request.json').write_text(json.dumps(clean,indent=2))
        timestamp=now()
        with self.connect() as db:
            db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)',
                (identifier,title.strip()[:120] or identifier,json.dumps(clean),'queued',timestamp,timestamp,None,None,None))
            for shard in clean['shards']:
                subset = {k:v for k,v in clean.items() if k != 'shards'}
                subset.update(tasks=shard['tasks'], settings=shard['settings'])
                if 'fixed_scenes' in clean:
                    subset['fixed_scenes'] = [r for r in clean['fixed_scenes']
                        if r['task'] in subset['tasks'] and r['setting'] in subset['settings']]
                db.execute('INSERT INTO job_shards VALUES (?,?,?,?,?,?,?,?,?)',
                    (identifier, shard['id'], shard['id'] if len(clean['shards']) > 1 else None,
                     json.dumps(subset), 'queued', None, None, None, timestamp))
                if len(clean['shards']) > 1:
                    child = folder/'shards'/str(shard['id'])
                    child.mkdir(parents=True)
                    (child/'request.json').write_text(json.dumps(subset,indent=2))
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
        shard=self.claim_shard(gpu,pid,lane=0)
        return self.job(shard['job_id']) if shard else None

    def update(self, identifier, status, error=None):
        with self.connect() as db:
            db.execute('UPDATE jobs SET status=?,error=?,updated=? WHERE id=?',(status,error,now(),identifier))

    def shards(self, identifier):
        with self.connect() as db:
            return [self.decode(r) for r in db.execute('SELECT * FROM job_shards WHERE job_id=? ORDER BY shard', (identifier,))]

    def pending(self, lane, identifier=None):
        with self.connect() as db:
            return db.execute("""SELECT 1 FROM job_shards s JOIN jobs j ON j.id=s.job_id
                WHERE s.status='queued' AND j.status IN ('queued','running') AND j.error IS NULL
                AND (s.lane IS NULL OR s.lane=?) AND (? IS NULL OR j.id=?) LIMIT 1""",
                (lane, identifier, identifier)).fetchone() is not None

    def claim_shard(self, gpu, pid, lane, identifier=None):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT 1 FROM job_shards WHERE gpu=? AND status='running'", (gpu,)).fetchone():
                return None
            if db.execute("""SELECT 1 FROM jobs j WHERE gpu=? AND status IN ('running','cancel_requested')
                AND NOT EXISTS (SELECT 1 FROM job_shards s WHERE s.job_id=j.id)""", (gpu,)).fetchone():
                return None
            candidates = db.execute("""SELECT s.* FROM job_shards s JOIN jobs j ON j.id=s.job_id
                WHERE s.status='queued' AND j.status IN ('queued','running') AND j.error IS NULL
                AND (s.lane IS NULL OR s.lane=?) AND (? IS NULL OR j.id=?)
                ORDER BY j.created,s.shard""", (lane, identifier, identifier)).fetchall()
            row=None
            for candidate in candidates:
                ids=json.loads(candidate['config']).get('gpu_uuids')
                if ids and gpu != ids[candidate['shard']]:
                    continue
                row=candidate
                break
            if row is None:
                return None
            db.execute("UPDATE job_shards SET status='running',gpu=?,pid=?,updated=? WHERE job_id=? AND shard=?",
                       (gpu, pid, now(), row['job_id'], row['shard']))
            devices = [r[0] for r in db.execute('SELECT DISTINCT gpu FROM job_shards WHERE job_id=? AND gpu IS NOT NULL', (row['job_id'],))]
            db.execute("UPDATE jobs SET status='running',gpu=?,updated=? WHERE id=?", (','.join(devices),now(),row['job_id']))
        result = self.decode(row)
        result.update(status='running',gpu=gpu,pid=pid)
        return result

    def finish_shard(self, identifier, shard, status, error=None):
        from .reports import report, csv_report
        from .planning import report_complete
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            parent = db.execute('SELECT status,error,config FROM jobs WHERE id=?',(identifier,)).fetchone()
            db.execute('UPDATE job_shards SET status=?,error=?,updated=? WHERE job_id=? AND shard=?',
                       (status,error,now(),identifier,shard))
            failure = error or parent['error']
            if status in ('failed','interrupted') or failure or parent['status']=='cancel_requested':
                db.execute("UPDATE job_shards SET status='cancelled',updated=? WHERE job_id=? AND status='queued'",(now(),identifier))
            states = [r[0] for r in db.execute('SELECT status FROM job_shards WHERE job_id=?',(identifier,))]
            if any(s in ('queued','running') for s in states):
                final = 'cancel_requested' if parent['status']=='cancel_requested' else 'running'
            elif failure or any(s in ('failed','interrupted') for s in states):
                final = 'failed'
            elif parent['status']=='cancel_requested' or 'cancelled' in states:
                final = 'cancelled'
            else:
                final = 'completed'
            if final in ('completed','cancelled','failed'):
                folder = self.run_folder(identifier)
                data = report(folder,json.loads(parent['config']))
                if final == 'completed' and not report_complete(data):
                    final, failure = 'failed', 'Aggregate report is incomplete or invalid; inspect shard records'
                # The DB transaction serializes finalizers; atomic replacement
                # gives readers either the old or the complete new report.
                for filename, text in [('report.json',json.dumps(data,indent=2)),('scores.csv',csv_report(data))]:
                    temporary=folder/(filename+'.tmp')
                    temporary.write_text(text,encoding='utf-8')
                    temporary.replace(folder/filename)
            db.execute('UPDATE jobs SET status=?,error=?,updated=? WHERE id=?',(final,failure,now(),identifier))
        return self.job(identifier)

    def shard_folder(self, identifier, shard):
        config=self.job(identifier)['config']
        if shard not in [s['id'] for s in config.get('shards', [])]:
            raise ValueError('Unknown shard')
        folder=self.run_folder(identifier)
        from .resume import shard_folder, load_resume
        return shard_folder(folder,shard) if len(config['shards'])>1 or load_resume(folder) else folder

    def cancel(self, identifier):
        if self.job(identifier)['config'].get('history_import'):
            raise ValueError('Imported historical jobs are read-only')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT status FROM jobs WHERE id=?',(identifier,)).fetchone()
            if not row: raise ValueError('Unknown job')
            status = 'cancelled' if row[0]=='queued' else 'cancel_requested' if row[0]=='running' else row[0]
            db.execute('UPDATE jobs SET status=?,updated=? WHERE id=?',(status,now(),identifier))
            if status in ('cancelled','cancel_requested'):
                db.execute("UPDATE job_shards SET status='cancelled',updated=? WHERE job_id=? AND status='queued'",(now(),identifier))
        return self.job(identifier)

    def heartbeat(self,gpu,pid,job=None):
        with self.connect() as db:
            if job is None and db.execute("SELECT 1 FROM job_shards WHERE gpu=? AND status='running' AND pid!=?",(gpu,pid)).fetchone():
                return
            db.execute('INSERT OR REPLACE INTO workers VALUES (?,?,?,?)',(gpu,pid,now(),job))

    def workers(self):
        with self.connect() as db:return [dict(r) for r in db.execute('SELECT * FROM workers')]

    def run_folder(self, identifier):
        if not re.fullmatch('[a-f0-9]{12}',identifier): raise ValueError('Invalid job ID')
        return self.root/'runs'/f'eval-{identifier}'
