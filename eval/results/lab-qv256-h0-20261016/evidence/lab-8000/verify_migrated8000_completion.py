"""CPU completion audit only: fixed inputs, PNG pixels, action traces and video metadata."""
import sys,json,hashlib,math,csv,io,subprocess
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
from PIL import Image
ROOT=Path('/data/lingbotvla/eval-team-lab8000-migrated-20261013')
DEST=ROOT/'artifacts/lab8000-completed-20261014-0705'
sys.path.insert(0,str(ROOT/'src'))
from eval_platform.store import Store
from eval_platform.resume import load_resume
from eval_platform.reports import report
store=Store.__new__(Store);store.root=ROOT;store.state=ROOT/'platform-state';store.db=store.state/'queue.sqlite3'
JOB='7cc771a96d5c';MODEL='f8ba918c658a'
job=store.job(JOB);config=job['config'];run=store.run_folder(JOB)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
key=lambda r:(r['task'],r['setting'],int(r['seed']))
issues=[];checks=Counter();rows=[];outcomes=[];groups=Counter();success_by_setting=Counter()
hosts=Counter();host_success=Counter();camera_equal_by_host=Counter();camera_diff_by_host=Counter()
runtimes=[];ledgers=[];actions=chunks=rgb_pngs=terminal_pngs=video_files=0
last_times=[]
def check(value,label,scene=None):
    checks[label]+=1
    if not value:issues.append(dict(check=label,scene=scene))
def read(path):return json.loads(Path(path).read_text())
def png_sha(path):
    with Image.open(path) as image:return hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
check(job['status']=='completed' and job.get('error') is None,'job_completed')
ref_path=Path(config['paired_reference']['path']);reference=read(ref_path);refs={key(r):r for r in reference['scenes']}
check(len(reference['scenes'])==len(refs)==500,'reference_unique500')
check(sha(ref_path)==config['paired_reference']['sha256']=='ec14ae2092bca38a689feca77b98dc89317bf9add20b820dd67e2d4bd503dd20','reference_sha')
check(len(config['fixed_scenes'])==500 and {key(r) for r in config['fixed_scenes']}==set(refs),'config_fixed500')
for field,value in [('precision','fp32'),('checkpoint_id',MODEL),('expert_check',False),('observer_video',True),('video',True),('video_fps',10),('execute',50),('release_unused_cuda_cache',False),('protocol','e0_fixed500_scene_replay_policy_only_v4'),('inference_profile','official_defaults_v1'),('moe_backend','official')]:
    check(config.get(field)==value,'config_'+field)
replay_path=Path(config['scene_replay']['path'])
check(sha(replay_path)==config['scene_replay']['sha256']=='e7d34a4f042d81bee8c0bf5183f982032f6fe7c0ee9c68f43e8d294526612397','scene_replay_sha')
camera_path=Path(config['paired_camera_policy']['path'])
check(sha(camera_path)==config['paired_camera_policy']['sha256']=='aa8f07f4b5a7971bd37795e6cb5d4d83b7a99d282d57d8867333f1ac97594954','camera_authorization_sha')
migration_path=Path(config['host_migration']['path'])
check(sha(migration_path)==config['host_migration']['sha256'],'host_migration_receipt')
host_proof=read(migration_path)
check(host_proof['passed'] and host_proof['original_parameter_hashes_equal'] and host_proof['original_all_14_shards_byte_equal'],'original_model_restore_bound')
restore_path=ROOT/'incoming/Lab-qv256-step-8000-20261013/export8000/checkpoints/global_step_8000/hf_ckpt_fp32/model-restore-complete.json'
restore=read(restore_path)
check(sha(restore_path)==host_proof['model_restore_receipt_sha256'] and restore['passed'] and restore['original_all_14_shards_byte_equal'],'restoration_receipt_unchanged')
order_path=ROOT/'artifacts/migration-soarm-20261013/e0-enumeration.json'
check(sha(order_path)=='a8007f917479f259ac1191595dce5761ab5b4125f295ca8e051ac119deeca3d9','E0_asset_enumeration_binding')
manifest=load_resume(run)
canonical=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
check(canonical==manifest['config_sha256'],'current_manifest_config')
prerequisites=[]
for item in manifest['required_files']:
    p=ROOT/item['path']
    check(p.is_file() and p.stat().st_size<8*1024*1024,'metadata_prerequisite_bounded',item['path'])
    if p.is_file() and p.stat().st_size<8*1024*1024:
        actual=sha(p);check(actual==item['sha256'],'metadata_prerequisite_unchanged',item['path'])
        prerequisites.append(dict(path=item['path'],sha256=actual))
expected_prefix=['f22cac660381d6f6b83f598f64c24b653e51b4853c4a663dd1d8f805c68d6e24','9a763fbeed25ad087a997da7853059c04af60261ca7605b0c0f5c8c7f7574bbb']
for shard in store.shards(JOB):
    sid=shard['shard'];folder=store.shard_folder(JOB,sid);out=folder/'evaluation'
    check(shard['status']=='completed' and shard.get('error') is None,'shard_completed',sid)
    rt_path=folder/'server-runtime.json';rt=read(rt_path);contract=rt['robot_action_contract']
    check(rt['dtype']=='fp32' and rt['actual_parameter_dtypes_numel']=={'torch.float32':6375906359} and rt['actual_parameter_dtypes_tensors']=={'torch.float32':1672},'runtime_fp32_all_parameters',sid)
    pc=rt['learned_parameter_precision_check']
    check(pc['exact_saved_fp32_values'] is True and pc['learned_tensors']==1672,'runtime_saved_parameter_values',sid)
    check(contract['normalization_sha256']==rt['normalization_sha256']=='b24e6a7ccf44fa3ecc9dbbe060cf9b0937cf3307bc2a68232bc98c4578c22f6d','runtime_own_norm',sid)
    check(contract['action_subtract_state']=={'action.arm.position':True,'action.effector.position':False},'relative_arm_absolute_gripper',sid)
    check(contract['robot_config_sha256']=='1ef96ff0d213561174813345f8674fc693f4831cde144e00e5b9781d7147876b','runtime_robot_yaml',sid)
    check(rt['use_compile'] is True and rt['image_size']==256 and rt['num_steps']==10 and rt['execute']==50 and rt['flow_noise_generation']=='upstream_native' and rt['matmul_precision']=='high' and rt['moe_backend']=='official' and rt['vit_attn_implementation']=='flash_attention_2','runtime_inference_recipe',sid)
    runtimes.append(dict(shard=sid,path=str(rt_path),sha256=sha(rt_path),dtype=rt['dtype'],parameter_numel=rt['actual_parameter_dtypes_numel'],precision_check=pc,contract=contract))
    ledger=out/'episodes.jsonl';raw=ledger.read_bytes();lane=[json.loads(x) for x in raw.splitlines() if x.strip()]
    source=run/'shards'/str(sid)/'evaluation/episodes.jsonl';prefix=source.read_bytes();prefix_rows=[json.loads(x) for x in prefix.splitlines()];source_keys={key(r) for r in prefix_rows}
    check(raw.startswith(prefix) and sha(source)==expected_prefix[sid] and len(prefix_rows)==[80,92][sid],'original_prefix_exact_bytes',sid)
    check(len(lane)==250,'shard_250',sid);rows.extend(lane)
    ledgers.append(dict(shard=sid,path=str(ledger),sha256=sha(ledger),bytes=ledger.stat().st_size,original_prefix_sha256=sha(source),original_prefix_records=len(prefix_rows)))
    for r in lane:
        k=key(r);scene=list(k);expected=refs.get(k);groups[k[:2]]+=1
        check(expected is not None,'key_in_reference',scene)
        if expected is None:continue
        origin='rented5090' if k in source_keys else 'soarm101'
        hosts[origin]+=1;host_success[origin]+=int(r['success'])
        check(r['status']=='policy_completed','complete_policy',scene)
        check(r.get('expert_attempts')==0 and r.get('expert',{}).get('phase')=='not_launched' and r.get('expert',{}).get('attempts')==0,'expert_not_launched',scene)
        check(r.get('policy_episode_index')==expected['policy_episode_index'],'fixed_policy_index',scene)
        check(r.get('language_seed')==k[2],'fixed_seed',scene)
        check(r.get('instruction')==expected['instruction'] and r.get('generated_instruction')==expected['instruction'] and r.get('instruction_source')=='frozen_E0_final_episode','fixed_instruction',scene)
        d=out/k[0]/k[1]/str(k[2])
        check(read(d/'episode.json')==r,'ledger_episode_equal',scene)
        ins=read(d/'instruction.json')
        check(ins['instruction']==expected['instruction'] and ins['parameters']==expected['parameters'],'fixed_instruction_parameters',scene)
        p=r['paired_input_check']
        check(p['source_job']=='843758599804' and p['instruction_equal'] is True and p['initial_state_equal'] is True,'paired_input_guard',scene)
        check(p['reference_episode']==expected['e0_episode_record'],'reference_episode_binding',scene)
        check(read(d/'paired-input-check.json')==p,'paired_receipt_equal',scene)
        if origin=='soarm101':
            check(p.get('camera_policy')=='e0_cross_host_rgb_recorded_v1' and p.get('camera_policy_receipt_sha256')==config['paired_camera_policy']['sha256'],'cross_host_camera_authorized',scene)
        replay=r['scene_replay'];entry=r['policy_entry']
        check(replay.get('expert_called') is False and replay.get('simulation_steps_added')==0 and replay.get('instruction_regenerated') is False,'scene_replay_policy_only',scene)
        check(entry.get('flow')=='frozen_e0_no_recheck_v1' and entry.get('expert_called') is False and entry.get('simulation_steps_added')==0,'policy_entry_no_expert',scene)
        check(r['video_requested'] is True and r['observer_video_requested'] is True and r['requested_video_fps']==10,'original_video_request',scene)
        trace=[json.loads(x) for x in (d/'trajectory.jsonl').read_text().splitlines() if x.strip()]
        check(len(trace)==r['chunks'] and r['chunks']==math.ceil(r['actions']/50),'completed_chunk_count',scene)
        check(0<r['actions']<=r['action_limit'] and (r['success'] or r['actions']==r['action_limit']),'action_budget',scene)
        if trace:
            state=trace[0]['state_before'];target=expected['initial_state']
            check(len(state)==len(target)==14 and all(abs(a-b)<=1e-6 for a,b in zip(state,target)),'actual_initial_joints',scene)
        previous=0
        for n,c in enumerate(trace):
            a=c['predicted_actions']
            check(len(a)==50 and all(len(x)==14 and all(math.isfinite(v) for v in x) for x in a),'finite_50x14_prediction',scene)
            check(c['chunk']==n and c['action_count_before']==previous and 0<c['action_count_after']-previous<=50,'complete_action_block',scene)
            previous=c['action_count_after']
        check(previous==r['actions'],'final_action_count',scene)
        all_equal=True
        for camera,suffix in [('head_camera',''),('left_camera','-left-wrist'),('right_camera','-right-wrist')]:
            png=d/('chunk-000'+suffix+'.png');check(png.is_file(),'initial_png_exists',scene)
            if png.exists():
                rgb=png_sha(png);check(rgb==p['camera_sha256'][camera],'initial_png_actual_rgb_sha',scene);rgb_pngs+=1
                equal=rgb==expected['initial_camera_sha256'][camera]
                check(p['camera_equal'].get(camera) is equal,'camera_equality_record_truth',scene)
                camera_equal_by_host[(origin,camera)]+=int(equal);camera_diff_by_host[(origin,camera)]+=int(not equal);all_equal=all_equal and equal
        if origin=='rented5090':check(all_equal,'original_host_E0_pixel_identity',scene)
        for camera,terminal in [('head_camera','final.png'),('observer_camera','final-observer.png')]:
            video=r['videos'].get(camera);check(video is not None,'head_observer_video_metadata',scene)
            if video:
                cap=video['capture'];vp=out/video['path'];png=d/terminal
                check(cap['terminal_frame_included'] is True and cap['fps']==10 and cap['last_action_count']==r['actions'] and cap['frames']>0 and vp.is_file() and vp.stat().st_size>0,'complete_video_file_metadata',scene)
                check(cap['extra_physics_steps']==0 and cap['extra_policy_observations']==0,'video_no_extra_policy_steps',scene)
                check(png.is_file() and png_sha(png)==cap['terminal_rgb_sha256'],'terminal_png_actual_rgb_sha',scene)
                terminal_pngs+=int(png.is_file());video_files+=int(vp.is_file() and vp.stat().st_size>0)
        actions+=r['actions'];chunks+=r['chunks'];success_by_setting[k[1]]+=int(r['success'])
        last_times.append(r.get('time',''))
        outcomes.append(dict(task=k[0],setting=k[1],seed=k[2],host=origin,status=r['status'],success=r['success'],actions=r['actions'],chunks=r['chunks'],instruction_equal=True,parameters_equal=True,index_equal=True,initial_joints_equal=True,three_initial_camera_sha_equal=all_equal,expert_attempts=r['expert_attempts']))
check(len(rows)==500 and len({key(r) for r in rows})==500 and {key(r) for r in rows}==set(refs),'completed_unique500')
check(len(groups)==100 and all(v==5 for v in groups.values()),'100_groups_each5')
check(hosts=={'rented5090':172,'soarm101':328},'migration_partition_counts')
result=report(run,config)
check(result['scored']==result['valid']==500 and result['execution_errors']==0 and result['record_error'] is None and result['timeout_failures']==0,'report_no_unhandled_errors')
processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
check(not processes,'GPU_compute_exited')
old_maintenance=ROOT.parent/'eval-team-lab15000-20261011/platform-state/maintenance.json'
check(sha(old_maintenance)=='47dff756eec44e4585f1be13bb7c2e409dcce2e908eb73e958dcba0206f3d5d6','old_lab_maintenance_unchanged')
summary=dict(passed=not issues,verified_at=datetime.now(timezone.utc).isoformat(),job_id=JOB,model_id=MODEL,title=job['title'],status=job['status'],completed_at=job.get('updated'),unique_keys=len({key(r) for r in rows}),groups=len(groups),completed=sum(r['status']=='policy_completed' for r in rows),timeout_zero=sum(r['status']=='policy_timeout' for r in rows),success=sum(bool(r['success']) for r in rows),success_by_setting=dict(success_by_setting),denominator_by_setting=dict(Counter(r['setting'] for r in rows)),host_counts=dict(hosts),host_success=dict(host_success),camera_equal_by_host={origin:{cam:camera_equal_by_host[(origin,cam)] for cam in ('head_camera','left_camera','right_camera')} for origin in hosts},camera_different_by_host={origin:{cam:camera_diff_by_host[(origin,cam)] for cam in ('head_camera','left_camera','right_camera')} for origin in hosts},total_actions=actions,finite_prediction_chunks=chunks,initial_rgb_pngs_checked=rgb_pngs,terminal_pngs_checked=terminal_pngs,nonempty_video_files_metadata_checked=video_files,all_expert_attempts=sum(r.get('expert_attempts',0) for r in rows),checks=dict(checks),issues=issues,current_segment=manifest['segment'],runtime=runtimes,source_ledgers=ledgers,metadata_prerequisites=prerequisites,reference_sha256=sha(ref_path),scene_replay_sha256=sha(replay_path),camera_authorization_sha256=sha(camera_path),model_restore_receipt_sha256=sha(restore_path),gpu_compute_processes=processes,limits=['172 original rented-host episodes retained E0 initial RGB identity; the 328 soarm episodes use authorized recorded cross-host RGB differences. Fixed instruction, seed, parameters, index and joints remain checked.','E0 historical inference used BF16; this model has FP32 parameters, so precision is not identical.','The before/after merge and reload fixed-input/noise action audit used a common reference backend; it is not a claim of bitwise equality to training mixed precision or the production official backend.','Migration and model process restart did not restore the old model RNG stream.','Development evaluation, not an official competition submission.','The enumeration repair does not establish the cause or permanent resolution of the historical Xid109 fault.'])
DEST.mkdir(parents=True,exist_ok=True)
assert not (DEST/'verification.json').exists(),'Do not overwrite existing completion evidence'
(DEST/'verification.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(outcomes[0]));w.writeheader();w.writerows(outcomes);(DEST/'paired-outcomes.csv').write_text(buf.getvalue())
buf=io.StringIO();w=csv.writer(buf);w.writerow(['task','setting','scored','completed','timeout_zero','success'])
for k,count in sorted(groups.items()):w.writerow([*k,count,count,0,sum(r['success'] for r in rows if key(r)[:2]==k)])
(DEST/'by-group.csv').write_text(buf.getvalue())
manifest_out=dict(passed=summary['passed'],files=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(DEST.iterdir()) if p.is_file()])
(DEST/'receipt-manifest.json').write_text(json.dumps(manifest_out,indent=2))
print(json.dumps(dict(passed=summary['passed'],artifact=str(DEST),counts={k:summary[k] for k in ('unique_keys','groups','completed','timeout_zero','success','success_by_setting','denominator_by_setting','host_counts','host_success','camera_equal_by_host','camera_different_by_host','finite_prediction_chunks','initial_rgb_pngs_checked','terminal_pngs_checked','nonempty_video_files_metadata_checked','all_expert_attempts')},issues=issues[:25],verification_sha256=sha(DEST/'verification.json')),ensure_ascii=False))
if issues:sys.exit(2)
