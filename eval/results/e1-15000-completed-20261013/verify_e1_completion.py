import sys,json,hashlib,math,csv,io,subprocess
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
from PIL import Image
ROOT=Path('/root/shared-nvme/lingbot-eval-team')
sys.path.insert(0,str(ROOT/'src'))
from eval_platform.store import Store
store=Store(ROOT);job_id='232fb68a8d15';job=store.job(job_id);config=job['config']
ref_path=Path(config['paired_reference']['path']);ref=json.loads(ref_path.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
key=lambda r:(r['task'],r['setting'],int(r['seed']))
refs={key(r):r for r in ref['scenes']}
issues=[];checks=Counter();rows=[];folders=[];files=[];outcomes=[];runtime=[];groups=Counter();succ=Counter();actions=chunks=0;pixel_count=0
def check(value,label,scene=None):
 checks[label]+=1
 if not value:issues.append({'check':label,'scene':scene})
check(job['status']=='completed' and job.get('error') is None,'job_completed')
check(len(refs)==500 and len(ref['scenes'])==500,'reference_unique500')
check(sha(ref_path)=='ec14ae2092bca38a689feca77b98dc89317bf9add20b820dd67e2d4bd503dd20','reference_sha')
check(len(config['fixed_scenes'])==500 and {key(r) for r in config['fixed_scenes']}==set(refs),'config_fixed500')
for field,value in [('precision','fp32'),('checkpoint_id','1b6e09f23423'),('expert_check',False),('observer_video',True),('video',True),('video_fps',10),('execute',50),('release_unused_cuda_cache',False),('protocol','e0_fixed500_scene_replay_policy_only_v4'),('inference_profile','official_defaults_v1'),('moe_backend','official')]:
 check(config.get(field)==value,'config_'+field)
scene_replay=ROOT/'incoming/E1-step-15000/scene-replay.json'
check(sha(scene_replay)=='e7d34a4f042d81bee8c0bf5183f982032f6fe7c0ee9c68f43e8d294526612397','scene_replay_sha')
for shard in store.shards(job_id):
 sid=shard['shard'];folder=store.shard_folder(job_id,sid);out=folder/'evaluation';folders.append(str(folder))
 check(shard['status']=='completed' and shard.get('error') is None,'shard_completed',sid)
 rt=json.loads((folder/'server-runtime.json').read_text());contract=rt['robot_action_contract']
 check(rt['dtype']=='fp32' and rt['actual_parameter_dtypes_numel']=={'torch.float32':6375906359},'runtime_fp32',sid)
 check(rt['learned_parameter_precision_check']['exact_saved_fp32_values'] is True,'runtime_original_learned_fp32',sid)
 check(contract['normalization_sha256']=='7e6ac279d147cb8422108d36216768111c19bb2113ca9eee1a23df22df238df0','runtime_own_norm',sid)
 check(contract['action_subtract_state']=={'action.arm.position':True,'action.effector.position':False},'runtime_relative_arm_absolute_gripper',sid)
 check(contract['robot_config_sha256']=='49d07e5dcff216df70abfed89c2eaf13c6a2aa12eba3d024d7b83a89e3bf70b5','runtime_robot_yaml',sid)
 runtime.append({'shard':sid,'path':str(folder/'server-runtime.json'),'sha256':sha(folder/'server-runtime.json'),'dtype':rt['dtype'],'parameter_numel':rt['actual_parameter_dtypes_numel'],'learned_precision':rt['learned_parameter_precision_check'],'contract':contract})
 ledger=out/'episodes.jsonl';lane=[json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
 check(len(lane)==250,'shard_250',sid);rows.extend(lane)
 files.append({'path':str(ledger),'bytes':ledger.stat().st_size,'sha256':sha(ledger)})
 for r in lane:
  k=key(r);scene=list(k);expected=refs.get(k);groups[k[:2]]+=1
  check(expected is not None,'key_in_reference',scene)
  if expected is None:continue
  check(r['status']=='policy_completed','complete_policy',scene)
  check(r.get('expert_attempts')==0 and r.get('expert',{}).get('phase')=='not_launched' and r.get('expert',{}).get('attempts')==0,'expert_not_launched',scene)
  check(r.get('policy_episode_index')==expected['policy_episode_index'],'policy_index',scene)
  check(r.get('language_seed')==k[2],'setup_language_seed',scene)
  check(r.get('instruction')==expected['instruction'],'instruction',scene)
  check(r.get('generated_instruction')==expected['instruction'] and r.get('instruction_source')=='frozen_E0_final_episode','fixed_instruction_source',scene)
  d=out/k[0]/k[1]/str(k[2]);episode=json.loads((d/'episode.json').read_text())
  check(episode==r,'ledger_episode_equal',scene)
  ins=json.loads((d/'instruction.json').read_text())
  check(ins['instruction']==expected['instruction'] and ins['parameters']==expected['parameters'],'instruction_parameters',scene)
  p=r['paired_input_check'];check(p['source_job']=='843758599804' and p['instruction_equal'] is True and p['initial_state_equal'] is True,'paired_guard',scene)
  check(p['camera_sha256']==expected['initial_camera_sha256'] and all(p['camera_equal'].get(cam) is True for cam in ['head_camera','left_camera','right_camera']),'three_camera_metadata_sha',scene)
  check(p['reference_episode']==expected['e0_episode_record'],'reference_episode_binding',scene)
  replay=r['scene_replay'];entry=r['policy_entry']
  check(replay.get('expert_called') is False and replay.get('simulation_steps_added')==0 and replay.get('instruction_regenerated') is False,'scene_replay_policy_only',scene)
  check(entry.get('flow')=='frozen_e0_no_recheck_v1' and entry.get('expert_called') is False and entry.get('simulation_steps_added')==0,'entry_policy_only',scene)
  check(r['video_requested'] is True and r['observer_video_requested'] is True and r['requested_video_fps']==10,'video_request',scene)
  tc=json.loads((d/'paired-input-check.json').read_text());check(tc==p,'paired_receipt_equal',scene)
  tl=d/'trajectory.jsonl';trace=[json.loads(x) for x in tl.read_text().splitlines() if x.strip()]
  check(len(trace)==r['chunks'] and r['chunks']==math.ceil(r['actions']/50),'completed_chunk_count',scene)
  check(0<r['actions']<=r['action_limit'] and (r['success'] or r['actions']==r['action_limit']),'action_budget',scene)
  if trace:
   state=trace[0]['state_before'];target=expected['initial_state']
   check(len(state)==len(target)==14 and all(abs(a-b)<=1e-6 for a,b in zip(state,target)),'actual_initial_joints',scene)
  previous=0
  for n,c in enumerate(trace):
   a=c['predicted_actions']
   check(len(a)==50 and all(len(x)==14 and all(math.isfinite(v) for v in x) for x in a),'finite_50x14_prediction',scene)
   check(c['chunk']==n and c['action_count_before']==previous and 0<c['action_count_after']-previous<=50,'action_block_progress',scene)
   previous=c['action_count_after']
  check(previous==r['actions'],'final_action_count',scene)
  for camera,suffix in [('head_camera',''),('left_camera','-left-wrist'),('right_camera','-right-wrist')]:
   png=d/('chunk-000'+suffix+'.png')
   check(png.is_file(),'initial_png_exists',scene)
   if png.exists():
    with Image.open(png) as image:rgb_sha=hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
    check(rgb_sha==expected['initial_camera_sha256'][camera],'actual_initial_rgb_sha',scene);pixel_count+=1
  for camera in ['head_camera','observer_camera']:
   video=r['videos'].get(camera)
   check(video is not None,'both_video_metadata',scene)
   if video:
    cap=video['capture'];vp=out/video['path']
    check(cap['terminal_frame_included'] is True and cap['fps']==10 and cap['last_action_count']==r['actions'] and cap['frames']>0 and vp.is_file() and vp.stat().st_size>0,'complete_video_recorded',scene)
  actions+=r['actions'];chunks+=r['chunks'];succ[k[1]]+=int(r['success'])
  outcomes.append({'task':k[0],'setting':k[1],'seed':k[2],'status':r['status'],'success':r['success'],'actions':r['actions'],'chunks':r['chunks'],'instruction_equal':True,'parameters_equal':True,'index_equal':True,'initial_joints_equal':True,'three_initial_camera_sha_equal':True,'expert_attempts':r['expert_attempts']})
check(len(rows)==500 and len({key(r) for r in rows})==500 and {key(r) for r in rows}==set(refs),'completed_unique500')
check(len(groups)==100 and all(v==5 for v in groups.values()),'100_groups5')
processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True).stdout
summary={'passed':not issues,'verified_at':datetime.now(timezone.utc).isoformat(),'job_id':job_id,'model_id':'1b6e09f23423','title':job['title'],'status':job['status'],'job_time_fields':{k:v for k,v in job.items() if 'time' in k or 'finish' in k or 'end' in k},'unique_keys':len({key(r) for r in rows}),'groups':len(groups),'completed':sum(r['status']=='policy_completed' for r in rows),'timeout_zero':sum(r['status']=='policy_timeout' for r in rows),'success':sum(bool(r['success']) for r in rows),'success_by_setting':dict(succ),'denominator_by_setting':dict(Counter(r['setting'] for r in rows)),'total_actions':actions,'finite_prediction_chunks':chunks,'initial_rgb_pngs_checked':pixel_count,'all_expert_attempts':sum(r.get('expert_attempts',0) for r in rows),'checks':dict(checks),'issues':issues,'folders':folders,'reference_sha256':sha(ref_path),'scene_replay_sha256':sha(scene_replay),'gpu_compute_processes':processes,'runtime':runtime,'source_ledgers':files,'limits':['E0 historical inference used BF16; E1 actual parameter FP32 is not an identical-precision comparison','Training-side before/after export fixed-noise complete action audit was not supplied','Model processes started RNG independently; old model RNG stream was not restored','Development evaluation, not an official competition submission']}
destination=ROOT/'artifacts/e1-completed-20261013-0747';destination.mkdir(parents=True,exist_ok=False)
(destination/'verification.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=list(outcomes[0]));writer.writeheader();writer.writerows(outcomes);(destination/'paired-outcomes.csv').write_text(buf.getvalue())
buf=io.StringIO();w=csv.writer(buf);w.writerow(['task','setting','scored','completed','timeout_zero','success'])
for k,count in sorted(groups.items()):w.writerow([*k,count,count,0,sum(r['success'] for r in rows if key(r)[:2]==k)])
(destination/'by-group.csv').write_text(buf.getvalue())
print(json.dumps({'passed':summary['passed'],'artifact':str(destination),'counts':{k:summary[k] for k in ['unique_keys','groups','completed','timeout_zero','success','success_by_setting','denominator_by_setting','finite_prediction_chunks','initial_rgb_pngs_checked','all_expert_attempts']},'issues':issues[:20],'verification_sha256':sha(destination/'verification.json')},ensure_ascii=False))
if issues:sys.exit(2)

