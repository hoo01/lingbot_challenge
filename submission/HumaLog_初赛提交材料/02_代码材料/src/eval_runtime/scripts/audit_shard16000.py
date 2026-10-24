"""Full CPU audit of a complete16000 shard; no simulation or model access."""
import csv,hashlib,io,json,math,os,sys,time
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from eval_platform.fork2000_plan import validate_history,is_scored,retained_error
from eval_platform.expert_outcomes import acceptable_record

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv_bytes(rows,fields):
 s=io.StringIO();w=csv.DictWriter(s,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows);return s.getvalue().encode()

def main():
 scope=json.loads((ROOT/'artifacts/eval2000-scope.json').read_text());out=ROOT/'run/evaluation'
 assert (ROOT/'artifacts/finished-unverified.json').exists() and not (ROOT/'user-stop.json').exists()
 protocol=json.loads((out/'protocol.json').read_text());meta=protocol['server']
 assert meta['model_revision']==scope['model_revision'] and meta['frozen_protocol_sha256']==scope['protocol_sha256']
 assert meta['learned_parameter_precision_check']['binding_sha256']==scope['runtime_parameter_audit_sha256']
 assert meta['learned_parameter_precision_check']['exact_saved_fp32_values'] and meta['dtype']=='fp32'
 assert meta['normalization_sha256']==scope['norm_sha256'] and meta['robot_action_contract']['robot_config_sha256']==scope['robot_sha256']
 assert not protocol['recording']['enabled'] and protocol['recording']['observer_video'] is False
 rows=[json.loads(l) for l in (out/'episodes.jsonl').read_text().splitlines()]
 groups=validate_history(protocol,rows);assert len(groups)==len(scope['tasks'])
 assert all(sum(map(is_scored,g))==30 for g in groups.values())
 assert all(acceptable_record(protocol,r) for r in rows)
 assert not list(out.rglob('*.mp4'))
 scored=[];errors=[];predictions=0;pngs=0
 from PIL import Image
 for r in rows:
  folder=out/r['task']/r['setting']/str(r['seed'])
  assert json.loads((folder/'episode.json').read_text())==r
  if not is_scored(r):
   assert retained_error(r);errors.append(r);continue
  assert r.get('model_revision',scope['model_revision'])==scope['model_revision']
  assert r.get('actual_environment_seed',r['seed'])==r['seed'] and 100000<=r['seed']<100300
  initial=json.loads((folder/'initial-inputs.json').read_text());assert len(initial['initial_state'])==14 and len(initial['cameras'])==3
  for camera,name in [('head_camera','chunk-000.png'),('left_camera','chunk-000-left-wrist.png'),('right_camera','chunk-000-right-wrist.png')]:
   image=Image.open(folder/name).convert('RGB');assert hashlib.sha256(image.tobytes()).hexdigest()==initial['cameras'][camera]['rgb_sha256'];pngs+=1
  pred=[json.loads(l) for l in (folder/'predictions.jsonl').read_text().splitlines()]
  trace=[json.loads(l) for l in (folder/'trajectory.jsonl').read_text().splitlines()] if (folder/'trajectory.jsonl').exists() else []
  assert pred, 'No valid H50 evidence'
  for i,p in enumerate(pred):
   a=p['predicted_actions'];assert p['chunk']==i and len(a)==50 and all(len(v)==14 and all(math.isfinite(x) for x in v) for v in a)
   predictions+=1
  for p,t in zip(pred,trace):
   assert all(t[k]==v for k,v in p.items()) and 0<t['action_count_after']-t['action_count_before']<=25
  if r['status']=='policy_completed':
   assert len(pred)==len(trace)==r['chunks']
   for n in ['final.png','final-left-wrist.png','final-right-wrist.png']:Image.open(folder/n).verify();pngs+=1
  scored.append(dict(task=r['task'],setting=r['setting'],seed=r['seed'],success=int(r['success']),status=r['status'],seconds=r.get('seconds'),actions=r.get('actions'),mean_inference_ms=r.get('mean_inference_ms'),model_revision=scope['model_revision'],protocol_sha256=scope['protocol_sha256']))
 assert len(scored)==scope['planned']
 by=[dict(task=t,setting=s,scored=30,success=sum(r['success'] for r in g if is_scored(r)),policy_timeout=sum(r['status']=='policy_timeout' for r in g),non_scored=len(g)-30) for (t,s),g in groups.items()]
 verification=dict(passed=True,shard_id=scope['shard_id'],root=str(ROOT),revision=scope['model_revision'],protocol_sha256=scope['protocol_sha256'],scored=len(scored),success=sum(r['success'] for r in scored),groups=len(groups),per_group=30,timeout=sum(r['status']=='policy_timeout' for r in scored),non_scored=len(errors),finite_h50_blocks=predictions,rgb_pngs=pngs,video_disabled_verified=True,source_complete_sha256=scope['source_complete_sha256'],runtime_parameter_audit_sha256=scope['runtime_parameter_audit_sha256'],time=time.time())
 complete=ROOT/'artifacts/completion';assert not complete.exists();complete.mkdir()
 files={'verification.json':(json.dumps(verification,indent=2)+'\n').encode(),
  'outcomes.csv':csv_bytes(scored,['task','setting','seed','success','status','seconds','actions','mean_inference_ms','model_revision','protocol_sha256']),
  'by-group.csv':csv_bytes(by,['task','setting','scored','success','policy_timeout','non_scored']),
  'non-scored-errors.csv':csv_bytes(errors,['task','setting','seed','status','success','expert_attempts','native_initialization_error','policy_entered','error'])}
 for n,b in files.items():(complete/n).write_bytes(b)
 manifest=dict(passed=True,shard_id=scope['shard_id'],revision=scope['model_revision'],protocol_sha256=scope['protocol_sha256'],files={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in files.items()})
 (complete/'receipt-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 publication=dict(passed=True,shard_id=scope['shard_id'],files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in complete.iterdir()})
 temp=ROOT/'artifacts/completion-publication.tmp';temp.write_text(json.dumps(publication,indent=2));temp.replace(ROOT/'artifacts/completed.json')
 print(json.dumps(verification))

if __name__=='__main__':main()
