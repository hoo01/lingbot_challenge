from .fork2000_plan import active, TARGET
import csv
from io import StringIO
import json
from collections import defaultdict
from .resume import load_resume, shard_folder
from .planning import validate_records, group_seeds, planned_count
from .expert_outcomes import acceptable_record, exception_observed, settled_expert_error
from .timeout_outcomes import timeout_failure


def report(folder, config):
    output=folder/'evaluation'
    rows=[]
    parse_error=None
    progress_list=[]
    shards=config.get('shards', [])
    recovery=load_resume(folder)
    history=recovery.get('historical_interruptions',[]) if recovery else []
    sources=[(shard_folder(folder,s['id'])/'evaluation', s) for s in shards] if shards and (len(shards)>1 or recovery) else [(output, None)]
    seen=set()
    for source, shard in sources:
        path=source/'episodes.jsonl'
        if path.exists():
            raw=path.read_text();lines=raw.splitlines()
            for index,line in enumerate(lines):
                try:
                    row=json.loads(line)
                    key=(row['task'],row['setting'],row['seed'])
                except (json.JSONDecodeError,KeyError,TypeError):
                    if index==len(lines)-1 and not raw.endswith('\n'):break
                    parse_error=f'Malformed episode record in {path.relative_to(folder)} at line {index+1}'
                    break
                if key in seen:
                    parse_error=f'Duplicate episode record: {key}'
                    continue
                if shard and (row['task'] not in shard['tasks'] or row['setting'] not in shard['settings'] or (not config.get('valid_per_group') and row['seed'] not in config['seeds'])):
                    parse_error=f'Episode outside assigned shard: {key}'
                    continue
                seen.add(key)
                if shard:
                    row.update(shard=shard['id'],artifact_root=source.relative_to(folder).as_posix())
                rows.append(row)
        try:
            item=json.loads((source/'worker-progress.json').read_text())
            if shard:item['shard']=shard['id']
            progress_list.append(item)
        except (FileNotFoundError,json.JSONDecodeError):pass
    if config.get('valid_per_group') or config.get('fixed_scenes') is not None:
        try:
            validate_records(config, rows)
        except (ValueError, KeyError, TypeError) as error:
            parse_error = str(error)
    groups=[]
    for task in config['tasks']:
        for setting in config['settings']:
            planned = TARGET if active(config) else config.get('valid_per_group', len(group_seeds(config, task, setting)))
            if planned == 0:
                continue
            subset=[r for r in rows if r['task']==task and r['setting']==setting]
            valid=[r for r in subset if r['status']=='policy_completed']
            timeouts=[r for r in subset if timeout_failure(config,r)]
            interrupted=[r for r in config.get('unrecorded_execution_attempts',[]) if r['task']==task and r['setting']==setting]
            past=[r for r in history if r['task']==task and r['setting']==setting]
            groups.append(dict(task=task,setting=setting,planned=planned,attempted=len(subset)+len(interrupted),
                valid=len(valid),scored=len(valid)+len(timeouts),timeout_failures=len(timeouts),successes=sum(bool(r['success']) for r in valid),
                setup_errors=sum(r['status']=='setup_error' for r in subset),
                expert_attempts=sum(r.get('expert_attempts',0) for r in subset),
                expert_rejections=sum(r['status']=='expert_rejected' for r in subset),
                expert_exceptions=sum(exception_observed(config,r) for r in subset),
                resolved_original_expert_errors=sum(settled_expert_error(config,r) for r in subset),
                historical_interruptions=len(past),
                execution_errors=sum(not acceptable_record(config,r) for r in subset)+len(interrupted),
                mean_seconds=sum(r.get('seconds',0) for r in valid)/len(valid) if valid else None))
    valid=sum(g['valid'] for g in groups); success=sum(g['successes'] for g in groups)
    scored=sum(g['scored'] for g in groups)
    settings={}
    for setting in config['settings']:
        subset=[g for g in groups if g['setting']==setting]
        rates=[g['successes']/g['scored'] for g in subset if g['scored']]
        settings[setting]={'macro_success_rate':sum(rates)/len(rates) if rates else None,
            'tasks_with_valid_episodes':len(rates),'tasks_planned':len(subset),
            'complete':all(g['scored' if active(config) else ('valid' if config.get('valid_per_group') else 'attempted')]==g['planned'] for g in subset)}
    progress=progress_list[0] if progress_list else None
    stages={}
    for row in rows:
        stage=row.get('protocol_stage',row.get('quota_flow', 'legacy_screening'))
        part=stages.setdefault(stage, dict(attempted=0,valid=0,successes=0,expert_rejections=0,
                                           setup_errors=0,forced_policy=0,expert_exceptions=0,timeout_failures=0))
        part['attempted']+=1
        part['valid']+=row['status']=='policy_completed'
        part['timeout_failures']+=timeout_failure(config,row)
        part['successes']+=row['status']=='policy_completed' and bool(row['success'])
        part['expert_rejections']+=row['status']=='expert_rejected'
        part['expert_exceptions']+=exception_observed(config,row)
        part['setup_errors']+=row['status']=='setup_error'
        part['forced_policy']+=row['status']=='policy_completed' and row.get('policy_entry_reason')=='expert_budget_exhausted'
    return dict(planned=planned_count(config), attempted=sum(g['attempted'] for g in groups),
                protocol_stages=stages,
                completion_basis='scored' if active(config) else ('valid' if config.get('valid_per_group') else 'attempted'),
                valid=valid,scored=scored,timeout_failures=sum(g['timeout_failures'] for g in groups),
                successes=success,success_rate=success/scored if scored else None,
                setup_errors=sum(g['setup_errors'] for g in groups),
                non_scored_errors=sum(r['status'] not in ('policy_completed','policy_timeout') for r in rows),
                expert_attempts=sum(g['expert_attempts'] for g in groups),
                expert_rejections=sum(g['expert_rejections'] for g in groups),
                expert_exceptions=sum(g['expert_exceptions'] for g in groups),
                resolved_original_expert_errors=sum(g['resolved_original_expert_errors'] for g in groups),
                execution_errors=sum(g['execution_errors'] for g in groups), record_error=parse_error,
                historical_interruptions=len(history), historical_execution_attempts=history,
                historical_runtime_errors=sum(r['kind'] in ('policy_runtime_error','expert_runtime_error','infrastructure_error') for r in history),
                historical_timeouts=sum(r['kind'] in ('episode_wall_timeout','episode_stall_timeout') for r in history),
                actual_attempts=sum(g['attempted'] for g in groups)+sum(r.get('counts_as_attempt',True) for r in history),
                settings=settings,groups=groups,episodes=rows,progress=progress,progresses=progress_list)


def csv_report(data):
    stream=StringIO()
    fields=['task','setting','planned','attempted','valid','successes','setup_errors','execution_errors','mean_seconds','expert_attempts','expert_rejections','expert_exceptions','resolved_original_expert_errors','historical_interruptions','scored','timeout_failures']
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(data['groups'])
    return '\ufeff'+stream.getvalue()
