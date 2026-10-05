import csv
from io import StringIO
import json
from collections import defaultdict


def report(folder, config):
    output=folder/'evaluation'
    path=output/'episodes.jsonl'
    rows=[]
    parse_error=None
    if path.exists():
        raw=path.read_text();lines=raw.splitlines()
        for index,line in enumerate(lines):
            try: rows.append(json.loads(line))
            except json.JSONDecodeError:
                if index==len(lines)-1 and not raw.endswith('\n'):break
                parse_error=f'Malformed episode record at line {index+1}'
                break
    groups=[]
    for task in config['tasks']:
        for setting in config['settings']:
            subset=[r for r in rows if r['task']==task and r['setting']==setting]
            valid=[r for r in subset if r['status']=='policy_completed']
            interrupted=[r for r in config.get('unrecorded_execution_attempts',[]) if r['task']==task and r['setting']==setting]
            groups.append(dict(task=task,setting=setting,planned=config.get('valid_per_group',len(config['seeds'])),attempted=len(subset)+len(interrupted),
                valid=len(valid),successes=sum(bool(r['success']) for r in valid),
                setup_errors=sum(r['status']=='setup_error' for r in subset),
                execution_errors=sum(r['status'] not in ('policy_completed','setup_error') for r in subset)+len(interrupted),
                mean_seconds=sum(r.get('seconds',0) for r in valid)/len(valid) if valid else None))
    valid=sum(g['valid'] for g in groups); success=sum(g['successes'] for g in groups)
    settings={}
    for setting in config['settings']:
        subset=[g for g in groups if g['setting']==setting]
        rates=[g['successes']/g['valid'] for g in subset if g['valid']]
        settings[setting]={'macro_success_rate':sum(rates)/len(rates) if rates else None,
            'tasks_with_valid_episodes':len(rates),'tasks_planned':len(subset),
            'complete':all(g['valid' if config.get('valid_per_group') else 'attempted']==g['planned'] for g in subset)}
    progress=None
    try: progress=json.loads((output/'worker-progress.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):pass
    return dict(planned=len(config['tasks'])*len(config['settings'])*config.get('valid_per_group',len(config['seeds'])), attempted=sum(g['attempted'] for g in groups),
                completion_basis='valid' if config.get('valid_per_group') else 'attempted',
                valid=valid,successes=success,success_rate=success/valid if valid else None,
                setup_errors=sum(g['setup_errors'] for g in groups),
                execution_errors=sum(g['execution_errors'] for g in groups), record_error=parse_error,
                settings=settings,groups=groups,episodes=rows,progress=progress)


def csv_report(data):
    stream=StringIO()
    fields=['task','setting','planned','attempted','valid','successes','setup_errors','execution_errors','mean_seconds']
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(data['groups'])
    return '\ufeff'+stream.getvalue()
