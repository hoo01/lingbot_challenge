import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .store import Store
from .reports import report, csv_report
from .servers import server_state

ROOT=Path(os.environ['LINGBOT_ROOT']).resolve()
store=Store(ROOT)
app=FastAPI(title='LingBot Evaluation Lab',docs_url='/api/docs')
web=Path(__file__).parent/'web'


@app.middleware('http')
async def protect_local(request: Request,call_next):
    # Reachable only by an authenticated SSH tunnel. Also block cross-site writes.
    origin=request.headers.get('origin')
    if request.method not in ('GET','HEAD','OPTIONS') and origin:
        from urllib.parse import urlparse
        if urlparse(origin).netloc != request.headers.get('host'):
            return Response('Cross-origin write blocked',status_code=403)
    expected = request.headers.get('x-lingbot-server')
    if expected and expected != server_state(ROOT)['current']['id']:
        return Response('Server differs from selected target; reconnect before writing', status_code=409)
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Referrer-Policy']='same-origin'
    return response


@app.exception_handler(ValueError)
async def invalid(request,error):
    from fastapi.responses import JSONResponse
    return JSONResponse({'detail':str(error)},status_code=400)


@app.get('/')
def index():return FileResponse(web/'index.html')


@app.get('/api/state')
def state():
    try:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5)
        gpus=[dict(zip(('uuid','name','used_mb','total_mb','utilization'),[v.strip() for v in line.split(',')])) for line in raw.splitlines()]
    except (OSError,subprocess.SubprocessError):gpus=[]
    jobs=store.jobs()
    for job in jobs:
        data=report(store.run_folder(job['id']),job['config']);data.pop('episodes');data.pop('groups');job['report']=data
    usage=shutil.disk_usage(ROOT)
    return dict(jobs=jobs,checkpoints=store.checkpoints(),gpus=gpus,workers=store.workers(),tasks=store.tasks(),
                server=server_state(ROOT),
                disk={'free':usage.free,'total':usage.total},root=str(ROOT),mode='evaluation_only')


class CheckpointRequest(BaseModel):
    name:str; checkpoint:str; cli:str; norm:str; notes:str=''


@app.post('/api/checkpoints')
def checkpoint(request:CheckpointRequest):return store.register(**request.model_dump())


class JobRequest(BaseModel):
    title:str='Evaluation'; checkpoint_id:str
    tasks:list[str]; settings:list[str]; seeds:list[int]
    execute:int=50; precision:str='fp32';video:bool=False;video_fps:int=10
    parallelism:int|None=None
    expert_check:bool=False


@app.post('/api/jobs')
def submit(request:JobRequest):
    data=request.model_dump();title=data.pop('title');return store.submit(title,data)


@app.post('/api/jobs/{identifier}/cancel')
def cancel(identifier:str):return store.cancel(identifier)


@app.post('/api/uploads')
def upload_directory():
    identifier=uuid.uuid4().hex
    folder=ROOT/'incoming'/identifier;folder.mkdir(parents=True)
    return {'id':identifier,'path':str(folder)}


@app.put('/api/uploads/{identifier}/{relative:path}')
async def upload_file(identifier:str,relative:str,request:Request):
    import re
    if not re.fullmatch('[a-f0-9]{32}',identifier):raise HTTPException(400,'Invalid upload ID')
    folder=ROOT/'incoming'/identifier
    if not folder.is_dir():raise HTTPException(404,'Upload not found')
    target=(folder/relative).resolve()
    if not target.is_relative_to(folder.resolve()) or target.suffix not in ('.safetensors','.json','.yaml','.yml'):
        raise HTTPException(400,'Only safetensors, JSON and YAML inside the upload folder are accepted')
    if target.exists():raise HTTPException(409,'File already exists')
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.uploading')
    length=int(request.headers.get('content-length','0'))
    if length>shutil.disk_usage(ROOT).free-10*1024**3:raise HTTPException(507,'Keep at least 10 GiB free')
    count=0
    try:
        with temporary.open('xb') as stream:
            async for block in request.stream():
                count+=len(block)
                if count>40*1024**3:raise HTTPException(413,'One file exceeds 40 GiB')
                stream.write(block)
        if length and count!=length:raise HTTPException(400,'Incomplete file')
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {'path':str(target),'bytes':count}


@app.get('/api/jobs/{identifier}')
def detail(identifier:str):
    job=store.job(identifier);job['report']=report(store.run_folder(identifier),job['config'])
    job['shards']=store.shards(identifier)
    return job


@app.get('/api/jobs/{identifier}/scores.csv')
def scores(identifier:str):
    job=store.job(identifier)
    return Response(csv_report(report(store.run_folder(identifier),job['config'])),media_type='text/csv',
                    headers={'Content-Disposition':f'attachment; filename="{identifier}-scores.csv"'})


@app.get('/api/jobs/{identifier}/files/{relative:path}')
def artifact(identifier:str,relative:str):
    folder=store.run_folder(identifier).resolve();store.job(identifier)
    file=(folder/relative).resolve()
    if not file.is_relative_to(folder) or not file.is_file():raise HTTPException(404,'File not found')
    if file.suffix not in ('.mp4','.png','.json','.jsonl','.csv','.log','.yaml'):raise HTTPException(403,'File type not served')
    return FileResponse(file)


@app.get('/api/jobs/{identifier}/logs/{kind}')
def logs(identifier:str,kind:str):
    if kind not in ('worker','server','evaluation'):raise HTTPException(404)
    job=store.job(identifier);folder=store.run_folder(identifier)
    shards=job['config'].get('shards',[])
    files=[(f'分组 {s["id"]+1}',store.shard_folder(identifier,s['id'])/f'{kind}.log') for s in shards] if shards else [('',folder/f'{kind}.log')]
    chunks=[]
    for label,file in files:
        if file.exists():
            with file.open('rb') as stream:
                stream.seek(max(0,file.stat().st_size-24000));tail=stream.read().decode(errors='replace')
            chunks.append(label+'\n'+tail)
    return {'text':'\n\n'.join(chunks) or '等待日志…'}


app.mount('/static',StaticFiles(directory=web),name='static')
