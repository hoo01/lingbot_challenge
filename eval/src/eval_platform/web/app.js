const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const statusNames={queued:'等待中',running:'运行中',external_running:'评测中（快照）',completed:'已完成',failed:'运行失败',interrupted:'已中断',cancel_requested:'正在取消',cancelled:'已取消'};
let state={jobs:[],checkpoints:[],tasks:[]},selected=null,details=null,view='queue',busy=false,serverReady=false;
const expectedServer=new URLSearchParams(location.search).get('server');
const fmtTime=s=>new Date(s).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false});
const pct=(n,d)=>d?`${(100*n/d).toFixed(1)}%`:'—';
const protocolLabel=c=>c.inference_profile==='official_defaults_v1'?'官方默认推理':c.protocol==='team_official_defaults_fixed_unfiltered_v1'?'官方默认推理':'旧适配推理';
const setting=s=>s==='demo_clean'?'Clean':'Randomized';
const simulationVideo=e=>e.video_capture?.version==='head_camera_simulation_time_v1';
const actionVideo=e=>e.video_capture?.version==='head_camera_after_action_v1';
const videoCaption=e=>simulationVideo(e)?`头部相机 · ${e.video_capture.fps} fps · 仿真时间 1× 原速。`:actionVideo(e)?'头部相机 · 10 fps · 按动作编号回放。':'历史录像：动作块内重复画面。';
const videoLabel=e=>simulationVideo(e)?'▶ 仿真原速':actionVideo(e)?'▶ 按动作回放':'▶ 旧版录像（画面不连续）';
function badge(s){return `<span class="badge ${esc(s)}">${esc(statusNames[s]||s)}</span>`;}
async function api(path,options={}){const r=await fetch(path,options);if(!r.ok){const body=await r.text();let m=body;try{m=JSON.parse(body).detail||body}catch{}throw Error((typeof m==='string'?m:JSON.stringify(m))||`HTTP ${r.status}`)}return r.json()}
const post=(path,body)=>{if(!serverReady)throw Error('服务器尚未核对，暂不能提交操作');return api(path,{method:'POST',headers:{'Content-Type':'application/json','X-Lingbot-Server':state.server.current.id},body:JSON.stringify(body)})};
function message(text){$('message').textContent=text;$('message').hidden=!text;}
function showView(name){view=name;for(const n of ['queue','models','guide','detail'])$(n+'-view').hidden=n!==name;document.querySelectorAll('.nav').forEach(b=>b.classList.toggle('active',b.dataset.view===name));$('page-title').textContent=({queue:'评测工作台',models:'模型与数据',guide:'使用说明',detail:'评测报告'})[name];if(name!=='detail')selected=null;}
function drawState(){
 const servers=state.server,current=servers.current;
 serverReady=Boolean(current.id)&&(!expectedServer||expectedServer===current.id);
 $('server-name').textContent=current.label;$('server-note').textContent=current.note;
 const previous=$('server-select').value;
 $('server-select').innerHTML=servers.servers.map(s=>`<option value="${esc(s.id)}">${esc(s.label)}${s.id===servers.default_server?'（默认）':''}</option>`).join('');
 $('server-select').value=servers.servers.some(s=>s.id===previous)?previous:(current.id||servers.default_server);
 $('server-select').disabled=false;$('switch-server').disabled=false;
 for(const id of ['new-job','register-model','empty-model','upload-directory','cancel-job'])$(id).disabled=!serverReady;
 if(!serverReady)message(expectedServer?`所选服务器为 ${expectedServer}，隧道实际连接到 ${current.id||'未绑定服务'}。请核对隧道后重试。`:'服务未绑定服务器配置，暂不能提交操作。');
 const active=state.jobs.filter(j=>['running','external_running'].includes(j.status)).length,queued=state.jobs.filter(j=>j.status==='queued').length;
 $('gpu-count').textContent=state.gpus.length+' 张';$('gpu-info').textContent=state.gpus.map(g=>g.name.replace('NVIDIA GeForce ','')).join(' / ')||'GPU 信息暂不可用';
 $('job-count').textContent=`${active} / ${queued}`;$('queued-count').textContent=queued;$('done-count').textContent=state.jobs.filter(j=>j.status==='completed').length;
 $('disk-free').textContent=(state.disk.free/1024**3).toFixed(1)+' GB';
 const query=$('job-search').value.trim().toLowerCase();const jobs=state.jobs.filter(j=>j.title.toLowerCase().includes(query));
 $('jobs').innerHTML=jobs.map(j=>{const r=j.report,c=state.checkpoints.find(c=>c.id===j.config.checkpoint_id),p=r.completion_basis==='valid'?r.valid:r.attempted;return `<tr><td><b>${esc(j.title)}</b><small>${esc(c?.name||j.config.checkpoint_id)} · ${j.config.tasks.length} 任务</small></td><td>${badge(j.status)}</td><td>${p} / ${r.planned}${r.completion_basis==='valid'?`<small>有效回合 · 已尝试 ${r.attempted} 次</small>`:''}<div class="progress-track"><i style="width:${Math.min(100,100*p/Math.max(r.planned,1))}%"></i></div></td><td><b>${r.successes} / ${r.valid}</b><small>${pct(r.successes,r.valid)} · ${r.execution_errors} 次运行错误</small></td><td>${r.setup_errors}</td><td>${fmtTime(j.created)}</td><td><button class="row-button" data-job="${j.id}">查看报告 →</button></td></tr>`}).join('');
 $('empty-jobs').hidden=state.jobs.length>0;
 $('gpu-cards').innerHTML=state.gpus.map(g=>{let w=state.workers.find(w=>w.gpu===g.uuid);return `<div class="gpu-card"><div><b>${esc(g.name.replace('NVIDIA GeForce ',''))}</b><span>${esc(g.uuid.slice(0,16))}…</span></div><div><b>${g.used_mb} / ${g.total_mb} MiB</b><span>利用率 ${g.utilization}% · ${w?(Date.now()-new Date(w.updated)<30000?'Worker 在线':'Worker 心跳过期'):'等待 Worker'}</span></div></div>`}).join('');
 $('models').innerHTML=state.checkpoints.map(c=>`<article class="model-card"><span class="badge ${c.read_only?'':'completed'}">${c.read_only?'只读':'已注册'}</span><h3 style="margin-top:14px">${esc(c.name)}</h3><p>${esc(c.notes||'暂无备注')}</p><p class="muted">${(c.files.reduce((n,f)=>n+f.bytes,0)/1024**3).toFixed(2)} GiB · ${c.files.length} 个权重文件</p><code>模型 ID：${esc(c.id)}</code><p class="muted">${fmtTime(c.created)}</p></article>`).join('')||'<div class="empty">尚未注册模型。上传后填写权重与配置路径。</div>';
}
async function refresh(){if(busy)return;busy=true;try{state=await api('/api/state');$('connection').textContent='● 已连接 · '+new Date().toLocaleTimeString('zh-CN');drawState();if(selected){details=await api('/api/jobs/'+selected);drawDetail()}}catch(e){serverReady=false;$('connection').textContent='连接中断';message(e.message)}finally{busy=false}}
function artifact(relative){return `/api/jobs/${selected}/files/`+relative.split('/').map(encodeURIComponent).join('/')}
function drawDetail(){
 const j=details,r=j.report,c=state.checkpoints.find(c=>c.id===j.config.checkpoint_id);
 $('detail-title').textContent=j.title;$('detail-subtitle').textContent=`${state.server.current.label} · ${protocolLabel(j.config)} · ${c?.name||''} · ${j.config.precision.toUpperCase()} · chunk ${j.config.execute} · ${statusNames[j.status]}${j.config.video===false?' · 当前不录像':j.config.video_fps?` · 录像 ${j.config.video_fps} fps`:''}`;
 $('download-csv').href=`/api/jobs/${j.id}/scores.csv`;$('cancel-job').hidden=!['queued','running','cancel_requested'].includes(j.status);
 $('cancel-job').disabled=!serverReady||j.status==='cancel_requested';
 $('detail-metrics').innerHTML=[[r.completion_basis==='valid'?'有效回合进度':'评测进度',`${r.completion_basis==='valid'?r.valid:r.attempted} / ${r.planned}`,r.completion_basis==='valid'?`已尝试 ${r.attempted} 次 · 初始化失败按顺序补足`:'固定任务、setting 与 seed'],['成功 / 有效',`${r.successes} / ${r.valid}`,pct(r.successes,r.valid)],['初始化失败',r.setup_errors,'与策略失败分别统计'],['运行错误',r.execution_errors,'出现错误后停止，保留现场']].map(([a,b,c])=>`<article><span>${a}</span><strong>${b}</strong><small>${c}</small></article>`).join('');
 const p=r.progress;$('current-progress').textContent=r.record_error||j.error|| (p?`${p.task} / ${setting(p.setting)} / seed ${p.seed} · ${p.event} ${p.actions?'· '+p.actions+' actions':''}`:'等待 GPU 领取任务');
 $('task-results').innerHTML=r.groups.map(g=>`<tr><td>${esc(g.task)}</td><td>${setting(g.setting)}</td><td>${r.completion_basis==='valid'?g.valid:g.attempted}/${g.planned}${r.completion_basis==='valid'?`<small>有效 · 尝试 ${g.attempted}</small>`:''}</td><td>${g.successes}/${g.valid}</td><td>${pct(g.successes,g.valid)}</td><td>${g.setup_errors}</td><td>${g.execution_errors}</td><td>${g.mean_seconds===null?'—':(g.mean_seconds/60).toFixed(1)+' min'}</td></tr>`).join('');
 drawEpisodes();
}
function drawEpisodes(){
 const filter=$('episode-filter').value;
 $('episodes').innerHTML=details.report.episodes.filter(e=>filter==='all'||(filter==='success'?e.status==='policy_completed'&&e.success:!e.success)).map(e=>{
 const folder=`evaluation/${e.task}/${e.setting}/${e.seed}`;const label=e.status==='setup_error'?'初始化失败':e.status==='policy_completed'?(e.success?'成功':'未成功'):'运行错误';
 return `<article class="episode-card">${e.video?`<button class="video-thumb" data-video="${esc(artifact('evaluation/'+e.video))}" data-title="${esc(e.task+' · '+setting(e.setting)+' · '+e.seed)}" data-caption="${esc(videoCaption(e))}"><img loading="lazy" src="${artifact(folder+'/final.png')}" alt="${esc(e.task)} 终帧"><span>${videoLabel(e)}</span></button>`:''}<div class="episode-body"><h3>${esc(e.task)}</h3>${e.video_requested===false?`<p class="muted">未录像${e.status==='policy_completed'?' · 三相机关键帧与动作轨迹已保存':''}</p>`:''}<p>${setting(e.setting)} · seed ${e.seed} · ${((e.seconds||0)/60).toFixed(1)} min</p><span class="badge ${e.success?'completed':e.status==='policy_completed'?'':'failed'}">${label}</span><div class="episode-actions"><a href="${artifact(folder+'/episode.json')}" target="_blank" rel="noopener">回合记录</a>${e.status==='policy_completed'?`<a href="${artifact(folder+'/trajectory.jsonl')}" target="_blank" rel="noopener">动作轨迹</a><a href="${artifact(folder+'/final.png')}" target="_blank" rel="noopener">头部相机</a><a href="${artifact(folder+'/final-left-wrist.png')}" target="_blank" rel="noopener">左腕</a><a href="${artifact(folder+'/final-right-wrist.png')}" target="_blank" rel="noopener">右腕</a>`:''}</div></div></article>`}).join('')||'<p class="muted">尚无符合条件的已完成回合。</p>';
}
function newJob(){if(!serverReady)return;const usable=state.checkpoints.filter(c=>!c.read_only);if(!usable.length){showView('models');$('model-dialog').showModal();return}
 $('checkpoint-select').innerHTML=usable.map(c=>`<option value="${c.id}">${esc(c.name)}</option>`).join('');
 $('task-picker').innerHTML=state.tasks.map(t=>`<label><input type="checkbox" name="tasks" value="${esc(t)}" checked>${esc(t)}</label>`).join('');updateCount();$('job-dialog').showModal();
}
function updateCount(){const f=$('job-form'),count=f.querySelectorAll('[name=tasks]:checked').length,seeds=f.elements.seeds.value.split(',').filter(x=>x.trim()).length,settings=f.elements.settings.value==='both'?2:1;$('planned-count').textContent=`${count} 任务 × ${settings} settings × ${seeds} seeds = ${count*settings*seeds} 个计划回合`;}
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>showView(b.dataset.view));
$('switch-server').onclick=()=>{const target=state.server?.servers.find(s=>s.id===$('server-select').value);if(target)location.assign(target.url)};
document.querySelectorAll('[data-close]').forEach(b=>b.onclick=()=>$(b.dataset.close).close());
$('video-dialog').addEventListener('close',()=>{$('player').pause();$('player').removeAttribute('src');$('player').load()});
$('new-job').onclick=newJob;$('register-model').onclick=()=>$('model-dialog').showModal();$('empty-model').onclick=()=>{showView('models');$('model-dialog').showModal()};
$('job-search').oninput=drawState;$('back').onclick=()=>showView('queue');$('episode-filter').onchange=drawEpisodes;
$('job-form').onchange=updateCount;$('job-form').oninput=updateCount;$('select-all').onclick=()=>{const boxes=[...$('task-picker').querySelectorAll('input')],check=boxes.some(b=>!b.checked);boxes.forEach(b=>b.checked=check);updateCount()};
$('jobs').onclick=async e=>{const b=e.target.closest('[data-job]');if(!b)return;selected=b.dataset.job;showView('detail');await refresh()};
$('episodes').onclick=e=>{const b=e.target.closest('[data-video]');if(!b)return;$('video-title').textContent=b.dataset.title;$('player').src=b.dataset.video;$('video-caption').textContent=b.dataset.caption;$('video-dialog').showModal();$('player').play().catch(()=>{})};
$('job-form').onsubmit=async e=>{e.preventDefault();const f=e.target,b=f.querySelector('.submit'),error=f.querySelector('.form-error');b.disabled=true;error.textContent='';try{
 const seeds=f.elements.seeds.value.split(',').map(s=>Number(s.trim()));if(seeds.some(s=>!Number.isInteger(s)))throw Error('seed 必须为整数');
 const result=await post('/api/jobs',{title:f.elements.title.value,checkpoint_id:f.elements.checkpoint_id.value,tasks:[...f.querySelectorAll('[name=tasks]:checked')].map(x=>x.value),settings:f.elements.settings.value==='both'?['demo_clean','demo_randomized']:[f.elements.settings.value],seeds,execute:Number(f.elements.execute.value),precision:f.elements.precision.value,video:f.elements.video.checked,video_fps:10});
 $('job-dialog').close();selected=result.id;showView('detail');await refresh();
 }catch(e){error.textContent=e.message}finally{b.disabled=false}};
$('model-form').onsubmit=async e=>{e.preventDefault();const f=e.target,b=f.querySelector('.submit'),error=f.querySelector('.form-error');b.disabled=true;b.textContent='正在注册…';error.textContent='';try{await post('/api/checkpoints',Object.fromEntries(new FormData(f)));$('model-dialog').close();showView('models');await refresh()}catch(e){error.textContent=e.message}finally{b.disabled=false;b.textContent='注册模型'}};
$('cancel-job').onclick=async()=>{try{await post('/api/jobs/'+selected+'/cancel',{});await refresh()}catch(e){message(e.message)}};
async function loadLog(){if(!selected)return;try{$('log-output').textContent=(await api(`/api/jobs/${selected}/logs/${$('log-kind').value}`)).text}catch(e){message(e.message)}}
$('refresh-log').onclick=loadLog;$('log-kind').onchange=loadLog;
$('upload-directory').onchange=async e=>{
 const files=[...e.target.files].filter(f=>/\.(safetensors|json|ya?ml)$/.test(f.name));if(!files.length)return;
 e.target.disabled=true;$('upload-progress').hidden=false;const total=files.reduce((s,f)=>s+f.size,0);let completed=0;
 try{const bundle=await post('/api/uploads',{});for(const f of files){$('upload-status').textContent=`上传 ${f.webkitRelativePath} (${(completed/1024**3).toFixed(2)} / ${(total/1024**3).toFixed(2)} GiB)`;
 await new Promise((resolve,reject)=>{if(!serverReady)return reject(Error('服务器连接状态已改变，上传暂停'));const xhr=new XMLHttpRequest();xhr.open('PUT',`/api/uploads/${bundle.id}/`+f.webkitRelativePath.split('/').map(encodeURIComponent).join('/'));xhr.setRequestHeader('X-Lingbot-Server',state.server.current.id);xhr.upload.onprogress=event=>{$('upload-progress').value=100*(completed+event.loaded)/total};xhr.onload=()=>xhr.status<300?resolve():reject(Error(xhr.responseText));xhr.onerror=()=>reject(Error('上传中断；已上传的完整文件仍保留'));xhr.send(f)});completed+=f.size;}
 $('upload-status').textContent=`上传完成：${bundle.path}\n请注册模型，并选择对应的权重目录、CLI YAML 与归一化 JSON。`;$('upload-progress').value=100;
 }catch(error){$('upload-status').textContent=error.message}finally{e.target.disabled=false}
};
refresh();setInterval(refresh,6000);
