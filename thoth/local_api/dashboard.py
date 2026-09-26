"""Node dashboard — single-page UI served at ``GET /`` (§24).

Zero-dependency vanilla HTML/JS: the page is static, all data comes from
the authenticated JSON API on the same origin. The browser passes the
local token once via ``/?token=<local_token>``; the page stores it in
``localStorage`` and sends it as ``Authorization: Bearer`` afterwards.
"""

from __future__ import annotations

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Thoth Node</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0e1117;--panel:#171c26;--line:#2a3140;--fg:#dbe2ee;--dim:#8592a8;
      --acc:#4fa3ff;--ok:#3fce8c;--warn:#ffb84d;--err:#ff6b6b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;align-items:center;gap:16px;padding:10px 18px;
background:var(--panel);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:9}
header b{font-size:15px}
nav{display:flex;gap:4px}
nav button{background:none;border:1px solid transparent;color:var(--dim);
padding:6px 12px;border-radius:6px;cursor:pointer;font-size:13px}
nav button.on{color:var(--fg);border-color:var(--acc);background:#1d2634}
#token{display:flex;gap:6px;margin-left:auto}
input,select,textarea{background:#10141c;border:1px solid var(--line);
color:var(--fg);border-radius:6px;padding:6px 9px;font:inherit}
textarea{width:100%;min-height:110px;font-family:ui-monospace,Consolas,monospace;font-size:12px}
button{background:#233047;border:1px solid var(--line);color:var(--fg);
border-radius:6px;padding:6px 12px;cursor:pointer;font-size:13px}
button:hover{border-color:var(--acc)}
button.go{background:#1d4a30;border-color:var(--ok)}
button.danger{background:#452024;border-color:var(--err)}
main{padding:18px;max-width:1150px;margin:auto}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:14px 16px;margin-bottom:14px}
.card h3{margin:0 0 10px;font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim)}
table{width:100%;border-collapse:collapse;font-size:13px}
th{color:var(--dim);font-weight:600;text-align:left;padding:4px 8px;border-bottom:1px solid var(--line)}
td{padding:5px 8px;border-bottom:1px solid #20283a;vertical-align:top}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;background:#243049}
.pill.ok{background:#1d4a30;color:var(--ok)}.pill.err{background:#452024;color:var(--err)}
pre{background:#10141c;border:1px solid var(--line);border-radius:6px;padding:10px;
overflow:auto;max-height:300px;font-size:12px;margin:6px 0}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.muted{color:var(--dim)}.page{display:none}.page.on{display:block}
#gate{position:fixed;inset:0;background:var(--bg);z-index:99;
display:none;align-items:center;justify-content:center}
#gate.on{display:flex}
.gatecard{background:var(--panel);border:1px solid var(--line);
border-radius:14px;padding:34px 38px;width:340px;text-align:center;
box-shadow:0 18px 50px #0008}
.gatecard h2{margin:0 0 4px;font-size:20px}
.gatecard .sub{color:var(--dim);font-size:12px;margin-bottom:22px}
.gatecard input{width:100%;text-align:center;margin-bottom:12px}
.gatecard button{width:100%}
.gatecard .err{color:var(--err);font-size:12px;min-height:16px;margin-top:10px}
#liveOut{max-height:260px}
small{font-size:11px}
</style></head><body>
<div id="gate"><div class="gatecard">
<h2>&#9672; thoth node</h2>
<div class="sub" id="gateSub">sign in to this node</div>
<div id="acctMode">
<input id="guser" placeholder="portal username or email"
  autocomplete="username">
<input id="gpass" type="password" placeholder="portal password"
  autocomplete="current-password">
<button class="go" onclick="unlockAcct()">sign in</button>
<small class="muted"><a href="#" onclick="gateMode('token');return false"
  style="color:var(--dim)">use node token</a></small>
</div>
<div id="tokMode" style="display:none">
<input id="gtok" type="password" placeholder="node token"
  autocomplete="off">
<button class="go" onclick="unlock()">unlock</button>
<small class="muted"><a href="#" onclick="gateMode('acct');return false"
  style="color:var(--dim)">portal sign-in</a> instead</small>
</div>
<div class="err" id="gateErr"></div>
</div></div>
<header><b>◈ thoth node</b><span id="devname" class="muted"></span>
<nav>
<button data-p="status" class="on">Status</button>
<button data-p="live">Live</button>
<button data-p="captures">Captures</button>
<button data-p="models">Models</button>
<button data-p="automations">Automations</button>
</nav>
<div id="token"><input id="tok" placeholder="local token" size="22"><button onclick="saveTok()">unlock</button></div>
</header><main>

<section id="status" class="page on">
  <div class="card"><h3>Device</h3><div id="statusBody" class="muted">…</div></div>
  <div class="card"><h3>Sensors</h3><table id="statusSensors"></table></div>
  <div class="card"><h3>Actuators</h3><table id="statusActuators"></table></div>
  <div class="card"><h3>Recent predictions</h3><table id="statusPreds"></table></div>
</section>

<section id="live" class="page">
  <div class="card"><h3>Sensors — live stream</h3>
    <div class="row">
      <select id="liveSensor"></select>
      <button onclick="liveStart()">stream</button>
      <button onclick="liveStop()">stop</button>
      <span id="liveMeta" class="muted"></span>
    </div>
    <pre id="liveOut"></pre>
  </div>
</section>

<section id="captures" class="page">
  <div class="card"><h3>New capture</h3>
    <div class="row" id="capSensors"></div>
    <div class="row" style="margin-top:8px"><button class="go" onclick="capStart()">start</button>
    <span class="muted">no selection = all sensors</span></div>
  </div>
  <div class="card"><h3>Captures</h3>
    <table id="capTable"></table>
    <div class="row" style="margin-top:8px">
      <input id="capLabel" placeholder="label text" size="18">
      <input id="capId" placeholder="capture id" size="14">
      <button onclick="addLabel()">label</button>
      <button onclick="autoLabel()">auto-label from predictions</button>
      <button class="danger" onclick="clearLabels()">clear labels</button>
    </div>
  </div>
</section>

<section id="models" class="page">
  <div class="card"><h3>Catalog</h3><table id="catTable"></table></div>
  <div class="card"><h3>Installed</h3><table id="modelTable"></table></div>
  <div class="card"><h3>Install / upload</h3>
    <div class="row"><input id="mName" placeholder="name" size="16">
      <input id="mProc" placeholder="processor (e.g. rule, pca-face-recognizer)" size="34">
      <button onclick="installModel()">install+activate</button></div>
    <div style="margin-top:8px"><span class="muted">config (JSON)</span>
      <textarea id="mCfg" placeholder='{"rules":[{"when":"snr_mean > 4","label":"occupied"}],"else":"empty","actions":[]}'></textarea></div>
    <div style="margin-top:8px"><span class="muted">or upload a whispy-model/v1 manifest (JSON)</span>
      <textarea id="mDeploy" placeholder='{"manifest":{"model_id":"…","version":"…","processor":{"type":"rule","config":{…}},"inputs":[…],"bindings":[…]}}'></textarea>
      <button onclick="deployModel()">deploy</button></div>
    <pre id="modelOut"></pre>
  </div>
</section>

<section id="automations" class="page">
  <div class="card"><h3>Automations</h3><table id="autoTable"></table></div>
  <div class="card"><h3>Create</h3>
    <div class="row"><input id="aName" placeholder="name" size="22">
    <select id="aTpl" onchange="aTemplate()">
      <option value="">— template —</option>
      <option value="condition">condition → HA light</option>
      <option value="event">event (label) → lan actuator</option>
      <option value="time">time interval → webhook</option>
    </select></div>
    <div class="row" style="margin-top:8px;align-items:flex-start">
      <div style="flex:1"><span class="muted">trigger (JSON)</span><textarea id="aTrig"></textarea></div>
      <div style="flex:1"><span class="muted">action (JSON)</span><textarea id="aAct"></textarea></div>
    </div>
    <div style="margin-top:8px"><button class="go" onclick="createAuto()">create</button></div>
    <pre id="autoOut"></pre>
  </div>
</section>
</main>
<script>
let TOKEN = new URLSearchParams(location.search).get('token')
        || localStorage.getItem('thoth_tok') || '';
if (TOKEN) localStorage.setItem('thoth_tok', TOKEN);
document.getElementById('tok').value = TOKEN;
const gate=document.getElementById('gate'),
      gateErr=document.getElementById('gateErr'),
      gtok=document.getElementById('gtok'),
      guser=document.getElementById('guser'),
      gpass=document.getElementById('gpass');
function showGate(msg){gate.classList.add('on');gateErr.textContent=msg||'';
  guser.focus();}
function hideGate(){gate.classList.remove('on');}
function gateMode(m){
  document.getElementById('acctMode').style.display=m==='acct'?'':'none';
  document.getElementById('tokMode').style.display=m==='token'?'':'none';
  gateErr.textContent='';}
gtok.addEventListener('keydown',e=>{if(e.key==='Enter')unlock();});
gpass.addEventListener('keydown',e=>{if(e.key==='Enter')unlockAcct();});
async function unlockAcct(){
  const u=guser.value.trim(), p=gpass.value;
  if(!u||!p)return;
  const r=await fetch('/api/auth/login',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({username:u,password:p})}).catch(()=>null);
  if(!r){gateErr.textContent='node unreachable';return;}
  if(r.status===401){gateErr.textContent='invalid username or password';return;}
  const b=await r.json().catch(()=>({}));
  if(!b.token){gateErr.textContent='sign-in failed — check Brain connection';return;}
  TOKEN=b.token; localStorage.setItem('thoth_tok',TOKEN);
  document.getElementById('tok').value=TOKEN; hideGate(); refresh();
}
async function unlock(){
  const t=gtok.value.trim(); if(!t)return;
  TOKEN=t; const r=await api('/api/status');
  if(r.status===401||r.status===403){TOKEN='';gateErr.textContent='invalid token';return;}
  if(r.status>=400){TOKEN='';gateErr.textContent='node error ('+r.status+')';return;}
  localStorage.setItem('thoth_tok',TOKEN);
  document.getElementById('tok').value=TOKEN;
  hideGate(); refresh();
}
function saveTok(){TOKEN=document.getElementById('tok').value;
  localStorage.setItem('thoth_tok',TOKEN); refresh();}
if(!TOKEN) showGate();
async function api(path, opt){
  opt = opt||{}; opt.headers = Object.assign(
    {'Authorization':'Bearer '+TOKEN}, opt.headers||{});
  const r = await fetch(path, opt);
  let body=null; try{body=await r.json();}catch(e){}
  if(r.status===401||r.status===403){localStorage.removeItem('thoth_tok');
    showGate('session expired — sign in again');}
  return {status:r.status, body};
}
const post=(p,b)=>api(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const ts=t=>t?new Date(t*1000).toLocaleTimeString():'–';

/* nav */
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('nav button').forEach(x=>x.classList.remove('on'));
  document.querySelectorAll('.page').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');document.getElementById(b.dataset.p).classList.add('on');
  if(b.dataset.p=='live')liveSensors(); if(b.dataset.p=='captures')loadCaptures();
  if(b.dataset.p=='models')loadModels(); if(b.dataset.p=='automations')loadAutos();});

/* ---------- status ---------- */
async function loadStatus(){
  let s=await api('/api/status');
  let inf=await api('/api/v1/device');
  let loc=await api('/api/v1/location');
  let d=(s.body||{}),i=(inf.body||{});
  document.getElementById('devname').textContent=' '+(i.name||i.device_name||'');
  document.getElementById('statusBody').innerHTML=
    `<b>${esc(i.name||i.device_name||d.device_name||'?')}</b>
     <span class="pill">${esc(d.device_id||i.id||'')}</span>
     uptime ${Math.round(d.uptime_s||0)}s · sensors ${d.sensors??'–'} ·
     models ${d.active_models??'–'} active · captures ${d.captures??'–'}
     <div class="muted">${esc(loc.body&&loc.body.postal_code||loc.body&&loc.body.city||'')}</div>`;
  const sn=await api('/api/sensors');
  document.getElementById('statusSensors').innerHTML='<tr><th>id</th><th>type</th><th>caps</th></tr>'+
    ((sn.body&&sn.body.sensors)||[]).map(x=>`<tr><td>${esc(x.id)}</td><td>${esc(x.type||'')}</td><td class="muted">${esc((x.capabilities||[]).join(', '))}</td></tr>`).join('');
  const ac=await api('/api/actuators');
  document.getElementById('statusActuators').innerHTML='<tr><th>id</th><th>kind</th><th>ops</th></tr>'+
    ((ac.body&&ac.body.actuators)||[]).map(x=>`<tr><td>${esc(x.id)}</td><td>${esc(x.kind||'')}</td><td class="muted">${esc((x.operations||[]).join(', '))}</td></tr>`).join('');
  const pr=await api('/api/predictions?limit=12');
  document.getElementById('statusPreds').innerHTML='<tr><th>time</th><th>label</th><th>conf</th><th>model</th></tr>'+
    ((pr.body&&pr.body.predictions)||[]).slice(-12).reverse().map(p=>
    `<tr><td>${ts(p.timestamp)}</td><td>${esc(p.label)}</td><td>${(p.confidence??0).toFixed(2)}</td><td class="muted">${esc(p.runtime_model_id||'')}</td></tr>`).join('');
}

/* ---------- live ---------- */
let liveTimer=null, liveCursor=0, liveCount=0;
async function liveSensors(){
  const r=await api('/api/sensors'); const sel=document.getElementById('liveSensor');
  sel.innerHTML=((r.body&&r.body.sensors)||[]).map(s=>`<option>${esc(s.id)}</option>`).join('');
}
function liveStart(){
  liveStop(); liveCursor=0; liveCount=0;
  const sid=document.getElementById('liveSensor').value; if(!sid)return;
  document.getElementById('liveOut').textContent='';
  liveTimer=setInterval(async()=>{
    const r=await api('/api/sensors/'+sid+'/tail?cursor='+liveCursor);
    const out=document.getElementById('liveOut');
    if(!r.body||r.status!=200)return;
    const list=r.body.samples||[];
    liveCursor=r.body.cursor||liveCursor; liveCount+=list.length;
    document.getElementById('liveMeta').textContent=liveCount+' samples';
    list.forEach(s=>{out.textContent+=ts(s.timestamp)+' '+JSON.stringify(s.payload).slice(0,400)+'\n';});
    out.scrollTop=out.scrollHeight;
    if(out.textContent.length>20000)out.textContent=out.textContent.slice(-10000);
  },700);
}
function liveStop(){if(liveTimer){clearInterval(liveTimer);liveTimer=null;}}

/* ---------- captures ---------- */
async function loadCaptures(){
  const sn=await api('/api/sensors');
  document.getElementById('capSensors').innerHTML=((sn.body&&sn.body.sensors)||[]).map(s=>
    `<label><input type="checkbox" value="${esc(s.id)}"> ${esc(s.id)}</label>`).join('');
  const r=await api('/api/captures');
  const t=document.getElementById('capTable');
  const rows=((r.body&&r.body.captures)||[]).slice().reverse();
  t.innerHTML='<tr><th>id</th><th>state</th><th>started</th><th>samples</th><th>labels</th><th></th></tr>'+
    rows.map(c=>{
      const samples=Object.values(c.sample_counts||{}).reduce((a,b)=>a+b,0);
      const labels=(c.labels||[]).map(l=>`${esc(l.label)}<span class="muted">/${esc(l.source)}</span>`).join(', ')||'–';
      return `<tr><td><a href="#" onclick="document.getElementById('capId').value='${esc(c.id)}';return false">${esc(c.id)}</a></td>
        <td><span class="pill ${c.state=='active'?'ok':''}">${esc(c.state)}</span></td>
        <td>${ts(c.started_at)}</td><td>${samples}</td><td>${labels}</td>
        <td class="row">${c.state=='active'?`<button onclick="capStop('${esc(c.id)}')">stop</button>`:''}
        <button onclick="dl('${esc(c.id)}')">zip</button>
        <button class="danger" onclick="delCap('${esc(c.id)}')">delete</button></td></tr>`;}).join('');
}
async function capStart(){
  const ids=[...document.querySelectorAll('#capSensors input:checked')].map(x=>x.value);
  await post('/api/captures/start',{sensors:ids.length?ids:null}); loadCaptures();
}
async function capStop(id){await post('/api/captures/stop',{capture_id:id});loadCaptures();}
async function delCap(id){if(!confirm('delete capture '+id+'?'))return;
  const r=await fetch('/api/captures/'+id,{method:'DELETE',headers:{'Authorization':'Bearer '+TOKEN}});
  loadCaptures();}
function dl(id){window.open('/api/captures/'+id+'/download?token='+TOKEN);}
async function addLabel(){
  const id=document.getElementById('capId').value,l=document.getElementById('capLabel').value;
  if(!id||!l)return; await post('/api/captures/'+id+'/label',{label:l}); loadCaptures();}
async function autoLabel(){const id=document.getElementById('capId').value;if(!id)return;
  await post('/api/captures/'+id+'/autolabel',{}); loadCaptures();}
async function clearLabels(){const id=document.getElementById('capId').value;if(!id)return;
  await post('/api/captures/'+id+'/clear-labels',{}); loadCaptures();}

/* ---------- models ---------- */
async function loadModels(){
  const c=await api('/api/v1/model-catalog');
  document.getElementById('catTable').innerHTML='<tr><th>name</th><th>kind</th><th>source</th><th>available</th></tr>'+
    ((c.body&&c.body.models)||[]).map(m=>`<tr><td>${esc(m.name||m.model_id)}</td><td>${esc(m.kind||m.processor||'')}</td>
     <td class="muted">${esc(m.builtin?'builtin':(m.package||m.source||'cloud'))}</td>
     <td><span class="pill ${m.available===false?'err':'ok'}">${m.available===false?'no':'yes'}</span></td></tr>`).join('');
  const r=await api('/api/models');
  document.getElementById('modelTable').innerHTML='<tr><th>runtime id</th><th>name</th><th>processor</th><th>active</th><th></th></tr>'+
    ((r.body&&r.body.models)||[]).map(m=>`<tr><td class="muted">${esc(m.runtime_model_id)}</td><td>${esc(m.name)}</td>
     <td>${esc(m.processor)}</td><td><span class="pill ${m.active?'ok':''}">${m.active?'active':'idle'}</span></td>
     <td><button onclick="actModel('${esc(m.runtime_model_id)}',${!m.active})">${m.active?'deactivate':'activate'}</button></td></tr>`).join('');
}
async function installModel(){
  let cfg={};try{cfg=JSON.parse(document.getElementById('mCfg').value||'{}')}catch(e){alert('bad config JSON');return}
  const r=await post('/api/models/install',{name:document.getElementById('mName').value||'model',
    processor:document.getElementById('mProc').value||'rule',config:cfg});
  document.getElementById('modelOut').textContent=JSON.stringify(r.body,null,2);
  if(r.body&&r.body.runtime_model_id)
    await post('/api/models/activate',{runtime_model_id:r.body.runtime_model_id,active:true});
  loadModels();
}
async function actModel(id,on){await post('/api/models/activate',{runtime_model_id:id,active:on});loadModels();}
async function deployModel(){
  let body;try{body=JSON.parse(document.getElementById('mDeploy').value)}catch(e){alert('bad manifest JSON');return}
  const r=await post('/api/deployments',Object.assign({deployment_id:'dash-'+Date.now()},body));
  document.getElementById('modelOut').textContent=JSON.stringify(r.body,null,2); loadModels();
}

/* ---------- automations ---------- */
function aTemplate(){
  const t=document.getElementById('aTpl').value;
  const T={condition:{
    trig:'{"type":"condition","when":"snr_mean > 4","labels":["occupied"],"for_s":10}',
    act:'{"type":"home_assistant","config":{"ha_url":"http://10.0.0.22:8123","ha_token":"…","entity_id":"light.gad_room_hue","service":"turn_on","data":{"rgb_color":[0,255,0]}},"cooldown_seconds":30}'},
   event:{
    trig:'{"type":"event","on":"label","label":"gad"}',
    act:'{"type":"lan","config":{"host":"10.0.0.88","port":5001,"token":"…","actuator":"matrix","operation":"scroll","params":{"text":"{label}"}},"cooldown_seconds":60}'},
   time:{
    trig:'{"type":"time","interval_s":3600}',
    act:'{"type":"webhook","config":{"url":"https://example.com/hook","body":{"text":"tick from {device_id}"}}}'}}
  if(T[t]){document.getElementById('aTrig').value=T[t].trig;document.getElementById('aAct').value=T[t].act;}
}
async function loadAutos(){
  const r=await api('/api/automations');
  document.getElementById('autoTable').innerHTML='<tr><th>name</th><th>trigger</th><th>action</th><th>enabled</th><th>fires</th><th></th></tr>'+
    ((r.body&&r.body.automations)||[]).map(a=>`<tr><td>${esc(a.name)}</td>
     <td class="muted">${esc((a.trigger||{}).type)}</td><td class="muted">${esc((a.action||{}).type)}</td>
     <td><span class="pill ${a.enabled?'ok':''}">${a.enabled?'on':'off'}</span></td>
     <td>${(a.state||{}).fires||0}</td>
     <td><button onclick="toggleAuto('${esc(a.id)}',${!a.enabled})">${a.enabled?'disable':'enable'}</button>
      <button class="danger" onclick="delAuto('${esc(a.id)}')">delete</button></td></tr>`).join('');
}
async function createAuto(){
  let trig,act;try{trig=JSON.parse(document.getElementById('aTrig').value);
    act=JSON.parse(document.getElementById('aAct').value)}catch(e){alert('bad JSON');return}
  const r=await post('/api/automations',{name:document.getElementById('aName').value||'automation',trigger:trig,action:act});
  document.getElementById('autoOut').textContent=JSON.stringify(r.body,null,2); loadAutos();
}
async function toggleAuto(id,en){await post('/api/automations/'+id,{enabled:en});loadAutos();}
async function delAuto(id){await fetch('/api/automations/'+id,{method:'DELETE',headers:{'Authorization':'Bearer '+TOKEN}});loadAutos();}

async function refresh(){if(!TOKEN){showGate();return;} await loadStatus();}
refresh(); setInterval(loadStatus,5000);
</script></body></html>
"""
