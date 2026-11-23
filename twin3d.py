"""
twin3d.py - 3D view of the wind turbine digital twin (used by digital_twin_app.py)
=================================================================================
Self-contained WebGL renderer (no Three.js / no internet needed), embedded in
Streamlit with st.components.v1.html.

What it shows
  * Tower, nacelle, hub and three blades. The rotor turns at the measured rotor
    speed, each blade turns about its own axis with the measured pitch angle
    (feathered at standstill, ~0 deg when producing).
  * Inside the see-through nacelle: main shaft, gearbox, generator, converter and
    hydraulic (pitch) unit; HV transformer at the tower base. Each part is coloured
    by the health the fault detection computes (green >= 70 %, amber 30-70 %,
    red < 30 %, grey = no data). Red parts pulse.
  * Wind streaks whose speed follows the measured wind speed.
  * Beacon on the nacelle: amber = warning, flashing red = alarm.
  * Its own player: play / pause, speed, timeline slider, camera presets,
    X-ray nacelle and labels on/off. Mouse: drag = rotate, wheel = zoom.

Python side
  timeline = build_timeline(data, cursor, steps, rated, parts)
  html     = turbine3d_html(timeline, title)
"""
import json

import numpy as np
import pandas as pd

PART_LABELS = {"hub": "Hub / pitch + hydraulics", "blades": "Blades (power performance)",
               "gearbox": "Gearbox", "generator": "Generator", "converter": "Converter / cabinets",
               "transformer": "HV transformer", "nacelle": "Whole turbine"}


def _health(score, thr):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(100.0 * (1.0 - score / thr), 0, 100)


def build_timeline(data, cursor, steps, rated, parts, max_points=1500, health_cols=None):
    """Window of `steps` samples ending at `cursor` -> compact dict for the 3D player.
    health_cols: optional {part: column with a ready 0-100 health} (e.g. the combined hybrid health)."""
    start = max(0, cursor - int(steps) + 1)
    w = data.iloc[start:cursor + 1]
    if len(w) == 0:
        w = data.tail(1)
    stride = max(1, int(np.ceil(len(w) / max_points)))
    w = w.iloc[::-1].iloc[::stride].iloc[::-1]          # keep the last (current) sample

    n = len(w)
    col = lambda c: pd.to_numeric(w[c], errors="coerce").to_numpy(float) if c in w else np.full(n, np.nan)
    thr = col("threshold")
    thr = np.where(np.isfinite(thr) & (thr > 0), thr, np.nan)
    power = col("power")
    pfrac = np.clip(power / (rated or 1.0), -0.05, 1.1)

    rpm = col("rotor_rpm")
    est = np.where(pfrac > 0.02, 14.9 * np.clip(0.67 + 0.33 * pfrac / 0.6, 0, 1), 0.0)   # fallback
    rpm = np.where(np.isfinite(rpm), rpm, est)
    pitch = col("pitch")
    pitch = np.where(np.isfinite(pitch), pitch, np.where(pfrac > 0.02, 0.0, 60.0))
    pitch = np.clip(pitch, -5, 90)

    part_h = {}
    for part, subs in parts.items():
        if health_cols and part in health_cols:
            h = col(health_cols[part])
            part_h[part] = [None if not np.isfinite(v) else round(float(v), 1) for v in h]
            continue
        hs = [_health(col(s), thr) for s in subs if s in w]
        if hs:
            with np.errstate(all="ignore"):
                h = np.nanmin(np.vstack(hs), axis=0) if len(hs) > 1 else hs[0]
        else:
            h = np.full(n, np.nan)
        part_h[part] = [None if not np.isfinite(v) else round(float(v), 1) for v in h]

    score = col("anomaly_score")
    crit = col("criticality")
    alarm = w["alarm"].fillna(False).astype(bool).to_numpy() if "alarm" in w else np.zeros(n, bool)
    status = np.where(alarm, 3, np.where((score > 0.5 * thr) | (np.nan_to_num(crit) > 0), 2,
                      np.where(np.isfinite(score), 1, 0)))           # 0 no data 1 normal 2 warning 3 alarm
    ev = w["in_event_window"].fillna(False).astype(bool).to_numpy() if "in_event_window" in w else np.zeros(n, bool)
    lab = w["fault_label"].fillna("").astype(str).tolist() if "fault_label" in w else [""] * n

    r = lambda a, d=2: [None if not np.isfinite(v) else round(float(v), d) for v in a]
    return {"t": pd.to_datetime(w["time_stamp"]).dt.strftime("%d %b %Y %H:%M").tolist(),
            "wind": r(col("wind")), "p": r(pfrac, 3), "rpm": r(rpm), "pitch": r(pitch, 1),
            "status": status.astype(int).tolist(), "ev": ev.astype(int).tolist(), "lab": lab,
            "parts": part_h, "labels": {k: PART_LABELS.get(k, k) for k in parts}}


def turbine3d_html(timeline, title="", autoplay=True, height=660):
    payload = json.dumps(timeline, separators=(",", ":"))
    return (_TEMPLATE.replace("__DATA__", payload)
            .replace("__TITLE__", title.replace("<", "&lt;"))
            .replace("__AUTOPLAY__", "true" if autoplay else "false")
            .replace("__H__", str(int(height))))


_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<style>
:root{--ink:#1F2D3D;--muted:#6B7C93;--line:#E6EEF5;--teal:#0E9F8E;--green:#16A34A;--amber:#F59E0B;--red:#E5484D;--grey:#A3B1C2}
*{box-sizing:border-box}
html,body{margin:0;height:100%;font-family:Inter,'Segoe UI',Roboto,Arial,sans-serif;color:var(--ink);background:transparent}
#wrap{position:relative;height:__H__px;border-radius:16px;overflow:hidden;border:1px solid var(--line);
  background:linear-gradient(180deg,#9cc9ea 0%,#cfe6f5 48%,#e9f3ea 70%,#e9f3ea 100%)}
canvas{display:block;width:100%;height:100%;cursor:grab}
canvas:active{cursor:grabbing}
.card{position:absolute;background:#ffffffe6;border:1px solid var(--line);border-radius:14px;
  box-shadow:0 6px 20px rgba(30,80,120,.10);padding:10px 12px;font-size:12.5px;backdrop-filter:blur(4px)}
#hud{left:12px;top:12px;width:226px}
#hud .t{font-weight:700;font-size:13px}
#hud .row{display:flex;justify-content:space-between;margin-top:4px}
#hud .row b{font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:11.5px;font-weight:700;color:#fff}
#parts{right:12px;top:12px;width:236px}
#parts .p{display:flex;align-items:center;gap:7px;margin-top:6px}
#parts .dot{width:10px;height:10px;border-radius:3px;flex:none}
#parts .nm{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#parts .bar{width:56px;height:6px;border-radius:3px;background:#edf1f5;overflow:hidden;flex:none}
#parts .bar i{display:block;height:100%}
#parts .v{width:36px;text-align:right;font-variant-numeric:tabular-nums;color:var(--muted)}
#ctl{left:12px;right:12px;bottom:12px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
button,select{font:inherit;font-size:12px;border:1px solid var(--line);background:#fff;border-radius:9px;padding:5px 9px;
  color:var(--ink);cursor:pointer}
button.on{background:#DDF4F0;border-color:var(--teal);color:#0b6e63}
button.main{background:var(--teal);color:#fff;border-color:var(--teal);min-width:72px;font-weight:700}
#seek{flex:1;min-width:120px;accent-color:var(--teal)}
#time{font-variant-numeric:tabular-nums;color:var(--muted);min-width:118px;text-align:right}
#banner{left:50%;transform:translateX(-50%);top:12px;display:none;color:#fff;background:var(--red);border:none;font-weight:700}
.lbl{position:absolute;transform:translate(-50%,-120%);pointer-events:none;background:#ffffffee;border:1px solid var(--line);
  border-radius:8px;padding:2px 7px;font-size:11px;white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,.08)}
.lbl i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px;vertical-align:0}
#title{position:absolute;left:12px;bottom:62px;font-size:11.5px;color:#2b3b4d;background:#ffffffb0;padding:3px 8px;border-radius:8px}
#nogl{position:absolute;inset:0;display:none;align-items:center;justify-content:center;font-size:14px;color:var(--muted)}
@media (max-width:640px){#parts{display:none}}
</style></head><body>
<div id="wrap">
  <canvas id="c"></canvas>
  <div id="labels"></div>
  <div class="card" id="hud">
    <div class="t" id="hT">–</div>
    <div style="margin-top:6px"><span class="pill" id="hS">–</span></div>
    <div class="row"><span>Wind</span><b id="hW">–</b></div>
    <div class="row"><span>Power</span><b id="hP">–</b></div>
    <div class="row"><span>Rotor speed</span><b id="hR">–</b></div>
    <div class="row"><span>Blade pitch</span><b id="hB">–</b></div>
  </div>
  <div class="card" id="parts"><div style="font-weight:700">Component health</div><div id="plist"></div></div>
  <div class="card" id="banner"></div>
  <div id="title">__TITLE__</div>
  <div class="card" id="ctl">
    <button class="main" id="play">▶ Play</button>
    <select id="spd" title="10-min steps per second">
      <option value="1">1 step/s</option><option value="3">3 steps/s</option><option value="6" selected>6 steps/s</option>
      <option value="18">18 steps/s</option><option value="60">60 steps/s</option></select>
    <input type="range" id="seek" min="0" max="0" value="0" step="1">
    <span id="time">–</span>
    <button id="now" title="Jump to the current twin time">⏭ Now</button>
    <button id="xray" class="on">X-ray</button>
    <button id="lbls" class="on">Labels</button>
    <button id="wnd" class="on">Wind</button>
    <select id="cam"><option value="q">View: 3/4</option><option value="front">View: facing the rotor</option><option value="side">View: side</option>
      <option value="close">View: nacelle close-up</option><option value="farm">View: farm</option><option value="orbit">View: auto-orbit</option></select>
  </div>
  <div id="nogl">3D view needs WebGL (enable hardware acceleration in the browser).</div>
</div>
<script>
"use strict";
const D = __DATA__;
const AUTOPLAY = __AUTOPLAY__;
const N = D.t.length;
const COL = {green:[0.09,0.64,0.29], amber:[0.96,0.62,0.04], red:[0.90,0.28,0.30], grey:[0.64,0.69,0.76]};
const HEX = {green:'#16A34A', amber:'#F59E0B', red:'#E5484D', grey:'#A3B1C2'};
const STAT = [['No data','#A3B1C2'],['Normal','#16A34A'],['Warning','#F59E0B'],['Alarm','#E5484D']];
const cls = h => h==null ? 'grey' : h>=70 ? 'green' : h>=30 ? 'amber' : 'red';

// ------------------------------------------------------------------ math (column-major 4x4)
const M = {
  I(){return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];},
  mul(a,b){const o=new Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+r]*b[c*4+k];o[c*4+r]=s;}return o;},
  T(x,y,z){const m=M.I();m[12]=x;m[13]=y;m[14]=z;return m;},
  S(s){return [s,0,0,0, 0,s,0,0, 0,0,s,0, 0,0,0,1];},
  Rx(a){const c=Math.cos(a),s=Math.sin(a);return [1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1];},
  Ry(a){const c=Math.cos(a),s=Math.sin(a);return [c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1];},
  Rz(a){const c=Math.cos(a),s=Math.sin(a);return [c,s,0,0, -s,c,0,0, 0,0,1,0, 0,0,0,1];},
  chain(...ms){return ms.reduce((a,b)=>M.mul(a,b));},
  persp(f,a,n,fa){const t=1/Math.tan(f/2),nf=1/(n-fa);return [t/a,0,0,0, 0,t,0,0, 0,0,(fa+n)*nf,-1, 0,0,2*fa*n*nf,0];},
  look(e,c,u){let z=norm(sub(e,c)),x=norm(cross(u,z)),y=cross(z,x);
    return [x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -dot(x,e),-dot(y,e),-dot(z,e),1];},
  apply(m,p){const x=p[0],y=p[1],z=p[2],w=m[3]*x+m[7]*y+m[11]*z+m[15];
    return [(m[0]*x+m[4]*y+m[8]*z+m[12])/w,(m[1]*x+m[5]*y+m[9]*z+m[13])/w,(m[2]*x+m[6]*y+m[10]*z+m[14])/w,w];}
};
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]], dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm=a=>{const l=Math.hypot(a[0],a[1],a[2])||1;return [a[0]/l,a[1]/l,a[2]/l];};
const lerp=(a,b,t)=>a+(b-a)*t;

// ------------------------------------------------------------------ geometry
function smoothNormals(pos, idx){
  const n=new Float32Array(pos.length);
  for(let i=0;i<idx.length;i+=3){const a=idx[i]*3,b=idx[i+1]*3,c=idx[i+2]*3;
    const u=[pos[b]-pos[a],pos[b+1]-pos[a+1],pos[b+2]-pos[a+2]], v=[pos[c]-pos[a],pos[c+1]-pos[a+1],pos[c+2]-pos[a+2]];
    const f=cross(u,v);for(const k of [a,b,c]){n[k]+=f[0];n[k+1]+=f[1];n[k+2]+=f[2];}}
  for(let i=0;i<n.length;i+=3){const l=Math.hypot(n[i],n[i+1],n[i+2])||1;n[i]/=l;n[i+1]/=l;n[i+2]/=l;}
  return n;
}
function box(w,h,d){
  const x=w/2,y=h/2,z=d/2, P=[],Nn=[],I=[];
  const F=[[[1,0,0],[[x,-y,-z],[x,y,-z],[x,y,z],[x,-y,z]]],[[-1,0,0],[[-x,-y,z],[-x,y,z],[-x,y,-z],[-x,-y,-z]]],
           [[0,1,0],[[-x,y,-z],[-x,y,z],[x,y,z],[x,y,-z]]],[[0,-1,0],[[-x,-y,z],[-x,-y,-z],[x,-y,-z],[x,-y,z]]],
           [[0,0,1],[[x,-y,z],[x,y,z],[-x,y,z],[-x,-y,z]]],[[0,0,-1],[[-x,-y,-z],[-x,y,-z],[x,y,-z],[x,-y,-z]]]];
  F.forEach(([n,q],f)=>{q.forEach(p=>{P.push(...p);Nn.push(...n);});const o=f*4;I.push(o,o+1,o+2,o,o+2,o+3);});
  return {p:new Float32Array(P),n:new Float32Array(Nn),i:new Uint16Array(I)};
}
function cyl(r0,r1,h,seg=28,cap=true){            // along +y, centred
  const P=[],Nn=[],I=[],sl=(r0-r1)/h;
  for(let k=0;k<=seg;k++){const a=k/seg*2*Math.PI,c=Math.cos(a),s=Math.sin(a),nn=norm([c,sl,s]);
    P.push(r0*c,-h/2,r0*s, r1*c,h/2,r1*s);Nn.push(...nn,...nn);
    if(k<seg){const o=k*2;I.push(o,o+1,o+3,o,o+3,o+2);}}
  if(cap){for(const [y,r,ny] of [[h/2,r1,1],[-h/2,r0,-1]]){const c0=P.length/3;P.push(0,y,0);Nn.push(0,ny,0);
    for(let k=0;k<=seg;k++){const a=k/seg*2*Math.PI;P.push(r*Math.cos(a),y,r*Math.sin(a));Nn.push(0,ny,0);
      if(k<seg){ny>0?I.push(c0,c0+2+k,c0+1+k):I.push(c0,c0+1+k,c0+2+k);}}}}
  return {p:new Float32Array(P),n:new Float32Array(Nn),i:new Uint16Array(I)};
}
function sphere(r,seg=24){
  const P=[],I=[];
  for(let j=0;j<=seg;j++){const v=j/seg*Math.PI;for(let k=0;k<=seg;k++){const u=k/seg*2*Math.PI;
    P.push(r*Math.sin(v)*Math.cos(u),r*Math.cos(v),r*Math.sin(v)*Math.sin(u));}}
  for(let j=0;j<seg;j++)for(let k=0;k<seg;k++){const a=j*(seg+1)+k,b=a+seg+1;I.push(a,a+1,b,b,a+1,b+1);}
  const p=new Float32Array(P),n=new Float32Array(P.length);for(let i=0;i<P.length;i++)n[i]=P[i]/r;
  return {p,n,i:new Uint16Array(I)};
}
function disc(r,seg=64){
  const P=[0,0,0],Nn=[0,1,0],I=[];
  for(let k=0;k<=seg;k++){const a=k/seg*2*Math.PI;P.push(r*Math.cos(a),0,r*Math.sin(a));Nn.push(0,1,0);if(k<seg)I.push(0,k+2,k+1);}
  return {p:new Float32Array(P),n:new Float32Array(Nn),i:new Uint16Array(I)};
}
function blade(L){                                 // along +y; chord along z, thickness along x
  const S=30,K=16,P=[],I=[];
  for(let j=0;j<=S;j++){
    const s=j/S, y=0.6+s*L;
    const root=Math.max(0,1-s/0.09);                         // round root -> airfoil
    const chord=lerp(lerp(3.6,0.55,Math.pow(s,0.8)),1.6,root*0.6), th=lerp(0.18*chord,1.5,root);
    const tw=lerp(14,-1,s)*Math.PI/180;
    for(let k=0;k<K;k++){const a=k/K*2*Math.PI;
      let cz=-Math.cos(a)*chord*0.5 + chord*0.12, cx=Math.sin(a)*th*0.5*(a<Math.PI?1.0:0.55);
      const rz=lerp(cz,Math.cos(a)*0.8,root), rx=lerp(cx,Math.sin(a)*0.8,root);
      P.push(rx*Math.cos(tw)-rz*Math.sin(tw), y, rx*Math.sin(tw)+rz*Math.cos(tw));}
  }
  for(let j=0;j<S;j++)for(let k=0;k<K;k++){const a=j*K+k,b=j*K+(k+1)%K;I.push(a,a+K,b,b,a+K,b+K);}
  const tip=P.length/3;P.push(0,0.6+L+0.15,0);for(let k=0;k<K;k++)I.push(S*K+k,tip,S*K+(k+1)%K);
  const p=new Float32Array(P);return {p,n:smoothNormals(p,I),i:new Uint16Array(I)};
}

// ------------------------------------------------------------------ GL setup
const cv=document.getElementById('c');
const gl=cv.getContext('webgl',{antialias:true,alpha:true,premultipliedAlpha:false});
if(!gl){document.getElementById('nogl').style.display='flex';throw new Error('no webgl');}
const VS=`attribute vec3 aP;attribute vec3 aN;uniform mat4 uVP;uniform mat4 uM;varying vec3 vN;varying vec3 vW;
void main(){vec4 w=uM*vec4(aP,1.0);vW=w.xyz;vN=mat3(uM)*aN;gl_Position=uVP*w;}`;
const FS=`precision mediump float;varying vec3 vN;varying vec3 vW;uniform vec3 uC;uniform vec3 uE;uniform float uA;
uniform vec3 uL;uniform vec3 uCam;uniform float uSpec;uniform vec3 uFog;
void main(){vec3 n=normalize(vN);if(!gl_FrontFacing)n=-n;vec3 v=normalize(uCam-vW);
 float d=max(dot(n,uL),0.0);vec3 hemi=mix(vec3(0.55,0.52,0.45),vec3(0.80,0.88,1.0),n.y*0.5+0.5);
 float s=pow(max(dot(reflect(-uL,n),v),0.0),28.0)*uSpec;
 vec3 c=uC*(hemi*0.62+d*vec3(1.0,0.97,0.9)*0.62)+s+uE;
 float f=clamp((length(uCam-vW)-220.0)/1000.0,0.0,1.0);gl_FragColor=vec4(mix(c,uFog,f),uA);}`;
function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);
  if(!gl.getShaderParameter(o,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(o));return o;}
const prog=gl.createProgram();gl.attachShader(prog,sh(gl.VERTEX_SHADER,VS));gl.attachShader(prog,sh(gl.FRAGMENT_SHADER,FS));
gl.linkProgram(prog);gl.useProgram(prog);
const A={P:gl.getAttribLocation(prog,'aP'),N:gl.getAttribLocation(prog,'aN')};
const U={};['uVP','uM','uC','uE','uA','uL','uCam','uSpec','uFog'].forEach(k=>U[k]=gl.getUniformLocation(prog,k));
function upload(g){const m={n:g.i.length};
  m.p=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,m.p);gl.bufferData(gl.ARRAY_BUFFER,g.p,gl.STATIC_DRAW);
  m.nb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,m.nb);gl.bufferData(gl.ARRAY_BUFFER,g.n,gl.STATIC_DRAW);
  m.i=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,m.i);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,g.i,gl.STATIC_DRAW);return m;}
gl.enable(gl.DEPTH_TEST);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
gl.enableVertexAttribArray(A.P);gl.enableVertexAttribArray(A.N);

// ------------------------------------------------------------------ scene dimensions (metres)
const HUB_Y=80, NAC_X=0.6, ROTOR_X=-6.9, R=43;
const G={
  ground:upload(disc(3200)), pad:upload(cyl(6,6,0.6)), shadow:upload(disc(1)),
  tower:upload(cyl(2.4,1.5,HUB_Y-2,32,false)), nacelle:upload(box(13.2,4.4,4.6)), roof:upload(box(13.2,0.12,4.6)),
  hub:upload(sphere(1.9)), cone:upload(cyl(2.0,0.15,3.0)), blade:upload(blade(R-2)),
  shaft:upload(cyl(0.45,0.45,3.6)), bearing:upload(box(1.0,1.6,1.6)), gearbox:upload(box(2.8,2.7,2.7)),
  hss:upload(cyl(0.22,0.22,1.0)), generator:upload(cyl(1.3,1.3,3.2)), converter:upload(box(1.3,2.3,1.6)),
  hyd:upload(box(1.1,1.0,1.2)), acc:upload(cyl(0.22,0.22,1.1)), trafo:upload(box(3.6,3.2,2.8)), fin:upload(box(3.4,2.4,0.1)),
  bush:upload(cyl(0.18,0.18,0.9)), beacon:upload(sphere(0.45)), streak:upload(box(1,0.12,0.12)), frame:upload(box(12.6,0.35,3.6))
};
const QX = M.Rz(-Math.PI/2);     // y-axis meshes -> x-axis

// ------------------------------------------------------------------ state
let idx=Math.max(0,N-1), fidx=idx, playing=false, speed=6, rotorA=0, xray=true, showLbl=true, showWind=true;
let cam={az:-0.62,el:0.18,r:165,tx:-4,ty:58,tz:0}, camGoal={...cam}, camMode='q', time0=performance.now();
const presets={q:{az:-0.62,el:0.18,r:165,tx:-4,ty:58,tz:0},front:{az:-1.5708,el:0.06,r:150,tx:0,ty:62,tz:0},side:{az:0,el:0.10,r:150,tx:0,ty:58,tz:0},
  close:{az:-0.95,el:0.33,r:34,tx:1,ty:HUB_Y,tz:0},farm:{az:-0.75,el:0.30,r:620,tx:60,ty:40,tz:-60}};
const farm=[[-230,0,-260,1.7],[150,0,-330,2.6],[-60,0,-520,4.1],[330,0,-160,0.9],[-420,0,-120,3.3]];
const streaks=[...Array(70)].map(()=>({x:-200+Math.random()*320,y:12+Math.random()*112,z:-60+Math.random()*120,k:0.7+Math.random()*0.6}));

// ------------------------------------------------------------------ helpers
function val(arr,i){const a=arr[Math.floor(i)],b=arr[Math.min(N-1,Math.floor(i)+1)],t=i-Math.floor(i);
  if(a==null)return b;if(b==null)return a;return lerp(a,b,t);}
function partCol(name,i,pulse){const h=D.parts[name]?D.parts[name][Math.round(i)]:null,c=cls(h),base=COL[c];
  const e=c==='red'?0.35*pulse:0;return {c:base,e:[e,e*0.15,e*0.15],h,k:c};}
let VP=null, camPos=[0,0,0];
function draw(m,model,color,alpha=1,emis=[0,0,0],spec=0.25){
  gl.uniformMatrix4fv(U.uM,false,new Float32Array(model));gl.uniform3fv(U.uC,color);gl.uniform3fv(U.uE,emis);
  gl.uniform1f(U.uA,alpha);gl.uniform1f(U.uSpec,spec);
  gl.bindBuffer(gl.ARRAY_BUFFER,m.p);gl.vertexAttribPointer(A.P,3,gl.FLOAT,false,0,0);
  gl.bindBuffer(gl.ARRAY_BUFFER,m.nb);gl.vertexAttribPointer(A.N,3,gl.FLOAT,false,0,0);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,m.i);gl.drawElements(gl.TRIANGLES,m.n,gl.UNSIGNED_SHORT,0);
}
function turbineDraws(base,scale,angle,pitchDeg,opts){
  const B=M.chain(M.T(...base),M.S(scale)), out=[], white=[0.95,0.96,0.97];
  out.push([G.tower,M.chain(B,M.T(0,(HUB_Y-2)/2,0)),[0.93,0.94,0.95]]);
  out.push([G.pad,M.chain(B,M.T(0,0.3,0)),[0.70,0.71,0.70]]);
  const H=M.chain(B,M.T(ROTOR_X,HUB_Y,0),M.Rx(angle)), pr=pitchDeg*Math.PI/180;
  for(let k=0;k<3;k++)out.push([G.blade,M.chain(H,M.Rx(k*2*Math.PI/3),M.T(0,1.2,0),M.Ry(pr)),opts.bladeCol||white,1,opts.bladeE||[0,0,0],0.35]);
  out.push([G.hub,H,opts.hubCol||white,1,opts.hubE||[0,0,0]]);
  out.push([G.cone,M.chain(B,M.T(ROTOR_X-1.6,HUB_Y,0),M.Rz(Math.PI/2)),white]);
  return out;
}

// ------------------------------------------------------------------ labels
const LBL={};const lblBox=document.getElementById('labels');
function label(key,text){if(!LBL[key]){const d=document.createElement('div');d.className='lbl';lblBox.appendChild(d);LBL[key]=d;}return LBL[key];}

// ------------------------------------------------------------------ frame
let last=performance.now();
function frame(now){
  const dt=Math.min(0.1,(now-last)/1000);last=now;
  if(playing&&N>1){fidx=Math.min(N-1,fidx+dt*speed);if(fidx>=N-1){playing=false;syncPlay();}}
  idx=Math.round(fidx);
  const W=cv.clientWidth,H=cv.clientHeight,dpr=Math.min(2,window.devicePixelRatio||1);
  if(cv.width!==Math.round(W*dpr)||cv.height!==Math.round(H*dpr)){cv.width=Math.round(W*dpr);cv.height=Math.round(H*dpr);}
  gl.viewport(0,0,cv.width,cv.height);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);

  // values at the current (interpolated) time
  const wind=val(D.wind,fidx)||0, p=val(D.p,fidx)||0, rpm=val(D.rpm,fidx)||0, pitch=val(D.pitch,fidx)||0;
  rotorA+=rpm/60*2*Math.PI*dt;
  const t=(now-time0)/1000, pulse=0.5+0.5*Math.sin(t*5);

  // camera
  if(camMode==='orbit'){camGoal.az+=dt*0.12;}
  for(const k of ['az','el','r','tx','ty','tz'])cam[k]=lerp(cam[k],camGoal[k],Math.min(1,dt*3.5));
  camPos=[cam.tx+cam.r*Math.cos(cam.el)*Math.sin(cam.az),cam.ty+cam.r*Math.sin(cam.el),cam.tz+cam.r*Math.cos(cam.el)*Math.cos(cam.az)];
  camPos[1]=Math.max(camPos[1],2);
  VP=M.mul(M.persp(0.78,W/H,0.5,4000),M.look(camPos,[cam.tx,cam.ty,cam.tz],[0,1,0]));
  gl.uniformMatrix4fv(U.uVP,false,new Float32Array(VP));gl.uniform3fv(U.uCam,camPos);
  gl.uniform3fv(U.uL,norm([-0.45,0.85,0.55]));gl.uniform3fv(U.uFog,[0.81,0.90,0.96]);

  const pc={};for(const k of ['hub','blades','gearbox','generator','converter','transformer','nacelle'])pc[k]=partCol(k,fidx,pulse);
  const tint=(c,k)=>k==='green'||k==='grey'?[0.95,0.96,0.97]:[lerp(0.95,c[0],0.55),lerp(0.96,c[1],0.55),lerp(0.97,c[2],0.55)];
  const opaque=[], trans=[];

  opaque.push([G.ground,M.T(0,0,0),[0.62,0.76,0.52],1,[0,0,0],0.0]);
  trans.push([G.shadow,M.chain(M.T(20,0.05,6),M.S(1),[18,0,0,0, 0,1,0,0, 0,0,10,0, 0,0,0,1]),[0.1,0.15,0.1],0.18,[0,0,0],0]);
  // background farm
  for(const [x,y,z,ph] of farm)opaque.push(...turbineDraws([x,y,z],0.9,rotorA*0.92+ph,pitch,{}).map(d=>{d[2]=[0.88,0.90,0.92];return d;}));
  // main turbine
  opaque.push(...turbineDraws([0,0,0],1,rotorA,pitch,{bladeCol:tint(pc.blades.c,pc.blades.k),bladeE:pc.blades.e,hubCol:pc.hub.c,hubE:pc.hub.e}));
  const NB=M.T(NAC_X,HUB_Y,0);
  opaque.push([G.frame,M.chain(NB,M.T(0,-1.95,0)),[0.55,0.58,0.62]]);
  opaque.push([G.shaft,M.chain(NB,M.T(-4.6,0,0),QX),[0.62,0.64,0.68]]);
  opaque.push([G.bearing,M.chain(NB,M.T(-4.4,-0.4,0)),[0.55,0.58,0.62]]);
  opaque.push([G.gearbox,M.chain(NB,M.T(-1.7,-0.3,0)),pc.gearbox.c,1,pc.gearbox.e]);
  opaque.push([G.hss,M.chain(NB,M.T(0.2,0.2,0),QX),[0.62,0.64,0.68]]);
  opaque.push([G.generator,M.chain(NB,M.T(2.3,0.1,0),QX),pc.generator.c,1,pc.generator.e]);
  opaque.push([G.converter,M.chain(NB,M.T(5.0,-0.6,1.0)),pc.converter.c,1,pc.converter.e]);
  opaque.push([G.hyd,M.chain(NB,M.T(-5.6,-1.1,1.4)),pc.hub.c,1,pc.hub.e]);
  for(const dz of [-0.35,0.35])opaque.push([G.acc,M.chain(NB,M.T(-5.6,-0.2,1.4+dz)),pc.hub.c,1,pc.hub.e]);
  const TB=M.T(9,1.6,8);
  opaque.push([G.trafo,TB,pc.transformer.c,1,pc.transformer.e]);
  for(const dz of [-1.5,1.5])opaque.push([G.fin,M.chain(TB,M.T(0,-0.1,dz)),[0.5,0.53,0.56]]);
  for(const dx of [-1,0,1])opaque.push([G.bush,M.chain(TB,M.T(dx,2.0,0)),[0.85,0.55,0.3]]);
  // beacon
  const st=D.status[idx];
  const bc=st===3?(pulse>0.5?[1,0.2,0.2]:[0.35,0.05,0.05]):st===2?[1,0.7,0.1]:[0.75,0.78,0.8];
  opaque.push([G.beacon,M.chain(NB,M.T(5.4,2.55,0)),bc,1,st>=2?[bc[0]*0.6,bc[1]*0.6,bc[2]*0.6]:[0,0,0]]);
  // nacelle shell (x-ray = see-through)
  const ncol=tint(pc.nacelle.c,pc.nacelle.k);
  const shell=[G.nacelle,NB,ncol,xray?0.22:1,pc.nacelle.e,0.5];
  (xray?trans:opaque).push(shell);
  // wind streaks
  if(showWind&&wind>0.2){const sp=wind*3.2;
    for(const s of streaks){s.x+=sp*s.k*dt;if(s.x>140){s.x=-220-Math.random()*40;s.y=12+Math.random()*112;s.z=-60+Math.random()*120;}
      const L=Math.max(1.5,wind*1.6)*s.k;trans.push([G.streak,M.chain(M.T(s.x,s.y,s.z),[L,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),[1,1,1],0.55,[0.35,0.35,0.35],0]);}}

  for(const d of opaque)draw(d[0],d[1],d[2],d[3]??1,d[4]||[0,0,0],d[5]??0.25);
  gl.depthMask(false);for(const d of trans)draw(d[0],d[1],d[2],d[3],d[4]||[0,0,0],d[5]??0.25);gl.depthMask(true);

  // labels
  const anchors={hub:[ROTOR_X,HUB_Y+2.2,0],gearbox:[NAC_X-1.7,HUB_Y+1.2,0],generator:[NAC_X+2.3,HUB_Y+1.5,0],
    converter:[NAC_X+5.0,HUB_Y+0.8,1.0],transformer:[9,3.4,8],blades:null,nacelle:null};
  const showL=showLbl&&cam.r<120, placed=[];
  for(const [k,pos] of Object.entries(anchors)){
    const el=label(k);if(!pos||!showL||!(k in D.parts)){el.style.display='none';continue;}
    const q=M.apply(VP,pos);if(q[3]<=0){el.style.display='none';continue;}
    const h=pc[k].h,name=D.labels[k]||k;
    let x=(q[0]*0.5+0.5)*W, y=(1-(q[1]*0.5+0.5))*H;
    for(let tries=0;tries<6;tries++){const hit=placed.find(r=>Math.abs(r[0]-x)<130&&Math.abs(r[1]-y)<22);if(!hit)break;y=hit[1]-24;}
    placed.push([x,y]);
    el.style.display='block';el.style.left=x+'px';el.style.top=y+'px';
    el.innerHTML=`<i style="background:${HEX[pc[k].k]}"></i>${name.split(' (')[0].split(' +')[0]} ${h==null?'–':Math.round(h)+'%'}`;
  }
  hud(wind,p,rpm,pitch,pc);
  requestAnimationFrame(frame);
}

// ------------------------------------------------------------------ HUD + controls
const $=id=>document.getElementById(id);
const seek=$('seek');seek.max=Math.max(0,N-1);
let lastHud=-1;
function hud(wind,p,rpm,pitch,pc){
  seek.value=Math.round(fidx);
  $('hW').textContent=wind.toFixed(1)+' m/s';$('hP').textContent=Math.round(p*100)+' % of rated';
  $('hR').textContent=rpm.toFixed(1)+' rpm';$('hB').textContent=pitch.toFixed(1)+'°';
  if(idx===lastHud)return;lastHud=idx;
  $('hT').textContent=D.t[idx];$('time').textContent=(idx+1)+' / '+N;
  const [s,c]=STAT[D.status[idx]||0];$('hS').textContent=s;$('hS').style.background=c;
  $('plist').innerHTML=Object.keys(D.parts).map(k=>{const h=pc[k]?pc[k].h:null,cl=cls(h);
    return `<div class="p"><span class="dot" style="background:${HEX[cl]}"></span><span class="nm" title="${D.labels[k]}">${D.labels[k]}</span>`+
      `<span class="bar"><i style="width:${h==null?0:h}%;background:${HEX[cl]}"></i></span><span class="v">${h==null?'–':Math.round(h)+'%'}</span></div>`;}).join('');
  const b=$('banner');
  if(D.ev[idx]){b.style.display='block';b.textContent=D.lab[idx]?('Injected fault: '+D.lab[idx]):'Inside the recorded fault window';}
  else b.style.display='none';
}
function syncPlay(){$('play').textContent=playing?'❚❚ Pause':'▶ Play';}
$('play').onclick=()=>{if(!playing&&fidx>=N-1)fidx=0;playing=!playing;syncPlay();};
$('spd').onchange=e=>speed=+e.target.value;
seek.oninput=e=>{fidx=+e.target.value;};
$('now').onclick=()=>{fidx=N-1;playing=false;syncPlay();};
const tog=(id,f)=>{$(id).onclick=()=>{const v=f();$(id).classList.toggle('on',v);};};
tog('xray',()=>xray=!xray);tog('lbls',()=>showLbl=!showLbl);tog('wnd',()=>showWind=!showWind);
$('cam').onchange=e=>{camMode=e.target.value;if(camMode!=='orbit')Object.assign(camGoal,presets[camMode]);};
// mouse
let drag=null;
cv.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];cv.setPointerCapture(e.pointerId);});
cv.addEventListener('pointermove',e=>{if(!drag)return;camGoal.az-=(e.clientX-drag[0])*0.006;
  camGoal.el=Math.max(-0.05,Math.min(1.35,camGoal.el+(e.clientY-drag[1])*0.005));drag=[e.clientX,e.clientY];
  cam.az=camGoal.az;cam.el=camGoal.el;});
cv.addEventListener('pointerup',()=>drag=null);
cv.addEventListener('wheel',e=>{e.preventDefault();camGoal.r=Math.max(18,Math.min(900,camGoal.r*Math.exp(e.deltaY*0.0012)));},{passive:false});

if(AUTOPLAY&&N>1){fidx=0;playing=true;}
syncPlay();
requestAnimationFrame(frame);
</script></body></html>"""
