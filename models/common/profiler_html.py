"""
HTML timeline profiler for WebGPU model inference.

Generates an interactive HTML report showing CPU and GPU events on a
shared timeline with zoom/pan, tooltips, step markers, and summary tables.
GPU events use hardware timestamp queries for exact GPU-side timings.

Usage:
    from common.profiler_html import generate_html_report
    generate_html_report(profiler, "profile.html")
"""
import json
from typing import List, Optional

from common.profiler import InferenceProfiler, CPUEvent, GPUDispatchEvent, GPUTimestamp


def generate_html_report(profiler: InferenceProfiler,
                         output_path: str = "profile.html",
                         title: str = "WebGPU Inference Profile",
                         adapter_info: Optional[dict] = None,
                         memory_info: Optional[dict] = None):
    """Generate an interactive HTML timeline from profiler data."""
    cpu_events = profiler._cpu.events
    gpu_timestamps = profiler._gpu.timestamps if profiler._gpu else []

    all_ns = []
    for e in cpu_events:
        all_ns.extend([e.begin_ns, e.end_ns])
    dispatch_events = profiler._dispatch_events if hasattr(profiler, '_dispatch_events') else []
    for de in dispatch_events:
        all_ns.extend([de.begin_ns, de.end_ns])

    if not all_ns:
        print("No profiling data to report.")
        return

    t0 = min(all_ns)

    # Build step markers from "total" events
    steps_json = []
    for e in cpu_events:
        if e.name == "total":
            steps_json.append({
                "name": e.scope,
                "start_us": (e.begin_ns - t0) / 1000.0,
                "dur_us": (e.end_ns - e.begin_ns) / 1000.0,
            })

    # CPU events (exclude total/scope_total meta-events)
    cpu_json = []
    for e in cpu_events:
        if e.name in ("total", "scope_total"):
            continue
        entry = {
            "name": e.name,
            "scope": e.scope,
            "start_us": (e.begin_ns - t0) / 1000.0,
            "dur_us": (e.end_ns - e.begin_ns) / 1000.0,
            "gpu": bool(e.gpu_dispatch),
        }
        if hasattr(e, 'link_id') and e.link_id:
            entry["link"] = e.link_id
        cpu_json.append(entry)

    # GPU dispatch events (CPU-timed)
    gpu_json = []
    for de in dispatch_events:
        entry = {
            "name": de.name,
            "start_us": (de.begin_ns - t0) / 1000.0,
            "dur_us": (de.end_ns - de.begin_ns) / 1000.0,
        }
        if hasattr(de, 'link_id') and de.link_id:
            entry["link"] = de.link_id
        gpu_json.append(entry)

    # GPU hardware timestamps — map to CPU timeline
    # When D3D12 clock calibration is available, gpu_timestamps already have
    # begin_ns/end_ns in the CPU perf_counter_ns domain (mapped during
    # resolve_and_read).  Otherwise, fall back to the approximate anchor
    # method using cpu_submit_ns of the first timestamp.
    gpu_hw_json = []
    if gpu_timestamps:
        has_calibration = profiler.has_clock_calibration
        if has_calibration:
            # Calibrated: timestamps are already in CPU ns
            for ts in gpu_timestamps:
                gpu_hw_json.append({
                    "name": ts.name,
                    "start_us": (ts.begin_ns - t0) / 1000.0,
                    "dur_us": (ts.end_ns - ts.begin_ns) / 1000.0,
                })
        else:
            # Fallback: anchor first GPU timestamp to its cpu_submit_ns
            first = gpu_timestamps[0]
            gpu_to_cpu_offset = first.cpu_submit_ns - first.begin_ns
            for ts in gpu_timestamps:
                mapped_begin_ns = ts.begin_ns + gpu_to_cpu_offset
                gpu_hw_json.append({
                    "name": ts.name,
                    "start_us": (mapped_begin_ns - t0) / 1000.0,
                    "dur_us": ts.duration_us,
                })

    summary = _compute_summary(cpu_events, dispatch_events, gpu_timestamps,
                                steps_json)

    replacements = {
        "{{TITLE}}": title,
        "{{CPU_EVENTS}}": json.dumps(cpu_json),
        "{{GPU_EVENTS}}": json.dumps(gpu_json),
        "{{GPU_HW_EVENTS}}": json.dumps(gpu_hw_json),
        "{{STEPS}}": json.dumps(steps_json),
        "{{SUMMARY}}": json.dumps(summary),
        "{{ADAPTER_INFO}}": json.dumps(adapter_info or {}),
        "{{MEMORY_INFO}}": json.dumps(memory_info or {}),
    }
    html = _TEMPLATE
    for k, v in replacements.items():
        html = html.replace(k, v)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Profile report saved to {output_path}")


def _compute_summary(cpu_events: List[CPUEvent],
                     dispatch_events: List[GPUDispatchEvent],
                     gpu_timestamps: List[GPUTimestamp] = None,
                     steps_json: list = None) -> dict:
    cpu_agg = {}
    total_cpu_ms = 0
    for e in cpu_events:
        if e.name in ("total", "scope_total"):
            if e.name == "total" and "/" not in e.scope:
                total_cpu_ms += e.duration_ms
            continue
        if e.name not in cpu_agg:
            cpu_agg[e.name] = {"total_ms": 0, "count": 0}
        cpu_agg[e.name]["total_ms"] += e.duration_ms
        cpu_agg[e.name]["count"] += 1

    # Fallback when model code doesn't emit top-level "total" events.
    if total_cpu_ms <= 0:
        total_cpu_ms = sum(d["total_ms"] for d in cpu_agg.values())

    cpu_summary = []
    for name, d in sorted(cpu_agg.items(), key=lambda x: -x[1]["total_ms"]):
        avg = d["total_ms"] / d["count"] if d["count"] else 0
        pct = (d["total_ms"] / total_cpu_ms * 100) if total_cpu_ms > 0 else 0
        cpu_summary.append({"name": name, "total_ms": round(d["total_ms"], 2),
                            "count": d["count"], "avg_ms": round(avg, 3),
                            "pct": round(pct, 1)})

    gpu_agg = {}
    total_gpu_ms = 0
    for de in dispatch_events:
        name = de.name.split("/")[-1] if "/" in de.name else de.name
        if name not in gpu_agg:
            gpu_agg[name] = {"total_ms": 0, "count": 0}
        dur = de.duration_ms
        gpu_agg[name]["total_ms"] += dur
        gpu_agg[name]["count"] += 1
        total_gpu_ms += dur

    gpu_summary = []
    for name, d in sorted(gpu_agg.items(), key=lambda x: -x[1]["total_ms"]):
        avg = d["total_ms"] / d["count"] if d["count"] else 0
        pct = (d["total_ms"] / total_gpu_ms * 100) if total_gpu_ms > 0 else 0
        gpu_summary.append({"name": name, "total_ms": round(d["total_ms"], 2),
                            "count": d["count"], "avg_ms": round(avg, 3),
                            "pct": round(pct, 1)})

    # GPU HW timestamp summary (actual GPU execution times)
    gpu_hw_agg = {}
    total_gpu_hw_ms = 0
    if gpu_timestamps:
        for ts in gpu_timestamps:
            name = ts.name.split("/")[-1] if "/" in ts.name else ts.name
            if name not in gpu_hw_agg:
                gpu_hw_agg[name] = {"total_ms": 0, "count": 0}
            dur = ts.duration_ms
            gpu_hw_agg[name]["total_ms"] += dur
            gpu_hw_agg[name]["count"] += 1
            total_gpu_hw_ms += dur

    gpu_hw_summary = []
    for name, d in sorted(gpu_hw_agg.items(), key=lambda x: -x[1]["total_ms"]):
        avg = d["total_ms"] / d["count"] if d["count"] else 0
        pct = (d["total_ms"] / total_gpu_hw_ms * 100) if total_gpu_hw_ms > 0 else 0
        gpu_hw_summary.append({"name": name, "total_ms": round(d["total_ms"], 2),
                               "count": d["count"], "avg_ms": round(avg, 3),
                               "pct": round(pct, 1)})

    # Per-step timing
    step_times = []
    for e in cpu_events:
        if e.name == "total" and "/" not in e.scope:
            step_times.append({"name": e.scope, "ms": round(e.duration_ms, 2)})

    # Op counts per phase (prefill vs decode)
    op_counts = _compute_op_counts(cpu_events, dispatch_events, steps_json)

    return {"total_cpu_ms": round(total_cpu_ms, 2),
            "total_gpu_ms": round(total_gpu_ms, 2),
            "total_gpu_hw_ms": round(total_gpu_hw_ms, 2),
            "cpu": cpu_summary, "gpu": gpu_summary,
            "gpu_hw": gpu_hw_summary, "steps": step_times,
            "op_counts": op_counts}


def _compute_op_counts(cpu_events, dispatch_events, steps_json):
    """Compute op counts and GPU dispatch counts per phase (prefill vs decode)."""
    if not steps_json:
        return {}

    # Build phase time ranges from steps
    phases = []  # (name, start_us, end_us)
    for s in steps_json:
        phases.append((s["name"], s["start_us"], s["start_us"] + s["dur_us"]))

    def classify_phase(start_us):
        for name, ps, pe in phases:
            if ps <= start_us < pe:
                return "prefill" if name == "prefill" else "decode"
        return "other"

    # Count CPU ops per phase
    cpu_counts = {"prefill": 0, "decode": 0}
    for e in cpu_events:
        if e.name in ("total", "scope_total"):
            continue
        phase = classify_phase((e.begin_ns - _get_t0(cpu_events, dispatch_events)) / 1000.0)
        if phase in cpu_counts:
            cpu_counts[phase] += 1

    # Count GPU dispatches per phase
    gpu_counts = {"prefill": 0, "decode": 0}
    for de in dispatch_events:
        phase = classify_phase((de.begin_ns - _get_t0(cpu_events, dispatch_events)) / 1000.0)
        if phase in gpu_counts:
            gpu_counts[phase] += 1

    # Decode ops per token (average)
    decode_steps = [s for s in steps_json if s["name"] != "prefill"]
    n_decode = len(decode_steps) if decode_steps else 1

    return {
        "prefill_cpu_ops": cpu_counts["prefill"],
        "prefill_gpu_ops": gpu_counts["prefill"],
        "decode_cpu_ops": cpu_counts["decode"],
        "decode_gpu_ops": gpu_counts["decode"],
        "decode_cpu_ops_per_token": round(cpu_counts["decode"] / n_decode, 1),
        "decode_gpu_ops_per_token": round(gpu_counts["decode"] / n_decode, 1),
        "decode_tokens": n_decode,
    }


def _get_t0(cpu_events, dispatch_events):
    """Get the minimum timestamp (ns) across all events."""
    all_ns = []
    for e in cpu_events:
        all_ns.append(e.begin_ns)
    for de in dispatch_events:
        all_ns.append(de.begin_ns)
    return min(all_ns) if all_ns else 0


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{{TITLE}}</title>
<style>
:root {
  --bg: #1a1b26; --bg2: #1f2335; --bg3: #292e42; --fg: #c0caf5;
  --fg2: #a9b1d6; --fg3: #565f89; --blue: #7aa2f7; --purple: #bb9af7;
  --green: #9ece6a; --orange: #ff9e64; --red: #f7768e; --cyan: #7dcfff;
  --yellow: #e0af68; --teal: #73daca;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:'Segoe UI',system-ui,sans-serif; background:var(--bg); color:var(--fg); font-size:13px; }
header { background:var(--bg2); padding:10px 16px; border-bottom:1px solid var(--bg3);
         display:flex; align-items:center; gap:16px; }
header h1 { font-size:15px; color:var(--purple); font-weight:600; }
header .chip { background:var(--bg3); padding:2px 8px; border-radius:4px; font-size:11px; color:var(--fg2); }
#controls { background:var(--bg2); padding:6px 16px; border-bottom:1px solid var(--bg3);
            display:flex; gap:8px; align-items:center; font-size:12px; }
#controls button { background:var(--bg3); color:var(--fg); border:none; padding:3px 10px;
                   border-radius:3px; cursor:pointer; font-size:11px; }
#controls button:hover { background:#3b4261; }
#controls label { color:var(--fg2); display:flex; align-items:center; gap:6px; font-size:11px; }
#controls .range { color:var(--fg3); margin-left:auto; }
#timeline-wrap { position:relative; overflow:hidden; cursor:grab; }
#timeline-wrap.dragging { cursor:grabbing; }
#flame-wrap { position:relative; overflow:hidden; border-top:1px solid var(--bg3); border-bottom:1px solid var(--bg3); }
.flame-title { padding:6px 16px; color:var(--fg2); background:var(--bg2); border-top:1px solid var(--bg3); font-size:12px; }
canvas { display:block; }
#tooltip { position:fixed; background:var(--bg3); color:var(--fg); padding:6px 10px;
           border-radius:4px; font-size:11px; pointer-events:none; display:none;
           border:1px solid #3b4261; z-index:100; line-height:1.5;
           box-shadow:0 2px 8px rgba(0,0,0,.5); max-width:360px; }
#tooltip b { color:var(--blue); }
#tooltip .dim { color:var(--fg3); }
#summary { padding:16px; display:flex; gap:16px; flex-wrap:wrap; }
.panel { background:var(--bg2); border:1px solid var(--bg3); border-radius:6px;
         padding:12px; min-width:280px; flex:1; }
.panel h2 { font-size:13px; color:var(--blue); margin-bottom:8px; }
.panel table { width:100%; border-collapse:collapse; font-size:11px; }
.panel th { text-align:left; padding:3px 6px; color:var(--fg3); border-bottom:1px solid var(--bg3); }
.panel td { padding:3px 6px; border-bottom:1px solid var(--bg); }
.panel td.r { text-align:right; font-variant-numeric:tabular-nums; }
.bar { height:4px; border-radius:2px; margin-top:1px; }
.step-cards { display:flex; gap:8px; flex-wrap:wrap; padding:8px 16px; }
.step-card { background:var(--bg2); border:1px solid var(--bg3); border-radius:6px;
             padding:8px 12px; min-width:90px; text-align:center; }
.step-card .label { font-size:10px; color:var(--fg3); }
.step-card .value { font-size:16px; font-weight:600; color:var(--green); }
.step-card .unit { font-size:10px; color:var(--fg3); }
</style>
</head>
<body>
<header>
  <h1>{{TITLE}}</h1>
  <span class="chip" id="gpu-chip"></span>
  <span class="chip" id="backend-chip"></span>
</header>
<div class="step-cards" id="step-cards"></div>
<div id="controls">
  <button onclick="resetZoom()">Reset</button>
  <button onclick="zoomIn()">+ Zoom</button>
  <button onclick="zoomOut()">- Zoom</button>
  <label><input id="toggle-flame" type="checkbox" checked onchange="toggleFlame()"> CPU Flamechart</label>
  <span class="range" id="range-label"></span>
</div>
<div id="timeline-wrap"><canvas id="c"></canvas></div>
<div class="flame-title">CPU Flamechart</div>
<div id="flame-wrap"><canvas id="fc"></canvas></div>
<div id="tooltip"></div>
<div id="summary">
  <div class="panel" id="overview-panel"><h2>Overview</h2><div id="overview-info"></div></div>
  <div class="panel"><h2>CPU Ops</h2><table id="cpu-tbl"><thead><tr>
    <th>Op</th><th>Total</th><th>#</th><th>Avg</th><th>%</th><th></th>
  </tr></thead><tbody></tbody></table></div>
  <div class="panel"><h2>GPU Dispatches</h2><table id="gpu-tbl"><thead><tr>
    <th>Op</th><th>Total</th><th>#</th><th>Avg</th><th>%</th><th></th>
  </tr></thead><tbody></tbody></table></div>
  <div class="panel" id="mem-panel"><h2>Memory</h2><div id="mem-info"></div></div>
</div>
<script>
const C=document.getElementById('c'), X=C.getContext('2d'),
      W=document.getElementById('timeline-wrap'),
      TT=document.getElementById('tooltip');
const FC=document.getElementById('fc'), FX=FC.getContext('2d'), FW=document.getElementById('flame-wrap');
const cpu=/*CPU*/{{CPU_EVENTS}};
const gpu=/*GPU*/{{GPU_EVENTS}};
const gpuHW=/*GPU_HW*/{{GPU_HW_EVENTS}};
const steps=/*STEPS*/{{STEPS}};
const S=/*SUM*/{{SUMMARY}};
const A=/*ADP*/{{ADAPTER_INFO}};
const M=/*MEM*/{{MEMORY_INFO}};
const dpr=devicePixelRatio||1;

if(A.device)document.getElementById('gpu-chip').textContent=A.device;
if(A.backend)document.getElementById('backend-chip').textContent=A.backend;

// Step cards + op counts
const sc=document.getElementById('step-cards');
if(S.steps&&S.steps.length){
  const decode=S.steps.filter(s=>s.name.startsWith('decode'));
  if(decode.length>0){
    const avgDec=decode.reduce((a,s)=>a+s.ms,0)/decode.length;
    const tps=1000/avgDec;
    let h='';
    h+=`<div class="step-card"><div class="label">Decode</div><div class="value">${tps.toFixed(1)}</div><div class="unit">tok/s</div></div>`;
    h+=`<div class="step-card"><div class="label">Avg/token</div><div class="value">${avgDec.toFixed(1)}</div><div class="unit">ms</div></div>`;
    const pf=S.steps.find(s=>s.name==='prefill');
    if(pf)h+=`<div class="step-card"><div class="label">Prefill</div><div class="value">${pf.ms.toFixed(0)}</div><div class="unit">ms</div></div>`;
    h+=`<div class="step-card"><div class="label">Steps</div><div class="value">${S.steps.length}</div><div class="unit">total</div></div>`;
    const oc=S.op_counts||{};
    if(oc.prefill_gpu_ops)h+=`<div class="step-card"><div class="label">Prefill ops</div><div class="value">${oc.prefill_gpu_ops}</div><div class="unit">GPU / ${oc.prefill_cpu_ops} CPU</div></div>`;
    if(oc.decode_gpu_ops_per_token)h+=`<div class="step-card"><div class="label">Decode ops/tok</div><div class="value">${oc.decode_gpu_ops_per_token}</div><div class="unit">GPU / ${oc.decode_cpu_ops_per_token} CPU</div></div>`;
    if(oc.decode_gpu_ops)h+=`<div class="step-card"><div class="label">Decode total</div><div class="value">${oc.decode_gpu_ops}</div><div class="unit">GPU (${oc.decode_tokens} tok)</div></div>`;
    sc.innerHTML=h;
  }
}else{
  // Fallback for profiles that don't emit step markers.
  const pre=(S.cpu||[]).find(d=>d.name==='prefill');
  const fd=(S.gpu||[]).find(d=>d.name==='fast_decode');
  let h='';
  if(fd&&fd.avg_ms>0){
    const tps=1000/fd.avg_ms;
    h+=`<div class="step-card"><div class="label">Decode</div><div class="value">${tps.toFixed(1)}</div><div class="unit">tok/s</div></div>`;
    h+=`<div class="step-card"><div class="label">Avg/token</div><div class="value">${fd.avg_ms.toFixed(2)}</div><div class="unit">ms</div></div>`;
    h+=`<div class="step-card"><div class="label">Decode calls</div><div class="value">${fd.count}</div><div class="unit">fast_decode</div></div>`;
  }
  if(pre){
    h+=`<div class="step-card"><div class="label">Prefill</div><div class="value">${pre.total_ms.toFixed(0)}</div><div class="unit">ms</div></div>`;
  }
  if(h)sc.innerHTML=h;
}

// Colors
const OC={qkv_linear:'#7aa2f7',qkv:'#7aa2f7',gate_up:'#f7768e',down_proj:'#9ece6a',down:'#9ece6a',
  o_proj:'#ff9e64',attention_cpu:'#bb9af7',attn:'#bb9af7',norm1:'#7dcfff',norm2:'#7dcfff',
  res_add_norm:'#73daca',res1:'#73daca',res2:'#73daca',final_norm:'#7dcfff',
  silu_mul_cpu:'#e0af68',silu_mul:'#e0af68',mlp:'#e0af68',
  lm_head:'#f7768e',embed:'#73daca',sampling:'#414868',
  rope_q:'#ff9e64',rope_kv:'#ff9e64',upload_weights:'#414868'};
function oc(n){if(OC[n])return OC[n];let h=0;for(let i=0;i<n.length;i++)h=(h*31+n.charCodeAt(i))&0xffffff;return`hsl(${h%360},55%,60%)`;}

// Layout: 2 lanes — CPU (with step markers as background), GPU
const LW=80, TH=20;
const CPU_Y=6;
const GPU_Y=CPU_Y+TH+4;
const CH=GPU_Y+TH+20;
const FRH=16;

const flameEvents=cpu.map(e=>{
  const depth=(e.scope?e.scope.split('/').filter(Boolean).length:0);
  return {...e, depth};
});
const flameMaxDepth=flameEvents.length?Math.max(...flameEvents.map(e=>e.depth)):0;
const FCH=(flameMaxDepth+1)*FRH+20;
// Merge GPU events: prefer HW timestamps, fall back to CPU-timed
const gpuMerged=gpuHW.length?gpuHW:gpu;

let allS=cpu.map(e=>e.start_us).concat(gpuMerged.map(e=>e.start_us)).concat(steps.map(e=>e.start_us));
let allE=cpu.map(e=>e.start_us+e.dur_us).concat(gpuMerged.map(e=>e.start_us+e.dur_us)).concat(steps.map(e=>e.start_us+e.dur_us));
let gMin=allS.length?Math.min(...allS):0, gMax=allE.length?Math.max(...allE):1000;

let vS=gMin, vE=gMax, drag=false, dX=0, dVS=0;

function resize(){
  const w=W.clientWidth;
  C.width=w*dpr; C.height=CH*dpr;
  C.style.width=w+'px'; C.style.height=CH+'px';
  X.setTransform(dpr,0,0,dpr,0,0); draw();

  FC.width=w*dpr; FC.height=FCH*dpr;
  FC.style.width=w+'px'; FC.style.height=FCH+'px';
  FX.setTransform(dpr,0,0,dpr,0,0); drawFlame();
}

function t2x(t){return LW+(t-vS)/(vE-vS)*(C.clientWidth-LW);}
function x2t(x){return vS+(x-LW)/(C.clientWidth-LW)*(vE-vS);}

function niceStep(r,mx){
  const rg=r/mx,p=Math.pow(10,Math.floor(Math.log10(rg))),n=rg/p;
  return(n<=1.5?1:n<=3.5?2:n<=7.5?5:10)*p;
}

function drawLane(events, y, getName){
  const w=C.clientWidth;
  for(const e of events){
    const n=getName?getName(e):e.name;
    const x1=t2x(e.start_us),x2=t2x(e.start_us+e.dur_us),bw=Math.max(x2-x1,1);
    if(x2<LW||x1>w)continue;
    X.fillStyle=oc(n); X.fillRect(x1,y+1,bw,TH-2);
    if(bw>24){X.fillStyle='#1a1b26';X.font='8px system-ui';
      X.save();X.beginPath();X.rect(x1,y,bw,TH);X.clip();
      X.fillText(n,x1+2,y+12);X.restore();}
  }
}

function draw(){
  const w=C.clientWidth, range=vE-vS;
  X.clearRect(0,0,w,CH);
  X.fillStyle='#1a1b26'; X.fillRect(0,0,w,CH);

  // Grid + time axis
  const ts=niceStep(range,(w-LW)/80);
  X.font='9px system-ui';
  for(let t=Math.ceil(vS/ts)*ts;t<=vE;t+=ts){
    const x=t2x(t);
    X.fillStyle='#292e42'; X.fillRect(x,0,1,CH);
    X.fillStyle='#565f89';
    const lb=t>=1e3?(t/1e3).toFixed(1)+'ms':t.toFixed(0)+'us';
    X.fillText(lb,x+2,CH-3);
  }

  // Lane labels
  X.fillStyle='#565f89'; X.font='10px system-ui';
  X.fillText('CPU',4,CPU_Y+14);
  X.fillText('GPU',4,GPU_Y+14);

  // Lane bg
  X.fillStyle='#1f2335';
  X.fillRect(LW,CPU_Y,w-LW,TH);
  X.fillRect(LW,GPU_Y,w-LW,TH);

  // Step markers as background bands in BOTH lanes
  // Prefill = green-tinted, Decode = blue-tinted (obvious differentiation)
  for(const s of steps){
    const x1=t2x(s.start_us),x2=t2x(s.start_us+s.dur_us),bw=Math.max(x2-x1,1);
    if(x2<LW||x1>w)continue;
    const isPrefill=s.name==='prefill';
    const bandColor=isPrefill?'rgba(158,206,106,0.12)':'rgba(122,162,247,0.08)';
    const borderColor=isPrefill?'rgba(158,206,106,0.35)':'rgba(122,162,247,0.25)';
    const labelColor=isPrefill?'#9ece6a':'#7aa2f7';
    // CPU lane band
    X.fillStyle=bandColor;
    X.fillRect(x1,CPU_Y,bw,TH);
    // GPU lane band (same phase coloring)
    X.fillRect(x1,GPU_Y,bw,TH);
    // Phase border line
    X.strokeStyle=borderColor; X.lineWidth=1;
    X.beginPath();X.moveTo(x1,CPU_Y);X.lineTo(x1,GPU_Y+TH);X.stroke();
    // Phase label — show at top of CPU lane
    if(bw>24){X.fillStyle=labelColor;X.font='bold 9px system-ui';X.fillText(s.name,x1+3,CPU_Y+10);}
  }

  // CPU lane (ops on top of step bands)
  drawLane(cpu, CPU_Y, e=>e.name);

  // GPU lane (merged: HW timestamps if available, else CPU-timed)
  drawLane(gpuMerged, GPU_Y, e=>e.name.split('/').pop());

  // Connector lines: link CPU recording to GPU execution
  // Draws dashed lines from CPU bar bottom-center to GPU bar top-center
  // for events that share a "link" id
  const cpuLinks={}, gpuLinks={};
  for(const e of cpu){if(e.link){const cx=t2x(e.start_us+e.dur_us/2);if(cx>=LW&&cx<=w)cpuLinks[e.link]={x:cx,x1:t2x(e.start_us),x2:t2x(e.start_us+e.dur_us),name:e.name};}}
  for(const e of gpuMerged){if(e.link){const cx=t2x(e.start_us+e.dur_us/2);if(cx>=LW&&cx<=w)gpuLinks[e.link]={x:cx,x1:t2x(e.start_us),x2:t2x(e.start_us+e.dur_us),name:e.name};}}
  X.save();
  X.setLineDash([3,3]);
  X.lineWidth=0.8;
  for(const id in cpuLinks){
    if(!gpuLinks[id])continue;
    const c=cpuLinks[id],g=gpuLinks[id];
    // Color: match the op color
    X.strokeStyle=oc(c.name);
    X.globalAlpha=0.6;
    // Draw from CPU bar bottom to GPU bar top
    X.beginPath();
    X.moveTo(c.x, CPU_Y+TH);
    X.lineTo(g.x, GPU_Y);
    X.stroke();
  }
  X.restore();

  document.getElementById('range-label').textContent=
    range>=1e3?(range/1e3).toFixed(1)+'ms':range.toFixed(0)+'us';

  drawFlame();
}

function toggleFlame(){
  const on=document.getElementById('toggle-flame').checked;
  const fw=document.getElementById('flame-wrap');
  const ft=document.querySelector('.flame-title');
  fw.style.display=on?'block':'none';
  ft.style.display=on?'block':'none';
  resize();
}

function drawFlame(){
  const w=FC.clientWidth;
  FX.clearRect(0,0,w,FCH);
  FX.fillStyle='#1a1b26'; FX.fillRect(0,0,w,FCH);

  const range=vE-vS;
  const ts=niceStep(range,(w-LW)/80);
  FX.font='9px system-ui';
  for(let t=Math.ceil(vS/ts)*ts;t<=vE;t+=ts){
    const x=t2x(t);
    FX.fillStyle='#292e42'; FX.fillRect(x,0,1,FCH);
  }

  for(const e of flameEvents){
    const x1=t2x(e.start_us),x2=t2x(e.start_us+e.dur_us),bw=Math.max(x2-x1,1);
    if(x2<LW||x1>w)continue;
    const y=4+e.depth*FRH;
    const n=e.name.includes('/')?e.name.split('/').pop():e.name;
    FX.fillStyle=oc(n); FX.fillRect(x1,y,bw,FRH-2);
    if(bw>28){
      FX.fillStyle='#1a1b26'; FX.font='8px system-ui';
      FX.save(); FX.beginPath(); FX.rect(x1,y,bw,FRH-2); FX.clip();
      FX.fillText(n,x1+2,y+11); FX.restore();
    }
  }

  FX.fillStyle='#565f89'; FX.font='10px system-ui';
  FX.fillText('Depth',4,12);
}

// Zoom/pan
C.addEventListener('wheel',ev=>{
  ev.preventDefault();
  const f=ev.deltaY>0?1.25:0.8,mx=ev.offsetX,t=x2t(mx);
  const nr=(vE-vS)*f,fr=(mx-LW)/(C.clientWidth-LW);
  vS=t-fr*nr;vE=vS+nr;draw();
});
C.addEventListener('mousedown',ev=>{drag=true;dX=ev.clientX;dVS=vS;W.classList.add('dragging');});
addEventListener('mousemove',ev=>{
  if(drag){const dx=ev.clientX-dX,r=vE-vS;vS=dVS-dx/(C.clientWidth-LW)*r;vE=vS+r;draw();}
  const rc=C.getBoundingClientRect(),mx=ev.clientX-rc.left,my=ev.clientY-rc.top,t=x2t(mx);
  let hit=null;
  if(my>=CPU_Y&&my<CPU_Y+TH){
    for(const e of cpu)if(t>=e.start_us&&t<=e.start_us+e.dur_us){hit={...e,type:'CPU'};break;}
    if(!hit){for(const s of steps)if(t>=s.start_us&&t<=s.start_us+s.dur_us){hit={name:s.name,dur_us:s.dur_us,type:'Step'};break;}}
  }else if(my>=GPU_Y&&my<GPU_Y+TH){
    for(const e of gpuMerged)if(t>=e.start_us&&t<=e.start_us+e.dur_us){hit={name:e.name.split('/').pop(),scope:e.name,dur_us:e.dur_us,type:gpuHW.length?'GPU (hw)':'GPU (sub)'};break;}
  }
  if(hit){
    const d=hit.dur_us>=1000?(hit.dur_us/1000).toFixed(2)+' ms':hit.dur_us.toFixed(1)+' us';
    let h=`<b>${hit.name}</b>`;
    if(hit.scope)h+=`<br><span class="dim">${hit.scope}</span>`;
    h+=`<br>${hit.type}: ${d}`;
    TT.innerHTML=h;TT.style.display='block';
    TT.style.left=(ev.clientX+12)+'px';TT.style.top=(ev.clientY-10)+'px';
  }else TT.style.display='none';

  // Flamechart hover
  const frc=FC.getBoundingClientRect();
  const fmx=ev.clientX-frc.left, fmy=ev.clientY-frc.top;
  if(fmx>=LW&&fmx<=FC.clientWidth&&fmy>=0&&fmy<=FCH){
    const ft=x2t(fmx);
    let fHit=null;
    for(const e of flameEvents){
      const y=4+e.depth*FRH;
      if(fmy>=y&&fmy<y+FRH-2&&ft>=e.start_us&&ft<=e.start_us+e.dur_us){fHit=e;break;}
    }
    if(fHit){
      const d=fHit.dur_us>=1000?(fHit.dur_us/1000).toFixed(2)+' ms':fHit.dur_us.toFixed(1)+' us';
      TT.innerHTML=`<b>${fHit.name}</b><br><span class="dim">${fHit.scope||''}</span><br>CPU Flame: ${d}`;
      TT.style.display='block';
      TT.style.left=(ev.clientX+12)+'px';TT.style.top=(ev.clientY-10)+'px';
    }
  }
});
addEventListener('mouseup',()=>{drag=false;W.classList.remove('dragging');});

function resetZoom(){vS=gMin;vE=gMax;draw();}
function zoomIn(){const m=(vS+vE)/2,r=(vE-vS)/2;vS=m-r/2;vE=m+r/2;draw();}
function zoomOut(){const m=(vS+vE)/2,r=(vE-vS)*2;vS=m-r/2;vE=m+r/2;draw();}

// Summary tables
function fillTbl(id,data,tCol,aCol){
  const tb=document.querySelector('#'+id+' tbody');
  if(!data||!data.length){tb.innerHTML='<tr><td colspan="6" style="color:var(--fg3)">No data</td></tr>';return;}
  tb.innerHTML=data.map(d=>{
    const tv=d[tCol],av=d[aCol];
    const tu=tCol.includes('ms')?'ms':'us',au=aCol.includes('ms')?'ms':'us';
    return`<tr><td>${d.name}</td><td class="r">${tv>=1?tv.toFixed(1):tv.toFixed(2)} ${tu}</td>
    <td class="r">${d.count}x</td><td class="r">${av>=1?av.toFixed(2):av.toFixed(3)} ${au}</td>
    <td class="r">${d.pct.toFixed(1)}%</td>
    <td><div class="bar" style="width:${Math.max(d.pct,1)}%;background:${oc(d.name)}"></div></td></tr>`;
  }).join('');
}
fillTbl('cpu-tbl',S.cpu,'total_ms','avg_ms');
// Use HW timestamp summary if available, else CPU-timed dispatch summary
fillTbl('gpu-tbl',S.gpu_hw&&S.gpu_hw.length?S.gpu_hw:S.gpu,'total_ms','avg_ms');

// Overview panel
const ov=document.getElementById('overview-info');
const gpuRows=(S.gpu_hw&&S.gpu_hw.length?S.gpu_hw:S.gpu)||[];
const cpuTop=(S.cpu||[]).slice(0,1)[0];
const gpuTop=gpuRows.slice(0,1)[0];
const ovRows=[
  ['CPU total', (S.total_cpu_ms||0).toFixed(2)+' ms'],
  ['GPU total', (S.total_gpu_ms||0).toFixed(2)+' ms'],
  ['GPU HW total', (S.total_gpu_hw_ms||0).toFixed(2)+' ms'],
  ['Top CPU op', cpuTop?`${cpuTop.name} (${cpuTop.total_ms.toFixed(2)} ms)`:'N/A'],
  ['Top GPU op', gpuTop?`${gpuTop.name} (${gpuTop.total_ms.toFixed(2)} ms)`:'N/A'],
];
ov.innerHTML='<table style="width:100%;font-size:11px;border-collapse:collapse">'+ovRows.map(([k,v])=>
  `<tr><td style="padding:3px 6px;color:var(--fg2);border-bottom:1px solid var(--bg)">${k}</td><td class="r" style="padding:3px 6px;border-bottom:1px solid var(--bg);font-variant-numeric:tabular-nums">${v}</td></tr>`
).join('')+'</table>';

// Memory info panel
const mp=document.getElementById('mem-info');
if(M&&M.gpu_total_mb){
  const rows=[
    ['GPU Total Allocated',M.gpu_total_mb>=1024?(M.gpu_total_mb/1024).toFixed(2)+' GB':M.gpu_total_mb+' MB'],
    ['GPU Weights',M.gpu_weight_mb>=1024?(M.gpu_weight_mb/1024).toFixed(2)+' GB':M.gpu_weight_mb+' MB'],
    ['GPU Buffer Cache',M.gpu_buffer_cache_mb.toFixed(1)+' MB'],
    ['GPU Allocations',M.gpu_alloc_count],
    ['Weight Tensors',M.gpu_weight_count],
    ['Compiled Pipelines',M.gpu_pipeline_count],
    ['CPU Weights',M.cpu_weight_mb>=1024?(M.cpu_weight_mb/1024).toFixed(2)+' GB':M.cpu_weight_mb+' MB'],
  ];
  mp.innerHTML='<table style="width:100%;font-size:11px;border-collapse:collapse">'+rows.map(([k,v])=>
    `<tr><td style="padding:3px 6px;color:var(--fg2);border-bottom:1px solid var(--bg)">${k}</td><td class="r" style="padding:3px 6px;border-bottom:1px solid var(--bg);font-variant-numeric:tabular-nums">${v}</td></tr>`
  ).join('')+'</table>';
}else{
  mp.innerHTML='<span style="color:var(--fg3);font-size:11px">No memory data</span>';
}

addEventListener('resize',resize);
resize();
</script>
</body>
</html>
"""
