#include "profile_html.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

// JSON helpers
static std::string jsonEscape(const std::string& s) {
    std::string r;
    for (char c : s) {
        if (c == '"') r += "\\\"";
        else if (c == '\\') r += "\\\\";
        else r += c;
    }
    return r;
}

struct JsonEvent { std::string name; double startUs; double durUs; };

static std::string eventsToJson(const std::vector<JsonEvent>& v) {
    std::ostringstream s;
    s << "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i) s << ",";
        s << "{\"name\":\"" << jsonEscape(v[i].name)
          << "\",\"start_us\":" << v[i].startUs
          << ",\"dur_us\":" << v[i].durUs << "}";
    }
    s << "]";
    return s.str();
}

struct AggEntry { double totalMs = 0; uint32_t count = 0; };

static std::string summaryToJson(const std::map<std::string, AggEntry>& agg,
                                  double totalMs) {
    std::vector<std::pair<std::string, AggEntry>> sorted(agg.begin(), agg.end());
    std::sort(sorted.begin(), sorted.end(),
              [](auto& a, auto& b) { return a.second.totalMs > b.second.totalMs; });
    std::ostringstream s;
    s << "[";
    for (size_t i = 0; i < sorted.size(); i++) {
        if (i) s << ",";
        auto& [name, e] = sorted[i];
        double avgMs = e.count > 0 ? e.totalMs / e.count : 0;
        double pct = totalMs > 0 ? e.totalMs / totalMs * 100.0 : 0;
        s << "{\"name\":\"" << jsonEscape(name)
          << "\",\"total_ms\":" << e.totalMs
          << ",\"count\":" << e.count
          << ",\"avg_ms\":" << avgMs
          << ",\"pct\":" << pct << "}";
    }
    s << "]";
    return s.str();
}

static void replaceAll(std::string& s, const std::string& from,
                        const std::string& to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.length(), to);
        pos += to.length();
    }
}

void generateProfileHTML(
        GPUContext& gpu, GPUProfiler& profiler,
        const ClockCalibration* cal, const uint64_t* ptr,
        int nDecodeTokens, int nPrefillTokens,
        double prefillMs, double decodeMs,
        const std::string& outputPath) {

    // Count dispatches per token
    int dispatchesPerToken = 0;
    for (size_t i = 0; i < profiler.entries.size(); i++) {
        dispatchesPerToken++;
        if (profiler.entries[i].name == "argmax") {
            break;
        }
    }
    if (dispatchesPerToken == 0)
        dispatchesPerToken = (int)profiler.entries.size() /
            std::max(1, nPrefillTokens + nDecodeTokens);

    // Build GPU HW events mapped to CPU timeline
    std::vector<JsonEvent> gpuHwEvents;
    uint64_t t0Gpu = UINT64_MAX;
    for (auto& e : profiler.entries) {
        uint64_t begin = ptr[e.beginIdx], end = ptr[e.endIdx];
        if (end > begin && begin > 0 && begin < t0Gpu) t0Gpu = begin;
    }
    uint64_t t0Cpu = cal && cal->valid ? cal->gpuNsToCpuNs(t0Gpu) : t0Gpu;

    for (auto& e : profiler.entries) {
        uint64_t begin = ptr[e.beginIdx], end = ptr[e.endIdx];
        if (end <= begin || begin == 0) continue;
        uint64_t cpuBegin = cal && cal->valid ? cal->gpuNsToCpuNs(begin) : begin;
        double startUs = (double)(cpuBegin - t0Cpu) / 1000.0;
        double durUs = (double)(end - begin) / 1000.0;
        gpuHwEvents.push_back({e.name, startUs, durUs});
    }

    // Build step markers
    std::vector<JsonEvent> steps;
    int dispatchIdx = 0, tokenIdx = 0;
    double curStart = 0;
    for (size_t i = 0; i < gpuHwEvents.size(); i++) {
        if (dispatchIdx == 0) curStart = gpuHwEvents[i].startUs;
        dispatchIdx++;
        if (dispatchIdx >= dispatchesPerToken || i == gpuHwEvents.size() - 1) {
            double endUs = gpuHwEvents[i].startUs + gpuHwEvents[i].durUs;
            std::string name = tokenIdx < nPrefillTokens
                ? "prefill" : "decode_" + std::to_string(tokenIdx - nPrefillTokens);
            steps.push_back({name, curStart, endUs - curStart});
            dispatchIdx = 0;
            tokenIdx++;
        }
    }

    // Summary
    std::map<std::string, AggEntry> gpuAgg;
    double totalGpuMs = 0;
    for (auto& ev : gpuHwEvents) {
        std::string kernel = ev.name;
        auto sl = kernel.find('/');
        if (sl != std::string::npos) kernel = kernel.substr(sl + 1);
        gpuAgg[kernel].totalMs += ev.durUs / 1000.0;
        gpuAgg[kernel].count++;
        totalGpuMs += ev.durUs / 1000.0;
    }

    std::ostringstream summaryJson;
    summaryJson << "{\"total_gpu_hw_ms\":" << totalGpuMs
                << ",\"gpu_hw\":" << summaryToJson(gpuAgg, totalGpuMs)
                << ",\"cpu\":[],\"gpu\":[],\"steps\":[]}";

    std::string backendName = gpu.backendType == WGPUBackendType_D3D12 ? "D3D12" : "Vulkan";
    char timeBuf[64];
    time_t now = time(nullptr);
    strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%d %H:%M:%S", localtime(&now));
    double tokPerSec = decodeMs > 0 ? nDecodeTokens * 1000.0 / decodeMs : 0;

    // Template with placeholders
    std::string html = R"TMPL(<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Backpack Engine GPU Profile</title>
<style>
:root{--bg:#1a1b26;--bg2:#1f2335;--bg3:#292e42;--fg:#c0caf5;--fg2:#a9b1d6;--fg3:#565f89;--blue:#7aa2f7;--purple:#bb9af7;--green:#9ece6a;--orange:#ff9e64;--red:#f7768e;--cyan:#7dcfff;--yellow:#e0af68;--teal:#73daca}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--fg);font-size:13px}
header{background:var(--bg2);padding:10px 16px;border-bottom:1px solid var(--bg3);display:flex;align-items:center;gap:16px}
header h1{font-size:15px;color:var(--purple);font-weight:600}
.chip{background:var(--bg3);padding:2px 8px;border-radius:4px;font-size:11px;color:var(--fg2)}
#controls{background:var(--bg2);padding:6px 16px;border-bottom:1px solid var(--bg3);display:flex;gap:8px;align-items:center;font-size:12px}
#controls button{background:var(--bg3);color:var(--fg);border:none;padding:3px 10px;border-radius:3px;cursor:pointer;font-size:11px}
#controls button:hover{background:#3b4261}
.range{color:var(--fg3);margin-left:auto}
#timeline-wrap{position:relative;overflow:hidden;cursor:grab}
#timeline-wrap.dragging{cursor:grabbing}
canvas{display:block}
#tooltip{position:fixed;background:var(--bg3);color:var(--fg);padding:6px 10px;border-radius:4px;font-size:11px;pointer-events:none;display:none;border:1px solid #3b4261;z-index:100;line-height:1.5;box-shadow:0 2px 8px rgba(0,0,0,.5);max-width:360px}
#tooltip b{color:var(--blue)} #tooltip .dim{color:var(--fg3)}
#summary{padding:16px;display:flex;gap:16px;flex-wrap:wrap}
.panel{background:var(--bg2);border:1px solid var(--bg3);border-radius:6px;padding:12px;min-width:280px;flex:1}
.panel h2{font-size:13px;color:var(--blue);margin-bottom:8px}
.panel table{width:100%;border-collapse:collapse;font-size:11px}
.panel th{text-align:left;padding:3px 6px;color:var(--fg3);border-bottom:1px solid var(--bg3)}
.panel td{padding:3px 6px;border-bottom:1px solid var(--bg)} .panel td.r{text-align:right;font-variant-numeric:tabular-nums}
.bar{height:4px;border-radius:2px;margin-top:1px}
.step-cards{display:flex;gap:8px;flex-wrap:wrap;padding:8px 16px}
.step-card{background:var(--bg2);border:1px solid var(--bg3);border-radius:6px;padding:8px 12px;min-width:90px;text-align:center}
.step-card .label{font-size:10px;color:var(--fg3)} .step-card .value{font-size:16px;font-weight:600;color:var(--green)} .step-card .unit{font-size:10px;color:var(--fg3)}
</style></head><body>
<header><h1>Backpack Engine GPU Profile</h1><span class="chip">{{TIMESTAMP}}</span><span class="chip">{{BACKEND}}</span></header>
<div class="step-cards">
<div class="step-card"><div class="label">Decode</div><div class="value">{{TPS}}</div><div class="unit">tok/s</div></div>
<div class="step-card"><div class="label">Prefill</div><div class="value">{{PREFILL_MS}}</div><div class="unit">ms</div></div>
<div class="step-card"><div class="label">Tokens</div><div class="value">{{N_TOKENS}}</div><div class="unit">decoded</div></div>
<div class="step-card"><div class="label">GPU Total</div><div class="value">{{GPU_TOTAL_MS}}</div><div class="unit">ms</div></div>
</div>
<div id="controls"><button onclick="resetZoom()">Reset</button><button onclick="zoomIn()">+ Zoom</button><button onclick="zoomOut()">- Zoom</button><span class="range" id="range-label"></span></div>
<div id="timeline-wrap"><canvas id="c"></canvas></div>
<div id="tooltip"></div>
<div id="summary"><div class="panel"><h2>GPU Kernels (Hardware Timestamps)</h2><table id="gpu-tbl"><thead><tr><th>Kernel</th><th>Total</th><th>#</th><th>Avg</th><th>%</th><th></th></tr></thead><tbody></tbody></table></div></div>
<script>
const C=document.getElementById('c'),X=C.getContext('2d'),W=document.getElementById('timeline-wrap'),TT=document.getElementById('tooltip');
const gpuHW={{GPU_HW_EVENTS}};
const steps={{STEPS}};
const S={{SUMMARY}};
const dpr=devicePixelRatio||1;
const OC={q8_gateup:'#f7768e',q8_down_add:'#9ece6a',lm_head:'#bb9af7',q8_qkv:'#7aa2f7',q8_oproj:'#ff9e64',attn_p1:'#7dcfff',attn_p2:'#73daca',add_rms:'#e0af68',rms_next:'#e0af68',rms_norm:'#e0af68',final_rms:'#e0af68',fused_rope:'#73daca',silu_mul:'#73daca',argmax:'#414868'};
function oc(n){if(OC[n])return OC[n];let h=0;for(let i=0;i<n.length;i++)h=(h*31+n.charCodeAt(i))&0xffffff;return'hsl('+h%360+',55%,60%)';}
const LW=80,TH=16,GPU_Y=6,CH=GPU_Y+TH+20;
let allS=gpuHW.map(e=>e.start_us).concat(steps.map(e=>e.start_us));
let allE=gpuHW.map(e=>e.start_us+e.dur_us).concat(steps.map(e=>e.start_us+e.dur_us));
let gMin=allS.length?Math.min(...allS):0,gMax=allE.length?Math.max(...allE):1000;
let vS=gMin,vE=gMax,drag=false,dX=0,dVS=0;
function resize(){const w=W.clientWidth;C.width=w*dpr;C.height=CH*dpr;C.style.width=w+'px';C.style.height=CH+'px';X.setTransform(dpr,0,0,dpr,0,0);draw();}
function t2x(t){return LW+(t-vS)/(vE-vS)*(C.clientWidth-LW);}
function x2t(x){return vS+(x-LW)/(C.clientWidth-LW)*(vE-vS);}
function niceStep(r,mx){const rg=r/mx,p=Math.pow(10,Math.floor(Math.log10(rg))),n=rg/p;return(n<=1.5?1:n<=3.5?2:n<=7.5?5:10)*p;}
function draw(){
const w=C.clientWidth,range=vE-vS;X.clearRect(0,0,w,CH);X.fillStyle='#1a1b26';X.fillRect(0,0,w,CH);
const ts=niceStep(range,(w-LW)/80);X.font='9px system-ui';
for(let t=Math.ceil(vS/ts)*ts;t<=vE;t+=ts){const x=t2x(t);X.fillStyle='#292e42';X.fillRect(x,0,1,CH);X.fillStyle='#565f89';const lb=t>=1e6?(t/1e6).toFixed(1)+'s':t>=1e3?(t/1e3).toFixed(1)+'ms':t.toFixed(0)+'us';X.fillText(lb,x+2,CH-3);}
X.fillStyle='#565f89';X.font='10px system-ui';X.fillText('GPU',4,GPU_Y+14);
X.fillStyle='#1f2335';X.fillRect(LW,GPU_Y,w-LW,TH);
for(const s of steps){const x1=t2x(s.start_us),x2=t2x(s.start_us+s.dur_us),bw=Math.max(x2-x1,1);if(x2<LW||x1>w)continue;const isPf=s.name==='prefill';X.fillStyle=isPf?'rgba(158,206,106,0.08)':'rgba(122,162,247,0.05)';X.fillRect(x1,GPU_Y,bw,TH);X.strokeStyle=isPf?'rgba(158,206,106,0.35)':'rgba(122,162,247,0.15)';X.lineWidth=1;X.beginPath();X.moveTo(x1,GPU_Y);X.lineTo(x1,GPU_Y+TH);X.stroke();}
for(const e of gpuHW){const n=e.name.split('/').pop();const x1=t2x(e.start_us),x2=t2x(e.start_us+e.dur_us),bw=Math.max(x2-x1,1);if(x2<LW||x1>w)continue;X.fillStyle=oc(n);X.fillRect(x1,GPU_Y+1,bw,TH-2);if(bw>24){X.fillStyle='#1a1b26';X.font='8px system-ui';X.save();X.beginPath();X.rect(x1,GPU_Y,bw,TH);X.clip();X.fillText(n,x1+2,GPU_Y+12);X.restore();}}
document.getElementById('range-label').textContent=range>=1e6?(range/1e6).toFixed(2)+'s':(range/1e3).toFixed(1)+'ms';
}
C.addEventListener('wheel',ev=>{ev.preventDefault();const f=ev.deltaY>0?1.25:0.8,mx=ev.offsetX,t=x2t(mx);const nr=(vE-vS)*f,fr=(mx-LW)/(C.clientWidth-LW);vS=t-fr*nr;vE=vS+nr;draw();});
C.addEventListener('mousedown',ev=>{drag=true;dX=ev.clientX;dVS=vS;W.classList.add('dragging');});
addEventListener('mousemove',ev=>{if(drag){const dx=ev.clientX-dX,r=vE-vS;vS=dVS-dx/(C.clientWidth-LW)*r;vE=vS+r;draw();}const rc=C.getBoundingClientRect(),mx=ev.clientX-rc.left,my=ev.clientY-rc.top,t=x2t(mx);let hit=null;if(my>=GPU_Y&&my<GPU_Y+TH){for(const e of gpuHW)if(t>=e.start_us&&t<=e.start_us+e.dur_us){hit={name:e.name.split('/').pop(),scope:e.name,start_us:e.start_us,dur_us:e.dur_us};break;}}if(hit){function fmt(us){return us>=1e6?(us/1e6).toFixed(3)+' s':us>=1000?(us/1000).toFixed(2)+' ms':us.toFixed(1)+' us';}TT.innerHTML='<b>'+hit.name+'</b><br><span class="dim">'+hit.scope+'</span><br>GPU: '+fmt(hit.dur_us)+'<br><span class="dim">'+fmt(hit.start_us)+'</span>';TT.style.display='block';TT.style.left=(ev.clientX+12)+'px';TT.style.top=(ev.clientY-10)+'px';}else TT.style.display='none';});
addEventListener('mouseup',function(){drag=false;W.classList.remove('dragging');});
function resetZoom(){vS=gMin;vE=gMax;draw();}
function zoomIn(){const m=(vS+vE)/2,r=(vE-vS)/2;vS=m-r/2;vE=m+r/2;draw();}
function zoomOut(){const m=(vS+vE)/2,r=(vE-vS)*2;vS=m-r/2;vE=m+r/2;draw();}
const tb=document.querySelector('#gpu-tbl tbody');const data=S.gpu_hw||[];
if(data.length){tb.innerHTML=data.map(function(d){return'<tr><td>'+d.name+'</td><td class="r">'+(d.total_ms>=1?d.total_ms.toFixed(1):d.total_ms.toFixed(2))+' ms</td><td class="r">'+d.count+'x</td><td class="r">'+(d.avg_ms>=1?d.avg_ms.toFixed(2):d.avg_ms.toFixed(3))+' ms</td><td class="r">'+d.pct.toFixed(1)+'%</td><td><div class="bar" style="width:'+Math.max(d.pct,1)+'%;background:'+oc(d.name)+'"></div></td></tr>';}).join('');}
addEventListener('resize',resize);resize();
</script></body></html>)TMPL";

    // Replace placeholders
    replaceAll(html, "{{TIMESTAMP}}", timeBuf);
    replaceAll(html, "{{BACKEND}}", backendName);

    char buf[32];
    snprintf(buf, sizeof(buf), "%.1f", tokPerSec);
    replaceAll(html, "{{TPS}}", buf);
    snprintf(buf, sizeof(buf), "%d", (int)prefillMs);
    replaceAll(html, "{{PREFILL_MS}}", buf);
    snprintf(buf, sizeof(buf), "%d", nDecodeTokens);
    replaceAll(html, "{{N_TOKENS}}", buf);
    snprintf(buf, sizeof(buf), "%d", (int)totalGpuMs);
    replaceAll(html, "{{GPU_TOTAL_MS}}", buf);
    replaceAll(html, "{{GPU_HW_EVENTS}}", eventsToJson(gpuHwEvents));
    replaceAll(html, "{{STEPS}}", eventsToJson(steps));
    replaceAll(html, "{{SUMMARY}}", summaryJson.str());

    std::ofstream out(outputPath);
    if (!out) {
        fprintf(stderr, "Failed to write profile HTML to %s\n", outputPath.c_str());
        return;
    }
    out << html;
    out.close();
    printf("  Profile saved to %s\n", outputPath.c_str());
}
