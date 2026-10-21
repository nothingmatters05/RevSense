from pathlib import Path
import io, base64, json, mimetypes, textwrap
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, windows
from rpm_estimator import analyse_recording, load_audio, estimate_window

ROOT=Path(__file__).resolve().parent
AUDIO=ROOT/'audio'; RESULTS=ROOT/'results'; ASSETS=ROOT/'assets'
for d in (AUDIO, RESULTS, ASSETS): d.mkdir(exist_ok=True)
st.set_page_config(page_title='RevSense | Acoustic RPM Lab', page_icon='🏁', layout='wide', initial_sidebar_state='collapsed')

DATA=[
 {'label':'1500–1750 RPM','file':'1500-1750.wav','kind':'steady','ref':1625,'note':'Low-speed steady recording'},
 {'label':'2250 RPM','file':'2250.wav','kind':'steady','ref':2250,'note':'Steady recording'},
 {'label':'3200 RPM','file':'3200.wav','kind':'steady','ref':3200,'note':'Steady recording'},
 {'label':'4000 RPM','file':'4000.wav','kind':'steady','ref':4000,'note':'Steady recording'},
 {'label':'5000 RPM','file':'5000.wav','kind':'steady','ref':5000,'note':'Steady recording'},
 {'label':'Revving run','file':'Revving.wav','kind':'revving','ref':None,'note':'Variable speed · idle to limiter'},
]
PREVIOUS={
 '1500-1750.wav':(1530,1500,1562,10,10),
 '2250.wav':(2120,2072,2148,7,7),
 '3200.wav':(3140,3052,3212,7,7),
 '4000.wav':(4080,4036,4080,10,10),
 '5000.wav':(5040,5028,5086,8,8),
}

st.markdown('''<style>
:root{color-scheme:dark}.stApp{background:linear-gradient(180deg,#0b1220,#101b2d 42%,#0b1220);color:#edf4ff}
[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none!important}
[data-testid="stHeader"]{background:transparent}
[data-testid="stToolbar"]{visibility:hidden!important;height:0!important}
#MainMenu{visibility:hidden!important}
footer{visibility:hidden!important}
.block-container{max-width:1250px;padding-top:1.25rem;padding-bottom:3rem}
.hero{border:1px solid #294b75;border-left:4px solid #55a6ff;border-radius:14px;padding:1.5rem 1.7rem;background:linear-gradient(110deg,#152a45,#101a2b 75%);margin-bottom:1rem}
.engine-photo{width:100%;height:620px;min-height:620px;max-height:620px;object-fit:cover;object-position:center 55%;display:block;border-radius:12px;border:1px solid #294b75}
.engine-card{min-height:620px;height:620px;border:1px solid #294b75;border-radius:12px;padding:1.15rem;background:linear-gradient(145deg,#172b45,#101a2b);display:flex;flex-direction:column;justify-content:space-between}
.spec-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:15px}
.spec-box{background:#0e1a2a;border:1px solid #2a4565;border-radius:10px;padding:12px;min-width:0}
.cycle-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:14px 0}
.cycle-step{border:1px solid #2a4565;background:#102037;border-radius:10px;padding:12px 8px;text-align:center;animation:cycleGlow 4s infinite}
.cycle-step:nth-child(2){animation-delay:1s}.cycle-step:nth-child(3){animation-delay:2s}.cycle-step:nth-child(4){animation-delay:3s}
.cycle-step b{display:block;color:#83c0ff;font-size:1rem}.cycle-step span{font-size:.75rem;color:#c4d5e9}
.engine-motion{display:grid;grid-template-columns:150px 1fr;gap:14px;align-items:center;margin:10px 0 12px;padding:12px;border:1px solid #294b75;border-radius:12px;background:radial-gradient(ellipse at 30% 50%,#1b3b5e,#0e1a2a 75%)}
.engine-motion svg{width:100%;max-width:150px;overflow:visible}.piston{animation:pistonTravel 2.4s linear infinite;transform-box:fill-box;transform-origin:center}.crank{animation:crankTurn 2.4s linear infinite;transform-origin:50px 105px}.spark{animation:sparkFlash 2.4s linear infinite}.motion-copy{font-size:.8rem;color:#c5d8ef;line-height:1.5}.motion-copy strong{color:#8bc5ff}.motion-track{height:5px;border-radius:99px;background:#203953;overflow:hidden;margin-top:8px}.motion-track span{display:block;width:25%;height:100%;background:linear-gradient(90deg,#4b9eff,#b9e1ff);animation:strokeProgress 2.4s linear infinite}
@keyframes pistonTravel{0%,100%{transform:translateY(-19px)}50%{transform:translateY(19px)}}@keyframes crankTurn{to{transform:rotate(360deg)}}@keyframes sparkFlash{0%,46%,100%{opacity:.2}50%,55%{opacity:1;filter:drop-shadow(0 0 5px #65b5ff)}}@keyframes strokeProgress{0%{transform:translateX(0)}100%{transform:translateX(400%)}}
@keyframes cycleGlow{0%,20%,100%{border-color:#2a4565;box-shadow:none}10%{border-color:#70b5ff;box-shadow:0 0 15px #438bd833}}
@media(max-width:760px){.frequency-panel{min-height:0}.cycle-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.engine-photo{height:300px;min-height:260px;max-height:300px}.engine-card{min-height:0;height:auto}.engine-motion{grid-template-columns:100px 1fr}.engine-motion svg{max-width:100px}}
.eyebrow{color:#79baff;font-weight:800;font-size:.7rem;letter-spacing:.15em;text-transform:uppercase}.hero h1{font-size:clamp(2.4rem,5vw,4rem);line-height:1;margin:.5rem 0 .7rem;font-weight:900;letter-spacing:-.05em;color:#fff;opacity:0;animation:heroReveal .95s cubic-bezier(.2,.8,.2,1) .12s forwards;text-shadow:0 0 0 rgba(100,180,255,0)}
.hero h1 span{animation:logoPulse 3s ease-in-out 1.1s infinite alternate}
.hero .eyebrow{animation:eyebrowReveal .8s ease-out both}
.hero p{animation:heroReveal .8s ease-out .35s both}
.hero .pill{animation:pillReveal .65s ease-out both}
.hero .pill:nth-of-type(2){animation-delay:.3s}.hero .pill:nth-of-type(3){animation-delay:.42s}.hero .pill:nth-of-type(4){animation-delay:.54s}
@keyframes heroReveal{from{opacity:0;transform:translateY(20px);filter:blur(5px)}to{opacity:1;transform:translateY(0);filter:blur(0)}}
@keyframes eyebrowReveal{from{opacity:0;letter-spacing:.35em}to{opacity:1;letter-spacing:.15em}}
@keyframes pillReveal{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
@keyframes logoPulse{0%,100%{text-shadow:0 0 5px rgba(70,155,255,.25),0 0 14px rgba(70,155,255,.18);filter:brightness(1)}20%{text-shadow:0 0 12px rgba(90,175,255,.7),0 0 30px rgba(70,155,255,.65),0 0 52px rgba(70,155,255,.3);filter:brightness(1.35)}45%{text-shadow:0 0 3px rgba(70,155,255,.2);filter:brightness(.92)}70%{text-shadow:0 0 16px rgba(90,175,255,.8),0 0 38px rgba(70,155,255,.5);filter:brightness(1.2)}}.hero p{color:#c5d5e9;max-width:760px;font-size:1rem;margin:0}.pill{display:inline-block;border:1px solid #365d88;border-radius:99px;padding:.25rem .6rem;margin:.7rem .25rem 0 0;color:#d8eaff;font-size:.72rem}
.sectionhead{border-bottom:1px solid #263d59;padding-bottom:.45rem;margin:1.3rem 0 .8rem}.sectionhead .eyebrow{font-size:.66rem}.sectionhead h2{margin:.2rem 0;font-size:1.45rem;color:#f4f8ff}.panel{border:1px solid #263b55;background:linear-gradient(145deg,#152238,#111b2b);border-radius:12px;padding:1rem 1.1rem;height:100%}.frequency-panel{min-height:700px;height:100%;display:flex;flex-direction:column;justify-content:space-between}.panel h3{margin:.1rem 0 .35rem;color:#f4f8ff}.muted{color:#b2c2d8;font-size:.86rem}.specnum{font-size:1.7rem;font-weight:850;color:#70b5ff;letter-spacing:-.03em}.specunit{font-size:.76rem;color:#b4c5da}.tag{font-size:.68rem;color:#7abaff;text-transform:uppercase;letter-spacing:.12em;font-weight:800}
div.stButton>button,div.stDownloadButton>button{border-radius:8px;border:1px solid #356ba4;background:#1c4f83;color:#fff;font-weight:750;min-height:2.45rem}div.stButton>button:hover,div.stDownloadButton>button:hover{background:#286db0;color:#fff;border-color:#77baff}div[data-testid="stMetricValue"]{color:#83c0ff}div[data-testid="stMetricLabel"]{color:#b9cbe0}
[data-testid="stExpander"]{border:1px solid #2c4564;border-radius:9px;background:#111d2e}.stDataFrame{border:1px solid #263b55;border-radius:8px}.smallnote{font-size:.78rem;color:#a7bbd3}[data-testid="stHorizontalBlock"]{gap:.6rem}
</style>''',unsafe_allow_html=True)

def heading(num,title,sub=''):
    st.markdown(f'<div class="sectionhead"><div class="eyebrow">{num} / REVSENSE LAB</div><h2>{title}</h2><div class="muted">{sub}</div></div>',unsafe_allow_html=True)

def styled_axes(fig,ax):
    fig.patch.set_facecolor('#111216'); ax.set_facecolor('#111216'); ax.tick_params(colors='#c8c8ce',labelsize=8)
    for s in ax.spines.values(): s.set_color('#44444b')
    ax.grid(alpha=.18,color='#aaaab0'); ax.xaxis.label.set_color('#d0d0d5'); ax.yaxis.label.set_color('#d0d0d5')

def make_wave(y,sr,offset=0):
    y=np.asarray(y); step=max(1,len(y)//5000); yy=y[::step]; tt=offset+np.arange(len(yy))*step/sr
    fig,ax=plt.subplots(figsize=(10,2.7)); styled_axes(fig,ax); ax.plot(tt,yy,color='#ef3d4c',lw=.55); ax.set_xlabel('Time (s)');ax.set_ylabel('Amplitude');fig.tight_layout();return fig

def make_spectrum(y,sr,max_hz=1200,f0=None):
    n=len(y); x=(y-np.mean(y))*windows.hann(n,sym=False); mag=np.abs(np.fft.rfft(x)); f=np.fft.rfftfreq(n,1/sr); keep=(f>0)&(f<=min(max_hz,sr/2)); f,mag=f[keep],mag[keep]
    db=20*np.log10(mag/(mag.max()+1e-12)+1e-8) if len(mag) else mag
    fig,ax=plt.subplots(figsize=(10,2.7));styled_axes(fig,ax);ax.plot(f,db,color='#5baeff',lw=.8)
    if len(mag) and mag.max()>0:
        ix,_=find_peaks(mag,prominence=mag.max()*.025); ix=sorted(ix,key=lambda i:mag[i],reverse=True)[:12]
        if ix: ax.scatter(f[ix],db[ix],s=17,color='#dbeaff',zorder=3,label='Spectral peaks')
    if f0 is not None and np.isfinite(f0):
        for h in range(1,9):
            if h*f0<=max_hz: ax.axvline(h*f0,color='#82bdff',ls='--',lw=.8,alpha=max(.12,.55/h))
        ax.legend(facecolor='#191a1e',edgecolor='#44444b',labelcolor='white',fontsize=8)
    ax.set_xlabel('Frequency (Hz)');ax.set_ylabel('Relative magnitude (dB)');fig.tight_layout();return fig

def gauge_html(value, label='MEASURED RPM', actual=None, revving=False, audio_path=None, video_path=None, trace=None, loop=True):
    # A self-contained HTML player lets media events and the needle share one browser clock.
    def data_uri(path):
        if not path or not path.exists(): return ''
        size=path.stat().st_size
        if size > 42*1024*1024: return ''
        mime=mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        return f'data:{mime};base64,'+base64.b64encode(path.read_bytes()).decode('ascii')
    aud=data_uri(audio_path); vid=data_uri(video_path)
    trace_json=json.dumps(trace or [])
    max_rpm=12000 if revving else 6500
    actual_html=(f'<div class="metric"><span>ACTUAL RPM · MEASURED</span><b>{actual:,.0f}<small> RPM</small></b></div>' if actual is not None else '')
    if not aud:
        return f'''<div style="background:#101114;color:#ddd;border:1px solid #333;border-radius:14px;padding:24px;font-family:Arial">Audio file is not embedded. Add the WAV to the audio/ folder. Gauge estimate: <b>{value or 0:,.0f} RPM</b>.</div>'''
    return f'''<!doctype html><html><head><meta charset="utf-8"><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#0d1727;color:#fff;font-family:Arial,sans-serif}}.wrap{{border:1px solid #294b75;border-radius:14px;background:radial-gradient(ellipse at 50% 0%,#172f4e,#101a2a 68%);padding:14px}}.playgrid{{display:grid;grid-template-columns:minmax(0,{'1fr' if revving else '1.6fr'}) minmax(0,1fr);gap:12px;align-items:stretch}}.gauges{{display:grid;grid-template-columns:repeat({2 if actual is not None else 1},minmax(0,1fr));gap:12px}}.gcard{{background:#122035;border:1px solid #2b4564;border-radius:12px;padding:10px;min-width:0}}canvas{{width:100%;height:auto;display:block;max-height:430px}}.metric span{{font-size:10px;letter-spacing:1.7px;color:#82bdff;font-weight:bold}}.metric b{{display:block;font-size:clamp(25px,4vw,38px);letter-spacing:-1px;margin-top:4px}}.metric small{{font-size:12px;color:#aaa}}.controls{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:12px}}button{{background:#1e5d99;color:#fff;border:1px solid #75baff;border-radius:9px;padding:11px 18px;font-weight:800;cursor:pointer}}button:disabled{{opacity:.45;cursor:not-allowed}}.state{{font-size:12px;color:#bcbcc4}}video{{width:100%;height:auto;max-height:390px;aspect-ratio:16/9;object-fit:contain;background:#050505;border-radius:10px;margin:0;display:block}}.videobox{{border:1px solid #2b4564;border-radius:12px;padding:10px;background:#122035;min-width:0}}.videotitle{{font-size:10px;letter-spacing:1.7px;color:#82bdff;font-weight:bold;margin-bottom:8px}}@media(max-width:760px){{.playgrid{{grid-template-columns:1fr}}.gauges{{grid-template-columns:repeat({2 if actual is not None else 1},minmax(0,1fr))}}}}.note{{font-size:11px;color:#8f9098;margin-top:9px}}@media(max-width:560px){{.gauges{{grid-template-columns:1fr}}}}
    </style></head><body><div class="wrap"><div class="playgrid"><div><div class="gauges"><div class="gcard"><div class="metric"><span>{label}</span><b id="rpm">0<small> RPM</small></b></div><canvas id="g" width="480" height="300"></canvas></div>{('<div class="gcard">'+actual_html+'<canvas id="ga" width="480" height="300"></canvas></div>') if actual is not None else ''}</div><div class="controls"><button id="play">▶ PLAY RUN</button><button id="stop" style="background:#23364d;border-color:#45617f">■ STOP</button><span id="state" class="state">Ready · click play</span></div><audio id="a" preload="auto" src="{aud}"></audio><div class="note">The needle follows the audio timeline. The second dial shows your measured RPM for steady recordings.</div></div>{('<div class="videobox"><div class="videotitle">REFERENCE VIDEO · MUTED</div><video id="v" muted preload="auto" playsinline src="'+vid+'"></video></div>') if vid else '<div class="videobox"><div class="videotitle">REFERENCE VIDEO</div><div class="note">No matching video file was found for this recording.</div></div>'}</div>
<script>
const audio=document.getElementById('a'),video=document.getElementById('v'),play=document.getElementById('play'),stop=document.getElementById('stop'),state=document.getElementById('state'),rpmText=document.getElementById('rpm');
const maxRpm={max_rpm},target={float(value or 0)},trace={trace_json},isRev={str(revving).lower()},shouldLoop={str(loop).lower()};let running=false,raf=null,cycleStart=0,zeroStart=0,previous=0;
function draw(canvasId,rpm){{const c=document.getElementById(canvasId);if(!c)return;const ctx=c.getContext('2d'),w=c.width,h=c.height,cx=w/2,cy=h*.82,r=Math.min(w*.43,h*.72);ctx.clearRect(0,0,w,h);const start=Math.PI,end=2*Math.PI;ctx.lineWidth=17;ctx.lineCap='butt';ctx.strokeStyle='#36373d';ctx.beginPath();ctx.arc(cx,cy,r,start,end);ctx.stroke();const frac=Math.max(0,Math.min(1,rpm/maxRpm));ctx.strokeStyle='#4ba3ff';ctx.beginPath();ctx.arc(cx,cy,r,start,start+Math.PI*frac);ctx.stroke();const redStart=.84;ctx.strokeStyle='#ff626e';ctx.lineWidth=20;ctx.beginPath();ctx.arc(cx,cy,r,start+Math.PI*redStart,end);ctx.stroke();for(let i=0;i<=maxRpm/500;i++){{const f=(i*500)/maxRpm,ang=start+Math.PI*f,inner=r-(i%2===0?22:13),outer=r+2;ctx.strokeStyle=f>=redStart?'#ff6c75':'#dedee2';ctx.lineWidth=i%2===0?3:1.4;ctx.beginPath();ctx.moveTo(cx+Math.cos(ang)*inner,cy+Math.sin(ang)*inner);ctx.lineTo(cx+Math.cos(ang)*outer,cy+Math.sin(ang)*outer);ctx.stroke();if(i%2===0){{ctx.fillStyle='#eaf4ff';ctx.font='bold 12px Arial';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(String(Math.round(maxRpm*f/1000)),cx+Math.cos(ang)*(r-39),cy+Math.sin(ang)*(r-39));}}}}const ang=start+Math.PI*frac;ctx.strokeStyle='#fff';ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(cx+Math.cos(ang)*(r-34),cy+Math.sin(ang)*(r-34));ctx.stroke();ctx.fillStyle='#eaf4ff';ctx.beginPath();ctx.arc(cx,cy,9,0,Math.PI*2);ctx.fill();ctx.fillStyle='#aaaab1';ctx.font='12px Arial';ctx.textAlign='center';ctx.fillText('×1000 RPM',cx,cy+27);}}
function showRpm(v){{v=Math.max(0,Math.min(maxRpm,v||0));rpmText.innerHTML=Math.round(v).toLocaleString()+'<small> RPM</small>';draw('g',v);if(document.getElementById('ga'))draw('ga',{actual or 0});}}
function interp(t){{if(!trace.length)return target*(Math.min(1,t/1.2));if(t<=trace[0][0])return trace[0][1];for(let i=1;i<trace.length;i++){{if(t<=trace[i][0]){{let [t0,v0]=trace[i-1],[t1,v1]=trace[i];let f=(t-t0)/Math.max(.001,t1-t0);return v0+(v1-v0)*f;}}}}return trace[trace.length-1][1];}}
function frame(now){{if(!running)return;const t=audio.currentTime;if(isRev)showRpm(interp(t));else{{const duration=audio.duration||5;const ramp=Math.min(1,t/Math.min(1.25,duration*.25));showRpm(target*ramp);}}raf=requestAnimationFrame(frame);}}
function stopCycle(replay){{running=false;if(raf)cancelAnimationFrame(raf);raf=null;audio.pause();try{{audio.currentTime=0}}catch(e){{}}if(video){{video.pause();try{{video.currentTime=0}}catch(e){{}}}}showRpm(0);play.disabled=false;state.textContent='RPM returned to zero';if(replay&&shouldLoop){{setTimeout(()=>{{if(shouldLoop)startCycle()}},1000)}}}}
async function startCycle(){{if(running)return;running=true;play.disabled=true;state.textContent='Playing · tracking audio';showRpm(0);try{{audio.currentTime=0;
  // Start audio immediately inside the user's click handler. Waiting for video metadata first
  // can consume the browser's transient user-activation permission and block audio.play().
  const audioStart=audio.play();
  await audioStart;
  if(video){{video.muted=true;video.playbackRate=1;try{{video.pause();video.currentTime=0;}}catch(e){{}}
    // Video is optional: never let a video sync/permission failure stop audio playback.
    try{{const vp=video.play();if(vp)vp.catch(()=>{{state.textContent='Audio playing · video playback blocked by browser';}});}}catch(e){{state.textContent='Audio playing · video playback unavailable';}}
  }}
}}catch(e){{running=false;play.disabled=false;state.textContent='Playback failed — check the recording and press PLAY again';return}}
if(raf)cancelAnimationFrame(raf);raf=requestAnimationFrame(frame);}}audio.addEventListener('ended',()=>stopCycle(true));play.addEventListener('click',()=>startCycle());stop.addEventListener('click',()=>stopCycle(false));draw('g',0);draw('ga',0);
</script></body></html>'''

# Load known previous summaries as a transparent fallback when audio files are not present.
def known_results():
    rows=[]
    for d in DATA:
        if d['file'] in PREVIOUS:
            m,p10,p90,accepted,total=PREVIOUS[d['file']]
            rows.append({'Dataset':d['label'],'File':d['file'],'Actual RPM (measured)':d['ref'],'Median estimate (RPM)':m,'P10 (RPM)':p10,'P90 (RPM)':p90,'Accepted windows':f'{accepted}/{total}','Status':'Previously analysed'})
        else: rows.append({'Dataset':d['label'],'File':d['file'],'Actual RPM (measured)':'—','Median estimate (RPM)':'—','P10 (RPM)':'—','P90 (RPM)':'—','Accepted windows':'—','Status':'Waiting for WAV'})
    return pd.DataFrame(rows)

st.markdown('''<div class="hero"><div class="eyebrow">DIGITAL SIGNAL PROCESSING · MOTORCYCLE ACOUSTICS</div><h1>REVSENSE<span style="color:#ed3342">.</span></h1><p>Hear the rev. See the RPM. RevSense turns engine sound into an RPM estimate using digital signal processing.</p><span class="pill">FFT + HARMONIC COMB</span><span class="pill">WINDOWED RPM TRACKING</span><span class="pill">AUDIO / VIDEO DEMO</span></div>''',unsafe_allow_html=True)

heading('01','Test bike','Yamaha MT-15 · the engine behind our recordings.')
vehicle_photo=ASSETS/'mt15.jpg'
left,right=st.columns([1.05,1],gap='large')
with left:
    if vehicle_photo.exists():
        photo_bytes=vehicle_photo.read_bytes()
        photo_mime=mimetypes.guess_type(vehicle_photo.name)[0] or 'image/jpeg'
        photo_uri=f'data:{photo_mime};base64,'+base64.b64encode(photo_bytes).decode('ascii')
        st.markdown(f'<img class="engine-photo" src="{photo_uri}" alt="Yamaha MT-15 test motorcycle">',unsafe_allow_html=True)
    else:
        st.markdown('''<div class="engine-photo" style="min-height:320px;display:flex;flex-direction:column;justify-content:center;align-items:center;background:radial-gradient(ellipse at 50% 45%,#24476c,#101a2b 72%);height:100%;box-sizing:border-box"><div class="tag">TEST MOTORCYCLE</div><div style="font-size:clamp(2.5rem,5vw,4.4rem);font-weight:950;line-height:1.05;margin:.65rem 0;text-align:center">YAMAHA<br><span style="color:#82bdff">MT-15</span></div><svg viewBox="0 0 650 220" style="width:100%;max-height:185px" xmlns="http://www.w3.org/2000/svg"><circle cx="140" cy="160" r="52" fill="none" stroke="#a9bfd8" stroke-width="8"/><circle cx="140" cy="160" r="31" fill="none" stroke="#45617f" stroke-width="4"/><circle cx="490" cy="160" r="52" fill="none" stroke="#a9bfd8" stroke-width="8"/><circle cx="490" cy="160" r="31" fill="none" stroke="#45617f" stroke-width="4"/><path d="M140 160 L235 95 L330 160 L490 160 L405 82 L330 160 L265 160 L220 122 L178 128" fill="none" stroke="#63b0ff" stroke-width="13" stroke-linejoin="round"/><path d="M235 95 L290 65 L365 65 L405 82 L452 105 L490 160" fill="none" stroke="#eaf4ff" stroke-width="8" stroke-linecap="round"/><path d="M290 65 L260 42 L335 38 L365 65" fill="#17283d" stroke="#63b0ff" stroke-width="4"/><path d="M402 78 L435 53 L462 57" fill="none" stroke="#eee" stroke-width="7" stroke-linecap="round"/><path d="M215 120 L280 117 L310 150 L258 161 Z" fill="#2878ba"/><rect x="319" y="130" width="48" height="35" rx="6" fill="#7893b2" stroke="#eaf4ff" stroke-width="3"/><path d="M365 65 L400 35 L437 38" fill="none" stroke="#a9bfd8" stroke-width="5"/></svg><div class="muted">Test platform · engine sound recordings</div></div>''',unsafe_allow_html=True)
with right:
    st.markdown('''<div class="engine-card"><div><div class="tag">ENGINE PROFILE</div><h3 style="font-size:1.45rem;margin:.4rem 0">155 cc · Liquid-cooled</h3><p class="muted">Fuel-injected, single-cylinder, 4-stroke SOHC engine with 4 valves. These recordings capture its sound at steady speeds and during acceleration.</p><div class="spec-grid"><div class="spec-box"><div class="specnum">18.4 PS</div><div class="specunit">13.5 kW · 10,000 rpm</div></div><div class="spec-box"><div class="specnum">14.1 Nm</div><div class="specunit">Peak torque · 7,500 rpm</div></div><div class="spec-box"><div class="specnum">4 valves</div><div class="specunit">2 intake + 2 exhaust</div></div><div class="spec-box"><div class="specnum">6-speed</div><div class="specunit">Constant-mesh gearbox</div></div><div class="spec-box"><div class="specnum">Liquid</div><div class="specunit">Cooling system</div></div><div class="spec-box"><div class="specnum">FI</div><div class="specunit">Fuel injection</div></div><div class="spec-box"><div class="specnum">58 × 58.7 mm</div><div class="specunit">Bore × stroke</div></div><div class="spec-box"><div class="specnum">11.6:1</div><div class="specunit">Compression ratio</div></div></div></div><div class="smallnote" style="margin-top:14px">Test data: five steady-speed recordings and one revving recording.</div></div>''',unsafe_allow_html=True)

heading('02','From firing frequency to RPM','A four-stroke cycle takes two crankshaft revolutions, with one firing event per cycle.')
ca,cb=st.columns([1,1],gap='large')
with ca:
    st.markdown('''<div class="panel frequency-panel"><div class="tag">THE FOUR-STROKE CYCLE</div><h3>Watch one cycle · 720° crank rotation</h3><div class="engine-motion"><svg viewBox="0 0 110 145" role="img" aria-label="Animated piston and crankshaft"><rect x="22" y="9" width="56" height="99" rx="7" fill="#152d47" stroke="#79baff" stroke-width="2"/><rect x="28" y="15" width="44" height="10" rx="3" fill="#5caaff"/><rect class="piston" x="29" y="40" width="42" height="20" rx="3" fill="#d6eaff" stroke="#72baff" stroke-width="2"/><path d="M50 60 L50 105" stroke="#9ab7d5" stroke-width="5" stroke-linecap="round"/><circle cx="50" cy="105" r="12" fill="#203e5f" stroke="#8fc6ff" stroke-width="3"/><g class="crank"><path d="M50 105 L68 105" stroke="#d6eaff" stroke-width="4" stroke-linecap="round"/><circle cx="68" cy="105" r="5" fill="#68b4ff"/></g><path class="spark" d="M39 29 L34 35 L40 35 L35 42 L47 32 L41 32 Z" fill="#78c4ff"/><path d="M12 18 L12 88 M8 23 L12 15 L16 23 M8 83 L12 91 L16 83" stroke="#72baff" stroke-width="1.5" fill="none"/><text x="8" y="105" fill="#9fcfff" font-size="6" transform="rotate(-90 8 105)">PISTON TRAVEL</text></svg><div class="motion-copy"><strong>Intake → Compression → Power → Exhaust</strong><br>The piston moves up and down while the crankshaft rotates. The spark flashes at the power stroke, driving the piston down. The animation is illustrative, not a real-time engine simulation.<div class="motion-track"><span></span></div></div></div><div class="cycle-grid"><div class="cycle-step"><b>01</b><span>Intake · piston down</span></div><div class="cycle-step"><b>02</b><span>Compression · piston up</span></div><div class="cycle-step"><b>03</b><span>Power · spark fires</span></div><div class="cycle-step"><b>04</b><span>Exhaust · piston up</span></div></div><div class="muted">One full four-stroke cycle takes two crankshaft revolutions. The spark occurs once per cycle in this simplified single-cylinder model.</div><div style="margin-top:14px;border-radius:10px;background:#0e1a2a;padding:12px;text-align:center"><span class="tag">KEY RELATIONSHIP</span><div style="font-size:1.5rem;font-weight:900;color:#82bdff;margin-top:4px">1 firing event / 2 revolutions</div></div></div>''',unsafe_allow_html=True)
with cb:
    st.markdown(r'''<div class="panel frequency-panel"><div><div class="tag">CONVERT FREQUENCY TO ENGINE SPEED</div><h3>From events per second to revolutions per minute</h3><p class="muted">Let <i>f₀</i> be the firing-event frequency measured in hertz (events per second).</p></div><div style="text-align:center;border:1px solid #294b75;background:#0e1a2a;border-radius:12px;padding:22px 12px;margin:14px 0"><div class="tag">STEP 1 · FIRING FREQUENCY</div><div style="font-size:clamp(1.2rem,2.2vw,1.8rem);font-weight:800;color:#d8eaff;margin:14px 0">f₀ = RPM / (2 × 60)</div><div class="muted">One firing event every two crankshaft revolutions; 60 converts minutes to seconds.</div></div><div style="text-align:center;color:#bcdcff;font-weight:800;letter-spacing:.08em">REARRANGE</div><div style="text-align:center;border:1px solid #3d6fa4;background:linear-gradient(135deg,#15375a,#0e1a2a);border-radius:12px;padding:22px 12px;margin:14px 0"><div class="tag">STEP 2 · ENGINE SPEED</div><div style="font-size:clamp(1.7rem,3vw,2.4rem);font-weight:950;color:#82bdff;margin:12px 0">RPM = 120 × f₀</div><div class="muted">Multiply the estimated firing frequency by 120 to obtain revolutions per minute.</div></div><div style="border-radius:10px;background:#0e1a2a;padding:16px;margin-top:auto"><div class="tag">QUICK EXAMPLE</div><div class="specnum">33.3 Hz × 120 ≈ 4,000 RPM</div><div class="muted">The harmonic-comb estimator finds the likely firing frequency from the spectrum. This equation converts it to engine speed.</div></div></div>''',unsafe_allow_html=True)

heading('03','Choose one of the six recordings','Select a recording below to update the player, waveform, spectrum and analysis.')
cols=st.columns(6)
if 'dataset_idx' not in st.session_state: st.session_state.dataset_idx=0
for i,d in enumerate(DATA):
    with cols[i]:
        label=('🏁 '+d['label'])
        if st.button(label,key=f'dataset_{i}',use_container_width=True,type='primary' if st.session_state.dataset_idx==i else 'secondary'):
            st.session_state.dataset_idx=i
selected=DATA[st.session_state.dataset_idx]; wav=AUDIO/selected['file']; stem=Path(selected['file']).stem
video=next((AUDIO/(stem+ext) for ext in ('.mp4','.mov','.webm','.m4v') if (AUDIO/(stem+ext)).exists()),None)
st.markdown(f'''<div class="panel"><div class="tag">SELECTED DATASET</div><h3>{selected['label']}</h3><div class="muted">{selected['note']} · File: <code>{selected['file']}</code></div></div>''',unsafe_allow_html=True)

if not wav.exists():
    st.warning(f'`audio/{selected["file"]}` is not in this package yet. Put your WAV files into the `audio/` folder to enable playback and live analysis. Previously calculated steady-recording results are shown below until the WAV is available.')
    audio=None; y=None; sr=None; duration=0; detail=None; summary=None
else:
    y,sr=load_audio(wav)
    audio=y
    duration=len(y)/sr
    if selected['kind']=='revving':
        win_sec=.50; hop_sec=.15; min_rpm=900; max_rpm=12000
    else:
        win_sec=1.5; hop_sec=.5; min_rpm=900; max_rpm=6500
    cache_key=f'{wav.name}-{wav.stat().st_mtime_ns}-{win_sec}-{hop_sec}-{max_rpm}'
    @st.cache_data(show_spinner='Analysing overlapping audio windows…')
    def run_analysis(pathstr, mtime, window_sec, hop_sec, max_rpm):
        return analyse_recording(pathstr,window_sec=window_sec,hop_sec=hop_sec,min_rpm=900,max_rpm=max_rpm)
    detail,summary=run_analysis(str(wav),wav.stat().st_mtime_ns,win_sec,hop_sec,max_rpm)

heading('04','Audio, video & RPM dashboard','Play the selected recording to hear the engine and follow the estimated RPM.')
if audio is not None:
    median=float(summary['median_rpm']) if pd.notna(summary['median_rpm']) else 0.0
    if selected['kind']=='revving':
        usable=detail[pd.notna(detail['rpm'])].copy()
        # Smooth isolated estimator jitter lightly, while preserving the overall rev-up/rev-down shape.
        vals=usable['rpm'].rolling(3,center=True,min_periods=1).median().clip(0,12000).to_numpy()
        trace=[[round(float(t),3),round(float(v),1)] for t,v in zip(usable['start_s'],vals)]
        gauge_value=median; actual=None
    else:
        trace=[]; gauge_value=median; actual=selected['ref']
    html=gauge_html(gauge_value,'ESTIMATED RPM',actual=actual,revving=selected['kind']=='revving',audio_path=wav,video_path=video,trace=trace,loop=True)
    components.html(html,height=(540 if selected['kind']=='revving' else 470) if video else (500 if selected['kind']=='revving' else 430),scrolling=False)
    if video is None: st.caption(f'No matching video found. Add `audio/{stem}.mp4` (or .mov/.webm/.m4v) to show the video in sync with this audio.')
    if selected['kind']=='revving':
        with st.expander('How revving RPM is tracked'):
            st.write('The audio is analysed in overlapping 0.50-second windows, with a new estimate every 0.15 seconds. Each estimate is plotted at its time position. The needle returns to zero after playback, waits one second, and repeats.')
    else:
        with st.expander('How to read the two gauges'):
            st.write('Estimated RPM is calculated from the audio. Actual RPM is the measured reference value for this steady recording.')
else:
    st.markdown('''<div class="panel"><h3>Playback area ready</h3><p class="muted">Once the WAV is added, this panel shows an automotive tachometer, a play/stop control, audio playback and the matching video if present.</p></div>''',unsafe_allow_html=True)

heading('05','Signal views','Waveform shows sound amplitude over time; spectrum shows its frequency content.')
if audio is not None:
    m1,m2,m3,m4=st.columns(4);m1.metric('Duration',f'{duration:.2f} s');m2.metric('Sample rate',f'{sr:,} Hz');m3.metric('Samples',f'{len(y):,}');m4.metric('Analysis window',f'{win_sec:.2f} s')
    wcol,scol=st.columns(2)
    with wcol:
        st.markdown('**Time-domain waveform**');fig=make_wave(y,sr);st.pyplot(fig,use_container_width=True);plt.close(fig)
    with scol:
        st.markdown('**Full-recording spectrum**');fig=make_spectrum(y,sr,1200);st.pyplot(fig,use_container_width=True);plt.close(fig)
else:
    st.dataframe(known_results(),use_container_width=True,hide_index=True)

heading('06','DSP workflow','The same harmonic-comb estimator is used for all recordings; the revving run adds faster tracking and continuity / octave checks. Select a step to expand its explanation.')
steps=[
 ('01','Load & prepare','Decode WAV, convert stereo to mono if needed, remove DC offset.','Audio samples x[n]'),
 ('02','Segment in time','Steady: 1.5 s windows / 0.5 s hop. Revving: 0.50 s windows / 0.15 s hop.','Local, overlapping frames'),
 ('03','Hann window','Taper each frame to reduce spectral leakage at the frame edges.','xw[n] = x[n] w[n]'),
 ('04','FFT','Convert the time-domain frame into frequency bins.','Δf = fs / N = 1 / T'),
 ('05','Harmonic comb','Test candidate f₀ and score energy near f₀, 2f₀, 3f₀…; do not simply pick the tallest peak.','Best adjusted harmonic score'),
 ('06','RPM conversion','For single-cylinder, four-stroke firing frequency f₀, multiply by 120.','RPM = 120 f₀'),
 ('07','Tracking & quality checks','Steady: summarize stable windows. Revving: correct likely octave errors, check continuity and track RPM over time.','Median + P10/P90, or RPM trace'),
]
stepcols=st.columns(2)
step_details=[
 'The WAV is loaded at its original sample rate. Stereo channels are combined to mono when needed, and the mean is removed so a DC offset does not dominate the spectrum.',
 'The signal is split into overlapping frames. Steady recordings use 1.5-second frames with 0.5-second hops; the revving run uses 0.50-second frames with 0.15-second hops to follow changing speed.',
 'The Hann window smoothly tapers the beginning and end of each frame. This reduces spectral leakage, which otherwise spreads energy into nearby FFT bins.',
 'The FFT converts each frame from samples versus time into magnitude versus frequency. Frequency-bin spacing is approximately 1 divided by the frame duration.',
 'The estimator tests possible firing frequencies and scores energy near their expected multiples: f₀, 2f₀, 3f₀ and so on. A candidate supported by several harmonics is preferred over simply taking the tallest peak.',
 'For a single-cylinder four-stroke engine, one firing event occurs per two crankshaft revolutions: f₀ = RPM / 120. Rearranging gives RPM = 120 × f₀.',
 'Steady recordings are summarized with median and P10/P90 across accepted windows. For the revving run, the same harmonic candidate estimator runs on shorter, more-overlapping frames; a half-frequency / octave ambiguity check and continuity flags help prevent implausible jumps. The resulting sequence is tracked over time rather than treated as one fixed RPM. These checks reduce, but cannot eliminate, errors caused by noise or engine harmonics.'
]
for row_start in range(0, len(steps), 2):
    rowcols=st.columns(2, gap='large')
    for col_idx, (n,t,d,formula) in enumerate(steps[row_start:row_start+2]):
        with rowcols[col_idx]:
            with st.expander(f'{n} · {t}', expanded=False):
                st.markdown(d)
                st.markdown(f'**Output:** {formula}')
                st.markdown(step_details[row_start+col_idx])
                if n=='04': st.latex(r'\Delta f=\frac{f_s}{N}=\frac{1}{T}')
                elif n=='06': st.latex(r'\boxed{\mathrm{RPM}=120f_0}')
with st.expander('Extra: why windowing and FFT matter'):
    st.markdown('A Hann window reduces spectral leakage. A longer frame gives finer frequency resolution, while shorter overlapping frames respond faster to changing RPM.')
    st.latex(r'X[k]=\sum_{n=0}^{N-1}x[n]w[n]e^{-j2\pi kn/N}')

heading('07','Inspect one analysis window','Choose a frame, inspect its waveform and spectrum first, then review the numeric results and spectral peaks.')
if audio is not None and len(detail):
    labs=[f'{r.start_s:.2f}–{r.end_s:.2f} s · {r.rpm:.0f} RPM · {r.quality}' for r in detail.itertuples()]
    idx=st.selectbox('Analysis window',range(len(labs)),format_func=lambda i:labs[i],key=f'win_{selected["file"]}')
    row=detail.iloc[idx]
    start=float(row.start_s); end=float(row.end_s)
    seg=y[int(start*sr):min(len(y),int(end*sr))]
    f0_value=row.get('f0_hz', row.get('corrected_f0_hz', np.nan))
    f0=float(f0_value) if pd.notna(f0_value) else np.nan
    rpm_value=row.get('rpm', np.nan)
    rpmw=float(rpm_value) if pd.notna(rpm_value) else np.nan

    st.markdown(f'<div class="panel"><div class="tag">SELECTED WINDOW</div><h3>{start:.2f}–{end:.2f} seconds</h3><div class="muted">Window {idx+1} of {len(detail)} · {selected["label"]}</div></div>',unsafe_allow_html=True)
    graph1,graph2=st.columns(2,gap='large')
    with graph1:
        st.markdown('**Window waveform**')
        fig=make_wave(seg,sr,start); st.pyplot(fig,use_container_width=True); plt.close(fig)
    with graph2:
        st.markdown('**Window spectrum**')
        fig=make_spectrum(seg,sr,1200,f0); st.pyplot(fig,use_container_width=True); plt.close(fig)

    st.markdown('**Window measurements**')
    q1,q2,q3,q4=st.columns(4)
    q1.metric('Firing frequency f₀',f'{f0:.2f} Hz' if np.isfinite(f0) else 'N/A')
    q2.metric('Window RPM',f'{rpmw:,.0f}' if np.isfinite(rpmw) else 'N/A')
    q3.metric('Harmonics matched',int(row.matched_harmonics) if pd.notna(row.matched_harmonics) else 'N/A')
    q4.metric('Harmonic score',f'{row.score:.3f}' if pd.notna(row.score) else 'N/A')
    st.markdown('**Detected spectral peaks**')
    peakrows=[]
    for item in str(row.peaks_hz_rel).split('; '):
        if ':' in item:
            f,a0=item.split(':',1)
            try: peakrows.append({'Peak (Hz)':float(f),'Relative magnitude':float(a0)})
            except ValueError: pass
    if peakrows:
        st.dataframe(pd.DataFrame(peakrows),use_container_width=True,hide_index=True)
    else:
        st.info('No peak list was recorded for this frame.')
    st.markdown(f"**Quality flag:** `{row.quality}`. PASS means the frame passed the estimator's implemented quality checks.")
    st.markdown('**RPM conversion for this window**')
    if np.isfinite(f0):
        st.latex(rf'\mathrm{{RPM}}=120\times {f0:.2f}\;\mathrm{{Hz}}\approx {rpmw:.0f}\;\mathrm{{RPM}}')
else:
    st.markdown('Add a WAV file to enable per-window analysis and spectral candidate inspection.')

heading('08','Results','Estimated RPM, measured RPM and variation across accepted windows.')
if audio is not None:
    accepted=detail[detail['quality'].astype(str).str.startswith('PASS')]
    med=float(summary['median_rpm']) if pd.notna(summary['median_rpm']) else np.nan
    p10=float(summary['p10_rpm']) if pd.notna(summary['p10_rpm']) else np.nan
    p90=float(summary['p90_rpm']) if pd.notna(summary['p90_rpm']) else np.nan
    k1,k2,k3,k4=st.columns(4);k1.metric('Median estimate',f'{med:,.0f} RPM' if np.isfinite(med) else 'N/A');k2.metric('P10',f'{p10:,.0f} RPM' if np.isfinite(p10) else 'N/A');k3.metric('P90',f'{p90:,.0f} RPM' if np.isfinite(p90) else 'N/A');k4.metric('Accepted windows',f"{summary['accepted_windows']}/{summary['total_windows']}")
    st.markdown('**RPM estimate by time window**')
    fig,ax=plt.subplots(figsize=(10,2.8));styled_axes(fig,ax);ax.plot(detail['start_s'],detail['rpm'],marker='o',ms=3,lw=1,color='#ef3d4c');ax.set_xlabel('Window start time (s)');ax.set_ylabel('Estimated RPM');ax.set_ylim(bottom=0);fig.tight_layout();st.pyplot(fig,use_container_width=True);plt.close(fig)
    st.markdown('**All window-level outputs**');st.dataframe(detail,use_container_width=True,hide_index=True)
else:
    st.dataframe(known_results(),use_container_width=True,hide_index=True)
    st.caption('These are prior steady-recording results from the project context. They are included for transparency; they will be recalculated from the WAVs when you add the audio files.')

heading('09','Conclusion','A concise summary of the selected recording and the estimate.')
if audio is not None:
    med=float(summary['median_rpm']) if pd.notna(summary['median_rpm']) else np.nan
    p10=float(summary['p10_rpm']) if pd.notna(summary['p10_rpm']) else np.nan; p90=float(summary['p90_rpm']) if pd.notna(summary['p90_rpm']) else np.nan
    accepted=detail[detail['quality'].astype(str).str.startswith('PASS')]
    chosen=detail[detail['quality'].astype(str).str.startswith('PASS')]
    f0med=float(chosen['f0_hz'].median()) if len(chosen) and chosen['f0_hz'].notna().any() else np.nan
    st.markdown(f"""<div class="panel"><div class="tag">AUTOMATED SUMMARY · CURRENT RECORDING</div><h3>{selected['label']}</h3><ul class="muted"><li><b>Frames analysed:</b> {summary['total_windows']} overlapping windows; {summary['accepted_windows']} passed the implemented checks ({summary['accepted_percent']:.1f}%).</li><li><b>Estimated RPM:</b> median {med:,.0f} RPM, with P10–P90 spread {p10:,.0f}–{p90:,.0f} RPM.</li><li><b>Firing frequency:</b> accepted-window median approximately {f0med:.2f} Hz. For a single-cylinder four-stroke engine, RPM = 120 × f₀.</li><li><b>Why this estimate:</b> candidate firing frequency is supported by energy at expected harmonics, not just the strongest peak. Median and percentile spread reduce the influence of isolated outliers.</li><li><b>Limitations:</b> PASS means a frame was not flagged by these checks, not that it is guaranteed correct. Revving estimates can still be affected by noise, exhaust resonances, limited window resolution or octave ambiguity.</li></ul></div>""",unsafe_allow_html=True)
else:
    st.markdown('''<div class="panel"><div class="tag">PROJECT CONCLUSION · PRIOR STEADY DATA</div><h3>Acoustic RPM estimation from six recordings</h3><ul class="muted"><li>Earlier steady-recording estimates: 1,530; 2,120; 3,140; 4,080; and 5,040 RPM.</li><li>Measured RPM is the reference for the five steady recordings; P10–P90 shows variation across accepted windows.</li><li>The revving run is tracked over time to show changing engine speed rather than a single fixed RPM.</li></ul></div>''',unsafe_allow_html=True)

st.markdown('''<div style="text-align:center;border-top:1px solid #343439;padding-top:20px;margin-top:40px;color:#777780;font-size:.76rem;letter-spacing:.14em">REVSENSE · ACOUSTIC ENGINE SPEED ESTIMATION</div>''',unsafe_allow_html=True)
