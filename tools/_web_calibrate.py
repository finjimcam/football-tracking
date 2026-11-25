"""Browser front-end for calibrate_pitch.py (not a CLI tool).

Replaces the OpenCV click window when there is no desktop (e.g. running on
the pphub server, reached from a laptop or phone over Tailscale). Same flow
as the OpenCV version — step through the landmark menu, click / skip / undo,
finish with 4+ points — plus:
  - a mini pitch map showing WHERE the current landmark is;
  - once 4+ points are placed, the solved pitch lines are drawn back onto the
    frame (green), with the reprojection error, so a bad click is obvious;
  - placed points can be dragged to fine-tune them;
  - zoom (wheel / pinch / +-) and pan (drag) for precise clicks.

collect_points_web() blocks until every keyframe is submitted (or the user
aborts) and returns one points list per frame, in the same
{"landmark", "image", "pitch"} shape the OpenCV loop returns, so
calibrate_pitch.py's homography + JSON writing is shared by both front-ends.
"""
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np


def pitch_lines(L, W):
    """Standard pitch markings as polylines in pitch meters, densely sampled
    (~1 m) so projected lines can be clipped where they pass behind the
    camera's horizon."""
    def seg(a, b):
        n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1])) + 1)
        return [[a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t] for t in np.linspace(0, 1, n)]

    def poly(pts):
        out = []
        for a, b in zip(pts, pts[1:]):
            out.extend(seg(a, b)[:-1])
        out.append(list(pts[-1]))
        return out

    def rect(x0, y0, x1, y1):
        return poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])

    def circle(cx, cy, r, a0=0.0, a1=2 * np.pi, n=72):
        return [[cx + r * np.cos(a), cy + r * np.sin(a)] for a in np.linspace(a0, a1, n)]

    pw, gw, cr = 40.32 / 2, 18.32 / 2, 9.15
    c = W / 2
    arc = np.arccos(5.5 / cr)  # penalty arc: the part of the 9.15 m circle outside the box
    return [
        rect(0, 0, L, W),
        poly([(L / 2, 0), (L / 2, W)]),
        circle(L / 2, c, cr),
        rect(0, c - pw, 16.5, c + pw),
        rect(L - 16.5, c - pw, L, c + pw),
        rect(0, c - gw, 5.5, c + gw),
        rect(L - 5.5, c - gw, L, c + gw),
        circle(11, c, cr, -arc, arc, 24),
        circle(L - 11, c, cr, np.pi - arc, np.pi + arc, 24),
        circle(11, c, 0.25, n=8),
        circle(L - 11, c, 0.25, n=8),
        circle(L / 2, c, 0.25, n=8),
    ]


def project_lines(H_img_to_pitch, lines):
    """Project pitch-meter polylines into the image; split wherever a point
    lands behind the camera (w <= 0) so the overlay never wraps around."""
    Hinv = np.linalg.inv(np.asarray(H_img_to_pitch, dtype=np.float64))
    out = []
    for line in lines:
        p = np.asarray(line, dtype=np.float64)
        h = np.hstack([p, np.ones((len(p), 1))]) @ Hinv.T
        cur = []
        for (x, y, w) in h:
            if w <= 1e-9:
                if len(cur) > 1:
                    out.append(cur)
                cur = []
                continue
            cur.append([round(x / w, 1), round(y / w, 1)])
        if len(cur) > 1:
            out.append(cur)
    return out


def try_solve(points):
    """Non-fatal homography solve for live feedback. Returns (H, err) or (None, None)."""
    if len(points) < 4:
        return None, None
    src = np.array([p["image"] for p in points], dtype=np.float64)
    dst = np.array([p["pitch"] for p in points], dtype=np.float64)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC if len(points) > 4 else 0)
    if H is None:
        return None, None
    h = np.hstack([src, np.ones((len(src), 1))]) @ H.T
    err = np.linalg.norm(h[:, :2] / h[:, 2:3] - dst, axis=1)
    return H, err


def _urls(host, port):
    if host not in ("0.0.0.0", "::", ""):
        return [f"http://{host}:{port}/"]
    names = [socket.gethostname()]
    try:  # Tailscale / LAN address the default route uses
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("100.100.100.100", 80))
        names.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return [f"http://{n}:{port}/" for n in names]


def collect_points_web(frames, menu, pitch_length, pitch_width, host="0.0.0.0", port=8790,
                       title="calibrate_pitch", on_ready=None):
    """frames: list of (frame_idx, BGR image). Blocks; returns list of points lists."""
    jpegs = []
    for _, img in frames:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise SystemExit("Failed to JPEG-encode frame for the web UI")
        jpegs.append(buf.tobytes())

    state = {
        "title": title,
        "frames": [{"frame": int(f), "width": int(img.shape[1]), "height": int(img.shape[0])}
                   for f, img in frames],
        "menu": [{"name": n, "pitch": [float(x), float(y)]} for n, (x, y) in menu],
        "pitch": {"length": pitch_length, "width": pitch_width},
        "pitch_lines": [[[round(a, 2), round(b, 2)] for a, b in ln]
                        for ln in pitch_lines(pitch_length, pitch_width)],
        "current": 0,
        "done": False,
        "aborted": False,
    }
    results = [None] * len(frames)
    finished = threading.Event()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if path == "/api/state":
                return self._send(200, state)
            if path.startswith("/frame/") and path.endswith(".jpg"):
                try:
                    k = int(path[len("/frame/"):-4])
                    return self._send(200, jpegs[k], "image/jpeg")
                except (ValueError, IndexError):
                    pass
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            path = self.path.split("?")[0]
            try:
                data = self._json()
            except ValueError:
                return self._send(400, {"error": "bad json"})
            if path == "/api/solve":
                H, err = try_solve(data.get("points", []))
                if H is None:
                    return self._send(200, {"ok": False})
                return self._send(200, {"ok": True, "err_mean": float(err.mean()), "err_max": float(err.max()),
                                        "per_point": [round(float(e), 2) for e in err],
                                        "lines": project_lines(H, state["pitch_lines"])})
            if path == "/api/submit":
                with lock:
                    k = int(data.get("frame_index", -1))
                    pts = data.get("points", [])
                    if k != state["current"] or state["done"]:
                        return self._send(409, {"error": "stale frame"})
                    if len(pts) < 4:
                        return self._send(400, {"error": "need 4+ points"})
                    H, err = try_solve(pts)
                    if H is None:
                        return self._send(400, {"error": "homography failed: points may be collinear"})
                    results[k] = [{"landmark": p["landmark"],
                                   "image": [round(float(p["image"][0]), 1), round(float(p["image"][1]), 1)],
                                   "pitch": [float(p["pitch"][0]), float(p["pitch"][1])]} for p in pts]
                    print(f"  frame {state['frames'][k]['frame']}: {len(pts)} points from browser, "
                          f"mean error {err.mean():.2f} m", flush=True)
                    state["current"] = k + 1
                    if state["current"] >= len(frames):
                        state["done"] = True
                        finished.set()
                return self._send(200, {"ok": True, "done": state["done"], "current": state["current"]})
            if path == "/api/abort":
                state["aborted"] = True
                finished.set()
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "not found"})

    server = ThreadingHTTPServer((host, port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    urls = _urls(host, port)
    print("Open the calibration page in a browser (laptop or phone):", flush=True)
    for u in urls:
        print(f"  {u}", flush=True)
    if on_ready:
        on_ready(urls)
    try:
        finished.wait()
    except KeyboardInterrupt:
        state["aborted"] = True
    finally:
        threading.Timer(2.0, server.shutdown).start()  # let the page get its last response
    if state["aborted"]:
        raise SystemExit("Aborted — no calibration written")
    return results


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>Pitch Calibration</title>
<style>
:root{--bg:#111418;--panel:#1b2026;--fg:#e8eaed;--muted:#9aa3ad;--accent:#ffd54a;--ok:#4caf7a;--bad:#e5534b;--line:#2c333b}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:14px/1.35 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
body{display:flex;flex-direction:column;overflow:hidden}
header{display:flex;gap:12px;align-items:center;padding:8px 12px;background:var(--panel);border-bottom:1px solid var(--line);flex-wrap:wrap}
#mini{background:#1f5f3a;border-radius:4px;flex:none}
#banner{flex:1;min-width:180px}#banner .k{color:var(--muted);font-size:12px}#banner .n{font-size:17px;font-weight:600;color:var(--accent)}
#err{font-size:12px;color:var(--muted)}#err.good{color:var(--ok)}#err.bad{color:var(--bad)}
#stage{flex:1;position:relative;touch-action:none;cursor:crosshair}#cv{position:absolute;inset:0;width:100%;height:100%}
footer{display:flex;gap:6px;padding:8px 12px;background:var(--panel);border-top:1px solid var(--line);flex-wrap:wrap;align-items:center}
button{background:#2a3139;color:var(--fg);border:1px solid #3a434d;border-radius:6px;padding:9px 14px;font:inherit;cursor:pointer}
button:disabled{opacity:.4;cursor:default}button.primary{background:#2f6f4a;border-color:#3d8a5d}button.danger{background:#5a2626;border-color:#7a3333}
.sp{flex:1}.hint{color:var(--muted);font-size:12px}
#overlay{position:fixed;inset:0;background:rgba(0,0,0,.8);display:none;align-items:center;justify-content:center;font-size:20px;text-align:center;padding:20px}
@media (max-width:700px){.hint{display:none}}
kbd{background:#2a3139;border:1px solid #3a434d;border-radius:3px;padding:0 4px;font-size:11px}
</style></head><body>
<header>
  <canvas id="mini" width="150" height="98"></canvas>
  <div id="banner"><div class="k" id="kf"></div><div class="n" id="lm">Loading…</div><div id="err"></div></div>
</header>
<div id="stage"><canvas id="cv"></canvas></div>
<footer>
  <button id="bUndo" title="Backspace">Undo</button>
  <button id="bSkip" title="Space">Skip</button>
  <button id="bLines">Lines: on</button>
  <button id="bZin">+</button><button id="bZout">−</button><button id="bFit">Fit</button>
  <span class="hint" id="hint"><kbd>click</kbd> place · <kbd>drag</kbd> pan / move point · <kbd>wheel</kbd> zoom · <kbd>Space</kbd> skip · <kbd>⌫</kbd> undo · <kbd>Enter</kbd> finish</span>
  <span class="sp"></span>
  <button class="danger" id="bAbort">Abort</button>
  <button class="primary" id="bDone" disabled title="Enter">Finish frame</button>
</footer>
<div id="overlay"></div>
<script>
const $=id=>document.getElementById(id);
const cv=$('cv'),ctx=cv.getContext('2d'),mini=$('mini'),mctx=mini.getContext('2d');
let S=null,img=new Image(),k=0,idx=0,points=[],hist=[],lines=null,showLines=true,errInfo=null;
let view={s:1,ox:0,oy:0},dpr=1;

async function api(path,body){const r=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});return r.json().then(j=>({status:r.status,...j}));}

async function init(){
  S=await api('/api/state');
  if(S.done){return msg('This calibration is already finished. You can close this tab.');}
  loadFrame(S.current);
}
function loadFrame(i){
  k=i;idx=0;points=[];hist=[];lines=null;errInfo=null;
  img=new Image();img.onload=()=>{fit();render();};img.src='/frame/'+k+'.jpg?'+Date.now();
  update();
}
function resize(){dpr=window.devicePixelRatio||1;const r=cv.getBoundingClientRect();cv.width=r.width*dpr;cv.height=r.height*dpr;render();}
function fit(){const r=cv.getBoundingClientRect();const s=Math.min(r.width/img.width,r.height/img.height);view={s,ox:(r.width-img.width*s)/2,oy:(r.height-img.height*s)/2};}
const toImg=(x,y)=>[(x-view.ox)/view.s,(y-view.oy)/view.s];
const toScr=(x,y)=>[x*view.s+view.ox,y*view.s+view.oy];
function zoomAt(f,cx,cy){const[ix,iy]=toImg(cx,cy);view.s=Math.min(Math.max(view.s*f,0.05),40);view.ox=cx-ix*view.s;view.oy=cy-iy*view.s;render();}

function render(){
  if(!S)return;ctx.setTransform(dpr,0,0,dpr,0,0);const r=cv.getBoundingClientRect();
  ctx.fillStyle='#000';ctx.fillRect(0,0,r.width,r.height);
  if(img.complete&&img.width){ctx.imageSmoothingEnabled=view.s<2;ctx.drawImage(img,view.ox,view.oy,img.width*view.s,img.height*view.s);}
  if(lines&&showLines){ctx.strokeStyle='rgba(80,255,140,.85)';ctx.lineWidth=1.5;
    for(const ln of lines){ctx.beginPath();ln.forEach((p,i)=>{const[x,y]=toScr(p[0],p[1]);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}}
  ctx.font='12px system-ui,sans-serif';
  points.forEach((p,i)=>{const[x,y]=toScr(p.image[0],p.image[1]);
    const e=errInfo&&errInfo.per_point?errInfo.per_point[i]:null;
    ctx.fillStyle=e!=null&&e>2?'#ff9a3c':'#ff3b3b';ctx.strokeStyle='#fff';ctx.lineWidth=1.5;
    ctx.beginPath();ctx.arc(x,y,5,0,7);ctx.fill();ctx.stroke();
    ctx.beginPath();ctx.moveTo(x-11,y);ctx.lineTo(x-7,y);ctx.moveTo(x+7,y);ctx.lineTo(x+11,y);ctx.moveTo(x,y-11);ctx.lineTo(x,y-7);ctx.moveTo(x,y+7);ctx.lineTo(x,y+11);ctx.stroke();
    const t=short(p.landmark)+(e!=null?` (${e.toFixed(1)}m)`:'');ctx.lineWidth=3;ctx.strokeStyle='rgba(0,0,0,.8)';ctx.strokeText(t,x+9,y-8);ctx.fillStyle='#fff';ctx.fillText(t,x+9,y-8);});
  drawMini();
}
const short=n=>n.replace('Center circle: ','Circle ').replace(' intersection with halfway line','').replace('penalty area','pen').replace('goal area','goal').replace(' on goal line',' (goal line)');
function drawMini(){
  const L=S.pitch.length,W=S.pitch.width,pad=5,sc=Math.min((mini.width-2*pad)/L,(mini.height-2*pad)/W);
  const m=(x,y)=>[pad+x*sc,pad+y*sc];
  mctx.fillStyle='#1f5f3a';mctx.fillRect(0,0,mini.width,mini.height);mctx.strokeStyle='rgba(255,255,255,.75)';mctx.lineWidth=1;
  for(const ln of S.pitch_lines){mctx.beginPath();ln.forEach((p,i)=>{const[x,y]=m(p[0],p[1]);i?mctx.lineTo(x,y):mctx.moveTo(x,y);});mctx.stroke();}
  const done=new Set(points.map(p=>p.landmark));
  S.menu.forEach((l,i)=>{const[x,y]=m(l.pitch[0],l.pitch[1]);if(done.has(l.name)){mctx.fillStyle='#ff3b3b';mctx.beginPath();mctx.arc(x,y,2.5,0,7);mctx.fill();}});
  if(idx<S.menu.length){const[x,y]=m(...S.menu[idx].pitch);mctx.strokeStyle=mctx.fillStyle='#ffd54a';mctx.lineWidth=2;mctx.beginPath();mctx.arc(x,y,6,0,7);mctx.stroke();mctx.beginPath();mctx.arc(x,y,2,0,7);mctx.fill();}
}
function update(){
  const nf=S.frames.length,f=S.frames[k];
  $('kf').textContent=(nf>1?`Keyframe ${k+1}/${nf} · `:'')+`frame ${f.frame} · ${points.length} recorded`+(idx<S.menu.length?` · landmark ${idx+1}/${S.menu.length}`:'');
  $('lm').textContent=idx<S.menu.length?'Click: '+S.menu[idx].name:(points.length>=4?'All landmarks done — press Finish':`All landmarks done — need ${4-points.length} more (Undo to go back)`);
  $('bDone').disabled=points.length<4;$('bSkip').disabled=idx>=S.menu.length;$('bUndo').disabled=!hist.length;
  const e=$('err');if(errInfo&&errInfo.ok){e.textContent=`Fit: mean ${errInfo.err_mean.toFixed(2)} m · max ${errInfo.err_max.toFixed(2)} m`+(errInfo.err_mean>2?' — something is off, check the orange points':'');e.className=errInfo.err_mean>2?'bad':'good';}
  else{e.textContent=points.length<4?`Place ${4-points.length} more to see the fitted lines (spread them out — corners are best)`:'Points look collinear — add one off the line';e.className='';}
  render();
}
let solveT=null,solveSeq=0;
function resolve(){clearTimeout(solveT);if(points.length<4){lines=null;errInfo=null;update();return;}
  solveT=setTimeout(async()=>{const my=++solveSeq;const r=await api('/api/solve',{points});if(my!==solveSeq)return;errInfo=r;lines=r.ok?r.lines:null;update();},120);}

function place(ix,iy){if(idx>=S.menu.length)return;
  if(ix<0||iy<0||ix>img.width||iy>img.height)return;
  const l=S.menu[idx];points.push({landmark:l.name,image:[+ix.toFixed(1),+iy.toFixed(1)],pitch:l.pitch});hist.push(true);idx++;update();resolve();}
function skip(){if(idx<S.menu.length){hist.push(false);idx++;update();}}
function undo(){if(!hist.length)return;const rec=hist.pop();idx--;if(rec)points.pop();update();resolve();}
async function finish(){if(points.length<4)return;$('bDone').disabled=true;
  const r=await api('/api/submit',{frame_index:k,points});
  if(r.status!==200){alert(r.error||'Submit failed');update();return;}
  if(r.done)msg('Calibration saved ✓<br><small>You can close this tab.</small>');else loadFrame(r.current);}
async function abort(){if(!confirm('Abort calibration? Nothing will be saved.'))return;await api('/api/abort',{});msg('Aborted — nothing saved.');}
function msg(h){const o=$('overlay');o.innerHTML=h;o.style.display='flex';}

// pointer handling: tap = place, drag = pan (or move a point), pinch = zoom
const ptrs=new Map();let g=null;
cv.addEventListener('pointerdown',e=>{cv.setPointerCapture(e.pointerId);ptrs.set(e.pointerId,{x:e.offsetX,y:e.offsetY});
  if(ptrs.size===1){let hit=-1,best=14;points.forEach((p,i)=>{const[x,y]=toScr(...p.image);const d=Math.hypot(x-e.offsetX,y-e.offsetY);if(d<best){best=d;hit=i;}});
    g={sx:e.offsetX,sy:e.offsetY,lx:e.offsetX,ly:e.offsetY,moved:false,hit};}
  else if(ptrs.size===2){const[a,b]=[...ptrs.values()];g={pinch:true,d:Math.hypot(a.x-b.x,a.y-b.y),cx:(a.x+b.x)/2,cy:(a.y+b.y)/2};}});
cv.addEventListener('pointermove',e=>{if(!ptrs.has(e.pointerId)||!g)return;ptrs.set(e.pointerId,{x:e.offsetX,y:e.offsetY});
  if(g.pinch&&ptrs.size===2){const[a,b]=[...ptrs.values()];const d=Math.hypot(a.x-b.x,a.y-b.y),cx=(a.x+b.x)/2,cy=(a.y+b.y)/2;
    view.ox+=cx-g.cx;view.oy+=cy-g.cy;zoomAt(d/g.d,cx,cy);g.d=d;g.cx=cx;g.cy=cy;return;}
  if(g.pinch)return;
  if(!g.moved&&Math.hypot(e.offsetX-g.sx,e.offsetY-g.sy)>6)g.moved=true;
  if(g.moved){if(g.hit>=0){const[ix,iy]=toImg(e.offsetX,e.offsetY);points[g.hit].image=[+ix.toFixed(1),+iy.toFixed(1)];render();}
    else{view.ox+=e.offsetX-g.lx;view.oy+=e.offsetY-g.ly;render();}}
  g.lx=e.offsetX;g.ly=e.offsetY;});
function up(e){if(!ptrs.has(e.pointerId))return;ptrs.delete(e.pointerId);if(!g)return;
  if(g.pinch){if(!ptrs.size)g=null;return;}
  if(!g.moved&&g.hit<0){const[ix,iy]=toImg(e.offsetX,e.offsetY);place(ix,iy);}
  else if(g.moved&&g.hit>=0)resolve();
  g=null;}
cv.addEventListener('pointerup',up);cv.addEventListener('pointercancel',e=>{ptrs.delete(e.pointerId);g=null;});
cv.addEventListener('wheel',e=>{e.preventDefault();zoomAt(Math.exp(-e.deltaY*0.0015),e.offsetX,e.offsetY);},{passive:false});
window.addEventListener('keydown',e=>{if(!S||$('overlay').style.display==='flex')return;
  if(e.key===' '){e.preventDefault();skip();}else if(e.key==='Backspace'){e.preventDefault();undo();}
  else if(e.key==='Enter'){e.preventDefault();finish();}else if(e.key==='q'||e.key==='Q')abort();
  else if(e.key==='+'||e.key==='=')zoomAt(1.25,cv.clientWidth/2,cv.clientHeight/2);else if(e.key==='-')zoomAt(0.8,cv.clientWidth/2,cv.clientHeight/2);
  else if(e.key==='l'||e.key==='L'){showLines=!showLines;$('bLines').textContent='Lines: '+(showLines?'on':'off');render();}});
$('bUndo').onclick=undo;$('bSkip').onclick=skip;$('bDone').onclick=finish;$('bAbort').onclick=abort;
$('bZin').onclick=()=>zoomAt(1.4,cv.clientWidth/2,cv.clientHeight/2);$('bZout').onclick=()=>zoomAt(1/1.4,cv.clientWidth/2,cv.clientHeight/2);
$('bFit').onclick=()=>{fit();render();};
$('bLines').onclick=()=>{showLines=!showLines;$('bLines').textContent='Lines: '+(showLines?'on':'off');render();};
window.addEventListener('resize',resize);new ResizeObserver(resize).observe($('stage'));
init();
</script></body></html>
"""
