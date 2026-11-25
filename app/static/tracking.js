// Tracking tab (issue #42): every player labelled on the video and on a
// top-down pitch, a lane per player showing where tracking holds, the problems
// worth checking, and hand fixes (swap, assign, draw a box, split, delete, not
// a player). Fixes are saved server-side to config/<id>_track_edits.json and
// every tool that reads tracks applies them (tools/track_edits.py).

const SVGNS = 'http://www.w3.org/2000/svg';
const CHUNK = 250;
const LANE_COL = { s: '#5DBB63', h: '#FFD84D', p: '#E8A33A', l: '#7A2E2E', o: '#2A2F36' };
const STATUS = { s: 'Tracked', h: 'Your box', p: 'Predicted', l: 'Lost', o: 'Off camera' };
const PROB_COL = { swap: '#FFD84D', lost: '#EF5B5B' };
const SEL = '#FFD84D';
const LABEL_W = 116, RULER_H = 24, ROW_H = 13, HEAD_H = 18;
const WINDOWS = [[60, '1 min'], [120, '2 min'], [300, '5 min'], [900, '15 min']];
const KEYS_HELP = [
  ['Click a box', 'select that player (or say who an unlabelled box is)'],
  ['b / Draw box', 'drag a box round a player the tracker missed'],
  ['k', 'play / pause'], [', / .', '−/+ 1 frame'], ['a / d', '−/+ 2 s'],
  ['n / p', 'next / previous open problem'], ['Esc', 'cancel drawing / deselect'],
];

function svg(tag, attrs) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null) el.setAttribute(k, v);
  return el;
}
function lum(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex || '');
  if (!m) return 0.5;
  const n = parseInt(m[1], 16);
  const c = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; });
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}

export async function viewTracking(view, parts, params, ui) {
  const { h, api, toast, renderTabs, setCrumb, pill, onLeave } = ui;
  renderTabs('tracking');
  const matchId = parts[1];

  if (!matchId) {
    setCrumb(null);
    const data = await api('/matches');
    const ok = data.matches.filter((m) => m.pipeline === 'broadcast');
    view.replaceChildren(h('section', { class: 'panel' },
      h('div', { class: 'panel-head' }, h('h1', {}, 'Tracking'), h('span', { class: 'muted' }, 'Pick a match')),
      ok.length ? h('div', {}, ok.map((m) => h('a', { class: 'match-row', href: '#/tracking/' + encodeURIComponent(m.match_id) },
        h('div', { class: 'match-title' }, `${m.team_a_name} v ${m.team_b_name}`),
        m.artifacts.player_map.exists ? pill('ready', 'ready') : pill('blocked', 'needs tracking'))))
        : h('div', { class: 'empty' }, 'No broadcast matches yet. Run a match through "Track identities" first.')));
    return;
  }

  const mid = encodeURIComponent(matchId);
  let match;
  try { match = await api('/matches/' + mid); } catch (e) {
    view.replaceChildren(h('div', { class: 'placeholder' }, h('h1', {}, 'Match not found'), h('p', {}, e.message)));
    return;
  }
  setCrumb(match);
  view.replaceChildren(h('div', { class: 'placeholder' }, h('h1', {}, 'Loading tracking…'),
    h('p', {}, 'The first load reads the whole half\'s tracks (a few seconds).')));
  let sum;
  try { sum = await api(`/matches/${mid}/tracking`); } catch (e) {
    view.replaceChildren(h('div', { class: 'placeholder' }, h('h1', {}, 'Tracking not ready'), h('p', {}, e.message),
      h('a', { class: 'btn', href: '#/match/' + mid }, 'Open the pipeline')));
    return;
  }

  const W = sum.width, H = sum.height, FPS = sum.fps;
  const [PL, PW] = sum.pitch;
  const IMG_W = Math.min(W, 1280);
  const byKey = {}, keyOfTid = {};
  for (const p of sum.players) { byKey[p.key] = p; for (const t of p.ids) keyOfTid[t] = p.key; }
  const color = {};
  for (const t of ['team_a', 'team_b']) {
    const c = sum.teams[t].color;
    color[t] = lum(c) < 0.12 ? (t === 'team_a' ? '#4C8DF6' : '#F08C2E') : c;
  }
  const pColor = (p) => (p ? color[p.team] || '#B4BAC4' : '#B4BAC4');
  const ink = (c) => (lum(c) > 0.35 ? '#0E1013' : '#FFFFFF');
  const label = (p) => (p ? (p.number != null ? `#${p.number} ${p.name}` : p.name) : '?');
  const who = (tid) => (keyOfTid[tid] ? label(byKey[keyOfTid[tid]]) : `box ${tid}`);
  const clamp = (f) => Math.max(sum.start, Math.min(sum.end, Math.round(f)));

  const firstOpen = sum.problems.find((p) => p.fixed == null);
  const st = {
    cur: clamp(+params.get('f') || (firstOpen ? firstOpen.frame : sum.start)),
    sel: params.get('sel') && byKey[params.get('sel')] ? params.get('sel') : (firstOpen && firstOpen.keys[0]) || null,
    playing: false, tool: 'select', drawFor: null, drawProblem: null, pending: null, drag: null,
    showBoxes: true, showNames: true, showTrails: false, showOthers: true,
    winLen: 120, win: null, lanes: null, filter: 'open', busy: false,
  };

  // ---------- layout ----------
  const img = h('img', { alt: 'Video frame', class: 'lines-img', draggable: 'false' });
  const overlay = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'lines-svg trk-svg', role: 'img',
    'aria-label': 'Video frame with every tracked player boxed and labelled' });
  const banner = h('div', { class: 'trk-banner', role: 'status', hidden: true });
  const frameChip = h('div', { class: 'trk-chip mono' });
  const clockChip = h('div', { class: 'trk-clock mono' });
  const toggleBtns = {};
  const toggles = h('div', { class: 'trk-toggles', role: 'group', 'aria-label': 'Overlays' },
    [['showBoxes', 'Boxes'], ['showNames', 'Names'], ['showTrails', 'All trails'], ['showOthers', 'Unlabelled']].map(([k, t]) => {
      toggleBtns[k] = h('button', { type: 'button', class: 'trk-toggle', 'aria-pressed': String(st[k]),
        onclick: () => { st[k] = !st[k]; toggleBtns[k].setAttribute('aria-pressed', String(st[k])); draw(); } }, t);
      return toggleBtns[k];
    }));
  const stage = h('div', { class: 'lines-stage', style: `aspect-ratio:${W}/${H}` }, img, overlay, toggles, frameChip, clockChip);

  const btn = (label_, aria, fn, cls) => h('button', { type: 'button', class: 'btn small ' + (cls || ''), 'aria-label': aria, onclick: fn }, label_);
  const playBtn = btn('▶', 'Play (k)', () => togglePlay());
  const selectBtn = btn('Select', 'Select tool', () => setTool('select'), 'primary');
  const drawBtn = btn('Draw box', 'Draw a box round a missed player (b)', () => setTool(st.tool === 'draw' ? 'select' : 'draw'));
  selectBtn.setAttribute('aria-pressed', 'true');
  drawBtn.setAttribute('aria-pressed', 'false');
  const msg = h('div', { class: 'note', role: 'status', 'aria-live': 'polite', style: 'min-height:18px' });
  const transport = h('div', { class: 'lines-tools' },
    btn('−2s', 'Back 2 seconds (a)', () => step(-2 * FPS)), btn('−1f', 'Previous frame (,)', () => step(-1)), playBtn,
    btn('+1f', 'Next frame (.)', () => step(1)), btn('+2s', 'Forward 2 seconds (d)', () => step(2 * FPS)),
    btn('‹ Problem', 'Previous open problem (p)', () => jumpProblem(-1)), btn('Problem ›', 'Next open problem (n)', () => jumpProblem(1)),
    h('div', { class: 'spacer' }), h('div', { class: 'lines-tools', role: 'group', 'aria-label': 'Tool' }, selectBtn, drawBtn));

  const winSel = h('select', { class: 'mini-select', 'aria-label': 'Timeline window', onchange: () => {
    st.winLen = +winSel.value; st.win = null; draw();
  } }, WINDOWS.map(([s, t]) => h('option', { value: s, selected: s === st.winLen || null }, t)));
  const winLabel = h('span', { class: 'muted mono', style: 'font-size:12px' });
  const lanes = h('canvas', { class: 'trk-lanes', role: 'img', 'aria-label': 'Track lanes: click a lane to seek, a name to select' });
  const legend = h('div', { class: 'legend' }, Object.entries(LANE_COL).map(([k, c]) =>
    h('span', {}, h('i', { style: `background:${c}` }), STATUS[k])),
  h('span', {}, h('i', { style: `background:${PROB_COL.swap};transform:rotate(45deg);width:8px` }), 'Possible swap'),
  h('span', {}, h('i', { style: `background:${PROB_COL.lost};transform:rotate(45deg);width:8px` }), 'Long loss'));

  const left = h('div', { class: 'col-main' },
    h('section', { class: 'panel' },
      h('div', { class: 'panel-body', style: 'display:flex;flex-direction:column;gap:10px' }, stage, banner, transport, msg)),
    h('section', { class: 'panel' },
      h('div', { class: 'panel-head' }, h('h2', {}, 'Tracks'), winLabel, h('div', { class: 'spacer' }), winSel),
      h('div', { class: 'panel-body', style: 'display:flex;flex-direction:column;gap:8px' }, lanes, legend)));

  // top-down
  const pitch = svg('svg', { viewBox: `-3 -3 ${PL + 6} ${PW + 6}`, class: 'trk-pitch', role: 'img',
    'aria-label': 'Top-down pitch: every player, the ball and the camera\'s view' });
  pitch.append(svg('rect', { x: -3, y: -3, width: PL + 6, height: PW + 6, fill: '#173B24' }));
  const lines = svg('g', { fill: 'none', stroke: '#E9F2EA', 'stroke-opacity': 0.65, 'stroke-width': 0.35 });
  const pa = 16.5, pw2 = 20.16, ga = 5.5, gw = 9.16;
  lines.append(svg('rect', { x: 0, y: 0, width: PL, height: PW }), svg('path', { d: `M${PL / 2} 0V${PW}` }),
    svg('circle', { cx: PL / 2, cy: PW / 2, r: 9.15 }),
    svg('path', { d: `M0 ${PW / 2 - pw2}H${pa}V${PW / 2 + pw2}H0M${PL} ${PW / 2 - pw2}H${PL - pa}V${PW / 2 + pw2}H${PL}` }),
    svg('path', { d: `M0 ${PW / 2 - gw}H${ga}V${PW / 2 + gw}H0M${PL} ${PW / 2 - gw}H${PL - ga}V${PW / 2 + gw}H${PL}` }),
    svg('path', { d: `M0 ${PW / 2 - 3.66}H-1.6V${PW / 2 + 3.66}H0M${PL} ${PW / 2 - 3.66}H${PL + 1.6}V${PW / 2 + 3.66}H${PL}` }));
  const clip = svg('clipPath', { id: 'trk-clip' });
  clip.append(svg('rect', { x: -3, y: -3, width: PL + 6, height: PW + 6 }));
  const camLayer = svg('g', { 'clip-path': 'url(#trk-clip)' });
  const dyn = svg('g', {});
  pitch.append(clip, camLayer, lines, dyn);
  const topClock = h('span', { class: 'muted mono', style: 'font-size:11px' });
  const topPanel = h('section', { class: 'panel' },
    h('div', { class: 'panel-head' }, h('h2', {}, 'Top-down'), h('span', { class: 'muted' }, 'from the pitch calibration'),
      h('div', { class: 'spacer' }), topClock),
    h('div', { class: 'panel-body' }, pitch,
      h('div', { class: 'legend', style: 'margin-top:8px' },
        h('span', {}, h('i', { style: `background:${color.team_a};border-radius:50%;width:10px;height:10px` }), sum.teams.team_a.name),
        h('span', {}, h('i', { style: `background:${color.team_b};border-radius:50%;width:10px;height:10px` }), sum.teams.team_b.name),
        h('span', {}, h('i', { style: 'border:1.5px dashed #E8A33A;border-radius:50%;width:10px;height:10px' }), 'Predicted'),
        h('span', {}, h('i', { style: 'border:1.5px dashed #EF5B5B;border-radius:50%;width:10px;height:10px' }), 'Lost (last seen)'),
        h('span', {}, h('i', { style: 'background:#9AA1AB;opacity:.4;border-radius:50%;width:10px;height:10px' }), 'Off camera'),
        h('span', {}, h('i', { style: 'border:1px dashed #fff;opacity:.6;width:12px;height:8px' }), 'Camera view'))));

  const pendingPanel = h('section', { class: 'panel trk-pending', hidden: true });
  const selPanel = h('section', { class: 'panel' });
  const probFilter = h('select', { class: 'mini-select', 'aria-label': 'Which problems', onchange: () => { st.filter = probFilter.value; drawProblems(); } },
    [['open', 'Open'], ['swap', 'Open swaps'], ['lost', 'Open long losses'], ['fixed', 'Fixed / checked']].map(([v, t]) => h('option', { value: v }, t)));
  const probCount = h('span', { class: 'muted', style: 'font-size:12px' });
  const probList = h('div', { class: 'scroll trk-list' });
  const probPanel = h('section', { class: 'panel' },
    h('div', { class: 'panel-head' }, h('h2', {}, 'Problems'), probCount, h('div', { class: 'spacer' }), probFilter), probList);
  const editCount = h('span', { class: 'muted', style: 'font-size:12px' });
  const editList = h('div', { class: 'scroll trk-list' });
  const staleNote = h('div', { class: 'note', style: 'padding:8px 14px', hidden: true });
  const editPanel = h('section', { class: 'panel' },
    h('div', { class: 'panel-head' }, h('h2', {}, 'Your fixes'), editCount), staleNote, editList,
    h('div', { class: 'muted', style: 'padding:10px 14px;font-size:11px;line-height:1.5' },
      `Saved to config/${sum.edits_file}, beside the tracking (never over it). The Lines tab uses them straight away; `,
      'possession, positions and physical show as "re-run" on the ', h('a', { href: '#/match/' + mid }, 'match page'), '.'));
  const keysHelp = h('details', { class: 'panel', style: 'padding:10px 14px' }, h('summary', {}, 'Keys'),
    h('dl', { class: 'kv', style: 'margin-top:8px' }, KEYS_HELP.flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, v)])));

  const right = h('div', { class: 'col-side' }, topPanel, pendingPanel, selPanel, probPanel, editPanel, keysHelp);
  view.replaceChildren(h('div', { class: 'cols' }, left, right));

  // ---------- data ----------
  const chunks = new Map();
  const chunkOf = (f) => Math.floor((f - sum.start) / CHUNK);
  function loadChunk(c) {
    if (c < 0 || sum.start + c * CHUNK > sum.end) return null;
    if (chunks.has(c)) return chunks.get(c);
    const s = sum.start + c * CHUNK, e = Math.min(sum.end, s + CHUNK - 1);
    const p = api(`/matches/${mid}/tracking/frames?start=${s}&end=${e}`)
      .then((d) => { p.data = d; return d; })
      .catch((err) => { chunks.delete(c); msg.textContent = err.message; });
    chunks.set(c, p);
    while (chunks.size > 24) chunks.delete(chunks.keys().next().value);
    return p;
  }
  function rec(f) {
    const p = chunks.get(chunkOf(f));
    return p && p.data ? p.data.frames[f - p.data.start] : null;
  }
  function need(f) {
    const p = loadChunk(chunkOf(f));
    if (p && !p.data) p.then(() => { if (Math.abs(chunkOf(st.cur) - chunkOf(f)) <= 1) draw(); });
  }
  let lanesReq = 0;
  async function loadLanes() {
    const [a, b] = st.win;
    const bins = Math.max(60, Math.round((lanes.clientWidth || 800) - LABEL_W - 8));
    const id = ++lanesReq;
    try {
      const d = await api(`/matches/${mid}/tracking/lanes?start=${a}&end=${b}&bins=${bins}`);
      if (id === lanesReq) { st.lanes = d; drawLanes(); }
    } catch (e) { msg.textContent = e.message; }
  }
  function ensureWindow() {
    const len = Math.round(st.winLen * FPS);
    if (st.win && st.cur >= st.win[0] && st.cur <= st.win[1] && st.win[1] - st.win[0] + 1 === Math.min(len, sum.end - sum.start + 1)) return;
    let a = st.cur - Math.round(len * 0.25);
    a = Math.max(sum.start, Math.min(a, sum.end - len + 1));
    st.win = [a, Math.min(sum.end, a + len - 1)];
    st.lanes = null;
    loadLanes();
  }
  async function refresh() {
    sum = await api(`/matches/${mid}/tracking`);
    chunks.clear();
    st.lanes = null;
    loadLanes();
    draw();
  }
  async function addEdit(edit, okText) {
    if (st.busy) return;
    if (edit.problem && !edit.note) {
      const p = sum.problems.find((q) => q.key === edit.problem);
      if (p) edit.note = p.text;
    }
    st.busy = true;
    msg.textContent = 'Saving…';
    try {
      await api(`/matches/${mid}/tracking/edits`, { method: 'POST', body: { edit } });
      st.pending = null;
      msg.textContent = okText || 'Saved';
      await refresh();
    } catch (e) { msg.textContent = ''; toast(e.message, true); } finally { st.busy = false; }
  }
  async function undo(e) {
    try {
      await api(`/matches/${mid}/tracking/edits/${e.id}`, { method: 'DELETE' });
      msg.textContent = 'Undone: ' + describe(e);
      await refresh();
    } catch (err) { toast(err.message, true); }
  }

  // ---------- frame + overlay ----------
  let wantFrame = null, loading = false;
  function loadFrame(f) {
    wantFrame = f;
    if (loading) return;
    loading = true;
    const pre = new Image();
    pre.onload = () => {
      img.src = pre.src;
      loading = false;
      if (wantFrame !== f) loadFrame(wantFrame);
      else if (st.playing) setTimeout(playTick, 0);
    };
    pre.onerror = () => { loading = false; };
    pre.src = `/api/matches/${mid}/frame/${f}.jpg?w=${IMG_W}`;
  }

  function trail(key, f, secs, useBox) {
    const pts = [];
    for (let g = f - Math.round(secs * FPS); g <= f; g += 2) {
      const r = rec(g);
      if (!r) continue;
      const p = r.p.find((q) => q[0] === key);
      if (!p || p[2] === 'l' || p[2] === 'o') continue;
      pts.push(useBox ? [(p[3] + p[5]) / 2, p[6]] : [p[7], p[8]]);
    }
    return pts;
  }

  function currentProblem() {
    const f = st.cur, pad = 1.5 * FPS;
    let best = null;
    for (const p of sum.problems) {
      if (p.fixed != null) continue;
      const a = p.kind === 'swap' ? p.frame - pad : p.frame, b = p.kind === 'swap' ? p.frame + pad : p.end;
      if (f >= a && f <= b && (!best || (p.kind === 'swap' && best.kind !== 'swap'))) best = p;
    }
    return best;
  }

  function draw() {
    ensureWindow();
    need(st.cur);
    if (st.playing) need(st.cur + CHUNK);
    if (st.showTrails || st.sel) need(st.cur - 3 * FPS);
    loadFrame(st.cur);
    const r = rec(st.cur);
    const clock = r && r.c ? r.c : '';
    clockChip.textContent = clock || `frame ${st.cur}`;
    topClock.textContent = clock;
    const tracked = r ? r.p.filter((p) => p[2] === 's' || p[2] === 'h' || p[2] === 'p').length : 0;
    frameChip.textContent = r ? `frame ${st.cur} · ${tracked}/${sum.players.length} tracked` : `frame ${st.cur} · loading…`;

    overlay.replaceChildren();
    overlay.classList.toggle('drawing', st.tool === 'draw');
    if (r && st.showOthers && st.showBoxes) {
      for (const o of r.o) {
        if (o[2] == null) continue;
        const isPend = st.pending && st.pending.other === o[0];
        overlay.append(svg('rect', { x: o[2], y: o[3], width: o[4] - o[2], height: o[5] - o[3], class: 'trk-other' + (o[1] === 'r' ? ' ref' : ''),
          style: isPend ? `stroke:${SEL};stroke-width:3;stroke-opacity:1` : null }));
      }
    }
    // trails
    const trailKeys = st.showTrails ? sum.players.map((p) => p.key) : (st.sel ? [st.sel] : []);
    for (const k of trailKeys) {
      const pts = trail(k, st.cur, 3, true);
      if (pts.length > 1) {
        overlay.append(svg('polyline', { points: pts.map((p) => p.join(',')).join(' '), fill: 'none',
          stroke: k === st.sel ? SEL : pColor(byKey[k]), 'stroke-width': k === st.sel ? 3 : 2, 'stroke-opacity': k === st.sel ? 0.95 : 0.5,
          'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
      }
    }
    const prob = currentProblem();
    const warn = new Set(prob && prob.kind === 'swap' ? prob.keys : []);
    if (r && st.showBoxes) {
      const order = r.p.slice().sort((a, b) => (a[0] === st.sel) - (b[0] === st.sel));
      for (const p of order) {
        if (p[3] == null) continue;
        const pl = byKey[p[0]], isSel = p[0] === st.sel;
        const c = isSel ? SEL : p[2] === 'h' ? SEL : p[2] === 'p' ? '#E8A33A' : pColor(pl);
        overlay.append(svg('rect', { x: p[3], y: p[4], width: p[5] - p[3], height: p[6] - p[4], fill: 'none', stroke: c,
          'stroke-width': isSel ? 3 : 2, 'stroke-dasharray': p[2] === 's' ? null : '5 4', rx: 2 }));
        if (st.showNames || isSel || warn.has(p[0])) {
          let t = st.showNames || isSel ? label(pl) : `#${pl.number}`;
          if (p[2] === 'p') t += ' ·?';
          const cx = (p[3] + p[5]) / 2;
          const txt = svg('text', { x: cx, y: p[4] - 6, class: 'trk-label', fill: c });
          txt.textContent = t;
          overlay.append(txt);
        }
        if (warn.has(p[0])) {
          const wt = svg('text', { x: (p[3] + p[5]) / 2, y: p[6] + 16, class: 'trk-label', fill: SEL });
          wt.textContent = 'swap?';
          overlay.append(wt);
        }
      }
    }
    if (st.pending && st.pending.box) {
      const b = st.pending.box;
      overlay.append(svg('rect', { x: b.x1, y: b.y1, width: b.x2 - b.x1, height: b.y2 - b.y1, fill: 'rgba(255,216,77,0.12)',
        stroke: SEL, 'stroke-width': 2.5, 'stroke-dasharray': '6 4' }));
    }
    if (st.drag) {
      const d = st.drag;
      overlay.append(svg('rect', { x: Math.min(d.x0, d.x1), y: Math.min(d.y0, d.y1), width: Math.abs(d.x1 - d.x0), height: Math.abs(d.y1 - d.y0),
        fill: 'rgba(255,216,77,0.12)', stroke: SEL, 'stroke-width': 2.5, 'stroke-dasharray': '6 4' }));
    }
    if (r && r.b && r.b[0] != null) overlay.append(svg('circle', { cx: r.b[0], cy: r.b[1], r: 5, fill: 'none', stroke: '#FFFFFF', 'stroke-width': 2 }));

    drawBanner(prob);
    drawTop(r);
    drawSel(r);
    drawPending(r);
    drawLanes();
    drawProblems();
    drawEdits();
  }

  function drawBanner(prob) {
    if (!prob || st.pending || st.tool === 'draw') { banner.hidden = true; return; }
    banner.hidden = false;
    banner.style.borderColor = PROB_COL[prob.kind];
    const acts = [];
    if (prob.kind === 'swap') {
      acts.push(btn('Swap them from here', 'Swap these two players from this frame on', () => addEdit(
        { kind: 'swap', frame: st.cur, a: prob.ids[0], b: prob.ids[1], problem: prob.key }, 'Swapped from here'), 'primary'));
      acts.push(btn('Not a swap', 'Mark this as checked', () => addEdit({ kind: 'ok', frame: st.cur, problem: prob.key }, 'Marked as checked')));
    } else {
      const pl = byKey[prob.keys[0]];
      const b = btn(`Draw box for #${pl ? pl.number : ''}`, 'Draw a box for this player', () => {
        st.sel = prob.keys[0]; st.drawFor = prob.keys[0]; st.drawProblem = prob.key; setTool('draw');
      });
      b.classList.add('primary');
      acts.push(b, btn('Ignore', 'Mark this as checked', () => addEdit({ kind: 'ok', frame: st.cur, problem: prob.key }, 'Marked as checked')));
    }
    banner.replaceChildren(h('span', { class: 'trk-dot', style: `background:${PROB_COL[prob.kind]}` }),
      h('span', { style: 'font-weight:600' }, prob.text), ...acts);
  }

  function drawTop(r) {
    dyn.replaceChildren();
    camLayer.replaceChildren();
    if (!r) return;
    if (r.v) camLayer.append(svg('polygon', { points: r.v.map((p) => p.join(',')).join(' '), fill: '#FFFFFF', 'fill-opacity': 0.07,
      stroke: '#FFFFFF', 'stroke-opacity': 0.4, 'stroke-width': 0.3, 'stroke-dasharray': '1.2 0.8' }));
    const trailKeys = st.showTrails ? sum.players.map((p) => p.key) : (st.sel ? [st.sel] : []);
    for (const k of trailKeys) {
      const pts = trail(k, st.cur, 3, false);
      if (pts.length > 1) dyn.append(svg('polyline', { points: pts.map((p) => p.join(',')).join(' '), fill: 'none',
        stroke: k === st.sel ? SEL : pColor(byKey[k]), 'stroke-width': 0.5, 'stroke-opacity': k === st.sel ? 0.95 : 0.5 }));
    }
    if (st.showOthers) {
      for (const o of r.o) {
        if (o[6] == null || o[1] === 'r') continue;
        dyn.append(svg('circle', { cx: o[6], cy: o[7], r: 0.7, fill: '#9AA1AB', 'fill-opacity': 0.6 }));
      }
    }
    const order = r.p.slice().sort((a, b) => (a[0] === st.sel) - (b[0] === st.sel));
    for (const p of order) {
      if (p[7] == null) continue;
      const pl = byKey[p[0]], c = pColor(pl), isSel = p[0] === st.sel;
      const g = svg('g', { class: 'trk-dot-g', tabindex: '0', role: 'button', 'aria-label': `${label(pl)}: ${STATUS[p[2]]}` });
      const solid = p[2] === 's' || p[2] === 'h';
      g.append(svg('circle', { cx: p[7], cy: p[8], r: 1.9,
        fill: solid ? c : p[2] === 'o' ? c : '#15181C', 'fill-opacity': p[2] === 'o' ? 0.35 : 1,
        stroke: isSel ? SEL : p[2] === 'p' ? '#E8A33A' : p[2] === 'l' ? '#EF5B5B' : p[2] === 'h' ? SEL : '#0E1013',
        'stroke-width': isSel ? 0.7 : 0.35, 'stroke-dasharray': p[2] === 'p' || p[2] === 'l' ? '0.7 0.45' : null }));
      const t = svg('text', { x: p[7], y: p[8] + 0.68, class: 'trk-num', fill: solid ? ink(c) : '#E6E8EB' });
      t.textContent = pl && pl.number != null ? pl.number : '';
      g.append(t);
      const title = svg('title', {}); title.textContent = `${label(pl)} (${STATUS[p[2]]})`; g.append(title);
      const pick = () => { st.sel = p[0]; st.pending = null; draw(); };
      g.addEventListener('click', pick);
      g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
      dyn.append(g);
    }
    if (r.b && r.b[2] != null) dyn.append(svg('circle', { cx: r.b[2], cy: r.b[3], r: 0.7, fill: '#FFFFFF', stroke: '#0E1013', 'stroke-width': 0.25 }));
  }

  function at(r, key) { return r ? r.p.find((p) => p[0] === key) : null; }

  function drawSel(r) {
    const pl = st.sel && byKey[st.sel];
    const picker = h('select', { class: 'mini-select', 'aria-label': 'Select a player', style: 'max-width:180px', onchange: (e) => {
      st.sel = e.target.value || null; st.pending = null; draw();
    } }, h('option', { value: '' }, 'Pick a player…'),
    ['team_a', 'team_b'].map((t) => h('optgroup', { label: sum.teams[t].name },
      sum.players.filter((p) => p.team === t).map((p) => h('option', { value: p.key, selected: p.key === st.sel || null }, label(p))))));
    if (!pl) {
      selPanel.replaceChildren(h('div', { class: 'panel-head' }, h('h2', {}, 'Selected player'), h('div', { class: 'spacer' }), picker),
        h('div', { class: 'empty' }, 'Click a box on the video, a dot on the pitch or a name in the lanes.'));
      return;
    }
    const p = at(r, pl.key);
    const s = p ? p[2] : null;
    const tid = p ? p[1] : pl.tid;
    const hasBox = s === 's' || s === 'h';
    const mates = sum.players.filter((q) => q.team === pl.team && q.key !== pl.key);
    const swapSel = h('select', { id: 'trk-swap', class: 'mini-select', style: 'flex:1 1 0;min-width:0;height:34px' },
      mates.map((q) => h('option', { value: q.key }, label(q))));
    const myProblems = sum.problems.filter((q) => q.fixed == null && q.keys.includes(pl.key)).length;
    const statusPill = h('span', { class: 'pill ' + (s === 's' ? 'done' : s === 'h' || s === 'p' ? 'stale' : s === 'l' ? 'failed' : 'blocked') }, s ? STATUS[s] : '…');
    const drawB = btn(hasBox ? 'Redraw box here' : 'Draw a box here', 'Draw this player\'s box on this frame', () => {
      st.drawFor = pl.key; st.drawProblem = null; setTool('draw');
    }, hasBox ? '' : 'primary');
    selPanel.replaceChildren(
      h('div', { class: 'panel-head' }, h('span', { class: 'chip', style: `background:${pColor(pl)};border-radius:50%` }),
        h('h2', {}, label(pl)), h('span', { class: 'muted' }, sum.teams[pl.team] ? sum.teams[pl.team].name : ''),
        h('div', { class: 'spacer' }), statusPill),
      h('div', { class: 'panel-body', style: 'display:flex;flex-direction:column;gap:10px' },
        h('dl', { class: 'kv' }, h('dt', {}, 'Track id'), h('dd', {}, String(tid)),
          h('dt', {}, 'Role'), h('dd', {}, pl.role || '—'),
          h('dt', {}, 'Open problems'), h('dd', {}, String(myProblems))),
        drawB,
        h('label', { for: 'trk-swap', class: 'section-label' }, 'Swap with, from this frame on'),
        h('div', { style: 'display:flex;gap:6px' }, swapSel, btn('Swap', 'Swap the two players from this frame on', () => {
          const other = byKey[swapSel.value];
          const op = at(r, other.key);
          addEdit({ kind: 'swap', frame: st.cur, a: tid, b: op ? op[1] : other.tid }, `Swapped ${label(pl)} and ${label(other)} from here`);
        }, 'primary')),
        h('div', { class: 'trk-actions' },
          btn('Not him from here', 'Split: from this frame to the end of this piece of track, the box is not this player', () => {
            if (!hasBox) { toast('No box for this player on this frame', true); return; }
            addEdit({ kind: 'split', frame: st.cur, tid }, `Split ${label(pl)}'s track here`);
          }),
          btn('Delete box', 'Delete this player\'s box on this frame only', () => {
            if (!hasBox) { toast('No box for this player on this frame', true); return; }
            addEdit({ kind: 'delete', frame: st.cur, tid }, `Deleted ${label(pl)}'s box here`);
          }),
          btn('Not a player', 'This piece of track is a referee, coach or fan', () => {
            if (!hasBox) { toast('No box for this player on this frame', true); return; }
            addEdit({ kind: 'not_player', frame: st.cur, tid }, 'Marked as not a player');
          })),
        h('div', { style: 'display:flex;justify-content:flex-end' }, picker)));
  }

  function drawPending(r) {
    const pd = st.pending;
    if (!pd) { pendingPanel.hidden = true; return; }
    pendingPanel.hidden = false;
    const noBox = r ? r.p.filter((p) => p[2] !== 's' && p[2] !== 'h') : [];
    // drawn box: nearest last-known spots first is not possible client-side (no camera model), so
    // order by how long they've been missing: lost before predicted before off camera
    const rank = { l: 0, p: 1, o: 2 };
    noBox.sort((a, b) => rank[a[2]] - rank[b[2]]);
    const chooseTo = (key) => {
      const pl = byKey[key];
      if (pd.box) {
        const p = at(r, key);
        const run = sum.problems.find((q) => q.kind === 'lost' && q.fixed == null && q.keys.includes(key)
          && q.frame <= st.cur && st.cur <= q.end);
        addEdit({ kind: 'box', frame: st.cur, tid: p ? p[1] : pl.tid, ...pd.box, ...(run ? { problem: run.key } : {}) },
          `Drew a box for ${label(pl)}`);
      } else {
        addEdit({ kind: 'assign', frame: st.cur, tid: pd.other, to: (at(r, key) || [0, pl.tid])[1] }, `That box is ${label(pl)} now`);
      }
      st.sel = key;
    };
    const quick = noBox.slice(0, 8).map((p) => {
      const pl = byKey[p[0]], c = pColor(pl);
      return h('button', { type: 'button', class: 'btn small', style: `background:${c};color:${ink(c)};border-color:${c}`,
        onclick: () => chooseTo(p[0]) }, `${label(pl)}`);
    });
    const any = h('select', { class: 'mini-select', 'aria-label': 'Someone else', style: 'flex:1 1 0;min-width:0;height:34px' },
      ['team_a', 'team_b'].map((t) => h('optgroup', { label: sum.teams[t].name },
        sum.players.filter((p) => p.team === t).map((p) => h('option', { value: p.key }, label(p))))));
    pendingPanel.replaceChildren(
      h('div', { class: 'panel-head' }, h('h2', {}, pd.box ? 'New box: who is this?' : `Unlabelled box ${pd.other}: who is this?`)),
      h('div', { class: 'panel-body', style: 'display:flex;flex-direction:column;gap:10px' },
        noBox.length ? h('span', { class: 'muted', style: 'font-size:12px' }, 'Players with no box on this frame') : null,
        h('div', { class: 'trk-actions' }, quick),
        h('div', { style: 'display:flex;gap:6px' }, any, btn('Use', 'This box is the player picked in the list', () => chooseTo(any.value), 'primary')),
        h('div', { class: 'trk-actions' },
          pd.other != null ? btn('Not a player', 'This box is a referee, coach or fan', () =>
            addEdit({ kind: 'not_player', frame: st.cur, tid: pd.other }, 'Marked as not a player')) : null,
          btn('Cancel', 'Cancel (Esc)', () => { st.pending = null; draw(); })),
        pd.box ? h('span', { class: 'muted', style: 'font-size:11px;line-height:1.5' },
          'The box is carried to nearby frames until the tracker finds the player again (up to 5 s).') : null));
  }

  function rowLayout() {
    const rows = [];
    let y = RULER_H;
    for (const t of ['team_a', 'team_b']) {
      rows.push({ head: t, y }); y += HEAD_H;
      for (const p of sum.players.filter((q) => q.team === t)) { rows.push({ key: p.key, y }); y += ROW_H; }
    }
    return { rows, height: y + 4 };
  }

  function drawLanes() {
    if (!st.win) return;
    const dpr = window.devicePixelRatio || 1;
    const cssW = lanes.clientWidth || 800;
    const { rows, height } = rowLayout();
    lanes.style.height = height + 'px';
    lanes.width = Math.round(cssW * dpr);
    lanes.height = Math.round(height * dpr);
    const g = lanes.getContext('2d');
    g.scale(dpr, dpr);
    g.fillStyle = '#0B0D10';
    g.fillRect(0, 0, cssW, height);
    const [a, b] = st.win;
    const x0 = LABEL_W, x1 = cssW - 4, span = Math.max(b - a, 1);
    const fx = (f) => x0 + ((f - a) / span) * (x1 - x0);
    winLabel.textContent = `${clockAt(a)} – ${clockAt(b)}`;
    // ruler
    g.font = '500 10px JetBrains Mono, monospace';
    g.fillStyle = '#858D98';
    const stepS = st.winLen <= 120 ? 15 : st.winLen <= 300 ? 30 : 120;
    let lastX = -1e9;
    for (let f = a - ((a - sum.start) % Math.round(stepS * FPS)); f <= b; f += Math.round(stepS * FPS)) {
      if (f < a) continue;
      const x = fx(f);
      g.fillRect(x, RULER_H - 6, 1, 4);
      const t = clockAt(f);
      if (t && x - lastX >= 44) { g.fillText(t, Math.min(x + 2, cssW - 40), 10); lastX = x; }
    }
    // problem markers
    for (const p of sum.problems) {
      if (p.frame < a || p.frame > b) continue;
      const x = fx(p.frame);
      g.save();
      g.translate(x, RULER_H - 8);
      g.rotate(Math.PI / 4);
      g.fillStyle = p.fixed != null ? '#5DBB63' : PROB_COL[p.kind];
      g.fillRect(-3.5, -3.5, 7, 7);
      g.restore();
    }
    for (const row of rows) {
      if (row.head) {
        g.fillStyle = color[row.head];
        g.fillRect(8, row.y + 6, 8, 8);
        g.fillStyle = '#C9CED6';
        g.font = '600 10px Inter Tight, sans-serif';
        g.fillText(sum.teams[row.head].name.toUpperCase(), 22, row.y + 13);
        continue;
      }
      const pl = byKey[row.key];
      const isSel = row.key === st.sel;
      if (isSel) { g.fillStyle = '#20242A'; g.fillRect(0, row.y, cssW, ROW_H); }
      g.fillStyle = isSel ? SEL : '#C9CED6';
      g.font = (isSel ? '600 ' : '400 ') + '11px Inter Tight, sans-serif';
      g.fillText(label(pl).slice(0, 18), 8, row.y + 10);
      const lane = st.lanes && st.lanes.lanes[row.key];
      if (!lane) { g.fillStyle = '#1A1D22'; g.fillRect(x0, row.y + 2, x1 - x0, ROW_H - 4); continue; }
      const n = lane.length, w = (x1 - x0) / n;
      for (let i = 0; i < n; i++) {
        g.fillStyle = LANE_COL[lane[i]];
        g.fillRect(x0 + i * w, row.y + 2, Math.ceil(w), ROW_H - 4);
      }
    }
    g.fillStyle = SEL;
    g.fillRect(fx(st.cur) - 1, 0, 2, height);
  }

  function clockAt(f) {
    const r = rec(f);
    if (r && r.c) return r.c;
    const keys = Object.keys(sum.clocks || {}).map(Number).filter((k) => k <= f).sort((x, y) => y - x);
    if (!keys.length) return '';
    const base = sum.clocks[keys[0]];
    const m = /^(\d+):(\d+)$/.exec(base || '');
    if (!m) return base || '';
    const s = (+m[1]) * 60 + (+m[2]) + Math.floor((f - keys[0]) / FPS);
    return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  }

  lanes.addEventListener('click', (ev) => {
    const rect = lanes.getBoundingClientRect();
    const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
    if (x < LABEL_W) {
      const row = rowLayout().rows.find((r) => r.key && y >= r.y && y < r.y + ROW_H);
      if (row) { st.sel = row.key; st.pending = null; draw(); }
      return;
    }
    const [a, b] = st.win;
    const f = a + ((x - LABEL_W) / (rect.width - 4 - LABEL_W)) * (b - a);
    if (y < RULER_H) {
      const near = sum.problems.filter((p) => p.frame >= a && p.frame <= b)
        .map((p) => [Math.abs(p.frame - f), p]).sort((p, q) => p[0] - q[0])[0];
      if (near && near[0] < (b - a) / 80) { gotoProblem(near[1]); return; }
    }
    const row = rowLayout().rows.find((r) => r.key && y >= r.y && y < r.y + ROW_H);
    if (row) st.sel = row.key;
    stopPlay();
    seek(f);
  });

  function drawProblems() {
    const list = sum.problems.filter((p) => (st.filter === 'fixed' ? p.fixed != null : p.fixed == null)
      && (st.filter === 'swap' || st.filter === 'lost' ? p.kind === st.filter : true));
    probCount.textContent = `${sum.open_problems} open · ${sum.problems.length - sum.open_problems} fixed`;
    // the 40 around the playhead
    let i = list.findIndex((p) => p.frame >= st.cur);
    if (i < 0) i = list.length;
    const shown = list.slice(Math.max(0, i - 5), Math.max(0, i - 5) + 40);
    probList.replaceChildren(...(shown.length ? shown.map((p) => {
      const here = st.cur >= p.frame - 1.5 * FPS && st.cur <= Math.max(p.end, p.frame) + 1.5 * FPS;
      return h('button', { type: 'button', class: 'job-row trk-prob' + (here ? ' here' : ''), 'aria-current': here ? 'true' : null,
        onclick: () => gotoProblem(p) },
      h('span', { style: 'display:flex;align-items:center;gap:8px;min-width:0' },
        h('span', { class: 'trk-dot', style: `background:${p.fixed != null ? '#5DBB63' : PROB_COL[p.kind]}` }),
        h('span', { class: 'mono muted', style: 'font-size:11px' }, p.clock || ''),
        h('span', { style: p.fixed != null ? 'color:var(--dim);text-decoration:line-through' : '' }, p.text)),
      h('span', { class: 'mono', style: `font-size:11px;color:${p.fixed != null ? 'var(--ok)' : 'var(--muted)'}` }, p.fixed != null ? 'fixed' : 'open'));
    }) : [h('div', { class: 'empty' }, st.filter === 'fixed' ? 'Nothing fixed yet.' : 'No open problems here.')]));
  }

  function describe(e) {
    switch (e.kind) {
      case 'swap': return `Swapped ${who(e.a)} and ${who(e.b)} from here${e.to != null ? ' (to ' + e.to + ')' : ''}`;
      case 'assign': return `Box ${e.tid} → ${who(e.to)}`;
      case 'box': return `Drew a box for ${who(e.tid)}`;
      case 'split': return `${who(e.tid)}: not him from here`;
      case 'delete': return `Deleted ${who(e.tid)}'s box`;
      case 'not_player': return keyOfTid[e.tid] ? `${who(e.tid)}'s box marked not a player` : `Box ${e.tid} marked not a player`;
      case 'ok': {
        const p = sum.problems.find((q) => q.key === e.problem);
        return 'Checked: ' + (p ? p.text : e.problem);
      }
      default: return e.kind;
    }
  }

  function drawEdits() {
    const es = sum.edits.slice().reverse();
    editCount.textContent = `${es.length} fix${es.length === 1 ? '' : 'es'}`;
    staleNote.hidden = !sum.stale;
    staleNote.textContent = sum.stale ? 'These fixes were made against different tracks (tracking was re-run), so they are not applied. Undo them or ask for a re-map.' : '';
    editList.replaceChildren(...(es.length ? es.map((e) => h('div', { class: 'trk-edit' },
      h('button', { type: 'button', class: 'job-row', style: 'border:0', onclick: () => { if (e.frame != null) { stopPlay(); seek(e.frame); } } },
        h('span', { style: 'display:flex;gap:10px;align-items:baseline;min-width:0' },
          h('span', { class: 'mono muted', style: 'font-size:11px' }, e.clock || (e.frame != null ? 'f' + e.frame : '')),
          h('span', {}, describe(e)))),
      btn('Undo', 'Undo: ' + describe(e), () => undo(e)))) : [h('div', { class: 'empty' }, 'No fixes yet.')]));
  }

  // ---------- interaction ----------
  function seek(f) { st.cur = clamp(f); draw(); }
  function step(n) { stopPlay(); seek(st.cur + n); }
  function stopPlay() { st.playing = false; playBtn.textContent = '▶'; playBtn.setAttribute('aria-label', 'Play (k)'); }
  function togglePlay() {
    if (st.playing) { stopPlay(); return; }
    st.playing = true;
    st.pending = null;
    playBtn.textContent = '❚❚';
    playBtn.setAttribute('aria-label', 'Pause (k)');
    lastTick = 0;
    playTick();
  }
  let lastTick = 0;
  function playTick() {
    if (!st.playing) return;
    if (st.cur >= sum.end) { stopPlay(); return; }
    const now = performance.now();
    const adv = lastTick ? Math.max(1, Math.min(6, Math.round(((now - lastTick) / 1000) * FPS))) : 1;
    lastTick = now;
    seek(st.cur + adv);
  }
  function gotoProblem(p) {
    stopPlay();
    st.pending = null;
    if (p.keys[0]) st.sel = p.keys[0];
    seek(p.frame);
  }
  function jumpProblem(dir) {
    const open = sum.problems.filter((p) => p.fixed == null);
    if (!open.length) { msg.textContent = 'No open problems'; return; }
    let t = dir > 0 ? open.find((p) => p.frame > st.cur + 1) : open.slice().reverse().find((p) => p.frame < st.cur - 1);
    if (!t) t = dir > 0 ? open[0] : open[open.length - 1];
    gotoProblem(t);
  }
  function setTool(tool) {
    st.tool = tool;
    if (tool === 'select') { st.drawFor = null; st.drawProblem = null; }
    selectBtn.classList.toggle('primary', tool === 'select');
    drawBtn.classList.toggle('primary', tool === 'draw');
    selectBtn.setAttribute('aria-pressed', String(tool === 'select'));
    drawBtn.setAttribute('aria-pressed', String(tool === 'draw'));
    if (tool === 'draw') {
      stopPlay();
      st.pending = null;
      msg.textContent = st.drawFor ? `Drag a box round ${label(byKey[st.drawFor])}, head to feet` : 'Drag a box round the player, head to feet';
    } else msg.textContent = '';
    draw();
  }

  function toVideo(ev) {
    const pt = overlay.createSVGPoint();
    pt.x = ev.clientX; pt.y = ev.clientY;
    const p = pt.matrixTransform(overlay.getScreenCTM().inverse());
    return [Math.max(0, Math.min(W, p.x)), Math.max(0, Math.min(H, p.y))];
  }
  function hitTest(x, y) {
    const r = rec(st.cur);
    if (!r) return null;
    let best = null, bd = 1e9;
    const consider = (kind, id, b) => {
      if (b[0] == null) return;
      const inside = b[0] <= x && x <= b[2] && b[1] <= y && y <= b[3];
      const d = inside ? (b[2] - b[0]) * (b[3] - b[1]) * 0.001 : Math.hypot(x - (b[0] + b[2]) / 2, y - b[3]);
      if ((inside || d < 30) && d < bd) { best = { kind, id }; bd = d; }
    };
    if (st.showBoxes) for (const p of r.p) if (p[2] === 's' || p[2] === 'h' || p[2] === 'p') consider('player', p[0], p.slice(3, 7));
    if (st.showOthers && st.showBoxes) for (const o of r.o) consider('other', o[0], o.slice(2, 6));
    return best;
  }
  overlay.addEventListener('pointerdown', (ev) => {
    if (st.tool !== 'draw') return;
    ev.preventDefault();
    overlay.setPointerCapture(ev.pointerId);
    const [x, y] = toVideo(ev);
    st.drag = { x0: x, y0: y, x1: x, y1: y };
  });
  overlay.addEventListener('pointermove', (ev) => {
    if (!st.drag) return;
    const [x, y] = toVideo(ev);
    st.drag.x1 = x; st.drag.y1 = y;
    draw();
  });
  overlay.addEventListener('pointerup', (ev) => {
    if (st.tool === 'draw' && st.drag) {
      const d = st.drag;
      st.drag = null;
      const box = { x1: Math.min(d.x0, d.x1), y1: Math.min(d.y0, d.y1), x2: Math.max(d.x0, d.x1), y2: Math.max(d.y0, d.y1) };
      if (box.x2 - box.x1 < 4 || box.y2 - box.y1 < 6) { msg.textContent = 'Drag a box round the player, head to feet'; draw(); return; }
      if (st.drawFor) {
        const pl = byKey[st.drawFor], p = at(rec(st.cur), st.drawFor);
        const e = { kind: 'box', frame: st.cur, tid: p ? p[1] : pl.tid, ...box };
        const run = sum.problems.find((q) => q.kind === 'lost' && q.fixed == null && q.keys.includes(st.drawFor)
          && q.frame <= st.cur && st.cur <= q.end);
        if (st.drawProblem || run) e.problem = st.drawProblem || run.key;
        setTool('select');
        addEdit(e, `Drew a box for ${label(pl)}`);
      } else {
        st.pending = { box };
        setTool('select');
        msg.textContent = 'Who is this? Pick them on the right.';
      }
      return;
    }
    if (st.tool !== 'select') return;
    const [x, y] = toVideo(ev);
    const hit = hitTest(x, y);
    if (!hit) { st.pending = null; draw(); return; }
    if (hit.kind === 'player') { st.sel = hit.id; st.pending = null; } else { st.pending = { other: hit.id }; }
    draw();
  });

  function onKey(e) {
    if (e.target && ['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key;
    if (k === 'k' || k === ' ') { e.preventDefault(); togglePlay(); }
    else if (k === ',') step(-1);
    else if (k === '.') step(1);
    else if (k === 'a') step(-2 * FPS);
    else if (k === 'd') step(2 * FPS);
    else if (k === 'n') jumpProblem(1);
    else if (k === 'p') jumpProblem(-1);
    else if (k === 'b') setTool(st.tool === 'draw' ? 'select' : 'draw');
    else if (k === 'Escape') { st.drag = null; st.pending = null; if (st.tool === 'draw') setTool('select'); else { st.sel = null; draw(); } }
  }
  document.addEventListener('keydown', onKey);
  const onResize = () => { st.lanes = null; loadLanes(); drawLanes(); };
  window.addEventListener('resize', onResize);
  onLeave(() => { st.playing = false; document.removeEventListener('keydown', onKey); window.removeEventListener('resize', onResize); });

  if (sum.stale) toast('Your saved fixes were made against different tracks: they are not applied', true);
  draw();
}
