// Calibration clips (issue #46): name the players in three 5-second clips per
// half, then re-run tracking; everything after it runs automatically.
// Each box shows the tracker's guess. Fix the wrong ones (click a box, pick the
// player), then "Accept the rest" turns the remaining guesses into names.
// Names are identity anchors (config/<id>_identity_anchors.json), the same file
// the Lines tab's Identify mode writes, so track_identities.py uses them.

const SVGNS = 'http://www.w3.org/2000/svg';
const KEYS_HELP = [
  ['Click a box', 'pick who it is (or type the shirt number, then Enter)'],
  ['A', 'accept the rest: every guess on this frame becomes a name'],
  ['k', 'play / pause'], [', / .', '−/+ 1 frame'], ['a / d', '−/+ 1 s'],
  ['n / p', 'next / previous clip'], ['Esc', 'deselect'],
];

function svg(tag, attrs) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null) el.setAttribute(k, v);
  return el;
}
function iou(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const i = ix * iy, u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
  return u > 0 ? i / u : 0;
}

export async function viewCalibrate(view, parts, params, ui) {
  const { h, api, toast, renderTabs, setCrumb, onLeave } = ui;
  renderTabs('tracking');
  const matchId = parts[1];
  const mid = encodeURIComponent(matchId || '');
  let match;
  try { match = await api('/matches/' + mid); } catch (e) {
    view.replaceChildren(h('div', { class: 'placeholder' }, h('h1', {}, 'Match not found'), h('p', {}, e.message)));
    return;
  }
  setCrumb(match);
  view.replaceChildren(h('div', { class: 'placeholder' }, h('h1', {}, 'Finding calibration clips…')));
  let sum;
  try { sum = await api(`/matches/${mid}/calibration`); } catch (e) {
    view.replaceChildren(h('div', { class: 'placeholder' }, h('h1', {}, 'Calibration not ready'), h('p', {}, e.message),
      h('a', { class: 'btn', href: '#/match/' + mid }, 'Open the pipeline')));
    return;
  }
  if (!sum.clips.length) {
    view.replaceChildren(h('div', { class: 'placeholder' }, h('h1', {}, 'No clips'),
      h('p', {}, 'config/<id>_match.json has no periods or analysis window to pick clips from.')));
    return;
  }

  const W = sum.width, H = sum.height, FPS = sum.fps;
  const IMG_W = Math.min(W, 1280);
  const LU = sum.lineups;
  const st = { clip: null, data: null, cur: 0, sel: null, typed: '', playing: false, busy: false, pickTeam: null };

  // ---------- layout ----------
  const img = h('img', { alt: 'Video frame', class: 'lines-img', draggable: 'false' });
  const overlay = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'lines-svg trk-svg', role: 'img', 'aria-label': 'Players: click a box to name them' });
  const frameChip = h('div', { class: 'trk-chip mono' });
  const stage = h('div', { class: 'lines-stage', style: `aspect-ratio:${W}/${H}` }, img, overlay, frameChip);
  const btn = (label, aria, fn, cls) => h('button', { type: 'button', class: 'btn small ' + (cls || ''), 'aria-label': aria, onclick: fn }, label);
  const playBtn = btn('▶', 'Play (k)', () => togglePlay());
  const slider = h('input', { type: 'range', min: 0, max: 0, value: 0, 'aria-label': 'Frame in clip', style: 'flex:1 1 160px;accent-color:var(--accent)',
    oninput: () => { stopPlay(); seek(st.clip.start + +slider.value); } });
  const msg = h('div', { class: 'note', role: 'status', 'aria-live': 'polite', style: 'min-height:18px' });
  const acceptBtn = btn('Accept the rest (A)', 'Every guess on this frame becomes a name', () => acceptRest(), 'primary');
  const transport = h('div', { class: 'lines-tools' },
    btn('−1s', 'Back 1 second (a)', () => step(-FPS)), btn('−1f', 'Previous frame (,)', () => step(-1)), playBtn,
    btn('+1f', 'Next frame (.)', () => step(1)), btn('+1s', 'Forward 1 second (d)', () => step(FPS)), slider, acceptBtn);
  const title = h('h1', { style: 'margin:0;font-size:14px' });
  const sub = h('span', { class: 'muted mono', style: 'font-size:12px' });
  const picker = h('div', { class: 'cal-picker', hidden: true });
  const left = h('div', { class: 'col-main' },
    h('section', { class: 'panel' },
      h('div', { class: 'panel-head' }, title, sub),
      h('div', { class: 'panel-body', style: 'display:flex;flex-direction:column;gap:10px' }, stage, transport, msg, picker)));

  const clipList = h('div', { class: 'scroll' });
  const rosterBox = h('div', { class: 'cal-roster' });
  const runInfo = h('div', { style: 'display:flex;flex-direction:column;gap:6px;font-size:12px' });
  const runBtn = h('button', { type: 'button', class: 'btn primary', onclick: () => run(false) }, 'Run tracking with these names');
  const guessBtn = h('button', { type: 'button', class: 'btn', onclick: () => run(true) }, 'Update guesses (re-track only)');
  const keysHelp = h('details', { class: 'panel', style: 'padding:10px 14px' }, h('summary', {}, 'Keys'),
    h('dl', { class: 'kv', style: 'margin-top:8px' }, KEYS_HELP.flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, v)])));
  const right = h('div', { class: 'col-side' },
    h('section', { class: 'panel' }, h('div', { class: 'panel-head' }, h('h2', {}, 'Clips'),
      h('span', { class: 'muted', style: 'font-size:12px' }, '5 s each: start, middle and end of each half')), clipList),
    h('section', { class: 'panel' }, h('div', { class: 'panel-head' }, h('h2', {}, 'Named in this clip')), rosterBox),
    h('section', { class: 'panel', style: 'padding:12px 14px;display:flex;flex-direction:column;gap:10px' },
      h('h2', { style: 'margin:0;font-size:13px' }, 'Run'),
      h('div', { class: 'muted', style: 'font-size:12px;line-height:1.5' },
        'Re-runs Track identities with your names, then ball, possession, physical, positions and verification, one after another. ',
        'Update guesses only re-tracks (about 4 min), so the next clip\'s guesses use the names you have so far.'),
      runBtn, guessBtn, runInfo,
      h('a', { href: '#/tracking/' + mid, class: 'btn small' }, 'Back to Tracking')),
    keysHelp);
  view.replaceChildren(h('div', { class: 'cols' }, left, right));

  // ---------- helpers ----------
  const teamName = (t) => LU[t].name;
  const person = (t, n) => LU[t].players.find((p) => p.number === n);
  const label = (t, n) => { const p = person(t, n); return p ? `${n} ${p.name.split(' ').slice(-1)[0]}` : `#${n}`; };
  function anchorFor(row) { // a name on this box: same track id, or the anchor's own box on this frame
    const tid = row[0];
    return st.data.anchors.find((a) => a.track_id === tid) ||
      st.data.anchors.find((a) => a.frame === st.cur && iou(a.box, row.slice(1, 5)) > 0.5) || null;
  }
  function namedSet(team) { return new Set(st.data.named[team] || []); }

  function drawClips() {
    clipList.replaceChildren(...sum.clips.map((c) => {
      const done = (c.named.team_a || 0) + (c.named.team_b || 0);
      const here = st.clip && st.clip.id === c.id;
      return h('button', { type: 'button', class: 'job-row cal-clip' + (here ? ' sel' : ''), 'aria-current': here ? 'true' : null,
        onclick: () => openClip(c.id) },
        h('div', { style: 'display:flex;flex-direction:column;align-items:flex-start;gap:2px;min-width:0' },
          h('span', { style: 'font-weight:600' }, c.label),
          h('span', { class: 'muted mono', style: 'font-size:11px' }, `${c.clock || ''}  ·  ~${c.players_avg} players in view`)),
        h('span', { class: 'pill ' + (done >= 20 ? 'done' : done ? 'stale' : 'todo') },
          `${c.named.team_a || 0} + ${c.named.team_b || 0} named`));
    }));
  }

  function drawRoster() {
    rosterBox.replaceChildren(...['team_a', 'team_b'].map((t) => {
      const named = namedSet(t);
      const starters = LU[t].players.filter((p) => !p.sub);
      return h('div', { class: 'cal-team' },
        h('div', { class: 'cal-team-head' }, h('i', { class: 'cal-swatch', style: `background:${LU[t].color}` }),
          `${teamName(t)} — ${starters.filter((p) => named.has(p.number)).length}/${starters.length}`),
        h('div', { class: 'cal-chips' }, LU[t].players.filter((p) => !p.sub || named.has(p.number)).map((p) =>
          h('span', { class: 'cal-chip' + (named.has(p.number) ? ' on' : ''), title: p.name }, `${p.number}`))));
    }));
  }

  function drawPicker() {
    const row = st.sel != null ? (st.data.frames[st.cur] || []).find((r) => r[0] === st.sel) : null;
    if (!row) { picker.hidden = true; return; }
    picker.hidden = false;
    const team = st.pickTeam || row[5] || 'team_a';
    const a = anchorFor(row);
    const named = namedSet(team);
    picker.replaceChildren(
      h('div', { class: 'cal-picker-head' },
        h('span', {}, a ? `Named ${label(a.team, a.number)}` : row[6] != null ? `Tracker thinks: ${label(row[5], row[6])}` : 'Unlabelled box'),
        h('div', { class: 'spacer' }),
        h('div', { class: 'seg', role: 'group', 'aria-label': 'Team' }, ['team_a', 'team_b'].map((t) =>
          h('button', { type: 'button', class: 'btn small', 'aria-pressed': String(t === team),
            onclick: () => { st.pickTeam = t; drawPicker(); } }, teamName(t)))),
        a ? btn('Remove name', 'Remove this name', () => unname(a.id)) : null,
        btn('Close', 'Deselect (Esc)', () => deselect())),
      h('div', { class: 'cal-grid' }, LU[team].players.map((p) =>
        h('button', { type: 'button', class: 'cal-num' + (named.has(p.number) ? ' used' : '') + (a && a.team === team && a.number === p.number ? ' me' : ''),
          title: named.has(p.number) ? `${p.name} (already named in this clip: naming again moves it here)` : p.name,
          onclick: () => nameBox(row, team, p.number) },
          h('b', {}, String(p.number)), h('span', {}, p.name.split(' ').slice(-1)[0] + (p.sub ? ' (sub)' : ''))))));
  }

  function draw() {
    if (!st.data) return;
    const rows = st.data.frames[st.cur] || [];
    overlay.replaceChildren();
    for (const r of rows) {
      const [tid, x1, y1, x2, y2, team, guess] = r;
      const a = anchorFor(r);
      const col = (a ? LU[a.team] : team ? LU[team] : null)?.color || '#9AA0A6';
      const g = svg('g', { class: 'trk-dot-g', 'data-tid': tid });
      g.append(svg('rect', { x: x1, y: y1, width: x2 - x1, height: y2 - y1, fill: 'transparent',
        stroke: tid === st.sel ? '#FFD84D' : col, 'stroke-width': tid === st.sel ? 3 : a ? 2.2 : 1.4,
        'stroke-dasharray': a ? null : '5 3', 'stroke-opacity': a ? 1 : 0.85 }));
      const text = a ? `✓ ${a.number}` : guess != null ? `${guess}?` : '?';
      const t = svg('text', { x: (x1 + x2) / 2, y: y1 - 5, class: 'trk-label', fill: col, 'fill-opacity': a ? 1 : 0.8 });
      t.textContent = text;
      g.append(t);
      g.addEventListener('click', (ev) => { ev.stopPropagation(); select(tid); });
      overlay.append(g);
    }
    const k = st.cur - st.clip.start;
    slider.value = k;
    frameChip.textContent = `${(k / FPS).toFixed(1)} / ${((st.clip.end - st.clip.start + 1) / FPS).toFixed(1)} s · frame ${st.cur}`;
    loadFrame(st.cur);
    drawPicker();
  }

  // ---------- frames ----------
  let wantFrame = null, loading = false;
  function loadFrame(f) {
    wantFrame = f;
    if (loading) return;
    loading = true;
    const pre = new Image();
    pre.onload = pre.onerror = () => {
      if (pre.naturalWidth) img.src = pre.src;
      loading = false;
      if (wantFrame !== f) loadFrame(wantFrame);
    };
    pre.src = `/api/matches/${mid}/frame/${f}.jpg?w=${IMG_W}`;
  }
  function seek(f) {
    st.cur = Math.max(st.clip.start, Math.min(st.clip.end, Math.round(f)));
    draw();
  }
  function step(n) { stopPlay(); seek(st.cur + n); }
  function stopPlay() { st.playing = false; playBtn.textContent = '▶'; }
  function togglePlay() {
    if (st.playing) { stopPlay(); return; }
    st.playing = true; playBtn.textContent = '❚❚';
    const tick = () => {
      if (!st.playing) return;
      if (st.cur >= st.clip.end) { stopPlay(); return; }
      seek(st.cur + 2);
      setTimeout(tick, 80);
    };
    tick();
  }

  // ---------- actions ----------
  function select(tid) { st.sel = tid; st.typed = ''; st.pickTeam = null; stopPlay(); draw(); }
  function deselect() { st.sel = null; st.typed = ''; draw(); }

  async function post(items) {
    if (st.busy) return null;
    st.busy = true;
    try {
      const r = await api(`/matches/${mid}/calibration/${st.clip.id}/names`, { method: 'POST', body: { items } });
      st.data.anchors = r.anchors; st.data.named = r.named;
      updateClipCount();
      return r;
    } catch (e) { msg.textContent = e.message; return null; } finally { st.busy = false; }
  }
  async function nameBox(row, team, number) {
    const r = await post([{ frame: st.cur, box: row.slice(1, 5), team, number }]);
    if (r) { msg.textContent = `Named ${label(team, number)}`; st.sel = null; draw(); drawRoster(); }
  }
  async function unname(id) {
    try {
      const r = await api(`/matches/${mid}/calibration/${st.clip.id}/names/${id}`, { method: 'DELETE' });
      st.data.anchors = r.anchors; st.data.named = r.named;
      updateClipCount(); msg.textContent = 'Name removed'; draw(); drawRoster();
    } catch (e) { msg.textContent = e.message; }
  }
  async function acceptRest() {
    const used = { team_a: namedSet('team_a'), team_b: namedSet('team_b') };
    const items = [];
    for (const r of st.data.frames[st.cur] || []) {
      const [, , , , , team, guess] = r;
      if (guess == null || !team || anchorFor(r) || used[team].has(guess) || !person(team, guess)) continue;
      used[team].add(guess);
      items.push({ frame: st.cur, box: r.slice(1, 5), team, number: guess });
    }
    if (!items.length) { msg.textContent = 'Nothing left to accept on this frame: every guessed box is named (or its player is already named)'; return; }
    const r = await post(items);
    if (r) { msg.textContent = `Accepted ${items.length} guess${items.length === 1 ? '' : 'es'}. Check for any still unnamed (no guess) and click them.`; draw(); drawRoster(); }
  }
  function updateClipCount() {
    const c = sum.clips.find((x) => x.id === st.clip.id);
    c.named = { team_a: (st.data.named.team_a || []).length, team_b: (st.data.named.team_b || []).length };
    drawClips();
  }

  async function openClip(id) {
    stopPlay();
    const c = sum.clips.find((x) => x.id === id);
    title.textContent = c.label;
    sub.textContent = `${c.clock || ''} · frames ${c.start}–${c.end}`;
    msg.textContent = 'Loading clip…';
    try {
      st.data = await api(`/matches/${mid}/calibration/${encodeURIComponent(id)}`);
    } catch (e) { msg.textContent = e.message; return; }
    st.clip = c; st.sel = null;
    slider.max = c.end - c.start;
    // start on the clip's busiest frame, the easiest one to name everyone on
    let best = c.start, bn = -1;
    for (let f = c.start; f <= c.end; f++) { const n = (st.data.frames[f] || []).length; if (n > bn) { best = f; bn = n; } }
    st.cur = best;
    msg.textContent = 'Fix any wrong guesses (click the box), then Accept the rest. Dashed = tracker\'s guess, solid ✓ = named.';
    history.replaceState(null, '', `#/calibrate/${mid}?clip=${encodeURIComponent(id)}`);
    drawClips(); drawRoster(); draw();
  }

  // ---------- run ----------
  function drawRun() {
    const jobs = (sum.jobs || []).slice().reverse();
    runInfo.replaceChildren(
      h('div', { class: 'muted' }, `${sum.anchors_total} name${sum.anchors_total === 1 ? '' : 's'} saved` +
        (sum.pending ? ` · ${sum.pending} not used by tracking yet` : ' · all used by the current tracking')),
      ...jobs.map((j) => h('div', { style: 'display:flex;justify-content:space-between;gap:8px' },
        h('span', {}, j.title), h('span', { class: 'pill ' + (j.status === 'done' ? 'done' : j.status === 'failed' ? 'failed' : 'stale') }, j.status === 'waiting for the step before' ? 'next' : j.status))));
  }
  let lastStamp = null;
  async function run(onlyIds) {
    try {
      const j = await api(`/matches/${mid}/calibration-run`, { method: 'POST', body: { only_identities: onlyIds } });
      toast(onlyIds ? 'Re-tracking: the guesses update when it finishes (about 4 min)'
        : 'Tracking queued: the stages after it start automatically');
      sum.jobs = [j, ...(sum.jobs || [])];
      drawRun();
    } catch (e) { toast(e.message, true); }
  }
  const poll = setInterval(async () => {
    try {
      const s = await api(`/matches/${mid}/calibration`);
      sum.jobs = s.jobs; sum.anchors_total = s.anchors_total; sum.pending = s.pending;
      drawRun();
      const idJob = (s.jobs || []).find((j) => j.stage === 'b_identities');
      const stamp = idJob && idJob.status === 'done' ? idJob.finished : null;
      if (lastStamp !== null && stamp && stamp !== lastStamp && st.clip && !st.busy) {
        msg.textContent = 'Tracking finished: guesses updated';
        const keep = st.cur; await openClip(st.clip.id); seek(keep);
      }
      if (stamp) lastStamp = stamp; else if (lastStamp === null) lastStamp = 0;
    } catch { /* ignore */ }
  }, 5000);

  // ---------- input ----------
  overlay.addEventListener('click', () => deselect());
  const onKey = (ev) => {
    const tag = ev.target.tagName;
    if (tag === 'INPUT' && ev.target.type !== 'range') return;
    if (tag === 'SELECT' || tag === 'TEXTAREA' || ev.ctrlKey || ev.metaKey || ev.altKey || !st.clip) return;
    const k = ev.key;
    if (st.sel != null && /^[0-9]$/.test(k)) { st.typed += k; msg.textContent = `Number: ${st.typed} (Enter to name)`; }
    else if (st.sel != null && k === 'Enter' && st.typed) {
      const row = (st.data.frames[st.cur] || []).find((r) => r[0] === st.sel);
      const team = st.pickTeam || (row && row[5]) || 'team_a';
      const n = +st.typed; st.typed = '';
      if (row && person(team, n)) nameBox(row, team, n); else msg.textContent = `No. ${n} isn't in ${teamName(team)}'s lineup`;
    }
    else if (k === ',') step(-1);
    else if (k === '.') step(1);
    else if (k === 'a') step(-FPS);
    else if (k === 'd') step(FPS);
    else if (k === 'k') togglePlay();
    else if (k === 'A') acceptRest();
    else if (k === 'n' || k === 'p') {
      const i = sum.clips.findIndex((c) => c.id === st.clip.id) + (k === 'n' ? 1 : -1);
      if (i >= 0 && i < sum.clips.length) openClip(sum.clips[i].id);
    }
    else if (k === 'Escape') deselect();
    else return;
    ev.preventDefault();
  };
  document.addEventListener('keydown', onKey);
  onLeave(() => { stopPlay(); clearInterval(poll); document.removeEventListener('keydown', onKey); });

  drawClips(); drawRun();
  const want = params.get('clip');
  openClip(sum.clips.find((c) => c.id === want) ? want : sum.clips[0].id);
}
