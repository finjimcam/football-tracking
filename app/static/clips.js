// Clips panel of the Lines tab (issue #58): set pieces, shots & key chances and
// passes into the final third, found by tools/detect_clips.py. Watch each one,
// keep or drop it, trim it, and export a team's kept clips as a zip of MP4s
// (tools/export_clips.py), in folders per heading.

const KEYS_HELP = [
  ['k', 'play / pause'], ['a / d', '−/+ 0.5 s'], ['A / D', '−/+ 2 s'], [', / .', '−/+ 1 frame'],
  ['[ / ]', 'start / end the clip here'], ['e', 'keep / drop this clip'], ['n / p', 'next / previous clip'],
];
const HOW = {
  'stoppage, dead ball': 'play stopped, then the ball was kicked from still',
  'stoppage, players': 'play stopped; told from where the players stood',
  'dead ball': 'the ball was kicked from still',
};

export async function viewClips(view, matchId, match, team, params, ui) {
  const { h, api, toast, setCrumb, onLeave } = ui;
  setCrumb(match);
  const enc = encodeURIComponent(matchId);
  const modeSeg = (cur) => h('div', { class: 'seg', role: 'group', 'aria-label': 'Lines tab view' },
    h('a', { class: 'btn small', href: `#/lines/${enc}?team=${team}`, 'aria-pressed': String(cur === 'phases') }, 'Phases'),
    h('a', { class: 'btn small', href: `#/lines/${enc}?team=${team}&view=clips`, 'aria-pressed': String(cur === 'clips') }, 'Clips'));
  const queueDetect = async () => {
    try {
      const job = await api('/jobs', { method: 'POST', body: { stage: 'b_clips', match_id: matchId, params: {} } });
      toast('Detect clips queued (a few seconds once it starts)');
      location.hash = `#/match/${enc}?job=${job.id}`;
    } catch (e) { toast(e.message, true); }
  };

  let sum;
  try { sum = await api(`/matches/${enc}/clips?team=${team}`); } catch (e) {
    view.replaceChildren(h('div', { class: 'placeholder' }, modeSeg('clips'), h('h1', {}, 'No clips yet'), h('p', {}, e.message),
      h('div', { style: 'display:flex;gap:8px;justify-content:center' },
        h('button', { type: 'button', class: 'btn primary', onclick: queueDetect }, 'Detect clips'),
        h('a', { class: 'btn', href: '#/match/' + enc }, 'Open the pipeline'))));
    return;
  }

  const FPS = sum.fps;
  const W = sum.width || 1920, H = sum.height || 1080;
  const IMG_W = Math.min(W, 1280);
  const all = () => sum.groups.flatMap((g) => g.kinds.flatMap((k) => k.clips));
  const st = { clip: null, cur: 0, playing: false };
  const open = new Set(JSON.parse(sessionStorageGet('clips-open') || '["set_pieces"]'));

  // ---------- layout ----------
  const teamSel = h('select', { 'aria-label': 'Team', class: 'mini-select', onchange: () => {
    location.hash = `#/lines/${enc}?team=${teamSel.value}&view=clips`;
  } }, ['team_a', 'team_b'].map((t) => h('option', { value: t, selected: t === team || null },
    t === 'team_a' ? match.team_a_name : match.team_b_name)));
  const list = h('div', { class: 'scroll lines-list clips-list' });
  const total = h('span', { class: 'muted mono', style: 'font-size:11px' });
  const title = h('h1', { style: 'margin:0;font-size:14px' }, 'Pick a clip');
  const sub = h('span', { class: 'muted mono', style: 'font-size:12px' });
  const img = h('img', { alt: 'Video frame', class: 'lines-img', draggable: 'false' });
  const stage = h('div', { class: 'lines-stage', style: `aspect-ratio:${W}/${H}` }, img);
  const strip = h('canvas', { class: 'lines-strip', height: 22, 'aria-label': 'Clip timeline: click to seek', role: 'img' });
  const btn = (label, aria, fn, cls) => h('button', { type: 'button', class: 'btn small ' + (cls || ''), 'aria-label': aria, title: aria, onclick: fn }, label);
  const playBtn = btn('▶', 'Play (k)', () => togglePlay());
  const transport = h('div', { class: 'lines-tools' },
    btn('|◀', 'Go to the clip start', () => { stopPlay(); seek(curEx().in); }),
    btn('−2s', 'Back 2 seconds', () => step(-2 * FPS)), btn('−½s', 'Back half a second', () => step(-FPS / 2)),
    btn('−1f', 'Previous frame', () => step(-1)), playBtn, btn('+1f', 'Next frame', () => step(1)),
    btn('+½s', 'Forward half a second', () => step(FPS / 2)), btn('+2s', 'Forward 2 seconds', () => step(2 * FPS)),
    h('div', { class: 'spacer' }),
    btn('Previous', 'Previous clip (p)', () => gotoIndex(idx() - 1)), btn('Next', 'Next clip (n)', () => gotoIndex(idx() + 1)));
  const keepBox = h('input', { type: 'checkbox', id: 'clip-keep', style: 'width:18px;height:18px;accent-color:var(--accent)',
    onchange: () => setExport(st.clip.id, { keep: keepBox.checked }) });
  const exportInfo = h('span', { class: 'mono', style: 'font-size:12px' });
  const clipTools = h('div', { class: 'lines-tools', role: 'group', 'aria-label': 'Export this clip' },
    h('label', { for: 'clip-keep', style: 'display:inline-flex;align-items:center;gap:6px;font-size:13px' }, keepBox, 'Keep this clip'),
    btn('Start here [', 'Start the exported clip at this frame ([)', () => setExport(st.clip.id, { trim_in: st.cur })),
    btn('End here ]', 'End the exported clip at this frame (])', () => setExport(st.clip.id, { trim_out: st.cur })),
    btn('Full clip', 'Undo the trim', () => setExport(st.clip.id, { reset_trim: true })),
    h('div', { class: 'spacer' }), exportInfo);
  const msg = h('div', { class: 'note', role: 'status', 'aria-live': 'polite', style: 'min-height:18px' });

  const exportSummary = h('span', { class: 'muted mono', style: 'font-size:11px' });
  const exportBtn = h('button', { type: 'button', class: 'btn small primary', onclick: async () => {
    const kept = all().filter((c) => c.export.keep).length;
    if (!kept) { toast('Every clip is dropped: tick at least one to export', true); return; }
    try {
      const job = await api('/jobs', { method: 'POST', body: { stage: 'b_clips_export', match_id: matchId, params: { team } } });
      toast(`Export of ${kept} clip${kept === 1 ? '' : 's'} queued: the zip appears on the match page when it's done`);
      location.hash = `#/match/${enc}?job=${job.id}`;
    } catch (e) { toast(e.message, true); }
  } }, 'Export clips (zip)');
  const zipLink = sum.zip_exists
    ? h('a', { class: 'btn small', href: `/api/matches/${enc}/files/event_clips_zip?team=${team}`, download: '' }, 'Download last zip')
    : null;

  view.replaceChildren(h('div', { class: 'cols' },
    h('div', { class: 'col-side lines-side' },
      h('section', { class: 'panel', 'aria-labelledby': 'h-clips' },
        h('div', { class: 'panel-head' }, h('h1', { id: 'h-clips' }, 'Clips'), total, h('div', { class: 'spacer' }), modeSeg('clips'), teamSel),
        list),
      h('section', { class: 'panel' }, h('div', { class: 'panel-body', style: 'display:flex;flex-direction:column;gap:10px' },
        h('div', { style: 'display:flex;align-items:center;gap:8px' }, h('span', { class: 'section-label' }, 'Export'), exportSummary),
        h('div', { style: 'display:flex;align-items:center;gap:8px;flex-wrap:wrap' }, exportBtn, zipLink,
          btn('Re-detect', 'Run Detect clips again (keeps your choices for clips it finds again)', queueDetect)),
        h('div', { class: 'muted', style: 'font-size:12px;line-height:1.4' },
          `Clips for ${sum.team_name}, found from the tracking (the ball is often lost, so some are missed or mislabelled: untick those). `,
          'Export makes one MP4 per kept clip, in folders per heading, in a zip you can drop into PowerPoint.'),
        h('details', {}, h('summary', { class: 'muted', style: 'cursor:pointer' }, 'Controls'),
          h('dl', { class: 'kv', style: 'margin-top:8px' }, KEYS_HELP.flatMap(([k, v]) => [h('dt', {}, k), h('dd', { style: 'font-family:var(--sans)' }, v)])))))),
    h('div', { class: 'col-main' },
      h('section', { class: 'panel' },
        h('div', { class: 'panel-head' }, title, sub),
        h('div', { class: 'panel-body', style: 'display:flex;flex-direction:column;gap:8px' },
          stage, strip, transport, clipTools, msg)))));

  // ---------- list ----------
  const fmtS = (f0, f1) => `${((f1 - f0 + 1) / FPS).toFixed(0)}s`;
  function clipLabel(c) {
    if (c.kind === 'pass') return [c.player_name, c.to_player_name].filter(Boolean).join(' → ');
    const p = c.published;
    if (p) return [p.type === 'goal' ? 'GOAL' : null, c.player_name, p.outcome && p.type !== 'goal' ? p.outcome : null, `${p.minute} published`]
      .filter(Boolean).join(' · ');
    return c.player_name || '';
  }
  function bulk(keep, group, kind) {
    return async (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      try {
        await api(`/matches/${enc}/clips/export`, { method: 'PUT', body: { team, keep, group, kind } });
        sum = await api(`/matches/${enc}/clips?team=${team}`);
        if (st.clip) st.clip = all().find((c) => c.id === st.clip.id) || st.clip;
        drawList();
        draw();
      } catch (e) { toast(e.message, true); }
    };
  }
  function drawList() {
    const clips = all();
    const kept = clips.filter((c) => c.export.keep);
    total.textContent = `${clips.length} found`;
    const secs = kept.reduce((a, c) => a + (c.export.out - c.export.in + 1) / FPS, 0);
    exportSummary.textContent = `${kept.length}/${clips.length} kept · ${Math.floor(secs / 60)}:${String(Math.round(secs % 60)).padStart(2, '0')}`;
    list.replaceChildren(...sum.groups.map((g) => {
      const n = g.kinds.reduce((a, k) => a + k.clips.length, 0);
      const det = h('details', { class: 'clips-group', open: open.has(g.group) || null,
        ontoggle: () => { if (det.open) open.add(g.group); else open.delete(g.group); sessionStorageSet('clips-open', JSON.stringify([...open])); } },
        h('summary', { class: 'clips-head' }, h('span', {}, g.title), h('span', { class: 'muted mono' }, String(n)),
          h('span', { class: 'spacer' }),
          h('button', { type: 'button', class: 'btn tiny', onclick: bulk(true, g.group) }, 'Keep all'),
          h('button', { type: 'button', class: 'btn tiny', onclick: bulk(false, g.group) }, 'Drop all')),
        ...g.kinds.map((k) => h('div', { class: 'clips-kind' },
          h('div', { class: 'clips-subhead' }, h('span', {}, k.title), h('span', { class: 'muted mono' }, String(k.clips.length)),
            k.clips.length && g.kinds.length > 1 ? h('span', { class: 'spacer' }) : null,
            k.clips.length && g.kinds.length > 1 ? h('button', { type: 'button', class: 'btn tiny', onclick: bulk(true, g.group, k.kind) }, 'Keep') : null,
            k.clips.length && g.kinds.length > 1 ? h('button', { type: 'button', class: 'btn tiny', onclick: bulk(false, g.group, k.kind) }, 'Drop') : null),
          k.clips.length ? k.clips.map((c) => h('div', { class: 'lines-row' + (c.export.keep ? '' : ' dropped') },
            h('input', { type: 'checkbox', checked: c.export.keep, 'aria-label': `Keep ${k.title} at ${c.clock}`,
              onchange: (ev) => setExport(c.id, { keep: ev.target.checked }) }),
            h('button', { type: 'button', class: 'job-row', 'aria-current': st.clip && st.clip.id === c.id ? 'true' : null,
              onclick: () => gotoClip(c.id) },
            h('span', { style: 'font-weight:600' }, h('span', { class: 'mono' }, c.clock || `f${c.frame}`),
              h('span', { class: 'dim', style: 'font-weight:400;margin-left:6px' }, fmtS(c.export.in, c.export.out))),
            h('span', { class: 'sub' }, [clipLabel(c), c.team ? null : 'team unknown', c.export.trimmed ? 'trimmed' : null].filter(Boolean).join(' · ') || ' '))))
            : h('div', { class: 'muted clips-none' }, 'None found'))));
      return det;
    }));
    const cur = list.querySelector('[aria-current="true"]');
    if (cur) cur.scrollIntoView({ block: 'nearest' });
  }

  // ---------- export ----------
  async function setExport(id, body) {
    try {
      const r = await api(`/matches/${enc}/clips/${encodeURIComponent(id)}/export`, { method: 'PUT', body: { team, ...body } });
      const c = all().find((q) => q.id === id);
      if (c) c.export = r;
      if (st.clip && st.clip.id === id) st.clip.export = r;
      drawList();
      draw();
    } catch (e) { toast(e.message, true); }
  }
  const curEx = () => (st.clip ? st.clip.export : { in: 0, out: 0 });

  // ---------- navigation ----------
  const idx = () => all().findIndex((c) => st.clip && c.id === st.clip.id);
  function gotoIndex(i) {
    const clips = all();
    if (i >= 0 && i < clips.length) gotoClip(clips[i].id);
  }
  function gotoClip(id) {
    const c = all().find((q) => q.id === id);
    if (!c) return;
    stopPlay();
    st.clip = c;
    const g = sum.groups.find((x) => x.group === c.group);
    if (g && !open.has(g.group)) { open.add(g.group); }
    st.cur = c.export.in;
    history.replaceState(null, '', `#/lines/${enc}?team=${team}&view=clips&clip=${encodeURIComponent(id)}`);
    const kind = (g && g.kinds.find((k) => k.kind === c.kind)) || { title: c.kind };
    const how = c.how ? ` · ${HOW[c.how] || c.how}` : '';
    msg.textContent = c.kind === 'pass' && c.end_x != null
      ? `Pass from (${c.x}, ${c.y}) to (${c.end_x}, ${c.end_y}) m`
      : (c.how === 'stoppage, players' ? HOW[c.how] : c.x != null ? `Ball at (${c.x}, ${c.y}) m${how}` : how.replace(/^ · /, ''));
    title.textContent = `${c.published && c.published.type === 'goal' ? 'Goal' : kind.title.replace(/s$/, '')} · ${c.clock || 'f' + c.frame}`;
    if (c.published) {
      const p = c.published;
      msg.textContent = [`${p.source || 'Published'}: ${p.minute}`, p.player, p.outcome, p.detail].filter(Boolean).join(' · ')
        + (c.how ? ` · ${HOW[c.how] || c.how}` : ' · matched to a detected moment');
    }
    drawList();
    draw();
    togglePlay();
  }

  // ---------- frames / playback ----------
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
    pre.src = `/api/matches/${enc}/frame/${f}.jpg?w=${IMG_W}`;
  }
  function draw() {
    const c = st.clip;
    if (!c) return;
    sub.textContent = `frame ${st.cur} · ${((st.cur - c.start) / FPS).toFixed(1)} s`;
    const ex = c.export;
    keepBox.checked = ex.keep;
    exportInfo.textContent = ex.keep ? `${((ex.out - ex.in + 1) / FPS).toFixed(1)} s${ex.trimmed ? ' · trimmed' : ''}` : 'dropped';
    drawStrip();
    loadFrame(st.cur);
  }
  function drawStrip() {
    const c = st.clip;
    const w = strip.clientWidth || 600;
    strip.width = w * devicePixelRatio;
    strip.height = 22 * devicePixelRatio;
    const g = strip.getContext('2d');
    g.scale(devicePixelRatio, devicePixelRatio);
    const x = (f) => ((f - c.start) / Math.max(c.end - c.start, 1)) * w;
    g.fillStyle = '#2A2F36';
    g.fillRect(0, 4, w, 14);
    g.fillStyle = c.export.keep ? '#4DA3FF' : '#5A6068';
    g.fillRect(x(c.export.in), 4, Math.max(2, x(c.export.out) - x(c.export.in)), 14);
    g.fillStyle = '#FFD84D';
    g.fillRect(x(c.frame) - 1, 0, 2, 22);  // the moment itself (restart / shot / pass)
    g.fillStyle = '#FFFFFF';
    g.fillRect(x(st.cur) - 1, 0, 3, 22);
  }
  strip.addEventListener('pointerdown', (ev) => {
    if (!st.clip) return;
    const r = strip.getBoundingClientRect();
    stopPlay();
    seek(st.clip.start + ((ev.clientX - r.left) / r.width) * (st.clip.end - st.clip.start));
  });
  function seek(f) {
    if (!st.clip) return;
    st.cur = Math.max(st.clip.start, Math.min(st.clip.end, Math.round(f)));
    draw();
  }
  function step(n) { stopPlay(); seek(st.cur + n); }
  function stopPlay() { st.playing = false; playBtn.textContent = '▶'; playBtn.setAttribute('aria-label', 'Play (k)'); }
  function togglePlay() {
    if (st.playing) { stopPlay(); return; }
    if (!st.clip) return;
    if (st.cur >= curEx().out) st.cur = curEx().in;
    st.playing = true;
    lastTick = 0;
    playBtn.textContent = '❚❚';
    playBtn.setAttribute('aria-label', 'Pause (k)');
    playTick();
  }
  let lastTick = 0;
  function playTick() {
    if (!st.playing || !st.clip) return;
    if (st.cur >= curEx().out) { stopPlay(); return; }
    const now = performance.now();
    const adv = lastTick ? Math.max(1, Math.min(6, Math.round(((now - lastTick) / 1000) * FPS))) : 1;
    lastTick = now;
    seek(st.cur + adv);
  }

  // ---------- keys ----------
  function onKey(ev) {
    if (ev.target.closest && ev.target.closest('input, select, textarea')) return;
    if (!st.clip) return;
    const k = ev.key;
    const acts = {
      k: () => togglePlay(), a: () => step(-FPS / 2), d: () => step(FPS / 2), A: () => step(-2 * FPS), D: () => step(2 * FPS),
      ',': () => step(-1), '.': () => step(1), '[': () => setExport(st.clip.id, { trim_in: st.cur }),
      ']': () => setExport(st.clip.id, { trim_out: st.cur }), e: () => setExport(st.clip.id, { keep: !st.clip.export.keep }),
      n: () => gotoIndex(idx() + 1), p: () => gotoIndex(idx() - 1),
    };
    if (acts[k]) { ev.preventDefault(); acts[k](); }
  }
  document.addEventListener('keydown', onKey);
  onLeave(() => { stopPlay(); document.removeEventListener('keydown', onKey); });

  drawList();
  const want = params.get('clip');
  const first = (want && all().find((c) => c.id === want)) || null;
  if (first) gotoClip(first.id);
  else msg.textContent = all().length ? 'Pick a clip from the list.' : 'No clips found for this team.';
}

function sessionStorageGet(k) { try { return sessionStorage.getItem(k); } catch (e) { return null; } }
function sessionStorageSet(k, v) { try { sessionStorage.setItem(k, v); } catch (e) { /* private mode */ } }
