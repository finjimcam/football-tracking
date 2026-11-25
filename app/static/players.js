// Players tab (issue #16): each player's full-match path, heatmap and running stats.
// Data comes from app/players.py -> tools/player_paths.py, computed from tracks.parquet
// after track_identities.py, so Lines-tab identity anchors are already applied.

const SVGNS = 'http://www.w3.org/2000/svg';
const PAD = 4; // metres of margin around the pitch
const BAND_LABEL = { walk: 'Walk <2', jog: 'Jog 2–4', run: 'Run 4–5.5', high: 'High 5.5–7', sprint: 'Sprint 7+' };
const BAND_COL = { walk: '#3A4350', jog: '#4C6A8F', run: '#4C8DF6', high: '#E8A33A', sprint: '#EF5B5B' };

function svg(tag, attrs) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null) el.setAttribute(k, v);
  return el;
}

function pitchLines(L, W) {
  const g = svg('g', { class: 'pl-lines' });
  const r = (x, y, w, hgt) => g.append(svg('rect', { x, y, width: w, height: hgt }));
  r(0, 0, L, W);
  g.append(svg('line', { x1: L / 2, y1: 0, x2: L / 2, y2: W }));
  g.append(svg('circle', { cx: L / 2, cy: W / 2, r: 9.15 }));
  g.append(svg('circle', { cx: L / 2, cy: W / 2, r: 0.3, class: 'pl-spot' }));
  for (const [x0, dir] of [[0, 1], [L, -1]]) {
    const box = (d, w) => r(dir > 0 ? x0 : x0 - d, (W - w) / 2, d, w);
    box(16.5, 40.32);
    box(5.5, 18.32);
    g.append(svg('circle', { cx: x0 + dir * 11, cy: W / 2, r: 0.3, class: 'pl-spot' }));
    r(dir > 0 ? x0 - 2 : x0, (W - 7.32) / 2, 2, 7.32);
  }
  return g;
}

function clockOf(frame, sum) {
  if (sum.kickoff_frame == null) return `f${frame}`;
  const s = Math.max(0, (frame - sum.kickoff_frame) / sum.fps);
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
}

export async function viewPlayers(view, parts, params, ui) {
  const { h, api, toast, renderTabs, setCrumb, pill } = ui;
  renderTabs('players');
  const matchId = parts[1];

  if (!matchId) {
    setCrumb(null);
    const data = await api('/matches');
    const ok = data.matches.filter((m) => m.pipeline === 'broadcast');
    view.replaceChildren(h('section', { class: 'panel' },
      h('div', { class: 'panel-head' }, h('h1', {}, 'Players'), h('span', { class: 'muted' }, 'Pick a match')),
      ok.length ? h('div', {}, ok.map((m) => h('a', { class: 'match-row', href: '#/players/' + encodeURIComponent(m.match_id) },
        h('div', { class: 'match-title' }, `${m.team_a_name} v ${m.team_b_name}`),
        m.artifacts.player_map.exists ? pill('ready', 'ready') : pill('blocked', 'needs tracking'))))
        : h('div', { class: 'empty' }, 'No broadcast matches yet.')));
    return;
  }

  let match;
  try { match = await api('/matches/' + encodeURIComponent(matchId)); } catch (e) {
    view.replaceChildren(h('div', { class: 'placeholder' }, h('h1', {}, 'Match not found'), h('p', {}, e.message)));
    return;
  }
  setCrumb(match);
  view.replaceChildren(h('div', { class: 'placeholder' }, h('h1', {}, 'Loading players…'),
    h('p', {}, 'The first load reads every tracked position in the match (a few seconds).')));
  let sum;
  try { sum = await api(`/matches/${encodeURIComponent(matchId)}/players`); } catch (e) {
    view.replaceChildren(h('div', { class: 'placeholder' }, h('h1', {}, 'No player data yet'), h('p', {}, e.message),
      h('a', { class: 'btn', href: '#/match/' + encodeURIComponent(matchId) }, 'Open the pipeline')));
    return;
  }

  const [L, W] = sum.pitch;
  const st = {
    pid: params.get('p') && sum.players.some((p) => p.id === params.get('p')) ? params.get('p') : sum.players.find((p) => p.role !== 'GK').id,
    start: Number(params.get('from')) || sum.start,
    end: Number(params.get('to')) || sum.end,
    show: params.get('show') || 'both',
    data: null, token: 0,
  };
  st.start = Math.max(sum.start, Math.min(st.start, sum.end - 1));
  st.end = Math.min(sum.end, Math.max(st.end, st.start + 1));

  // ---------- layout ----------
  const list = h('div', { class: 'scroll pl-list' });
  const head = h('div', { class: 'pl-head' });
  const pitch = svg('svg', { viewBox: `${-PAD} ${-PAD} ${L + 2 * PAD} ${W + 2 * PAD}`, class: 'pl-pitch', role: 'img' });
  const statsBox = h('div', { class: 'pl-stats' });
  const note = h('div', { class: 'note', role: 'status', 'aria-live': 'polite' });

  const minF = sum.start, maxF = sum.end;
  const fromIn = h('input', { type: 'range', min: minF, max: maxF, step: sum.fps * 15, value: st.start, 'aria-label': 'From' });
  const toIn = h('input', { type: 'range', min: minF, max: maxF, step: sum.fps * 15, value: st.end, 'aria-label': 'To' });
  const winLabel = h('span', { class: 'mono pl-win' });
  const presets = h('div', { class: 'pl-presets' });
  const ko = sum.kickoff_frame ?? sum.start;
  const preset = (label, a, b) => h('button', { type: 'button', class: 'btn small', onclick: () => setWindow(a, b) }, label);
  presets.append(preset('All', minF, maxF));
  for (let m = 0; m * 60 * sum.fps + ko < maxF; m += 15) {
    const a = Math.max(minF, Math.round(ko + m * 60 * sum.fps));
    const b = Math.min(maxF, Math.round(ko + (m + 15) * 60 * sum.fps));
    if (b - a > sum.fps * 60) presets.append(preset(`${m}–${m + 15 >= 45 && b === maxF ? 'end' : m + 15}'`, a, b));
  }
  const showSel = h('div', { class: 'seg', role: 'group', 'aria-label': 'Show' },
    ['both', 'heatmap', 'path'].map((k) => h('button', {
      type: 'button', class: 'btn small', 'data-k': k, 'aria-pressed': String(st.show === k),
      onclick: () => { st.show = k; showSel.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.k === k))); syncHash(); drawPitch(); },
    }, k === 'both' ? 'Both' : k === 'heatmap' ? 'Heatmap' : 'Path')));

  view.replaceChildren(h('div', { class: 'pl-layout' },
    h('section', { class: 'panel pl-side', 'aria-labelledby': 'h-pl' },
      h('div', { class: 'panel-head' }, h('h1', { id: 'h-pl' }, 'Players'),
        h('span', { class: 'muted mono', style: 'font-size:11px' }, `${sum.start_clock || ''}–${sum.end_clock || ''}`)),
      list),
    h('section', { class: 'panel pl-main', 'aria-label': 'Player detail' },
      h('div', { class: 'panel-head' }, head, h('div', { class: 'spacer' }), showSel),
      h('div', { class: 'pl-body' },
        h('div', { class: 'pl-pitch-wrap' }, pitch,
          h('div', { class: 'pl-window' },
            h('div', { class: 'pl-sliders' }, h('label', {}, 'From', fromIn), h('label', {}, 'To', toIn), winLabel),
            presets),
          note),
        statsBox))));

  // ---------- player list ----------
  function drawList() {
    const rows = [];
    for (const team of ['team_a', 'team_b']) {
      rows.push(h('div', { class: 'pl-team ' + team }, sum.teams[team]));
      for (const p of sum.players.filter((q) => q.team === team)) {
        rows.push(h('a', {
          class: 'pl-row' + (p.id === st.pid ? ' sel' : ''), href: '#', 'aria-current': p.id === st.pid ? 'true' : null,
          onclick: (ev) => { ev.preventDefault(); st.pid = p.id; syncHash(); drawList(); load(); },
        },
        h('span', { class: 'pl-num ' + team }, p.number ?? '?'),
        h('span', { class: 'pl-name' }, p.name || p.id, p.role === 'GK' ? h('span', { class: 'dim' }, ' GK') : null,
          p.manual ? h('span', { class: 'pl-manual', title: `Named from ${p.anchors} Lines-tab anchor(s)` }, '●') : null),
        h('span', { class: 'mono dim pl-km' }, (p.distance_m / 1000).toFixed(2) + ' km'),
        h('span', { class: 'mono dim pl-cov', title: 'Share of the window this player was tracked' }, Math.round(p.coverage * 100) + '%')));
      }
    }
    list.replaceChildren(...rows);
  }

  // ---------- window ----------
  function setWindow(a, b) {
    st.start = Math.max(minF, Math.min(a, maxF - 1));
    st.end = Math.min(maxF, Math.max(b, st.start + 1));
    fromIn.value = st.start; toIn.value = st.end;
    syncHash(); load();
  }
  function showWindow(a = st.start, b = st.end) { winLabel.textContent = `${clockOf(a, sum)} – ${clockOf(b, sum)}`; }
  let debounce;
  for (const el of [fromIn, toIn]) {
    el.addEventListener('input', () => {
      if (Number(fromIn.value) >= Number(toIn.value)) {
        if (el === fromIn) fromIn.value = Number(toIn.value) - sum.fps * 15; else toIn.value = Number(fromIn.value) + sum.fps * 15;
      }
      showWindow(Number(fromIn.value), Number(toIn.value));
      clearTimeout(debounce);
      debounce = setTimeout(() => setWindow(Number(fromIn.value), Number(toIn.value)), 250);
    });
  }

  function syncHash() {
    const q = new URLSearchParams({ p: st.pid });
    if (st.start !== sum.start) q.set('from', st.start);
    if (st.end !== sum.end) q.set('to', st.end);
    if (st.show !== 'both') q.set('show', st.show);
    history.replaceState(null, '', `#/players/${encodeURIComponent(matchId)}?${q}`);
  }

  // ---------- pitch ----------
  function drawPitch() {
    const d = st.data;
    pitch.replaceChildren(svg('rect', { x: -PAD, y: -PAD, width: L + 2 * PAD, height: W + 2 * PAD, class: 'pl-grass' }));
    if (!d) { pitch.append(pitchLines(L, W)); return; }
    const team = d.player.team;
    if (st.show !== 'path' && d.heatmap.max > 0) {
      const hm = d.heatmap, c = hm.cell_m, g = svg('g', { class: 'pl-heat' });
      // sqrt scale so a winger's long, thin coverage still shows next to their hot spots
      for (let j = 0; j < hm.ny; j++) {
        for (let i = 0; i < hm.nx; i++) {
          const v = hm.secs[j][i];
          if (v <= 0) continue;
          const a = Math.sqrt(v / hm.max);
          g.append(svg('rect', { x: i * c, y: j * c, width: c + 0.05, height: c + 0.05,
            fill: a > 0.66 ? '#EF5B5B' : a > 0.33 ? '#F08C2E' : '#FFD84D', 'fill-opacity': (0.15 + 0.75 * a).toFixed(2) }));
        }
      }
      pitch.append(g);
    }
    pitch.append(pitchLines(L, W));
    if (st.show !== 'heatmap') {
      const g = svg('g', { class: 'pl-path ' + team });
      for (const run of d.path) g.append(svg('polyline', { points: run.map((p) => `${p[1]},${p[2]}`).join(' ') }));
      pitch.append(g);
    }
    // team shape: everyone's average position in this window
    for (const q of d.team_shape) {
      const me = q.id === d.player.id;
      pitch.append(svg('circle', { cx: q.pos[0], cy: q.pos[1], r: me ? 2.2 : 1.6, class: 'pl-dot ' + team + (me ? ' me' : '') }));
      const t = svg('text', { x: q.pos[0], y: q.pos[1] + 0.7, class: 'pl-dot-num' + (me ? ' me' : '') });
      t.textContent = q.number ?? '';
      pitch.append(t);
    }
    // attacking direction
    const right = d.attacks_x > L / 2;
    const ax = L / 2 + (right ? 8 : -8);
    pitch.append(svg('path', { d: `M ${L / 2 - (right ? 8 : -8)} ${W + 2} L ${ax} ${W + 2} M ${ax} ${W + 2} l ${right ? -1.5 : 1.5} -1 M ${ax} ${W + 2} l ${right ? -1.5 : 1.5} 1`, class: 'pl-arrow' }));
    const t = svg('text', { x: L / 2, y: W + 3.6, class: 'pl-arrow-label' });
    t.textContent = 'attacking';
    pitch.append(t);
    pitch.setAttribute('aria-label', `${d.player.name || d.player.id}: ${st.show === 'path' ? 'path' : st.show === 'heatmap' ? 'heatmap' : 'heatmap and path'} on the pitch`);
  }

  // ---------- stats ----------
  function drawStats() {
    const d = st.data;
    if (!d) { statsBox.replaceChildren(); return; }
    const s = d.stats, p = d.player;
    head.replaceChildren(
      h('span', { class: 'pl-num big ' + p.team }, p.number ?? '?'),
      h('div', {}, h('h2', { style: 'margin:0;font-size:15px' }, p.name || p.id),
        h('div', { class: 'muted', style: 'font-size:11px' }, sum.teams[p.team], p.role ? ` · ${p.role}` : '')));
    const kmh = (v) => (v * 3.6).toFixed(1);
    const tile = (label, value, unit, title) => h('div', { class: 'pl-tile', title },
      h('div', { class: 'pl-tile-v' }, value, unit ? h('span', { class: 'pl-unit' }, unit) : null), h('div', { class: 'pl-tile-l' }, label));
    const total = Object.values(s.bands_m).reduce((a, b) => a + b, 0) || 1;
    statsBox.replaceChildren(
      h('div', { class: 'pl-window-label mono' }, `${d.start_clock || clockOf(d.start, sum)} – ${d.end_clock || clockOf(d.end, sum)}`),
      h('div', { class: 'pl-tiles' },
        tile('Distance', (s.distance_m / 1000).toFixed(2), 'km'),
        tile('Tracked', s.tracked_min.toFixed(0), 'min', `${Math.round(s.coverage * 100)}% of the window`),
        tile('Per 90', s.per_90_m == null ? '–' : (s.per_90_m / 1000).toFixed(1), s.per_90_m == null ? null : 'km', 'Distance per tracked minute, scaled to 90'),
        tile('Avg speed', kmh(s.avg_speed), 'km/h'),
        tile('Top speed', kmh(s.top_speed), 'km/h', '99th percentile of smoothed speed'),
        tile('Sprints', s.sprints, null, '7+ m/s (25 km/h) held for at least 1 s')),
      h('div', { class: 'pl-bands-h muted' }, 'Distance by speed (m/s)'),
      h('div', { class: 'pl-bands', role: 'img', 'aria-label': Object.entries(s.bands_m).map(([k, v]) => `${BAND_LABEL[k]}: ${Math.round(v)} m`).join(', ') },
        Object.entries(s.bands_m).map(([k, v]) => h('span', { style: `flex:${v / total};background:${BAND_COL[k]}`, title: `${BAND_LABEL[k]}: ${Math.round(v)} m` }))),
      h('div', { class: 'pl-bands-key' }, Object.entries(s.bands_m).map(([k, v]) =>
        h('span', {}, h('i', { style: `background:${BAND_COL[k]}` }), BAND_LABEL[k], h('b', { class: 'mono' }, ` ${Math.round(v)} m`)))),
      h('p', { class: 'muted pl-src' },
        p.manual
          ? `Identity confirmed by hand (${p.anchors} anchor${p.anchors === 1 ? '' : 's'} in the Lines tab).`
          : 'Name is role-inferred from average position. If it looks wrong, fix it in Lines → Identify and press "Apply to tracking".',
        p.tracklets ? ` Built from ${p.tracklets} tracklets; ${p.fragile_links} of the joins were uncertain${p.fragile_links > 10 ? ', so parts of this path may belong to a team-mate' : ''}.` : '',
        ' Coverage gaps (', Math.round((1 - s.coverage) * 100), '% here) are left out, so distance is a lower bound.'));
  }

  async function load() {
    const tok = ++st.token;
    note.textContent = 'Loading…';
    showWindow();
    try {
      const d = await api(`/matches/${encodeURIComponent(matchId)}/players/${encodeURIComponent(st.pid)}?start=${st.start}&end=${st.end}`);
      if (tok !== st.token) return;
      st.data = d;
      note.textContent = '';
      drawPitch(); drawStats();
    } catch (e) {
      if (tok !== st.token) return;
      note.textContent = e.message;
      toast(e.message, true);
    }
  }

  drawList();
  drawPitch();
  syncHash();
  load();
}
