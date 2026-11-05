// Lines tab: pick each phase's back line and midfield line (issue #32).
// Browser twin of tools/select_shape_players.py. Chain logic lives server-side
// (app/lines.py -> cut_phase_clips.py), so this page draws exactly what the
// renderer will draw.

const KEYS_HELP = [
  ['Click player', 'cycle none → back → midfield'],
  ['Right-click / Re-link mode', 're-attach a lost chain to this player'],
  ['x / Fix swap mode', 'click the chain that jumped to the wrong player, then the right player'],
  ['a / d', '−/+ 0.5 s'], ['A / D', '−/+ 2 s'], [', / .', '−/+ 1 frame'], ['k', 'play / pause'],
  ['Enter', 'save + next phase'], ['s', 'skip phase (no lines)'], ['r', 'reset to auto guess here'],
  ['c', 'clear picks'], ['p', 'previous phase'],
  ['[ / ]', 'clip export: start / end the clip here'], ['e', 'clip export: keep / drop this phase'],
  ['i / Identify mode', 'click a player, or drag a box round an unidentified one, then pick who it is'],
  ['Shift-click / long-press', 'change who a player is without leaving line picking'],
];
const STRIP_ROW = 12;
const COL = { seen: '#5DBB63', filled: '#E8A33A', lost: '#7A2E2E', link: '#FFFFFF', head: '#FFD84D' };
const SVGNS = 'http://www.w3.org/2000/svg';

function svg(tag, attrs) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null) el.setAttribute(k, v);
  return el;
}

export async function viewLines(view, parts, params, ui) {
  const { h, api, toast, every, renderTabs, setCrumb, pill, onLeave } = ui;
  renderTabs('lines');
  const matchId = parts[1];
  const team = params.get('team') === 'team_b' ? 'team_b' : 'team_a';

  if (!matchId) {
    setCrumb(null);
    const data = await api('/matches');
    const ok = data.matches.filter((m) => m.pipeline === 'broadcast');
    view.replaceChildren(h('section', { class: 'panel' },
      h('div', { class: 'panel-head' }, h('h1', {}, 'Lines'), h('span', { class: 'muted' }, 'Pick a match')),
      ok.length ? h('div', {}, ok.map((m) => h('a', { class: 'match-row', href: '#/lines/' + encodeURIComponent(m.match_id) },
        h('div', { class: 'match-title' }, `${m.team_a_name} v ${m.team_b_name}`),
        m.artifacts.possession.exists ? pill('ready', 'ready') : pill('blocked', 'needs tracking'))))
        : h('div', { class: 'empty' }, 'No broadcast matches yet. Ingest a video that has a config/<id>_match.json, then run its pipeline down to "Compute possession".')));
    return;
  }

  let match;
  try { match = await api('/matches/' + encodeURIComponent(matchId)); } catch (e) {
    view.replaceChildren(h('div', { class: 'placeholder' }, h('h1', {}, 'Match not found'), h('p', {}, e.message)));
    return;
  }
  setCrumb(match);
  view.replaceChildren(h('div', { class: 'placeholder' }, h('h1', {}, 'Loading tracking…'),
    h('p', {}, 'The first load reads the whole half\'s tracks (a few seconds).')));
  let sum;
  try { sum = await api(`/matches/${encodeURIComponent(matchId)}/lines?team=${team}`); } catch (e) {
    view.replaceChildren(h('div', { class: 'placeholder' }, h('h1', {}, 'Not ready for line picking'), h('p', {}, e.message),
      h('a', { class: 'btn', href: '#/match/' + encodeURIComponent(matchId) }, 'Open the pipeline')));
    return;
  }

  const W = sum.width, H = sum.height, FPS = sum.fps;
  const IMG_W = Math.min(W, 1280);
  const st = { phase: null, cur: 0, slots: [], chains: [], dirty: false, playing: false, mode: 'pick', swapChain: null, busy: false, loadedFrame: null };

  // ---------- layout ----------
  const list = h('div', { class: 'scroll lines-list' });
  const progress = h('span', { class: 'muted mono', style: 'font-size:11px' });
  const teamSel = h('select', { 'aria-label': 'Team', class: 'mini-select', onchange: () => {
    location.hash = `#/lines/${encodeURIComponent(matchId)}?team=${teamSel.value}`;
  } }, ['team_a', 'team_b'].map((t) => h('option', { value: t, selected: t === team || null },
    t === team ? sum.team_name : (t === 'team_a' ? match.team_a_name : match.team_b_name))));

  const title = h('h1', { style: 'margin:0;font-size:14px' });
  const sub = h('span', { class: 'muted mono', style: 'font-size:12px' });
  const dirtyChip = h('span', { class: 'pill stale', hidden: true }, 'unsaved');
  const msg = h('div', { class: 'note', role: 'status', 'aria-live': 'polite', style: 'min-height:18px' });

  const img = h('img', { alt: 'Video frame', class: 'lines-img', draggable: 'false' });
  const overlay = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'lines-svg', role: 'img', 'aria-label': 'Tracked players and picked lines' });
  const stage = h('div', { class: 'lines-stage', style: `aspect-ratio:${W}/${H}` }, img, overlay);
  const strip = h('canvas', { class: 'lines-strip', 'aria-label': 'Chain timeline: click to seek', role: 'img' });
  const counts = h('span', { class: 'mono', style: 'font-size:12px' });

  const btn = (label, aria, fn, cls) => h('button', { type: 'button', class: 'btn small ' + (cls || ''), 'aria-label': aria, onclick: fn }, label);
  const playBtn = btn('▶', 'Play', () => togglePlay());
  const MODE_HELP = {
    pick: 'Shift-click (or long-press) a player to change who they are.',
    relink: 'Re-link: tap the player a lost (red/amber) chain should continue on.',
    swap: 'Fix swap: tap the chain that jumped to the wrong player (or its B/M row in the strip), then tap the right player.',
    identify: 'Identify: click a player, or drag a box round one with no number (?), then pick who it is. Dashed grey boxes are other detections (other team, keepers, players the tracker never put on a team): you can name them too. 27• = changed by you (or swapped by your pick), not applied to tracking yet; 27? = an old guess for a number you gave someone else.',
  };
  const modeBtns = {};
  function setMode(mode) {
    st.mode = mode;
    st.swapChain = null;
    st.idBox = null;
    overlay.style.touchAction = mode === 'identify' ? 'none' : '';
    if (typeof drawIdPanel === 'function') drawIdPanel();
    for (const [k, b] of Object.entries(modeBtns)) {
      b.setAttribute('aria-pressed', String(k === mode));
      b.classList.toggle('primary', k === mode);
    }
    msg.textContent = MODE_HELP[mode];
    if (st.phase) draw();
  }
  const modeGroup = h('div', { role: 'group', 'aria-label': 'Click mode', class: 'lines-tools' },
    [['pick', 'Pick'], ['relink', 'Re-link'], ['swap', 'Fix swap'], ['identify', 'Identify']].map(([k, label]) => {
      modeBtns[k] = h('button', { type: 'button', class: 'btn small' + (k === 'pick' ? ' primary' : ''),
        'aria-pressed': String(k === 'pick'), onclick: () => setMode(k) }, label);
      return modeBtns[k];
    }));

  const transport = h('div', { class: 'lines-tools' },
    btn('−2s', 'Back 2 seconds', () => step(-2 * FPS)), btn('−½s', 'Back half a second', () => step(-FPS / 2)),
    btn('−1f', 'Previous frame', () => step(-1)), playBtn, btn('+1f', 'Next frame', () => step(1)),
    btn('+½s', 'Forward half a second', () => step(FPS / 2)), btn('+2s', 'Forward 2 seconds', () => step(2 * FPS)),
    h('div', { class: 'spacer' }), modeGroup);
  const actions = h('div', { class: 'lines-tools' },
    btn('Save + next', 'Save picks and go to the next phase (Enter)', () => save(false), 'primary'),
    btn('Skip phase', 'Save as no lines (s)', () => save(true)),
    btn('Auto guess here', 'Reset to the automatic guess on this frame (r)', () => edit('auto')),
    btn('Clear', 'Clear all picks (c)', () => edit('clear')),
    btn('Previous phase', 'Previous phase (p)', () => gotoIndex(idx() - 1)),
    h('div', { class: 'spacer' }), counts);

  // ---------- clip export (issue #37) ----------
  const exportInfo = h('span', { class: 'mono', style: 'font-size:12px' });
  const keepBox = h('input', { type: 'checkbox', id: 'keep-cur', style: 'width:18px;height:18px;accent-color:var(--accent)',
    onchange: () => setExport(st.phase.start, { keep: keepBox.checked }) });
  const clipTools = h('div', { class: 'lines-tools', role: 'group', 'aria-label': 'Clip export for this phase' },
    h('label', { for: 'keep-cur', style: 'display:inline-flex;align-items:center;gap:6px;font-size:13px' }, keepBox, 'Export this phase'),
    btn('Start here [', 'Start the exported clip at this frame ([)', () => setExport(st.phase.start, { trim_in: st.cur })),
    btn('End here ]', 'End the exported clip at this frame (])', () => setExport(st.phase.start, { trim_out: st.cur })),
    btn('Full phase', 'Undo the trim: export the whole phase', () => setExport(st.phase.start, { reset_trim: true })),
    h('div', { class: 'spacer' }), exportInfo);
  const exportSummary = h('span', { class: 'muted mono', style: 'font-size:11px' });
  const exportBtn = h('button', { type: 'button', class: 'btn small primary', onclick: async () => {
    const kept = sum.phases.filter((p) => p.export.keep).length;
    if (!kept) { toast('Every phase is dropped: tick at least one to export', true); return; }
    try {
      const job = await api('/jobs', { method: 'POST', body: { stage: 'b_clip_export', match_id: matchId, params: { team } } });
      toast(`Export of ${kept} clip${kept === 1 ? '' : 's'} queued: the zip appears on the match page when it's done`);
      location.hash = `#/match/${encodeURIComponent(matchId)}?job=${job.id}`;
    } catch (e) { toast(e.message, true); }
  } }, 'Export clips (zip)');
  async function setExportAll(keep) {
    if (!keep && !confirm('Drop every phase from the export? Trims are kept.')) return;
    try {
      await api(`/matches/${encodeURIComponent(matchId)}/lines/export`, { method: 'PUT', body: { team, keep } });
      sum = await api(`/matches/${encodeURIComponent(matchId)}/lines?team=${team}`);
      drawList();
      if (st.phase) draw();
    } catch (e) { toast(e.message, true); }
  }
  async function setExport(start, body) {
    try {
      const r = await api(`/matches/${encodeURIComponent(matchId)}/lines/${start}/export`, { method: 'PUT', body: { team, ...body } });
      const p = sum.phases.find((q) => q.start === start);
      if (p) p.export = r.export;
      drawList();
      if (st.phase) draw();
    } catch (e) { toast(e.message, true); if (st.phase) draw(); }
  }
  const curExport = () => (st.phase && (sum.phases.find((p) => p.start === st.phase.start) || {}).export) || null;
  function drawClipInfo() {
    const ex = curExport();
    if (!ex) return;
    keepBox.checked = ex.keep;
    const ph = st.phase;
    const t = (f) => ((f - ph.start) / FPS).toFixed(1);
    exportInfo.textContent = ex.keep
      ? `clip ${t(ex.in)}–${t(ex.out)} s (${ex.clip_s} s)${ex.trimmed ? ' · trimmed' : ''}`
      : 'not exported';
    const kept = sum.phases.filter((p) => p.export.keep);
    const secs = kept.reduce((a, p) => a + p.export.clip_s, 0);
    exportSummary.textContent = `${kept.length}/${sum.phases.length} kept · ${Math.floor(secs / 60)}:${String(Math.round(secs % 60)).padStart(2, '0')}`;
  }

  const renderBtn = h('button', { type: 'button', class: 'btn small primary', onclick: async () => {
    try {
      const job = await api('/jobs', { method: 'POST', body: { stage: 'b_shape_render', match_id: matchId, params: { team } } });
      toast('Render queued: it waits its turn and runs at low priority');
      location.hash = `#/match/${encodeURIComponent(matchId)}?job=${job.id}`;
    } catch (e) { toast(e.message, true); }
  } }, 'Render picked lines');

  // ---------- identify players (issue #16) ----------
  const idPanel = h('div', { class: 'id-panel', hidden: true, role: 'group', 'aria-label': 'Who is this player?' });
  const anchorList = h('div', { class: 'id-anchors' });
  const anchorInfo = h('span', { class: 'muted mono', style: 'font-size:11px' });
  // Who a box is: your not-yet-applied anchors win over the last tracking run, on
  // the stretch of track (tracklet) each one was made on. A box the tracker still
  // calls by an anchored number gets a "?" since that number is now someone else.
  const pendingAt = (f) => sum.anchors.filter((a) => a.pending && a.span && a.span[0] <= f && f <= a.span[1]);
  const nameOf = (tid, f) => {
    const here = f == null ? [] : pendingAt(f);
    const mine = here.filter((a) => a.track_id === tid).sort((a, b) => (b.created || 0) - (a.created || 0))[0];
    if (mine) return { label: `${mine.number}•`, full: `${mine.number} ${mine.name || ''}`.trim(), manual: true, pending: true, number: mine.number };
    const n = sum.names[String(tid)];
    if (!n) return null;
    const took = here.find((a) => a.number === n.number && a.track_id !== tid);
    if (took) {
      // your pick took this box's number: it most likely swaps to the number the picked box had
      const old = sum.names[String(took.track_id)];
      if (old && old.number !== n.number && !here.some((a) => a.number === old.number)) {
        return { label: `${old.number}•`, full: `${old.number} ${old.name || ''}`.trim(), manual: true, pending: true, swapped: true, number: old.number };
      }
      return { label: `${n.number}?`, full: `${n.number} ${n.name || ''}`.trim(), manual: false, doubt: true, number: n.number };
    }
    return { label: String(n.number), full: `${n.number} ${n.name || ''}`.trim(), manual: n.manual, number: n.number };
  };
  // Identify mode also shows the detections the picker leaves out (the other team,
  // goalkeepers, fragments never put on a team), fetched per frame when paused
  const others = new Map();
  let othersTimer = null;
  function othersAt(f) {
    if (st.mode !== 'identify') return [];
    if (others.has(f)) return others.get(f) || [];
    if (!st.playing) {
      clearTimeout(othersTimer);
      othersTimer = setTimeout(async () => {
        if (others.has(f)) return;
        others.set(f, null);
        try {
          const res = await api(`/matches/${encodeURIComponent(matchId)}/lines/others?team=${team}&frame=${f}`);
          others.set(f, res.rows);
          if (others.size > 400) others.delete(others.keys().next().value);
          if (st.cur === f) draw();
        } catch (e) { others.delete(f); }
      }, 120);
    }
    return [];
  }
  const nameFill = (nm) => (!nm || nm.doubt ? '#EF5B5B' : nm.manual ? '#FFD84D' : '#FFFFFF');
  function drawIdPanel() {
    idPanel.hidden = !st.idBox || st.idBox.dragging;
    if (idPanel.hidden) return;
    const cur = st.idBox.tid != null ? nameOf(st.idBox.tid, st.cur) : null;
    const oth = !cur && st.idBox.tid != null ? (others.get(st.cur) || []).find((r) => r[0] === st.idBox.tid) : null;
    const says = oth ? (oth[7] && oth[7] !== team ? ` · tracking has them on ${sum.opponent_name}${oth[8] != null ? ` as No. ${oth[8]}` : ''}`
      : oth[7] ? ` · tracking says ${oth[8]}` : ' · tracking never put them on a team')
      : !cur ? ' · unidentified' : cur.pending ? ` · you said ${cur.full} (not applied yet)` : ` · tracking says ${cur.full}`;
    idPanel.replaceChildren(
      h('div', { class: 'section-label' }, 'Who is this?', h('span', { class: 'dim' }, says)),
      h('div', { class: 'id-names' }, sum.lineup.length ? sum.lineup.map((p) => h('button', {
        type: 'button', class: 'id-name' + (cur && cur.number === p.number ? ' on' : ''),
        onclick: () => identify(p.number),
      }, `${p.number} ${p.name || ''}`, p.gk ? h('span', { class: 'dim' }, ' GK') : null, p.sub ? h('span', { class: 'dim' }, ' sub') : null))
        : h('span', { class: 'muted' }, 'No lineup for this team in config/<id>_reference.json')),
      h('div', {}, btn('Cancel', 'Cancel (Esc)', () => { st.idBox = null; drawIdPanel(); draw(); })));
  }
  async function identify(number) {
    const b = st.idBox;
    if (!b) return;
    try {
      const res = await api(`/matches/${encodeURIComponent(matchId)}/lines/identify`, {
        method: 'POST', body: { team, frame: st.cur, box: b.box, number } });
      sum.anchors = res.anchors;
      const a = res.anchor;
      st.idBox = null;
      drawIdPanel(); drawAnchors(); draw();
      const moved = a.frame !== st.cur;
      if (moved && st.phase && a.frame >= st.phase.start && a.frame <= st.phase.end) seek(a.frame);
      msg.textContent = (moved ? `They weren't detected on that frame, so No. ${a.number} ${a.name || ''} was anchored ${Math.abs((a.frame - st.cur) / sum.fps).toFixed(1)} s ${a.frame > st.cur ? 'later' : 'earlier'}, where they were. `
        : `No. ${a.number} ${a.name || ''} saved. `) + `The box shows ${a.number}• until you press Apply to tracking, which updates the whole match.`;
    } catch (e) { toast(e.message, true); }
  }
  async function unidentify(id) {
    try {
      const res = await api(`/matches/${encodeURIComponent(matchId)}/lines/identify/${encodeURIComponent(id)}?team=${team}`, { method: 'DELETE' });
      sum.anchors = res.anchors;
      drawAnchors(); draw();
    } catch (e) { toast(e.message, true); }
  }
  function drawAnchors() {
    const pend = sum.anchors.filter((a) => a.pending).length;
    anchorInfo.textContent = sum.anchors.length ? `${sum.anchors.length} anchor${sum.anchors.length === 1 ? '' : 's'}${pend ? ` · ${pend} not applied` : ''}` : '';
    applyBtn.disabled = !sum.anchors.length || null;
    anchorList.replaceChildren(...(sum.anchors.length ? sum.anchors.map((a) => h('div', { class: 'id-anchor' },
      h('button', { type: 'button', class: 'id-anchor-main', title: a.phase != null ? 'Go to this frame' : 'Outside the phases listed here',
        onclick: () => { if (a.phase != null) { gotoStart(a.phase).then(() => seek(a.frame)); } } },
        h('span', { class: 'mono muted' }, a.clock || `f${a.frame}`), h('b', {}, `${a.number} ${a.name || ''}`),
        a.pending ? pill('stale', 'not applied') : pill('done', 'applied')),
      h('button', { type: 'button', class: 'btn small danger', 'aria-label': `Remove the anchor for ${a.name || a.number}`, onclick: () => unidentify(a.id) }, '✕')))
      : [h('div', { class: 'muted', style: 'font-size:12px' }, 'None yet. Use Identify mode (i) on the frame.')]));
  }
  const applyBtn = h('button', { type: 'button', class: 'btn small primary', onclick: async () => {
    try {
      const job = await api('/jobs', { method: 'POST', body: { stage: 'b_identities', match_id: matchId, params: { reuse_tracklets: true } } });
      toast('Track identities queued with your anchors. Afterwards re-run Track ball, Compute possession and the stats stages.');
      location.hash = `#/match/${encodeURIComponent(matchId)}?job=${job.id}`;
    } catch (e) { toast(e.message, true); }
  } }, 'Apply to tracking');

  view.replaceChildren(h('div', { class: 'cols' },
    h('div', { class: 'col-side lines-side' },
      h('section', { class: 'panel', 'aria-labelledby': 'h-phases' },
        h('div', { class: 'panel-head' }, h('h1', { id: 'h-phases' }, 'Phases'), progress, h('div', { class: 'spacer' }), teamSel),
        list),
      h('section', { class: 'panel' }, h('div', { class: 'panel-body', style: 'display:flex;flex-direction:column;gap:10px' },
        h('div', { style: 'display:flex;align-items:center;gap:8px;flex-wrap:wrap' }, renderBtn,
          match.artifacts.shape_video && match.artifacts.shape_video.exists
            ? h('a', { class: 'btn small', href: `/api/matches/${encodeURIComponent(matchId)}/files/shape_video`, target: '_blank', rel: 'noopener' }, 'Last render') : null),
        h('div', { style: 'display:flex;flex-direction:column;gap:6px;border-top:1px solid var(--line);padding-top:10px' },
          h('div', { style: 'display:flex;align-items:center;gap:8px' }, h('span', { class: 'section-label' }, 'Clip export'), exportSummary),
          h('div', { style: 'display:flex;align-items:center;gap:8px;flex-wrap:wrap' }, exportBtn,
            btn('Keep all', 'Export every phase', () => setExportAll(true)),
            btn('Drop all', 'Export no phase (then tick the ones you want)', () => setExportAll(false)),
            match.artifacts.clips_zip && match.artifacts.clips_zip.exists
              ? h('a', { class: 'btn small', href: `/api/matches/${encodeURIComponent(matchId)}/files/clips_zip`, download: '' }, 'Download last zip') : null),
          h('div', { class: 'muted', style: 'font-size:12px;line-height:1.4' },
            'Tick the phases to export and trim each one with Start here / End here. You get one MP4 per phase, with your picked lines, in a zip you can drop into PowerPoint.')),
        h('div', { style: 'display:flex;flex-direction:column;gap:6px;border-top:1px solid var(--line);padding-top:10px' },
          h('div', { style: 'display:flex;align-items:center;gap:8px;flex-wrap:wrap' }, h('span', { class: 'section-label' }, 'Identified players'), anchorInfo,
            h('div', { class: 'spacer' }), applyBtn),
          anchorList,
          h('div', { class: 'muted', style: 'font-size:12px;line-height:1.4' },
            'Tell the tracker who players are: Shift-click (or long-press) a player while picking lines, or use Identify mode to click a player or drag a box round one with no number. ',
            'Apply to tracking re-runs Track identities (about 3 min) so the names, stats and paths for the whole match use your picks.')),
        h('details', {}, h('summary', { class: 'muted', style: 'cursor:pointer' }, 'Controls'),
          h('dl', { class: 'kv', style: 'margin-top:8px' }, KEYS_HELP.flatMap(([k, v]) => [h('dt', {}, k), h('dd', { style: 'font-family:var(--sans)' }, v)]))),
        h('div', { class: 'legend' },
          h('span', {}, h('i', { style: `background:${sum.color}` }), 'back line'),
          h('span', {}, h('i', { style: 'background:#fff' }), 'midfield line'),
          h('span', {}, h('i', { style: `background:${COL.seen}` }), 'detected'),
          h('span', {}, h('i', { style: `background:${COL.filled}` }), 'filled'),
          h('span', {}, h('i', { style: `background:${COL.lost}` }), 'lost'),
          h('span', {}, h('i', { style: 'background:#fff;width:2px' }), 'hand-off')),
      ))),
    h('div', { class: 'col-main' },
      h('section', { class: 'panel' },
        h('div', { class: 'panel-head' }, title, sub, dirtyChip),
        h('div', { class: 'panel-body', style: 'display:flex;flex-direction:column;gap:8px' },
          stage, idPanel, strip, transport, actions, clipTools, msg)))));

  // ---------- phase list ----------
  function drawList() {
    const done = sum.phases.filter((p) => p.status !== 'todo').length;
    progress.textContent = `${done}/${sum.phases.length} done`;
    list.replaceChildren(...sum.phases.map((p) => h('div', { class: 'lines-row' + (p.export.keep ? '' : ' dropped') },
      h('input', { type: 'checkbox', checked: p.export.keep, 'aria-label': `Export phase ${p.n}`,
        title: p.export.keep ? 'Exported: untick to drop it' : 'Dropped: tick to export it',
        onchange: (ev) => setExport(p.start, { keep: ev.target.checked }) }),
      h('button', {
        type: 'button', class: 'job-row', 'aria-current': st.phase && p.start === st.phase.start ? 'true' : null,
        onclick: () => gotoStart(p.start),
      },
      h('span', { style: 'font-weight:600' }, `${p.n}. `, h('span', { class: 'mono', style: 'font-weight:500' }, p.clock || ''),
        h('span', { class: 'dim', style: 'font-weight:400;margin-left:6px' }, `${p.secs}s`)),
      pill(p.status === 'picked' ? 'done' : p.status === 'skipped' ? 'skip' : 'ready', p.status === 'todo' ? 'to do' : p.status),
      p.status === 'picked' || p.export.trimmed
        ? h('span', { class: 'sub' }, [p.status === 'picked' ? `back ${p.n_back} · mid ${p.n_mid}` : null,
          p.export.trimmed ? `clip ${p.export.clip_s}s (trimmed)` : null].filter(Boolean).join(' · ')) : null))));
    drawClipInfo();
    const cur = list.querySelector('[aria-current="true"]');
    if (cur) cur.scrollIntoView({ block: 'nearest' });
  }
  const idx = () => sum.phases.findIndex((p) => st.phase && p.start === st.phase.start);

  // ---------- data ----------
  async function gotoStart(start) {
    if (st.dirty && !confirm('This phase has unsaved picks. Leave without saving?')) return;
    stopPlay();
    st.busy = true;
    title.textContent = 'Loading phase…';
    try {
      const d = await api(`/matches/${encodeURIComponent(matchId)}/lines/${start}?team=${team}`);
      st.phase = d;
      st.slots = d.slots;
      st.chains = d.chains;
      st.cur = d.keyframe;
      st.dirty = false;
      msg.textContent = d.source === 'auto' ? 'Pre-filled with the automatic guess: check it, fix it, then Save + next.' : '';
      history.replaceState(null, '', `#/lines/${encodeURIComponent(matchId)}?team=${team}&phase=${start}`);
      drawList();
      draw();
    } catch (e) { toast(e.message, true); } finally { st.busy = false; }
  }
  function gotoIndex(i) {
    if (i < 0 || i >= sum.phases.length) return;
    gotoStart(sum.phases[i].start);
  }

  async function edit(action, frame, tid, chain) {
    if (!st.phase || st.busy) return;
    st.busy = true;
    try {
      const r = await api(`/matches/${encodeURIComponent(matchId)}/lines/${st.phase.start}/edit`, {
        method: 'POST', body: { team, slots: st.slots, action, frame: frame ?? st.cur, tid, chain },
      });
      st.slots = r.slots;
      st.chains = r.chains;
      st.dirty = true;
      msg.textContent = r.msg || '';
      draw();
    } catch (e) { msg.textContent = e.message; } finally { st.busy = false; }
  }

  async function save(skip) {
    if (!st.phase || st.busy) return;
    st.busy = true;
    try {
      await api(`/matches/${encodeURIComponent(matchId)}/lines/${st.phase.start}`, {
        method: 'PUT', body: { team, slots: skip ? [] : st.slots, keyframe: st.phase.keyframe, skip },
      });
      st.dirty = false;
      sum = await api(`/matches/${encodeURIComponent(matchId)}/lines?team=${team}`);
      toast(skip ? `Phase ${st.phase.n} skipped` : `Phase ${st.phase.n} saved`);
    } catch (e) { toast(e.message, true); st.busy = false; return; }
    st.busy = false;
    const i = idx();
    const next = sum.phases.findIndex((p, j) => j > i && p.status === 'todo');
    if (next >= 0) gotoIndex(next);
    else if (i + 1 < sum.phases.length) gotoIndex(i + 1);
    else { drawList(); msg.textContent = 'That was the last phase. Render picked lines when you\'re ready.'; }
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
      st.loadedFrame = f;
      loading = false;
      if (wantFrame !== f) loadFrame(wantFrame);
      else if (st.playing) setTimeout(playTick, 0);
    };
    pre.onerror = () => { loading = false; };
    pre.src = `/api/matches/${encodeURIComponent(matchId)}/frame/${f}.jpg?w=${IMG_W}`;
  }

  function chainAt(ch, f) { return ch.pts[f] || null; }

  function draw() {
    const ph = st.phase;
    if (!ph) return;
    const f = st.cur;
    const clock = ph.clocks[f - ((f - ph.start) % Math.round(FPS))] || '';
    title.textContent = `Phase ${ph.n}/${sum.phases.length}`;
    sub.textContent = `${clock}  frame ${f}  (${((f - ph.start) / FPS).toFixed(1)} s)`;
    dirtyChip.hidden = !st.dirty;
    const nb = st.slots.filter((s) => s.line === 'back').length;
    counts.textContent = `BACK ${nb} · MID ${st.slots.length - nb}`;
    drawClipInfo();
    loadFrame(f);

    // overlay
    overlay.replaceChildren();
    const rows = ph.rows[f] || [];
    const taken = new Set();
    for (const ch of st.chains) { const p = chainAt(ch, f); if (p && p[4] !== 'f') taken.add(p[0]); }
    if (st.mode === 'identify') {
      const here = sum.anchors.filter((a) => a.frame === f);
      for (const r of rows) {
        const nm = nameOf(r[0], f);
        overlay.append(svg('rect', { x: r[1], y: r[2], width: r[3] - r[1], height: r[4] - r[2], class: 'ov-box' + (nm && !nm.doubt ? '' : ' unid') }));
        const t = svg('text', { x: (r[1] + r[3]) / 2, y: r[2] - 6, class: 'ov-label', fill: nameFill(nm) });
        t.textContent = nm ? nm.label : '?';
        overlay.append(t);
      }
      for (const r of othersAt(f)) {
        const nm = nameOf(r[0], f);
        overlay.append(svg('rect', { x: r[1], y: r[2], width: r[3] - r[1], height: r[4] - r[2], class: 'ov-box other' }));
        // the other team's players stay unlabelled (less clutter); click one to see who tracking says
        if (!nm && r[7] && r[7] !== team) continue;
        const t = svg('text', { x: (r[1] + r[3]) / 2, y: r[2] - 6, class: 'ov-label other', fill: nm ? nameFill(nm) : '#9AA0A8' });
        t.textContent = nm ? nm.label : r[8] != null ? String(r[8]) : '?';
        overlay.append(t);
      }
      for (const a of here) {
        overlay.append(svg('rect', { x: a.box[0] - 3, y: a.box[1] - 3, width: a.box[2] - a.box[0] + 6, height: a.box[3] - a.box[1] + 6,
          fill: 'none', stroke: '#FFD84D', 'stroke-width': 3 }));
      }
    } else {
      for (const r of rows) {
        if (taken.has(r[0])) continue;
        overlay.append(svg('rect', { x: r[1], y: r[2], width: r[3] - r[1], height: r[4] - r[2], class: 'ov-box' }));
        const nm = nameOf(r[0], f); // who the tracker (or your pending anchor) says this is, small and dim
        const t = svg('text', { x: (r[1] + r[3]) / 2, y: r[2] - 5, class: 'ov-num', fill: nameFill(nm) });
        t.textContent = nm ? nm.label : '?';
        overlay.append(t);
      }
    }
    if (st.idBox) {
      const b = st.idBox.box;
      overlay.append(svg('rect', { x: b[0], y: b[1], width: b[2] - b[0], height: b[3] - b[1], fill: 'rgba(255,216,77,0.12)', stroke: '#FFD84D', 'stroke-width': 2, 'stroke-dasharray': '6 4' }));
    }
    const byLine = { back: [], mid: [] };
    st.chains.forEach((ch, ci) => {
      const p = chainAt(ch, f);
      if (!p) return;
      const [, fx, fy, py, s] = p;
      const color = ch.line === 'back' ? sum.color : '#FFFFFF';
      if (ci === st.swapChain) {
        overlay.append(svg('ellipse', { cx: fx, cy: fy, rx: 24, ry: 10, fill: 'none', stroke: '#FFD84D', 'stroke-width': 3 }));
      }
      if (st.mode !== 'identify') { // identify mode labels every box with its shirt number instead
        overlay.append(svg('ellipse', { cx: fx, cy: fy, rx: 16, ry: 6, fill: 'none', stroke: color,
          'stroke-width': s === 'f' ? 1.5 : 2.5, 'stroke-dasharray': s === 'f' ? '4 4' : null }));
        const t = svg('text', { x: fx, y: fy - 22, class: 'ov-label', fill: color });
        t.textContent = (ch.line === 'back' ? 'B' : 'M') + (s === 'f' ? '~' : s === 'l' ? '*' : '');
        const nm = s !== 'f' ? nameOf(p[0], f) : null; // the shirt number of the player the chain is on
        if (nm) {
          const n = svg('tspan', { dx: 4, fill: nameFill(nm), class: 'ov-chain-num' });
          n.textContent = nm.label;
          t.append(n);
        }
        overlay.append(t);
      }
      byLine[ch.line].push([py, fx, fy]);
    });
    for (const [line, pts] of Object.entries(byLine)) {
      if (pts.length < 2) continue;
      pts.sort((a, b) => a[0] - b[0]);
      overlay.append(svg('polyline', { points: pts.map((p) => `${p[1]},${p[2]}`).join(' '), fill: 'none',
        stroke: line === 'back' ? sum.color : '#FFFFFF', 'stroke-width': 3, 'stroke-linejoin': 'round',
        opacity: st.mode === 'identify' ? 0.35 : 0.9 }));
    }
    drawStrip();
  }

  function drawStrip() {
    const ph = st.phase;
    const dpr = window.devicePixelRatio || 1;
    const cssW = strip.clientWidth || 600;
    const rowsN = Math.max(st.chains.length, 1);
    const cssH = rowsN * STRIP_ROW + 6;
    strip.style.height = cssH + 'px';
    strip.width = Math.round(cssW * dpr);
    strip.height = Math.round(cssH * dpr);
    const g = strip.getContext('2d');
    g.scale(dpr, dpr);
    g.fillStyle = '#0B0D10';
    g.fillRect(0, 0, cssW, cssH);
    const x0 = 18, x1 = cssW - 4, span = Math.max(ph.end - ph.start, 1);
    const fx = (f) => x0 + ((f - ph.start) / span) * (x1 - x0);
    st.chains.forEach((ch, i) => {
      const y = 3 + i * STRIP_ROW;
      if (i === st.swapChain) { g.fillStyle = '#FFD84D'; g.fillRect(0, y - 1, 16, STRIP_ROW - 1); }
      g.fillStyle = ch.line === 'back' ? sum.color : '#FFFFFF';
      g.font = '600 10px Inter Tight, sans-serif';
      g.fillText(ch.line === 'back' ? 'B' : 'M', 4, y + 9);
      for (let x = x0; x < x1; x++) {
        const f = ph.start + Math.round(((x - x0) / (x1 - x0)) * span);
        const p = ch.pts[f];
        g.fillStyle = !p ? COL.lost : p[4] === 'f' ? COL.filled : COL.seen;
        g.fillRect(x, y, 1, STRIP_ROW - 3);
      }
      g.fillStyle = COL.link;
      for (const [f, p] of Object.entries(ch.pts)) if (p[4] === 'l') g.fillRect(fx(+f) - 1, y, 2, STRIP_ROW - 3);
    });
    const ex = curExport();
    if (ex) { // clip export: shade what's trimmed off (or the whole phase if dropped)
      g.fillStyle = 'rgba(11,13,16,0.72)';
      if (!ex.keep) g.fillRect(x0, 0, x1 - x0, cssH);
      else {
        if (ex.in > ph.start) g.fillRect(x0, 0, fx(ex.in) - x0, cssH);
        if (ex.out < ph.end) g.fillRect(fx(ex.out), 0, x1 - fx(ex.out), cssH);
        g.fillStyle = COL.head;
        if (ex.in > ph.start) g.fillRect(fx(ex.in) - 1, 0, 2, cssH);
        if (ex.out < ph.end) g.fillRect(fx(ex.out) - 1, 0, 2, cssH);
      }
    }
    g.fillStyle = COL.head;
    g.fillRect(fx(st.cur) - 1, 0, 2, cssH);
    strip.setAttribute('aria-label', `Chain timeline, frame ${st.cur} of ${ph.start}–${ph.end}: click to seek`);
  }

  // ---------- interaction ----------
  function seek(f) {
    const ph = st.phase;
    if (!ph) return;
    st.cur = Math.max(ph.start, Math.min(ph.end, Math.round(f)));
    draw();
  }
  function step(n) { stopPlay(); seek(st.cur + n); }
  function stopPlay() { st.playing = false; playBtn.textContent = '▶'; playBtn.setAttribute('aria-label', 'Play'); }
  function togglePlay() {
    if (st.playing) { stopPlay(); return; }
    st.playing = true;
    playBtn.textContent = '❚❚';
    playBtn.setAttribute('aria-label', 'Pause');
    playTick();
  }
  let lastTick = 0;
  function playTick() {
    if (!st.playing || !st.phase) return;
    if (st.cur >= st.phase.end) { stopPlay(); return; }
    // ~real time: advance by however many frames have elapsed since the last shown one
    const now = performance.now();
    const adv = lastTick ? Math.max(1, Math.min(6, Math.round(((now - lastTick) / 1000) * FPS))) : 1;
    lastTick = now;
    seek(st.cur + adv);
  }

  function hitTest(x, y) {
    const rows = ((st.phase && st.phase.rows[st.cur]) || []).concat(st.mode === 'identify' ? (others.get(st.cur) || []) : []);
    let best = null, bd = 1e9;
    for (const r of rows) {
      if (r[1] <= x && x <= r[3] && r[2] <= y && y <= r[4]) {
        const d = Math.abs(x - (r[1] + r[3]) / 2) + Math.abs(y - r[6]) * 0.25;
        if (d < bd) { best = r[0]; bd = d; }
      }
    }
    if (best != null) return best;
    for (const r of rows) {
      const d = Math.hypot(x - r[5], y - r[6]);
      if (d < Math.min(bd, 40)) { best = r[0]; bd = d; }
    }
    return best;
  }
  function toVideo(ev) {
    const rect = overlay.getBoundingClientRect();
    return [((ev.clientX - rect.left) / rect.width) * W, ((ev.clientY - rect.top) / rect.height) * H];
  }
  function chainNear(x, y) {
    let best = null, bd = 30;
    st.chains.forEach((ch, i) => {
      const p = chainAt(ch, st.cur);
      if (!p) return;
      const d = Math.hypot(x - p[1], y - p[2]);
      if (d < bd) { best = i; bd = d; }
    });
    return best;
  }
  function selectSwapChain(i) {
    st.swapChain = i;
    const line = st.chains[i].line === 'back' ? 'back-line' : 'midfield';
    msg.textContent = `Fixing a ${line} chain: now tap the player it should be on.`;
    draw();
  }
  // identify: drag a box round a player (pointer events), or click one
  let drag = null, dragged = false;
  overlay.addEventListener('pointerdown', (ev) => {
    if (st.mode !== 'identify' || ev.button !== 0) return;
    drag = toVideo(ev); dragged = false;
    overlay.setPointerCapture(ev.pointerId);
  });
  overlay.addEventListener('pointermove', (ev) => {
    if (!drag) return;
    const [x, y] = toVideo(ev);
    if (!dragged && Math.hypot(x - drag[0], y - drag[1]) < 10) return;
    dragged = true;
    st.idBox = { box: [Math.min(drag[0], x), Math.min(drag[1], y), Math.max(drag[0], x), Math.max(drag[1], y)], tid: null, dragging: true };
    draw();
  });
  overlay.addEventListener('pointerup', () => {
    if (!drag) return;
    drag = null;
    if (dragged && st.idBox) { stopPlay(); st.idBox.dragging = false; drawIdPanel(); draw(); }
  });
  function askWho(x, y) {
    const tid = hitTest(x, y);
    const r = ((st.phase && st.phase.rows[st.cur]) || []).find((row) => row[0] === tid);
    if (!r) { msg.textContent = 'No tracked box there. Use Identify mode (i) to drag a box round the player.'; return; }
    st.idBox = { box: [r[1], r[2], r[3], r[4]], tid };
    drawIdPanel(); draw();
    msg.textContent = 'Who is this? Pick a name under the video.';
  }
  // long-press (touch) = shift-click: change who a player is without leaving line picking
  let press = null, pressed = false;
  overlay.addEventListener('pointerdown', (ev) => {
    if (st.mode === 'identify' || ev.button !== 0 || ev.pointerType === 'mouse') return;
    const at = toVideo(ev);
    pressed = false;
    press = setTimeout(() => { press = null; pressed = true; stopPlay(); askWho(...at); }, 550);
  });
  const endPress = () => { if (press) { clearTimeout(press); press = null; } };
  overlay.addEventListener('pointerup', endPress);
  overlay.addEventListener('pointercancel', endPress);
  overlay.addEventListener('pointermove', (ev) => { if (press && ev.pointerType !== 'mouse' && (Math.abs(ev.movementX) + Math.abs(ev.movementY) > 6)) endPress(); });
  overlay.addEventListener('click', (ev) => {
    stopPlay();
    const [x, y] = toVideo(ev);
    if (pressed) { pressed = false; return; }
    if (ev.shiftKey && st.mode !== 'identify') { askWho(x, y); return; }
    if (st.mode === 'identify') {
      if (dragged) { dragged = false; return; }
      const tid = hitTest(x, y);
      const r = ((st.phase && st.phase.rows[st.cur]) || []).concat(others.get(st.cur) || []).find((row) => row[0] === tid);
      if (!r) { msg.textContent = 'No tracked box there: drag a box round the player instead.'; return; }
      st.idBox = { box: [r[1], r[2], r[3], r[4]], tid };
      drawIdPanel(); draw();
      return;
    }
    if (st.mode === 'swap' && st.swapChain == null) {
      const ci = chainNear(x, y);
      if (ci == null) { msg.textContent = 'Tap the marker of the chain that jumped (or its row label in the strip).'; return; }
      selectSwapChain(ci);
      return;
    }
    const tid = hitTest(x, y);
    if (tid == null) { msg.textContent = 'No player there: click on a box or near the feet.'; return; }
    if (st.mode === 'swap') {
      const ci = st.swapChain;
      st.swapChain = null;
      edit('swap', st.cur, tid, ci).then(() => { const m = msg.textContent; setMode('pick'); msg.textContent = m; });
      return;
    }
    edit(st.mode === 'relink' ? 'relink' : 'cycle', st.cur, tid);
  });
  overlay.addEventListener('contextmenu', (ev) => {
    ev.preventDefault();
    stopPlay();
    const [x, y] = toVideo(ev);
    if (press || (ev.pointerType && ev.pointerType !== 'mouse')) { // a touch long-press, not a right-click
      endPress(); pressed = true; askWho(x, y); return;
    }
    const tid = hitTest(x, y);
    if (tid != null) edit('relink', st.cur, tid);
  });
  const stripSeek = (ev) => {
    const rect = strip.getBoundingClientRect();
    const x0 = 18, x1 = rect.width - 4;
    const frac = Math.max(0, Math.min(1, (ev.clientX - rect.left - x0) / (x1 - x0)));
    stopPlay();
    seek(st.phase.start + frac * (st.phase.end - st.phase.start));
  };
  strip.addEventListener('pointerdown', (ev) => {
    const rect = strip.getBoundingClientRect();
    if (st.mode === 'swap' && ev.clientX - rect.left < 18) { // the B/M labels pick a chain
      const i = Math.floor((ev.clientY - rect.top - 3) / STRIP_ROW);
      if (i >= 0 && i < st.chains.length) selectSwapChain(i);
      return;
    }
    stripSeek(ev);
    strip.setPointerCapture(ev.pointerId);
  });
  strip.addEventListener('pointermove', (ev) => { if (ev.buttons) stripSeek(ev); });

  const onKey = (ev) => {
    const tag = ev.target.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || ev.target.isContentEditable) return;
    if ((tag === 'BUTTON' || tag === 'A' || tag === 'SUMMARY') && (ev.key === 'Enter' || ev.key === ' ')) return;
    if (ev.ctrlKey || ev.metaKey || ev.altKey || !st.phase) return;
    const k = ev.key;
    const steps = { a: -FPS / 2, d: FPS / 2, A: -2 * FPS, D: 2 * FPS, ',': -1, '.': 1 };
    if (k in steps) step(Math.round(steps[k]));
    else if (k === 'Enter') save(false);
    else if (k === 's') save(true);
    else if (k === 'r') edit('auto');
    else if (k === 'c') edit('clear');
    else if (k === 'p') gotoIndex(idx() - 1);
    else if (k === 'k') togglePlay();
    else if (k === 'x') setMode(st.mode === 'swap' ? 'pick' : 'swap');
    else if (k === 'i') setMode(st.mode === 'identify' ? 'pick' : 'identify');
    else if (k === '[') setExport(st.phase.start, { trim_in: st.cur });
    else if (k === ']') setExport(st.phase.start, { trim_out: st.cur });
    else if (k === 'e') { const ex = curExport(); if (ex) setExport(st.phase.start, { keep: !ex.keep }); }
    else if (k === 'Escape') { if (st.idBox && st.mode !== 'identify') { st.idBox = null; drawIdPanel(); draw(); } else setMode('pick'); }
    else return;
    ev.preventDefault();
  };
  document.addEventListener('keydown', onKey);
  const onResize = () => { if (st.phase) drawStrip(); };
  window.addEventListener('resize', onResize);
  const onBeforeUnload = (ev) => { if (st.dirty) { ev.preventDefault(); ev.returnValue = ''; } };
  window.addEventListener('beforeunload', onBeforeUnload);
  onLeave(() => {
    stopPlay();
    document.removeEventListener('keydown', onKey);
    window.removeEventListener('resize', onResize);
    window.removeEventListener('beforeunload', onBeforeUnload);
  });

  drawList();
  drawAnchors();
  const want = Number(params.get('phase'));
  const first = sum.phases.find((p) => p.start === want) || sum.phases.find((p) => p.status === 'todo') || sum.phases[0];
  if (first) gotoStart(first.start);
  else msg.textContent = 'No phases found for this team.';
  every(30000, async () => { // pick up saves from another tab / the CLI picker
    if (st.dirty || st.busy) return;
    try { sum = await api(`/matches/${encodeURIComponent(matchId)}/lines?team=${team}`); drawList(); } catch { /* ignore */ }
  });
}
