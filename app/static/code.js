// Code tab (issue #3): a code window of tagging buttons beside the match video.
// Press a button (or its hotkey) while the video plays to make an instance
// covering [t - lead, t + lag]; label buttons then describe the selected
// instance (team, player, outcome, zone...). Layouts are templates saved in
// config/code_windows/, instances in config/<id>_instances.json (app/coding.py).

const KEYS_HELP = [
  ['Button key', 'tag an instance at the current time'],
  ['Label key', 'set / clear that label on the selected instance'],
  ['Space', 'play / pause'], ['← / →', '−/+ 2 s (Shift: 10 s)'], [', / .', '−/+ 1 frame'],
  ['[ / ]', 'selected clip starts / ends here'], ['Delete', 'delete the selected instance'],
  ['Ctrl+Z', 'undo the last delete'], ['Esc', 'deselect'],
];
const SPEEDS = [0.5, 1, 1.5, 2];
const LS_TEMPLATE = 'stopframe.code.template';

function lsGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function lsSet(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } }

const pad = (n) => String(n).padStart(2, '0');
function fmt(t) {
  if (t == null || !isFinite(t)) return '--:--';
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${pad(m)}:${s.toFixed(1).padStart(4, '0')}`;
}
// readable text on a coloured button
function ink(hex) {
  const n = parseInt(hex.slice(1), 16);
  const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
  return (0.299 * r + 0.587 * g + 0.114 * b) > 150 ? '#0E1013' : '#FFFFFF';
}

export async function viewCode(view, parts, params, ui) {
  const { h, api, toast, renderTabs, setCrumb, pill, onLeave } = ui;
  renderTabs('code');
  const matchId = parts[1];

  if (!matchId) {
    setCrumb(null);
    const data = await api('/matches');
    const ok = data.matches.filter((m) => m.video);
    view.replaceChildren(h('section', { class: 'panel' },
      h('div', { class: 'panel-head' }, h('h1', {}, 'Code'), h('span', { class: 'muted' }, 'Pick a match to tag')),
      ok.length ? h('div', {}, ok.map((m) => h('a', { class: 'match-row', href: '#/code/' + encodeURIComponent(m.match_id) },
        h('div', { class: 'match-title' }, `${m.team_a_name} v ${m.team_b_name}`),
        pill('ready', m.video_name || 'video'))))
        : h('div', { class: 'empty' }, 'No matches yet. Ingest a video from the Matches tab first.')));
    return;
  }

  let match, data, templates;
  try {
    [match, data, templates] = await Promise.all([
      api('/matches/' + encodeURIComponent(matchId)),
      api(`/matches/${encodeURIComponent(matchId)}/instances`),
      api('/code/templates'),
    ]);
  } catch (e) {
    view.replaceChildren(h('div', { class: 'placeholder' }, h('h1', {}, 'Match not found'), h('p', {}, e.message)));
    return;
  }
  setCrumb(match);
  const base = `/matches/${encodeURIComponent(matchId)}/instances`;
  const FPS = match.fps || 25;
  const ML = data.match_labels || { teams: [], lineup: [] };
  const st = {
    inst: data.instances || [], sel: null, tpl: null, editing: false, lastDeleted: null,
  };

  // ---------- video + transport ----------
  const video = h('video', {
    class: 'code-video', preload: 'metadata', playsinline: true,
    src: `/api/matches/${encodeURIComponent(matchId)}/video`, 'aria-label': 'Match video',
  });
  const timeLbl = h('span', { class: 'mono code-time' });
  const seek = h('input', { type: 'range', min: '0', max: String(match.duration_s || 0), step: '0.04', value: '0', class: 'code-seek', 'aria-label': 'Seek' });
  const btn = (label, aria, fn, cls) => h('button', { type: 'button', class: 'btn small ' + (cls || ''), 'aria-label': aria, title: aria, onclick: fn }, label);
  const playBtn = btn('▶', 'Play / pause (Space)', () => togglePlay());
  const speedSel = h('select', { class: 'mini-select', 'aria-label': 'Playback speed', onchange: () => { video.playbackRate = +speedSel.value; } },
    SPEEDS.map((s) => h('option', { value: s, selected: s === 1 || null }, `${s}×`)));
  const transport = h('div', { class: 'lines-tools' },
    btn('−10s', 'Back 10 s', () => jump(-10)), btn('−2s', 'Back 2 s (←)', () => jump(-2)), btn('◂', 'Back 1 frame (,)', () => step(-1)),
    playBtn, btn('▸', 'Forward 1 frame (.)', () => step(1)), btn('+2s', 'Forward 2 s (→)', () => jump(2)), btn('+10s', 'Forward 10 s', () => jump(10)),
    speedSel, timeLbl);
  const strip = h('canvas', { class: 'lines-strip code-strip', role: 'img', 'aria-label': 'Tagged instances along the match: click to seek' });

  function jump(d) { video.currentTime = Math.max(0, Math.min((video.duration || match.duration_s || 0), video.currentTime + d)); }
  function step(n) { video.pause(); jump(n / FPS); }
  function togglePlay() { if (video.paused) video.play().catch((e) => toast(e.message, true)); else video.pause(); }
  video.addEventListener('play', () => { playBtn.textContent = '❚❚'; });
  video.addEventListener('pause', () => { playBtn.textContent = '▶'; });
  video.addEventListener('error', () => toast('The browser could not play this video file (needs H.264 MP4).', true));
  let lastStripDraw = 0;
  video.addEventListener('timeupdate', () => {
    timeLbl.textContent = `${fmt(video.currentTime)} / ${fmt(video.duration || match.duration_s)}`;
    if (document.activeElement !== seek) seek.value = String(video.currentTime);
    const now = performance.now();
    if (now - lastStripDraw > 250) { lastStripDraw = now; drawStrip(); }
  });
  video.addEventListener('loadedmetadata', () => { seek.max = String(video.duration); drawStrip(); });
  seek.addEventListener('input', () => { video.currentTime = +seek.value; });

  // ---------- template ----------
  const tplSel = h('select', { class: 'mini-select', 'aria-label': 'Code window template', onchange: () => loadTemplate(tplSel.value) });
  function fillTplSel(selected) {
    tplSel.replaceChildren(...templates.map((t) => h('option', { value: t.name, selected: t.name === selected || null }, t.name)));
  }
  async function loadTemplate(name) {
    try { st.tpl = await api('/code/templates/' + encodeURIComponent(name)); } catch (e) { toast(e.message, true); return; }
    lsSet(LS_TEMPLATE, st.tpl.name);
    fillTplSel(st.tpl.name);
    st.editing = false;
    renderPad();
  }

  // ---------- code window ----------
  const padBody = h('div', { class: 'panel-body code-pad' });
  const editBtn = btn('Edit layout', 'Edit this code window', () => { st.editing = !st.editing; renderPad(); });
  const csvLink = h('a', { class: 'btn small', href: `/api${base}.csv`, download: '' }, 'CSV');

  function groupLabels(g) {
    if (g.source === 'teams') return ML.teams.map((x) => ({ label: x.label, team: x.team }));
    if (g.source === 'lineup') return ML.lineup.map((x) => ({ label: x.label, team: x.team }));
    return g.labels || [];
  }
  // the team an instance is labelled with (picks which lineup to show)
  function instTeam(inst) {
    if (!inst || !st.tpl) return null;
    for (const g of st.tpl.label_groups) {
      if (g.source !== 'teams') continue;
      const v = inst.labels[g.name];
      const hit = ML.teams.find((x) => x.label === v);
      if (hit) return hit.team;
    }
    return null;
  }

  function renderPad() {
    if (!st.tpl) return;
    editBtn.textContent = st.editing ? 'Close editor' : 'Edit layout';
    if (st.editing) { padBody.replaceChildren(editor()); return; }
    const grid = h('div', { class: 'code-grid' }, st.tpl.buttons.map((b) => h('button', {
      type: 'button', class: 'code-btn', style: `--c:${b.color};color:${ink(b.color)}`,
      title: `${b.label}: ${b.lead_s}s before, ${b.lag_s}s after${b.key ? ` (key ${b.key})` : ''}`,
      onclick: () => tag(b),
    }, h('span', { class: 'code-btn-label' }, b.label),
    h('span', { class: 'code-btn-meta mono' }, `${b.key ? b.key.toUpperCase() + ' · ' : ''}−${b.lead_s}/+${b.lag_s}s`))));
    const inst = selected();
    const team = instTeam(inst);
    const groups = st.tpl.label_groups.map((g) => {
      let labels = groupLabels(g);
      if (g.source === 'lineup' && team) labels = labels.filter((l) => l.team === team);
      const cur = inst ? inst.labels[g.name] : null;
      return h('div', { class: 'code-group' },
        h('div', { class: 'section-label' }, g.name, g.source === 'lineup' && !labels.length ? h('span', { class: 'dim' }, ' · no lineup in config/<id>_reference.json') : null),
        h('div', { class: 'code-labels' }, labels.map((l) => h('button', {
          type: 'button', class: 'code-label' + (cur === l.label ? ' on' : ''), disabled: !inst || null,
          'aria-pressed': cur === l.label ? 'true' : 'false',
          onclick: () => setLabel(g.name, cur === l.label ? null : l.label),
        }, l.label, l.key ? h('span', { class: 'mono dim' }, ` ${l.key}`) : null))));
    });
    const selInfo = inst
      ? h('div', { class: 'code-sel' },
        h('span', { class: 'chip', style: `background:${inst.color}` }), h('b', {}, inst.code),
        h('span', { class: 'mono muted' }, `${inst.clock ? inst.clock + ' · ' : ''}${fmt(inst.start)}–${fmt(inst.end)}`),
        btn('Start here [', 'Clip starts at the current time', () => trim('start')),
        btn('End here ]', 'Clip ends at the current time', () => trim('end')),
        btn('Delete', 'Delete this instance (Delete)', () => del(inst), 'danger'))
      : h('div', { class: 'muted' }, 'Tag something, or pick an instance below, to label it.');
    padBody.replaceChildren(grid, h('div', { class: 'code-labels-head' }, h('span', { class: 'section-label' }, 'Labels for'), selInfo), ...groups);
  }

  // ---------- layout editor ----------
  function editor() {
    const t = JSON.parse(JSON.stringify(st.tpl));
    const nameIn = h('input', { type: 'text', value: t.name, maxlength: '60', 'aria-label': 'Template name' });
    const rows = h('div', { class: 'code-edit-rows' });
    const groupsBox = h('div', { class: 'code-edit-rows' });
    const err = h('div', { class: 'form-error', role: 'alert' });

    function btnRow(b) {
      const row = h('div', { class: 'code-edit-row' },
        h('input', { type: 'color', value: b.color || '#4C8DF6', 'aria-label': 'Colour', 'data-f': 'color' }),
        h('input', { type: 'text', value: b.label || '', placeholder: 'Label', maxlength: '40', 'aria-label': 'Label', 'data-f': 'label' }),
        h('input', { type: 'text', value: b.key || '', placeholder: 'key', maxlength: '1', class: 'mono', 'aria-label': 'Hotkey', 'data-f': 'key' }),
        h('input', { type: 'number', value: b.lead_s ?? 5, min: '0', max: '120', step: '0.5', 'aria-label': 'Seconds before', title: 'Seconds before', 'data-f': 'lead_s' }),
        h('input', { type: 'number', value: b.lag_s ?? 5, min: '0', max: '120', step: '0.5', 'aria-label': 'Seconds after', title: 'Seconds after', 'data-f': 'lag_s' }),
        h('button', { type: 'button', class: 'btn small', 'aria-label': 'Move up', onclick: () => { if (row.previousSibling) rows.insertBefore(row, row.previousSibling); } }, '↑'),
        h('button', { type: 'button', class: 'btn small danger', 'aria-label': 'Remove button', onclick: () => row.remove() }, '✕'));
      if (b.id) row.dataset.id = b.id;
      return row;
    }
    function groupRow(g) {
      const src = h('select', { class: 'mini-select', 'aria-label': 'Labels come from', 'data-f': 'source' },
        [['', 'Listed labels'], ['teams', 'The two teams'], ['lineup', 'Match lineup (players)']].map(([v, n]) =>
          h('option', { value: v, selected: (g.source || '') === v || null }, n)));
      const labelsIn = h('input', {
        type: 'text', 'data-f': 'labels', 'aria-label': 'Labels, comma separated (label=key)', placeholder: 'Success=1, Fail=2',
        value: (g.labels || []).map((l) => l.key ? `${l.label}=${l.key}` : l.label).join(', '),
      });
      const sync = () => { labelsIn.hidden = !!src.value; };
      src.addEventListener('change', sync);
      sync();
      const row = h('div', { class: 'code-edit-row group' },
        h('input', { type: 'text', value: g.name || '', placeholder: 'Group', maxlength: '30', 'aria-label': 'Group name', 'data-f': 'name' }),
        src, labelsIn,
        h('button', { type: 'button', class: 'btn small danger', 'aria-label': 'Remove group', onclick: () => row.remove() }, '✕'));
      if (g.id) row.dataset.id = g.id;
      return row;
    }
    t.buttons.forEach((b) => rows.append(btnRow(b)));
    t.label_groups.forEach((g) => groupsBox.append(groupRow(g)));

    const val = (row, f) => row.querySelector(`[data-f="${f}"]`).value;
    function collect(name) {
      return {
        name,
        buttons: [...rows.children].map((r) => ({
          id: r.dataset.id, label: val(r, 'label'), key: val(r, 'key'), color: val(r, 'color'),
          lead_s: +val(r, 'lead_s'), lag_s: +val(r, 'lag_s'),
        })),
        label_groups: [...groupsBox.children].map((r) => {
          const g = { id: r.dataset.id, name: val(r, 'name') };
          const s = val(r, 'source');
          if (s) g.source = s;
          else {
            g.labels = val(r, 'labels').split(',').map((x) => x.trim()).filter(Boolean).map((x) => {
              const m = x.match(/^(.*?)\s*=\s*(\S)$/);
              return m ? { label: m[1], key: m[2] } : { label: x };
            });
          }
          return g;
        }),
      };
    }
    async function save(asNew) {
      let name = nameIn.value.trim();
      if (asNew) {
        name = prompt('Name for the new template', name === st.tpl.name ? name + ' copy' : name);
        if (!name) return;
        name = name.trim();
      }
      err.textContent = '';
      try {
        const saved = await api('/code/templates/' + encodeURIComponent(name), { method: 'PUT', body: collect(name) });
        templates = await api('/code/templates');
        st.tpl = saved;
        lsSet(LS_TEMPLATE, saved.name);
        fillTplSel(saved.name);
        st.editing = false;
        renderPad();
        toast(`Saved template "${saved.name}" to config/code_windows/`);
      } catch (e) { err.textContent = e.message; }
    }
    async function remove() {
      if (!confirm(`Delete the saved template "${st.tpl.name}"? Instances already tagged are kept.`)) return;
      try {
        await api('/code/templates/' + encodeURIComponent(st.tpl.name), { method: 'DELETE' });
        templates = await api('/code/templates');
        await loadTemplate(templates[0].name);
        toast('Template deleted');
      } catch (e) { err.textContent = e.message; }
    }
    const saved = templates.find((x) => x.name === st.tpl.name)?.saved;
    return h('div', { class: 'code-editor' },
      h('label', { class: 'field' }, 'Template name', nameIn),
      h('div', { class: 'section-label' }, 'Buttons ', h('span', { class: 'dim' }, '· colour, label, hotkey, seconds before, seconds after')),
      rows,
      h('div', {}, btn('+ Add button', 'Add a button', () => rows.append(btnRow({ color: '#4C8DF6', lead_s: 5, lag_s: 5 })))),
      h('div', { class: 'section-label' }, 'Label groups ', h('span', { class: 'dim' }, '· one label per group on each instance')),
      groupsBox,
      h('div', {}, btn('+ Add group', 'Add a label group', () => groupsBox.append(groupRow({ labels: [] })))),
      err,
      h('div', { class: 'form-actions' },
        btn('Save', 'Save this template', () => save(false), 'primary'),
        btn('Save as new…', 'Save as a new template', () => save(true)),
        btn('Cancel', 'Close without saving', () => { st.editing = false; renderPad(); }),
        saved ? btn('Delete template', 'Delete this saved template', () => remove(), 'danger') : null),
      h('p', { class: 'muted', style: 'margin:0;font-size:12px' },
        'Hotkeys are single characters; Space, arrows, comma, full stop, [ ] and Delete belong to the player. ',
        'Changing a layout never changes instances you already tagged.'));
  }

  // ---------- instances ----------
  const list = h('div', { class: 'scroll code-list', role: 'list' });
  const countLbl = h('span', { class: 'muted mono', style: 'font-size:12px' });
  const selected = () => st.inst.find((i) => i.id === st.sel) || null;

  function select(id, seekTo) {
    st.sel = id;
    const i = selected();
    if (i && seekTo) video.currentTime = i.start;
    renderList();
    renderPad();
    drawStrip();
  }

  function renderList() {
    countLbl.textContent = `${st.inst.length} instance${st.inst.length === 1 ? '' : 's'}`;
    if (!st.inst.length) {
      list.replaceChildren(h('div', { class: 'empty' }, 'Nothing tagged yet. Play the video and press a code button (or its key).'));
      return;
    }
    list.replaceChildren(...st.inst.map((i) => {
      const row = h('div', { class: 'code-row' + (i.id === st.sel ? ' on' : ''), role: 'listitem' },
        h('button', { type: 'button', class: 'code-row-main', 'aria-label': `${i.code} at ${i.clock || fmt(i.t)}: select and play from its start`, onclick: () => select(i.id, true) },
          h('span', { class: 'chip', style: `background:${i.color}` }),
          h('span', { class: 'code-row-code' }, i.code),
          h('span', { class: 'mono muted' }, i.clock ? `${i.clock}` : fmt(i.t)),
          h('span', { class: 'mono dim' }, `${(i.end - i.start).toFixed(0)}s`),
          h('span', { class: 'code-row-labels' }, Object.entries(i.labels).map(([g, v]) => h('span', { class: 'tag', title: g }, v)))),
        h('button', { type: 'button', class: 'btn small danger', 'aria-label': `Delete ${i.code} at ${i.clock || fmt(i.t)}`, onclick: () => del(i) }, '✕'));
      return row;
    }));
    const on = list.querySelector('.code-row.on');
    if (on) on.scrollIntoView({ block: 'nearest' });
  }

  function upsert(inst) {
    const k = st.inst.findIndex((x) => x.id === inst.id);
    if (k >= 0) st.inst[k] = inst; else st.inst.push(inst);
    st.inst.sort((a, b) => a.t - b.t || a.created - b.created);
  }

  async function tag(b) {
    const t = video.currentTime;
    try {
      const inst = await api(base, {
        method: 'POST',
        body: { code: b.label, color: b.color, t, lead_s: b.lead_s, lag_s: b.lag_s, template: st.tpl.name },
      });
      upsert(inst);
      select(inst.id, false);
      flash(b);
    } catch (e) { toast(e.message, true); }
  }
  function flash(b) {
    const el = [...padBody.querySelectorAll('.code-btn')].find((x) => x.firstChild.textContent === b.label);
    if (!el) return;
    el.classList.remove('hit'); void el.offsetWidth; el.classList.add('hit');
  }

  async function setLabel(group, label) {
    const i = selected();
    if (!i) { toast('Select an instance first'); return; }
    try {
      upsert(await api(`${base}/${i.id}`, { method: 'PATCH', body: { set_label: { group, label } } }));
      renderList(); renderPad();
    } catch (e) { toast(e.message, true); }
  }

  async function trim(which) {
    const i = selected();
    if (!i) return;
    const t = video.currentTime;
    try {
      upsert(await api(`${base}/${i.id}`, { method: 'PATCH', body: { [which]: t } }));
      renderList(); renderPad(); drawStrip();
    } catch (e) { toast(e.message, true); }
  }

  async function del(i) {
    try {
      await api(`${base}/${i.id}`, { method: 'DELETE' });
      st.inst = st.inst.filter((x) => x.id !== i.id);
      st.lastDeleted = i;
      if (st.sel === i.id) st.sel = null;
      renderList(); renderPad(); drawStrip();
      toast(`Deleted ${i.code} · Ctrl+Z to undo`);
    } catch (e) { toast(e.message, true); }
  }

  async function undo() {
    const i = st.lastDeleted;
    if (!i) return;
    st.lastDeleted = null;
    try {
      let inst = await api(base, { method: 'POST', body: { code: i.code, color: i.color, t: i.t, lead_s: 0, lag_s: 0, labels: i.labels } });
      inst = await api(`${base}/${inst.id}`, { method: 'PATCH', body: { start: i.start, end: i.end } });
      upsert(inst);
      select(inst.id, false);
      toast(`Restored ${i.code}`);
    } catch (e) { toast(e.message, true); }
  }

  // ---------- overview strip ----------
  function drawStrip() {
    const dpr = window.devicePixelRatio || 1;
    const codes = [];
    (st.tpl ? st.tpl.buttons.map((b) => b.label) : []).concat(st.inst.map((i) => i.code))
      .forEach((c) => { if (!codes.includes(c) && st.inst.some((i) => i.code === c)) codes.push(c); });
    const ROW = 8, rowsN = Math.max(1, codes.length);
    const cssW = strip.clientWidth || 600, cssH = rowsN * ROW + 6;
    strip.style.height = cssH + 'px';
    strip.width = Math.round(cssW * dpr); strip.height = Math.round(cssH * dpr);
    const ctx = strip.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.fillStyle = '#0B0D10'; ctx.fillRect(0, 0, cssW, cssH);
    const D = video.duration || match.duration_s || 1;
    const x = (t) => (t / D) * cssW;
    for (const i of st.inst) {
      const r = codes.indexOf(i.code);
      ctx.fillStyle = i.color;
      ctx.globalAlpha = i.id === st.sel ? 1 : 0.75;
      ctx.fillRect(x(i.start), 3 + r * ROW, Math.max(2, x(i.end) - x(i.start)), ROW - 2);
    }
    ctx.globalAlpha = 1;
    const sel = selected();
    if (sel) { ctx.strokeStyle = '#FFD84D'; ctx.lineWidth = 1.5; ctx.strokeRect(x(sel.start) - 1, 1, Math.max(4, x(sel.end) - x(sel.start)) + 2, cssH - 2); }
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(x(video.currentTime) - 1, 0, 2, cssH);
  }
  strip.addEventListener('click', (e) => {
    const r = strip.getBoundingClientRect();
    const D = video.duration || match.duration_s || 0;
    video.currentTime = Math.max(0, Math.min(D, ((e.clientX - r.left) / r.width) * D));
    drawStrip();
  });

  // ---------- keyboard ----------
  function onKey(e) {
    if (e.target.closest && e.target.closest('input, select, textarea')) return;
    if (st.editing) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); return; }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.length === 1 && /[a-z]/i.test(e.key) ? e.key.toLowerCase() : e.key;
    const handled = () => e.preventDefault();
    if (k === ' ') { handled(); togglePlay(); return; }
    if (k === 'ArrowLeft') { handled(); jump(e.shiftKey ? -10 : -2); return; }
    if (k === 'ArrowRight') { handled(); jump(e.shiftKey ? 10 : 2); return; }
    if (k === ',') { handled(); step(-1); return; }
    if (k === '.') { handled(); step(1); return; }
    if (k === '[') { handled(); trim('start'); return; }
    if (k === ']') { handled(); trim('end'); return; }
    if (k === 'Escape') { select(null); return; }
    if (k === 'Delete' || k === 'Backspace') { const i = selected(); if (i) { handled(); del(i); } return; }
    if (!st.tpl) return;
    const b = st.tpl.buttons.find((x) => x.key === k);
    if (b) { handled(); tag(b); return; }
    for (const g of st.tpl.label_groups) {
      const l = (g.labels || []).find((x) => x.key === k);
      if (l) {
        handled();
        const i = selected();
        setLabel(g.name, i && i.labels[g.name] === l.label ? null : l.label);
        return;
      }
    }
  }
  document.addEventListener('keydown', onKey);
  const onResize = () => drawStrip();
  window.addEventListener('resize', onResize);
  onLeave(() => {
    document.removeEventListener('keydown', onKey);
    window.removeEventListener('resize', onResize);
    video.pause();
    video.removeAttribute('src');
    video.load();
  });

  // ---------- page ----------
  const help = h('details', { class: 'panel' },
    h('summary', { class: 'panel-head', style: 'cursor:pointer' }, h('h2', {}, 'Keys')),
    h('dl', { class: 'kv panel-body' }, KEYS_HELP.flatMap(([k, v]) => [h('dt', { class: 'mono' }, k), h('dd', {}, v)])));

  view.replaceChildren(h('div', { class: 'cols code-layout' },
    h('div', { class: 'col-main' },
      h('section', { class: 'panel code-video-panel' },
        h('div', { class: 'panel-head' }, h('h1', {}, `${match.team_a_name} v ${match.team_b_name}`),
          h('span', { class: 'muted mono', style: 'font-size:12px' }, match.video_name)),
        h('div', { class: 'panel-body code-player' }, video, seek, transport, strip)),
      h('section', { class: 'panel code-list-panel' },
        h('div', { class: 'panel-head' }, h('h2', {}, 'Instances'), countLbl, h('div', { class: 'spacer' }), csvLink),
        list)),
    h('div', { class: 'col-side code-side' },
      h('section', { class: 'panel' },
        h('div', { class: 'panel-head' }, h('h2', {}, 'Code window'), tplSel, h('div', { class: 'spacer' }), editBtn),
        padBody),
      help)));

  fillTplSel(null);
  renderList();
  const want = data.template || lsGet(LS_TEMPLATE);
  await loadTemplate(templates.some((t) => t.name === want) ? want : templates[0].name);
  timeLbl.textContent = `${fmt(0)} / ${fmt(match.duration_s)}`;
  drawStrip();
}
