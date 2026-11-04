// Stopframe front end: plain ES module, no build step. Talks to app/server.py.
const REPO = 'https://github.com/finjimcam/football-tracking/issues/';
const WORKFLOW = 'https://github.com/finjimcam/football-tracking/blob/master/workflows/';
const TABS = [
  { id: 'matches', name: 'Matches' },
  { id: 'jobs', name: 'Jobs' },
  { id: 'code', name: 'Code', issue: 3, blurb: 'Configurable code buttons for tagging instances while the video plays.' },
  { id: 'review', name: 'Review', issue: 1, blurb: 'Frame-accurate video player with the tagged timeline underneath.' },
  { id: 'sorter', name: 'Sorter', issue: 4, blurb: 'Every tagged instance in one list: edit, relabel, trim and delete.' },
  { id: 'matrix', name: 'Matrix', issue: 11, blurb: 'Filter and cross-tabulate instances by code and label.' },
  { id: 'present', name: 'Present', issue: 12, blurb: 'Playlists and clip reels for team talks.' },
];
const ACTIVE = new Set(['queued', 'waiting', 'running']);
const TERMINAL = new Set(['done', 'failed', 'cancelled', 'interrupted']);

let STAGES = [];
let timers = [];
let routeToken = 0;

// ---------- helpers ----------
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'html') el.innerHTML = v; // only ever used with static SVG strings
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
const ICON = {
  play: '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M7 4.5v15l13-7.5z"></path></svg>',
  stop: '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="5" y="5" width="14" height="14" rx="2"></rect></svg>',
  down: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3v12M7 10l5 5 5-5M5 21h14"></path></svg>',
  check: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#5DBB63" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12l5 5L20 7"></path></svg>',
};
const icon = (name) => h('span', { html: ICON[name], style: 'display:inline-flex' });

async function api(path, opts = {}) {
  const res = await fetch('/api' + path, {
    ...opts,
    headers: opts.body ? { 'Content-Type': 'application/json' } : undefined,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) throw new Error((data && data.detail) || `${res.status} ${res.statusText}`);
  return data;
}

function toast(msg, isError) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast show' + (isError ? ' error' : '');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.className = 'toast'; }, 4000);
}

const pad = (n) => String(n).padStart(2, '0');
function dur(s) {
  if (s == null) return '—';
  s = Math.round(s);
  const hh = Math.floor(s / 3600), mm = Math.floor((s % 3600) / 60), ss = s % 60;
  return hh ? `${hh}:${pad(mm)}:${pad(ss)}` : `${pad(mm)}:${pad(ss)}`;
}
function when(ts) {
  if (!ts) return '—';
  const d = new Date(ts * 1000);
  const today = new Date().toDateString() === d.toDateString();
  return today ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : d.toLocaleDateString([], { day: 'numeric', month: 'short' }) + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
function bytes(n) {
  if (n == null) return '';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n >= 10 || i === 0 ? 0 : 1)} ${u[i]}`;
}
function every(ms, fn) {
  const token = routeToken;
  const id = setInterval(() => { if (token === routeToken) fn(); }, ms);
  timers.push(id);
  return id;
}
function stateLabel(state) {
  return { done: 'done', stale: 'stale', ready: 'to do', blocked: 'blocked', skip: 'n/a' }[state] || state;
}
const pill = (state, text) => h('span', { class: 'pill ' + state }, text || state);

// ---------- top bar ----------
function renderTabs(active) {
  const nav = document.getElementById('tabs');
  nav.replaceChildren(...TABS.map((t) => h('a', {
    class: 'tab', href: '#/' + t.id, 'aria-current': t.id === active ? 'page' : null,
  }, t.name, t.issue ? h('span', { class: 'soon' }, '#' + t.issue) : null)));
}
function setCrumb(match) {
  const c = document.getElementById('crumb');
  if (!match) { c.replaceChildren(); return; }
  c.replaceChildren(
    h('span', { class: 'sep' }, '/'),
    h('a', { href: '#/match/' + encodeURIComponent(match.match_id), style: 'color:inherit;text-decoration:none' },
      `${match.team_a_name} v ${match.team_b_name}`),
    h('span', { class: 'tag' }, dur(match.duration_s)),
  );
}
async function refreshActivity() {
  try {
    const jobs = await api('/jobs');
    const running = jobs.filter((j) => j.status === 'running').length;
    const waiting = jobs.filter((j) => j.status === 'queued' || j.status === 'waiting').length;
    const el = document.getElementById('activity');
    if (!running && !waiting) {
      el.replaceChildren(icon('check'), 'Idle');
    } else {
      el.replaceChildren(pill(running ? 'running' : 'queued',
        [running && `${running} running`, waiting && `${waiting} queued`].filter(Boolean).join(' · ')));
    }
  } catch { /* server away; next tick retries */ }
}
async function refreshDisk() {
  try {
    const sys = await api('/system');
    const d = sys.disk[0];
    const el = document.getElementById('disk');
    if (!d) return;
    el.textContent = `${bytes(d.free)} free`;
    el.title = `${d.path}: ${bytes(d.free)} free of ${bytes(d.total)}`;
    el.classList.toggle('low', d.free < 2 * 1024 ** 3);
  } catch { /* ignore */ }
}

// ---------- job widgets ----------
function progressBar(job) {
  const p = job.progress;
  if (job.status === 'running' && (!p || !p.total)) {
    return h('div', { class: 'progress indeterminate', role: 'progressbar', 'aria-label': 'Working' }, h('div'));
  }
  if (!p || !p.total) return null;
  const pct = Math.min(100, (100 * p.done) / p.total);
  return h('div', { class: 'progress', role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': 100, 'aria-valuenow': Math.round(pct), 'aria-label': p.label || 'Progress' },
    h('div', { style: `width:${pct}%` }));
}
function progressText(job) {
  const p = job.progress;
  if (!p || !p.total) return null;
  return `${p.label ? p.label + ' · ' : ''}${p.done.toLocaleString()} / ${p.total.toLocaleString()} (${Math.floor((100 * p.done) / p.total)}%)`;
}

function jobRow(job, href, current) {
  return h('a', { class: 'job-row', href, 'aria-current': current ? 'true' : null },
    h('span', { style: 'font-weight:600;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap' }, job.title),
    pill(job.status),
    h('span', { class: 'sub' }, [when(job.created), job.match_id && !current ? job.match_id : null, progressText(job)].filter(Boolean).join(' · ')),
    job.status === 'running' ? h('div', { style: 'grid-column:1/-1' }, progressBar(job)) : null,
  );
}

// Live job panel: status, progress, log tail, cancel / run-now.
function jobDetail(jobId, onChange) {
  const root = h('div', { class: 'job-detail' });
  const head = h('div', { style: 'display:flex;flex-wrap:wrap;align-items:center;gap:8px' });
  const prog = h('div');
  const info = h('dl', { class: 'kv' });
  const log = h('pre', { class: 'log', tabindex: '0', 'aria-label': 'Job log' });
  root.append(head, prog, info, log);
  let offset = 0, lastStatus = null, stopped = false;

  function draw(job) {
    const actions = [];
    if (ACTIVE.has(job.status)) {
      actions.push(h('button', { type: 'button', class: 'btn small danger', onclick: async () => {
        try { await api(`/jobs/${job.id}/cancel`, { method: 'POST' }); poll(); } catch (e) { toast(e.message, true); }
      } }, icon('stop'), 'Cancel'));
    }
    if (job.status === 'waiting') {
      actions.push(h('button', { type: 'button', class: 'btn small', onclick: async () => {
        try { await api(`/jobs/${job.id}/force`, { method: 'POST' }); poll(); } catch (e) { toast(e.message, true); }
      } }, 'Run now'));
    }
    head.replaceChildren(h('h2', { style: 'margin:0;font-size:14px' }, job.title), pill(job.status),
      h('div', { class: 'spacer' }), ...actions);
    prog.replaceChildren(...[progressBar(job), progressText(job) && h('div', { class: 'dim mono', style: 'font-size:11px;margin-top:4px' }, progressText(job)),
      job.note && h('div', { class: 'note' }, job.note)].filter(Boolean));
    const rows = [
      ['Tool', `tools/${job.tool}`],
      job.match_id && ['Match', job.match_id],
      ['Queued', when(job.created)],
      job.started && ['Started', when(job.started)],
      job.finished && ['Took', dur((job.finished) - (job.started || job.created))],
      job.returncode != null && ['Exit', String(job.returncode)],
    ].filter(Boolean);
    info.replaceChildren(...rows.flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, v)]));
  }

  async function poll() {
    if (stopped) return;
    try {
      const job = await api(`/jobs/${jobId}?offset=${offset}`);
      if (job.log) {
        const stick = log.scrollTop + log.clientHeight >= log.scrollHeight - 24;
        log.append(job.log);
        if (stick) log.scrollTop = log.scrollHeight;
      }
      offset = job.offset;
      draw(job);
      if (lastStatus && lastStatus !== job.status && onChange) onChange(job);
      lastStatus = job.status;
      if (TERMINAL.has(job.status) && !job.log) { stopped = true; clearInterval(timer); }
    } catch (e) {
      log.append(`\n[${e.message}]\n`);
      stopped = true; clearInterval(timer);
    }
  }
  const timer = every(1000, poll);
  poll();
  return root;
}

// ---------- param forms ----------
function stageForm(stage, { matchId, video, onSubmitted, onCancel }) {
  const err = h('div', { class: 'form-error', role: 'alert' });
  const inputs = {};
  const fields = stage.params.filter((p) => p.kind !== 'video').map((p) => {
    const id = `f-${stage.id}-${p.name}-${Math.random().toString(36).slice(2, 7)}`;
    if (p.kind === 'flag') {
      const input = h('input', { type: 'checkbox', id, checked: p.default === true || null });
      inputs[p.name] = input;
      return h('label', { class: 'field check', for: id, title: p.help || null }, input, p.label);
    }
    if (p.kind === 'choice') {
      const input = h('select', { id }, p.options.map((o) => h('option', { value: o, selected: o === p.default || null }, o)));
      inputs[p.name] = input;
      return h('label', { class: 'field', for: id }, p.label, input);
    }
    const input = h('input', {
      id, type: p.kind === 'int' || p.kind === 'float' ? 'number' : 'text',
      step: p.kind === 'float' ? 'any' : p.kind === 'int' ? '1' : null,
      placeholder: p.default != null ? String(p.default) : (p.help || 'default'),
      required: p.required || null, inputmode: p.kind === 'float' ? 'decimal' : null,
    });
    inputs[p.name] = input;
    return h('label', { class: 'field', for: id }, p.label, input);
  });
  const submit = h('button', { type: 'submit', class: 'btn primary small' }, icon('play'), stage.lane !== 'light' ? 'Queue' : 'Run');
  const form = h('form', { class: 'stage-form', onsubmit: async (ev) => {
    ev.preventDefault();
    err.textContent = '';
    const params = {};
    for (const [k, input] of Object.entries(inputs)) {
      if (input.type === 'checkbox') { if (input.checked) params[k] = true; }
      else if (String(input.value).trim() !== '') params[k] = String(input.value).trim();
    }
    if (video) params.video = video;
    submit.disabled = true;
    try {
      const job = await api('/jobs', { method: 'POST', body: { stage: stage.id, match_id: matchId, params } });
      toast(`${stage.title}: ${job.status}`);
      onSubmitted && onSubmitted(job);
    } catch (e) {
      err.textContent = e.message;
    } finally {
      submit.disabled = false;
    }
  } },
  fields.length ? h('div', { class: 'fields' }, fields) : h('div', { class: 'dim', style: 'font-size:12px' }, 'No options: runs with the tool\'s defaults.'),
  fields.length ? h('div', { class: 'dim', style: 'font-size:11px' }, 'Leave a box empty to use the tool\'s default.') : null,
  stage.lane !== 'light' ? h('div', { class: 'note' }, 'Heavy job: runs one at a time at low priority, and waits while the media server is playing.') : null,
  err,
  h('div', { class: 'form-actions' }, submit, h('button', { type: 'button', class: 'btn small', onclick: onCancel }, 'Close')));
  return form;
}

// ---------- views ----------
async function viewMatches(view) {
  renderTabs('matches');
  setCrumb(null);
  const matchesPanel = h('section', { class: 'panel', 'aria-labelledby': 'h-matches' });
  const videosPanel = h('section', { class: 'panel', 'aria-labelledby': 'h-videos' });
  view.replaceChildren(h('div', { class: 'cols' },
    h('div', { class: 'col-main' }, matchesPanel),
    h('div', { class: 'col-side' }, videosPanel)));
  const openForms = new Map(); // video name -> form element, kept across refreshes

  async function load() {
    let data;
    try { data = await api('/matches'); } catch (e) {
      matchesPanel.replaceChildren(h('div', { class: 'empty' }, `Couldn't reach the server: ${e.message}`));
      return;
    }
    const byId = Object.fromEntries(STAGES.map((s) => [s.id, s]));
    matchesPanel.replaceChildren(
      h('div', { class: 'panel-head' }, h('h1', { id: 'h-matches' }, 'Matches'), h('span', { class: 'muted' }, `${data.matches.length}`)),
      data.matches.length ? h('div', {}, data.matches.map((m) => h('a', { class: 'match-row', href: '#/match/' + encodeURIComponent(m.match_id) },
        h('div', { style: 'min-width:0;display:flex;flex-direction:column;gap:4px' },
          h('div', { class: 'match-title' },
            h('span', { class: 'chip', style: 'background:var(--home)' }), m.team_a_name,
            h('span', { class: 'dim', style: 'font-weight:400' }, 'v'),
            h('span', { class: 'chip', style: 'background:var(--away)' }), m.team_b_name),
          h('div', { class: 'match-meta' },
            h('span', {}, m.video_name), h('span', {}, dur(m.duration_s)),
            m.width && h('span', {}, `${m.height}p · ${Math.round(m.fps)} fps`),
            m.kickoff && h('span', {}, m.kickoff))),
        h('div', { style: 'display:flex;gap:6px;align-items:center' },
          m.active_stages.length ? pill('running', m.active_stages.length + ' job' + (m.active_stages.length > 1 ? 's' : '')) : null,
          m.artifacts.report.exists ? h('span', { class: 'tag' }, 'report') : null),
        h('div', { class: 'stage-dots', 'aria-hidden': 'true' }, m.stage_order.map((id) => byId[id]).map((s) => {
          const st = m.stages[s.id].state;
          const cls = m.active_stages.includes(s.id) ? 'active' : st;
          return h('span', { class: cls, title: `${s.title}: ${stateLabel(st)}` });
        })),
      )))
        : h('div', { class: 'empty' }, 'No matches yet. Ingest a video from the list on the right to start one.'),
    );

    videosPanel.replaceChildren(
      h('div', { class: 'panel-head' }, h('h2', { id: 'h-videos' }, 'Videos'), h('span', { class: 'muted mono', style: 'font-size:11px' }, 'videos/')),
      data.videos.length ? h('div', {}, data.videos.map((v) => {
        const row = h('div', { class: 'video-row' },
          h('span', { class: 'name' }, v.name, v.match_config ? h('span', { class: 'tag', style: 'margin-left:8px', title: v.match_config }, 'broadcast config') : null),
          h('span', { class: 'dim mono', style: 'font-size:11px' }, bytes(v.size)),
          v.match_id
            ? h('a', { class: 'btn small', href: '#/match/' + encodeURIComponent(v.match_id) }, 'Open')
            : h('button', { type: 'button', class: 'btn primary small', onclick: () => {
              if (openForms.has(v.name)) return;
              openForms.set(v.name, null);
              load();
            } }, 'Ingest'));
        const wrap = h('div', {}, row);
        if (openForms.has(v.name) && !v.match_id) {
          let form = openForms.get(v.name);
          if (!form) {
            form = stageForm(STAGES.find((s) => s.id === 'ingest'), {
              video: v.name,
              onSubmitted: () => { openForms.delete(v.name); load(); },
              onCancel: () => { openForms.delete(v.name); load(); },
            });
            form.style.paddingLeft = '14px';
            openForms.set(v.name, form);
          }
          wrap.append(form);
        }
        return wrap;
      }))
        : h('div', { class: 'empty' }, 'No videos found. Copy match videos (.mp4) into the repo\'s videos/ folder.'),
    );
  }
  await load();
  every(5000, () => { if (!document.activeElement || !document.activeElement.closest('form')) load(); });
}

async function viewMatch(view, matchId, params) {
  renderTabs('matches');
  let match;
  try { match = await api('/matches/' + encodeURIComponent(matchId)); } catch (e) {
    setCrumb(null);
    view.replaceChildren(h('div', { class: 'placeholder' }, h('h1', {}, 'Match not found'), h('p', {}, e.message), h('a', { class: 'btn', href: '#/matches' }, 'All matches')));
    return;
  }
  setCrumb(match);
  const selectedJob = params.get('job');
  const stageList = h('div');
  const outputs = h('div', { class: 'panel-body' });
  const jobsList = h('div', { class: 'scroll', style: 'max-height:280px;overflow:auto' });
  const detail = h('div', { class: 'panel-body' });
  const forms = new Map(); // stage id -> form (persistent while open)
  const formSlots = new Map();
  const rowSlots = new Map();

  const byId = Object.fromEntries(STAGES.map((s) => [s.id, s]));
  const perMatch = match.stage_order.map((id) => byId[id]);
  for (const s of perMatch) {
    const rowSlot = h('div');
    const formSlot = h('div');
    rowSlots.set(s.id, rowSlot);
    formSlots.set(s.id, formSlot);
    stageList.append(h('div', { class: 'stage' }, rowSlot, formSlot));
  }

  view.replaceChildren(h('div', { class: 'cols' },
    h('div', { class: 'col-main' },
      h('section', { class: 'panel', 'aria-labelledby': 'h-pipeline' },
        h('div', { class: 'panel-head' }, h('h1', { id: 'h-pipeline' }, 'Pipeline'),
          h('span', { class: 'tag' }, match.pipeline === 'broadcast' ? 'broadcast' : 'match report'),
          h('span', { class: 'muted mono', style: 'font-size:11px' }, match.match_id)),
        stageList),
      h('section', { class: 'panel', 'aria-labelledby': 'h-out' },
        h('div', { class: 'panel-head' }, h('h2', { id: 'h-out' }, 'Outputs')), outputs)),
    h('div', { class: 'col-side' },
      h('section', { class: 'panel', 'aria-labelledby': 'h-jobs' },
        h('div', { class: 'panel-head' }, h('h2', { id: 'h-jobs' }, 'Jobs')), jobsList),
      h('section', { class: 'panel', 'aria-label': 'Job detail' }, detail))));

  function toggleForm(stage) {
    const slot = formSlots.get(stage.id);
    if (forms.has(stage.id)) {
      forms.delete(stage.id);
      slot.replaceChildren();
      return;
    }
    const form = stageForm(stage, {
      matchId,
      onSubmitted: (job) => { forms.delete(stage.id); slot.replaceChildren(); location.hash = `#/match/${encodeURIComponent(matchId)}?job=${job.id}`; },
      onCancel: () => { forms.delete(stage.id); slot.replaceChildren(); },
    });
    forms.set(stage.id, form);
    slot.replaceChildren(form);
    const first = form.querySelector('input');
    if (first) first.focus();
  }

  function draw(m) {
    const active = new Map();
    for (const j of m.jobs) if (ACTIVE.has(j.status)) active.set(j.stage, j);
    for (const s of perMatch) {
      const st = m.stages[s.id];
      const job = active.get(s.id);
      let action;
      if (s.interactive) {
        action = h('a', {
          class: 'btn small', target: '_blank', rel: 'noopener', title: 'Not in the app yet: run it by hand as the workflow describes',
          href: WORKFLOW + (s.id === 'roster' ? 'curate_and_identify_players.md' : 'generate_match_report.md'),
        }, 'How to');
      } else if (job) {
        action = h('a', { class: 'btn small', href: `#/match/${encodeURIComponent(matchId)}?job=${job.id}` }, 'View job');
      } else {
        const blocked = st.state === 'blocked';
        action = h('button', {
          type: 'button', class: 'btn small' + (st.state === 'ready' ? ' primary' : ''),
          disabled: blocked || null, 'aria-expanded': forms.has(s.id) ? 'true' : 'false',
          title: blocked ? 'Needs: ' + st.missing.join(', ') : null,
          onclick: () => { toggleForm(s); draw(m); },
        }, st.state === 'done' || st.state === 'stale' ? 'Re-run' : 'Run');
      }
      rowSlots.get(s.id).replaceChildren(h('div', { class: 'stage-row' },
        h('span', { class: 'stage-num' }, String(s.step).padStart(2, '0')),
        h('div', { style: 'min-width:0' },
          h('div', { class: 'stage-title' }, s.title,
            s.heavy ? h('span', { class: 'tag heavy' }, 'GPU') : s.cpu_heavy ? h('span', { class: 'tag heavy' }, 'CPU') : null,
            s.interactive ? h('span', { class: 'tag' }, 'interactive') : null),
          h('div', { class: 'stage-summary' }, st.state === 'blocked' ? `Needs ${st.missing.join(', ')} first.` : (st.note || s.summary))),
        h('div', { class: 'stage-actions' }, job ? pill(job.status) : pill(st.state, stateLabel(st.state)), action)));
    }

    const outs = [
      ['report', 'Report PDF'], ['stats', 'Stats CSV'], ['tracking_video', 'Tracking video'],
      ['vs_reference', 'Vs published stats'], ['phase_index', 'Phase index CSV'],
      ['phase_video', 'Phase compilation'], ['shape_video', 'Picked-lines video'],
    ].filter(([k]) => m.artifacts[k].exists);
    outputs.replaceChildren(outs.length
      ? h('div', { class: 'outputs' }, outs.map(([k, label]) => h('a', {
        class: 'btn', href: `/api/matches/${encodeURIComponent(matchId)}/files/${k}`, target: k === 'stats' ? null : '_blank', rel: 'noopener',
      }, icon('down'), label, h('span', { class: 'dim mono', style: 'font-size:11px' }, bytes(m.artifacts[k].size)))))
      : h('div', { class: 'muted' }, 'Nothing yet. Run the pipeline down to "Build PDF report".'));

    const current = selectedJob || (m.jobs[0] && m.jobs[0].id);
    jobsList.replaceChildren(...(m.jobs.length
      ? m.jobs.map((j) => jobRow(j, `#/match/${encodeURIComponent(matchId)}?job=${j.id}`, j.id === current))
      : [h('div', { class: 'empty' }, 'No jobs for this match yet.')]));
    return current;
  }

  let shownJob = draw(match);
  if (shownJob) detail.replaceChildren(jobDetail(shownJob, () => refresh()));
  else detail.replaceChildren(h('div', { class: 'muted' }, 'Pick a stage and press Run. Its log shows up here.'));

  async function refresh() {
    try {
      const m = await api('/matches/' + encodeURIComponent(matchId));
      const cur = draw(m);
      if (cur && cur !== shownJob) {
        shownJob = cur;
        detail.replaceChildren(jobDetail(cur, () => refresh()));
      }
    } catch { /* keep the last view */ }
  }
  every(3000, refresh);
}

async function viewJobs(view, jobId) {
  renderTabs('jobs');
  setCrumb(null);
  const list = h('div', { class: 'scroll', style: 'max-height:70vh;overflow:auto' });
  const detail = h('div', { class: 'panel-body' });
  view.replaceChildren(h('div', { class: 'cols' },
    h('div', { class: 'col-side', style: 'flex:1 1 380px' },
      h('section', { class: 'panel', 'aria-labelledby': 'h-alljobs' },
        h('div', { class: 'panel-head' }, h('h1', { id: 'h-alljobs' }, 'Jobs')), list)),
    h('div', { class: 'col-main' }, h('section', { class: 'panel', 'aria-label': 'Job detail' }, detail))));
  async function load() {
    try {
      const jobs = await api('/jobs');
      list.replaceChildren(...(jobs.length ? jobs.map((j) => jobRow(j, '#/jobs/' + j.id, j.id === jobId))
        : [h('div', { class: 'empty' }, 'No jobs have run yet.')]));
    } catch (e) { list.replaceChildren(h('div', { class: 'empty' }, e.message)); }
  }
  await load();
  every(3000, load);
  detail.replaceChildren(jobId ? jobDetail(jobId, load) : h('div', { class: 'muted' }, 'Select a job to see its log.'));
}

function viewPlaceholder(view, tab) {
  renderTabs(tab.id);
  setCrumb(null);
  view.replaceChildren(h('div', { class: 'placeholder' },
    h('h1', {}, `${tab.name} isn't built yet`),
    h('p', {}, tab.blurb),
    h('a', { class: 'btn', href: REPO + tab.issue, target: '_blank', rel: 'noopener' }, `Tracked in issue #${tab.issue}`)));
}

// ---------- router ----------
async function route() {
  routeToken++;
  timers.forEach(clearInterval);
  timers = [];
  const view = document.getElementById('view');
  const raw = location.hash.replace(/^#\/?/, '');
  const [path, query] = raw.split('?');
  const parts = path.split('/').filter(Boolean).map(decodeURIComponent);
  const params = new URLSearchParams(query || '');
  const tab = TABS.find((t) => t.id === parts[0]);
  if (parts[0] === 'match' && parts[1]) await viewMatch(view, parts[1], params);
  else if (parts[0] === 'jobs') await viewJobs(view, parts[1]);
  else if (tab && tab.issue) viewPlaceholder(view, tab);
  else await viewMatches(view);
}

async function boot() {
  try { STAGES = await api('/stages'); } catch (e) {
    document.getElementById('view').replaceChildren(h('div', { class: 'placeholder' }, h('h1', {}, 'Server not reachable'), h('p', {}, e.message)));
    return;
  }
  window.addEventListener('hashchange', route);
  refreshActivity(); refreshDisk();
  setInterval(refreshActivity, 4000);
  setInterval(refreshDisk, 60000);
  route();
}
boot();
