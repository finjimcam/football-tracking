// Library tab (issue #16): teams, squads (numbers per season), competitions, seasons,
// and which match video belongs to which season and teams. Backed by config/library.json
// through app/library.py -> tools/library.py, which checks every change before saving.

export async function viewLibrary(view, parts, params, ui) {
  const { h, api, toast, renderTabs, setCrumb } = ui;
  renderTabs('library');
  setCrumb(null);
  let data;
  try { data = await api('/library'); } catch (e) {
    view.replaceChildren(h('div', { class: 'placeholder' }, h('h1', {}, 'Library unavailable'), h('p', {}, e.message)));
    return;
  }
  let videos = data.videos;
  const st = { sel: parts[1] || 'matches', season: params.get('season') || null, openMatch: params.get('match') || null };
  const nav = h('div', { class: 'lib-nav' });
  const main = h('div', { class: 'panel-body lib-main' });
  view.replaceChildren(h('div', { class: 'lib-layout' },
    h('section', { class: 'panel', 'aria-label': 'Library sections' },
      h('div', { class: 'panel-head' }, h('h1', {}, 'Library')), nav),
    h('section', { class: 'panel', 'aria-live': 'polite' }, main)));

  const L = () => data.library;
  const seasonLabel = (sid) => {
    const s = L().seasons[sid];
    return s ? `${(L().competitions[s.competition] || {}).name || '?'} ${s.name}` : '';
  };
  const byName = (obj) => Object.entries(obj).sort((a, b) => (a[1].name || '').localeCompare(b[1].name || ''));
  const go = (sel, extra = '') => { history.replaceState(null, '', `#/library/${sel}${extra}`); st.sel = sel; draw(); };
  async function write(path, method, body, okMsg) {
    try {
      const res = await api('/library' + path, { method, body });
      data = { ...data, ...res, videos };
      if (okMsg) toast(okMsg);
      return res;
    } catch (e) { toast(e.message, true); return null; }
  }
  async function reload() {
    try { data = await api('/library'); videos = data.videos; } catch (e) { toast(e.message, true); }
  }
  const field = (label, input) => h('label', { class: 'lib-field' }, h('span', {}, label), input);
  const input = (value, attrs = {}) => h('input', { class: 'lib-input', value: value ?? '', ...attrs });
  const select = (options, value, attrs = {}) => h('select', { class: 'lib-input', ...attrs },
    options.map(([v, label]) => h('option', { value: v, selected: v === (value ?? '') || null }, label)));

  // ---------- nav ----------
  function drawNav() {
    const item = (id, label, extra) => h('a', {
      class: 'lib-nav-item' + (st.sel === id ? ' sel' : ''), href: `#/library/${id}`, 'aria-current': st.sel === id ? 'page' : null,
      onclick: (ev) => { ev.preventDefault(); go(id); },
    }, label, extra ? h('span', { class: 'dim mono' }, extra) : null);
    nav.replaceChildren(
      item('matches', 'Matches', String(videos.length)),
      item('competitions', 'Competitions & seasons', String(Object.keys(L().seasons).length)),
      h('div', { class: 'lib-nav-head' }, 'Teams'),
      ...byName(L().teams).map(([tid, t]) => item('team:' + tid, h('span', { class: 'lib-team-name' },
        h('i', { class: 'lib-swatch', style: `background:${t.color || '#555'}` }), t.name))),
      h('button', { type: 'button', class: 'btn small lib-add', onclick: addTeam }, '+ Team'));
  }
  async function addTeam() {
    const name = prompt('Team name');
    if (!name) return;
    const res = await write('/teams', 'POST', { name }, `Added ${name}`);
    if (res) go('team:' + res.id);
  }

  // ---------- matches ----------
  function drawMatches() {
    const seasons = [['', '– season –'], ...Object.keys(L().seasons).map((sid) => [sid, seasonLabel(sid)])];
    const teams = [['', '– team –'], ...byName(L().teams).map(([tid, t]) => [tid, t.name])];
    const rows = videos.map((v) => {
      const m = L().matches[v.match_id] || {};
      const open = st.openMatch === v.match_id;
      const head = h('button', { type: 'button', class: 'lib-match-head', 'aria-expanded': String(open),
        onclick: () => { st.openMatch = open ? null : v.match_id; draw(); } },
      h('div', { class: 'lib-match-title' }, m.team_a && m.team_b ? `${L().teams[m.team_a].name} v ${L().teams[m.team_b].name}` : v.title),
      h('div', { class: 'muted lib-match-sub' }, [m.date, seasonLabel(m.season), m.round].filter(Boolean).join(' · ') || v.video_name),
      v.linked ? h('span', { class: 'pill done' }, 'linked') : h('span', { class: 'pill stale' }, 'not linked'));
      if (!open) return h('div', { class: 'lib-match' }, head);
      const f = {
        season: select(seasons, m.season), round: input(m.round, { placeholder: 'Gameweek 26' }),
        date: input(m.date, { type: 'date' }), venue: input(m.venue), result: input(m.result, { placeholder: 'Man City 2-3 Tottenham' }),
        team_a: select(teams, m.team_a), team_b: select(teams, m.team_b),
        home: select([['', '–'], ['team_a', 'Team A'], ['team_b', 'Team B']], m.home),
      };
      const lineup = (side) => {
        const lu = (m.lineups || {})[side] || [];
        if (!lu.length) return null;
        return h('div', { class: 'lib-lineup' }, h('div', { class: 'section-label' }, `${(L().teams[m[side]] || {}).name || side} lineup`),
          h('div', { class: 'lib-chips' }, lu.map((e) => h('span', { class: 'lib-chip' + (e.starter === false ? ' sub' : '') },
            h('b', { class: 'mono' }, e.number ?? '–'), ' ', (L().players[e.player] || {}).name || e.player))));
      };
      return h('div', { class: 'lib-match open' }, head,
        h('div', { class: 'lib-match-body' },
          h('div', { class: 'lib-grid' },
            field('Season', f.season), field('Round', f.round), field('Date', f.date), field('Venue', f.venue),
            field('Team A (first in the match config)', f.team_a), field('Team B', f.team_b), field('Home side', f.home), field('Result', f.result)),
          h('div', { class: 'lib-files' },
            h('div', { class: 'section-label' }, 'Video and data'),
            h('div', { class: 'mono dim' }, m.video || v.video_name || '–'),
            ...(m.data || []).map((p) => h('div', { class: 'mono dim' }, p))),
          lineup('team_a'), lineup('team_b'),
          h('div', { class: 'lib-actions' },
            h('button', { type: 'button', class: 'btn small primary', onclick: async () => {
              const body = Object.fromEntries(Object.entries(f).map(([k, el]) => [k, el.value || null]));
              if (await write(`/matches/${encodeURIComponent(v.match_id)}`, 'PUT', body, 'Match saved')) { v.linked = true; draw(); }
            } }, 'Save'),
            v.has_config ? h('button', { type: 'button', class: 'btn small', title: 'Teams, lineups, season and data files from config/<id>_match.json and _reference.json',
              onclick: async () => {
                const res = await write(`/import/${encodeURIComponent(v.match_id)}`, 'POST', null);
                if (res) {
                  const a = res.imported.added;
                  toast(`Imported: ${a.teams} new teams, ${a.players} new players, ${res.imported.lineup_players} in the lineups`);
                  v.linked = true; draw();
                }
              } }, 'Import from match config') : null)));
    });
    main.replaceChildren(
      h('div', { class: 'lib-title' }, h('h2', {}, 'Matches'),
        h('span', { class: 'muted' }, 'Link each match video to a season and its two teams.')),
      rows.length ? h('div', { class: 'lib-matches' }, rows) : h('div', { class: 'empty' }, 'No matches ingested yet.'));
  }

  // ---------- competitions ----------
  function drawCompetitions() {
    const comps = byName(L().competitions);
    main.replaceChildren(
      h('div', { class: 'lib-title' }, h('h2', {}, 'Competitions & seasons'),
        h('button', { type: 'button', class: 'btn small', onclick: async () => {
          const name = prompt('Competition name (e.g. Premier League)');
          if (name) { await write('/competitions', 'POST', { name }, `Added ${name}`); draw(); }
        } }, '+ Competition')),
      comps.length ? h('div', {}, comps.map(([cid, c]) => {
        const seasons = Object.entries(L().seasons).filter(([, s]) => s.competition === cid).sort((a, b) => b[1].name.localeCompare(a[1].name));
        return h('div', { class: 'lib-comp' },
          h('div', { class: 'lib-comp-head' }, h('b', {}, c.name), h('div', { class: 'spacer' }),
            h('button', { type: 'button', class: 'btn small', onclick: async () => {
              const name = prompt(`New season of ${c.name} (e.g. 2022-23)`);
              if (name) { await write('/seasons', 'POST', { competition: cid, name }, `Added ${c.name} ${name}`); draw(); }
            } }, '+ Season'),
            h('button', { type: 'button', class: 'btn small', onclick: async () => {
              const name = prompt('Rename competition', c.name);
              if (name && name !== c.name) { await write(`/competitions/${cid}`, 'PUT', { name }); draw(); }
            } }, 'Rename'),
            h('button', { type: 'button', class: 'btn small danger', 'aria-label': `Delete ${c.name}`, onclick: async () => {
              if (confirm(`Delete ${c.name}?`)) { await write(`/competitions/${cid}`, 'DELETE', null, 'Deleted'); draw(); }
            } }, '✕')),
          seasons.length ? h('div', { class: 'lib-seasons' }, seasons.map(([sid, s]) => {
            const squads = Object.keys(L().squads[sid] || {}).length;
            const nm = Object.values(L().matches).filter((m) => m.season === sid).length;
            return h('div', { class: 'lib-season' }, h('span', { class: 'mono' }, s.name),
              h('span', { class: 'dim' }, `${squads} squad${squads === 1 ? '' : 's'} · ${nm} match${nm === 1 ? '' : 'es'}`),
              h('div', { class: 'spacer' }),
              h('button', { type: 'button', class: 'btn small danger', 'aria-label': `Delete season ${s.name}`, onclick: async () => {
                if (confirm(`Delete ${c.name} ${s.name}?`)) { await write(`/seasons/${sid}`, 'DELETE', null, 'Deleted'); draw(); }
              } }, '✕'));
          })) : h('div', { class: 'muted', style: 'padding:6px 0' }, 'No seasons yet.'));
      })) : h('div', { class: 'empty' }, 'No competitions yet. Add one, or import a match from the Matches page.'));
  }

  // ---------- team + squad ----------
  function drawTeam(tid) {
    const t = L().teams[tid];
    if (!t) { go('matches'); return; }
    const f = { name: input(t.name), short: input(t.short, { maxlength: 4, style: 'max-width:90px' }),
      color: input(t.color || '#888888', { type: 'color', style: 'max-width:60px;padding:0' }), kit: input(t.kit, { placeholder: 'sky-blue shirts, white shorts' }) };
    const seasonIds = Object.keys(L().seasons).sort((a, b) => seasonLabel(b).localeCompare(seasonLabel(a)));
    if (!st.season || !L().seasons[st.season]) st.season = seasonIds.find((s) => (L().squads[s] || {})[tid]) || seasonIds[0] || null;
    const seasonSel = select(seasonIds.map((s) => [s, seasonLabel(s)]), st.season, { 'aria-label': 'Season', onchange: () => { st.season = seasonSel.value; draw(); } });
    const squad = ((L().squads[st.season] || {})[tid] || []).map((e) => ({ ...e, name: (L().players[e.player] || {}).name, positions: (L().players[e.player] || {}).positions || [] }));
    const body = h('tbody');
    const posOpts = [['', '–'], ...data.positions.map((p) => [p, p])];
    const listId = 'lib-players-' + tid;
    const addRow = (e = {}) => {
      const num = input(e.number, { type: 'number', min: 1, max: 99, class: 'lib-input lib-num', 'aria-label': 'Shirt number' });
      const nm = input(e.name, { list: listId, 'aria-label': 'Player name', placeholder: 'Player name' });
      const pos = select(posOpts, (e.positions || [])[0], { 'aria-label': 'Position' });
      const tr = h('tr', { 'data-player': e.player || '' },
        h('td', {}, num), h('td', {}, nm), h('td', {}, pos),
        h('td', {}, h('button', { type: 'button', class: 'btn small danger', 'aria-label': 'Remove from squad', onclick: () => tr.remove() }, '✕')));
      tr._get = () => {
        const name = nm.value.trim();
        const pid = tr.dataset.player && name === (e.name || '') ? tr.dataset.player : null;
        return { player: pid, name: pid ? null : name, number: num.value ? Number(num.value) : null, positions: pos.value ? [pos.value] : [] };
      };
      body.append(tr);
      return nm;
    };
    squad.sort((a, b) => (a.number ?? 999) - (b.number ?? 999)).forEach((e) => addRow(e));
    main.replaceChildren(
      h('div', { class: 'lib-title' }, h('h2', {}, t.name), h('div', { class: 'spacer' }),
        h('button', { type: 'button', class: 'btn small danger', onclick: async () => {
          if (confirm(`Delete ${t.name}? Only possible when no squad or match uses it.`)) {
            if (await write(`/teams/${tid}`, 'DELETE', null, 'Deleted')) go('matches');
          }
        } }, 'Delete team')),
      h('div', { class: 'lib-grid' }, field('Name', f.name), field('Short', f.short), field('Colour', f.color), field('Kit', f.kit)),
      h('div', { class: 'lib-actions' }, h('button', { type: 'button', class: 'btn small primary', onclick: async () => {
        await write(`/teams/${tid}`, 'PUT', Object.fromEntries(Object.entries(f).map(([k, el]) => [k, el.value])), 'Team saved');
        draw();
      } }, 'Save team')),
      h('div', { class: 'lib-title', style: 'margin-top:18px' }, h('h3', {}, 'Squad'),
        seasonIds.length ? seasonSel : h('span', { class: 'muted' }, 'Add a competition and season first.')),
      st.season ? h('div', {},
        h('datalist', { id: listId }, byName(L().players).map(([, p]) => h('option', { value: p.name }))),
        h('div', { class: 'lib-table-wrap' }, h('table', { class: 'lib-table' },
          h('thead', {}, h('tr', {}, h('th', {}, 'No.'), h('th', {}, 'Name'), h('th', {}, 'Position'), h('th', {}))), body)),
        h('div', { class: 'lib-actions' },
          h('button', { type: 'button', class: 'btn small', onclick: () => addRow().focus() }, '+ Player'),
          h('button', { type: 'button', class: 'btn small primary', onclick: async () => {
            const entries = [...body.children].map((tr) => tr._get()).filter((e) => e.player || e.name);
            await write(`/squads/${encodeURIComponent(st.season)}/${encodeURIComponent(tid)}`, 'PUT', { entries }, 'Squad saved');
            draw();
          } }, 'Save squad')),
        h('p', { class: 'muted', style: 'font-size:12px' },
          'Numbers belong to this season\'s squad. A name you type that\'s already in the library is linked to that player, so their history follows them between teams. ',
          'The Lines tab\'s Identify picker lists the match lineup and then the rest of the squad, so you can name substitutes too.')) : null);
  }

  function draw() {
    drawNav();
    if (st.sel.startsWith('team:')) drawTeam(st.sel.slice(5));
    else if (st.sel === 'competitions') drawCompetitions();
    else drawMatches();
  }
  draw();
}
