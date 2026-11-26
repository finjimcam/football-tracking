# Clips: set pieces, shots & key chances, passes into the final third

Issue #58. For any broadcast match tracked down to **Compute possession**, this
finds the moments worth a clip, lists them in the Lines tab's **Clips** panel
under headings, and exports the ones you keep as a zip of MP4s.

| Heading | Subheadings |
|---|---|
| Set pieces | Goal kicks, Throw-ins, Corners, Free kicks (Penalties when there are any) |
| Shots & key chances | Shots (goals included), Box entries |
| Passes into the final third | Passes |

## Inputs

- The match's `tracks.parquet`, `player_map.json`, `ball_track.parquet` and
  `possession_timeline.parquet`. These come from
  [track_identities.md](track_identities.md) down to `compute_possession.py
  --include-unidentified`. For a whole match tracked by period
  (`track_periods.py`), the stages run on the combined files.
- `config/<id>_match.json`: `clock_sync.kickoff_frame`, and for later periods
  `periods.<name>.kickoff_frame` and `clock_start_s`. These turn published
  minutes into frames and label the clips with the match clock.
- Optional, but worth the five minutes: published minute-stamped events in
  `config/<id>_reference.json` (see below).

## Steps

1. **Collect published events (recommended).** Find the match on ESPN (the
   match page's shot map, plus `/soccer/commentary/_/gameId/<id>`), BBC live
   text, FotMob or a minute-by-minute report. Add what has a minute to the
   reference as `match_events`, with a `match_events_note` saying what it covers
   and where it came from:

   ```json
   "match_events": [
     {"minute": "12'", "type": "shot", "team": "team_a", "player": "Son Heung-min", "outcome": "saved", "source": "ESPN shot map"},
     {"minute": "59'", "type": "goal", "team": "team_b", "player": "Ladislav Krejci", "source": "ESPN shot map"},
     {"minute": "77'", "type": "free_kick", "team": "team_b", "detail": "left channel, attacking half", "source": "ESPN commentary"},
     {"minute": "45'+2'", "type": "corner", "team": "team_a", "source": "BBC live text"}
   ]
   ```

   - `type` is one of `shot`, `goal`, `free_kick`, `throw_in`, `corner`,
     `goal_kick` or `penalty`.
   - `team` is `team_a` or `team_b`, as in the match config.
   - Added time is written `45'+2'`.
   - Write a shot map's outcome down as it is. Note it when it contradicts the
     score: ESPN's shot map labelled two non-goals as goals for Korea v Czechia.

2. **Detect.** Run *Detect clips* on the match page, or:

   ```
   python tools/detect_clips.py --match-id <id>
   ```

   It takes a few seconds and writes `.tmp/<id>/clips.json`. Re-running keeps
   your keep / drop / trim choices for any clip it finds again (same type and
   frame).

3. **Check in the Lines tab → Clips.** Pick the team at the top. Each clip
   plays from its start when you click it.
   - Untick the wrong ones. *Drop all* / *Keep all* work on a heading or a
     subheading.
   - Trim a clip with `[` / `]`.
   - Clips with a published event show its minute, player and outcome.

4. **Export.** Press *Export clips (zip)*, or run *Export clips (zip)* on the
   match page, or:

   ```
   python tools/export_clips.py --match-id <id> --team team_b
   ```

   It writes `outputs/videos/<id>_<team>_clips.zip`: one MP4 per kept clip in
   `Set pieces/Corners/…` style folders, plus `clips.csv`. Download it from
   the panel. Clips the detector couldn't give a team go to both teams.

## How the detection works (and where it fails)

- **Set pieces are restarts.**
  - **Stoppage:** play stops when the identified players' median speed stays
    under 1.3 m/s for 3 s.
  - **Restart:** then the first "dead ball" near the end of the stoppage: the
    ball still for 0.3 s, then kicked away without jumping.
  - **Type, from where the ball was:**
    - within 4 m of a corner flag: corner;
    - in or around a goal area: goal kick;
    - on or over a touchline: throw-in;
    - the penalty spot: penalty;
    - the centre: kick-off (not listed);
    - anywhere else: free kick.
  - **Ball not seen** (a corner taken off camera, a throw-in from the far
    side), so the type comes from where the players stand:
    - an outfield player held at a corner flag: corner;
    - one over a touchline: throw-in, but only when his team then has the ball
      within 20 m;
    - a keeper in his goal area with at most one opponent in the box: goal kick;
    - both teams in their own halves with someone on the centre spot: kick-off.
  - **Quick free kicks:** a dead ball outside any stoppage counts as a free
    kick if it was still for 1.5 s.
- **Clutter:** still "balls" that turn up again and again in the same spot (a
  painted penalty spot, a spare ball behind the boards) are ignored. That
  means a real penalty is missed too: add it from published events.
- **Shots** come from `detect_events.detect()`, run per period because teams
  change ends. **Box entries** are a team's spells with the ball in the box it
  attacks. A box entry followed by a shot within 6 s is one clip.
- **Final-third passes** are completed passes (same team, ball moved ≥ 4 m)
  from more than 35 m out to within 35 m of the goal line the team attacks.
- **Published events:**
  - Each event goes to a detected clip of that team inside its minute, ±30 s.
  - A set piece of another type is relabelled.
  - An event with no match becomes its own clip: at the team's closest
    approach to goal for a shot, or at the nearest stoppage end for a set
    piece.
  - A **goal** is placed from the kick-off that follows it. Play only stops
    once the celebration is over, 45–65 s after the goal, so the goal is the
    scoring team's last chance 20–90 s before that stop. Goal clips run from
    10 s before to 8 s after.
- **Wide views:** zoomed-out wide views (kick-offs, some replays) are skipped
  for the player-position cues, because positions there are metres out.

### Korea v Czechia (first run, 26 Nov 2026)

- **Coverage:** the ball was found in ~56% of frames, and ESPN had 22 shots
  with minutes but commentary only from 64'.
- **Clip counts:**
  - Korea: 4 goal kicks, 5 throw-ins, 5 corners, 6 free kicks, 26 shots, 58
    final-third passes.
  - Czechia: 4 goal kicks, 2 throw-ins, 10 free kicks, 18 shots, 10
    final-third passes.
- **Corners:** the real totals were 4 and 5. Corners were labelled to Korea
  only, so check the team.
- **Throw-ins:** most were missed. Far-side throw-ins are rarely seen
  (far-side ball detection is weak).
- **Published events:** 20 of 26 matched a detected moment. All three goals
  landed on the goal.
- **Main gap:** throw-ins and goal kicks have no published timings. A ball
  detector tuned for still balls near the lines, or a classifier trained on
  the clips kept and dropped in the panel, is the way to improve them (issue
  #7).

## Files

- **Tools:**
  - `tools/detect_clips.py`: detection, published-event matching; writes `.tmp/<id>/clips.json`.
  - `tools/export_clips.py`: the zip; also owns the export file helpers.
- **App:**
  - `app/clips.py`: Clips panel API.
  - `app/static/clips.js`: the panel; `#/lines/<id>?view=clips`.
- **Choices:** `config/<id>_clips_export.json`, `{"clips": {"<id>": {"keep", "in", "out"}}}`.
- **Tests:** `tests/test_detect_clips.py`.
