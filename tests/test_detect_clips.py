"""Issue #58: clips of set pieces, shots & key chances and passes into the final
third for the Lines tab, found from the tracking (tools/detect_clips.py),
checked and trimmed in the Clips panel (app/clips.py) and exported as a zip
(tools/export_clips.py)."""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import detect_clips as D  # noqa: E402
import export_clips as X  # noqa: E402
from test_lines import FPS, MID, build_synthetic  # noqa: E402

L, W = 105.0, 68.0


@pytest.mark.parametrize("x, y, want", [
    (104.0, 67.0, "corner"), (1.0, 1.5, "corner"), (5.5, 34.0, "goal_kick"), (100.0, 30.0, "goal_kick"),
    (52.5, 34.0, "kick_off"), (11.0, 34.0, "penalty"), (40.0, 0.5, "throw_in"), (60.0, 69.0, "throw_in"),
    (70.0, 40.0, "free_kick"), (30.0, 10.0, "free_kick"),
])
def test_restart_type_from_where_the_ball_was(x, y, want):
    assert D.classify(x, y, L, W) == want


def ball_track(segments):
    """[(n_frames, x0, y0, vx, vy)] in metres / frame -> frames, x, y arrays."""
    f, x, y, t = [], [], [], 0
    for n, x0, y0, vx, vy in segments:
        for i in range(n):
            f.append(t)
            x.append(x0 + vx * i)
            y.append(y0 + vy * i)
            t += 1
    return np.array(f), np.array(x, float), np.array(y, float)


def test_dead_ball_is_still_then_kicked_away_continuously():
    # still for 2 s at (70, 40), then kicked at 0.6 m/frame
    f, x, y = ball_track([(50, 70, 40, 0, 0), (30, 70.3, 40, 0.6, 0)])
    runs = D.still_runs(f, x, y, FPS, min_s=0.5)
    assert len(runs) == 1 and runs[0][0] == 0 and abs(runs[0][2] - 70) < 0.1
    assert D.kicked(f, x, y, runs[0], FPS)
    # the 'ball' jumps 20 m (tracker switched to another ball): not a kick
    f2, x2, y2 = ball_track([(50, 70, 40, 0, 0), (30, 90, 40, 0.6, 0)])
    assert not D.kicked(f2, x2, y2, D.still_runs(f2, x2, y2, FPS, min_s=0.5)[0], FPS)


def test_clutter_is_a_still_ball_seen_again_and_again_over_the_match():
    runs = [(k * 25 * 60, k * 25 * 60 + 50, 11.0, 34.1, 0, 0) for k in range(6)]  # penalty spot, every minute
    runs.append((9000, 9050, 70.0, 40.0, 0, 0))  # one real dead ball
    cells = D.clutter_cells(runs, FPS)
    assert D.in_clutter(11.2, 34.0, cells) and not D.in_clutter(70.0, 40.0, cells)


def test_stoppages_are_long_slow_spells():
    frames = np.arange(1000)
    speed = np.where((frames >= 200) & (frames < 400), 0.8, 3.0)  # 8 s of strolling at 25 fps
    speed[600:650] = 0.5  # 2 s: too short
    assert D.stoppages(speed, frames, FPS, 1.3, 3.0) == [(200, 399)]


def rows(players, n=30):
    """Identified-player rows over n frames: {track_id: (team, gk, x, y)}."""
    out = []
    for f in range(n):
        for tid, (team, gk, x, y) in players.items():
            out.append({"frame": f, "track_id": tid, "team": team, "gk": gk, "pitch_x": x, "pitch_y": y})
    return pd.DataFrame(out)


GOALS = {"team_a": 105.0, "team_b": 0.0}  # team_a attacks x = 105


def test_taker_cue_corner_throw_in_and_goal_kick():
    corner = rows({9: ("team_a", False, 104.5, 67.0), 3: ("team_b", False, 100.0, 34.0)})
    assert D.taker_cue(corner, GOALS, L, W)[:2] == ("corner", "team_a")
    throw = rows({5: ("team_b", False, 40.0, -0.8), 6: ("team_a", False, 42.0, 5.0)})
    k, team, x, y = D.taker_cue(throw, GOALS, L, W)
    assert (k, team) == ("throw_in", "team_b") and abs(x - 40) < 0.1
    # team_a's keeper in his goal area (team_a defends x = 0), no team_b player in that box
    gk = rows({1: ("team_a", True, 4.0, 33.0), 7: ("team_b", False, 25.0, 30.0)})
    assert D.taker_cue(gk, GOALS, L, W)[:2] == ("goal_kick", "team_a")
    # two opponents still in the box: open play (one could be a box mis-projected by a metre)
    gk2 = rows({1: ("team_a", True, 4.0, 33.0), 7: ("team_b", False, 10.0, 30.0), 8: ("team_b", False, 12.0, 40.0)})
    assert D.taker_cue(gk2, GOALS, L, W) is None


def test_kickoff_cue_needs_both_teams_in_their_own_halves():
    lineup = {}
    for i in range(7):
        lineup[i] = ("team_a", False, 30.0 + i, 10.0 + 7 * i)   # team_a defends x = 0
        lineup[20 + i] = ("team_b", False, 75.0 - i, 10.0 + 7 * i)
    lineup[99] = ("team_a", False, 52.0, 34.0)  # on the centre spot
    assert D.kickoff_cue(rows(lineup), GOALS, L)
    for i in range(3):  # three of team_b in team_a's half: open play
        lineup[20 + i] = ("team_b", False, 20.0 + 5 * i, 34.0)
    assert not D.kickoff_cue(rows(lineup), GOALS, L)


CFG = {"clock_sync": {"kickoff_frame": 20},
       "periods": {"first_half": {"start_frame": 20, "end_frame": 86746},
                   "second_half": {"start_frame": 86760, "end_frame": 179867, "kickoff_frame": 86760,
                                   "clock_start_s": 2700}}}


@pytest.mark.parametrize("m, want", [("12'", (12, 0)), ("45'+2'", (45, 2)), ("90+4", (90, 4)), (67, (67, 0))])
def test_parse_minute(m, want):
    assert D.parse_minute(m) == want


def test_minute_frames_use_each_period_kickoff():
    fps = 30.0
    assert D.minute_frames("12'", CFG, fps) == (20 + int(11 * 60 * fps), 20 + int(12 * 60 * fps))
    # added time runs on from the end of the first half
    assert D.minute_frames("45'+2'", CFG, fps)[0] == 20 + int(46 * 60 * fps)
    # 59' = 13:00-14:00 after the second-half kick-off (clock 45:00)
    assert D.minute_frames("59'", CFG, fps)[0] == 86760 + int(13 * 60 * fps)


def test_published_events_tie_to_detected_clips_or_become_their_own():
    fps = 30.0
    f12 = D.minute_frames("12'", CFG, fps)[0]
    items = [{"frame": f12 + 900, "kind": "box_entry", "team": "team_a", "first": f12 + 900},
             {"frame": f12 + 600, "kind": "free_kick", "team": "team_b"}]
    tl = pd.DataFrame({"frame": np.arange(f12, f12 + 3600), "state": "team_a",  # minutes 12 and 13
                       "ball_x": np.linspace(60, 100, 3600), "ball_y": 34.0})
    events = [{"minute": "12'", "type": "shot", "team": "team_a", "player": "Son", "outcome": "saved"},
              {"minute": "12'", "type": "corner", "team": "team_b"},  # no corner found: relabel the restart
              {"minute": "13'", "type": "shot", "team": "team_a"},  # nothing detected: added at the best moment
              {"minute": "", "type": "shot", "team": "team_a"}]  # unusable
    out, rep = D.apply_published(items, events, CFG, fps, tl, [], lambda f: GOALS)
    assert rep == {"matched": 2, "added": 1, "skipped": 1}
    assert out[0]["kind"] == "shot" and out[0]["published"]["player"] == "Son"
    assert out[1]["kind"] == "corner" and out[1]["team"] == "team_b"
    added = out[2]
    assert added["how"].startswith("published minute") and added["frame"] == f12 + 3599  # closest to goal in the 13th minute


def test_goal_is_the_last_chance_before_play_stopped_for_the_celebration():
    fps = 30.0
    w0 = D.minute_frames("59'", CFG, fps)[0]
    stop = w0 + 60 * 30  # play stops a minute after the goal, then a kick-off
    items = [{"frame": stop - 70 * 30, "kind": "box_entry", "team": "team_b", "first": stop - 70 * 30},
             {"frame": stop - 50 * 30, "kind": "shot", "team": "team_b", "first": stop - 50 * 30},
             {"frame": stop - 5 * 30, "kind": "box_entry", "team": "team_b", "first": stop - 5 * 30}]
    stops = [(stop, stop + 300)]
    kickoffs = [stop + 300]
    out, _ = D.apply_published(items, [{"minute": "59'", "type": "goal", "team": "team_b"}], CFG, fps,
                               pd.DataFrame(columns=["frame", "state", "ball_x", "ball_y"]), stops,
                               lambda f: GOALS, kickoffs=kickoffs)
    goal = [i for i in out if i.get("published")]
    assert len(goal) == 1 and goal[0]["frame"] == stop - 50 * 30 and goal[0]["how"].startswith("goal")


def test_final_third_passes_start_outside_and_end_inside():
    ev = pd.DataFrame([
        {"frame": 10, "end_frame": 30, "type": "pass", "team": "team_a", "player": 7, "to_player": 9,
         "x": 50.0, "y": 30.0, "end_x": 80.0, "end_y": 20.0},   # into the attacking third (x >= 70)
        {"frame": 40, "end_frame": 60, "type": "pass", "team": "team_a", "player": 7, "to_player": 9,
         "x": 75.0, "y": 30.0, "end_x": 90.0, "end_y": 20.0},   # already in it
        {"frame": 70, "end_frame": 90, "type": "pass", "team": "team_b", "player": 3, "to_player": 4,
         "x": 50.0, "y": 30.0, "end_x": 30.0, "end_y": 20.0},   # team_b attacks x = 0
        {"frame": 95, "end_frame": 99, "type": "pass_lost", "team": "team_b", "player": 3, "to_player": 7,
         "x": 50.0, "y": 30.0, "end_x": 20.0, "end_y": 20.0},   # not completed
    ])
    out = D.final_third_passes(ev, lambda f: GOALS)
    assert [(p["frame"], p["team"]) for p in out] == [(10, "team_a"), (70, "team_b")]


def test_box_entry_followed_by_a_shot_is_one_chance():
    ev = pd.DataFrame([{"frame": 300, "type": "shot", "team": "team_a", "player": 9, "x": 95.0, "y": 34.0}])
    out = D.chances(ev, [(200, 260, "team_a"), (600, 650, "team_b")], 25.0)
    assert [(c["kind"], c["frame"], c["first"]) for c in out] == [("shot", 300, 200), ("box_entry", 600, 600)]
    assert out[0]["lead_s"] == 4.0


def test_export_plan_orders_by_heading_and_names_files():
    clips = [{"id": "pass-500", "group": "final_third", "kind": "pass", "team": "team_a", "frame": 500,
              "start": 400, "end": 600, "clock": "00:20"},
             {"id": "corner-900", "group": "set_pieces", "kind": "corner", "team": "team_a", "frame": 900,
              "start": 820, "end": 1100, "clock": "00:36"},
             {"id": "throw_in-100", "group": "set_pieces", "kind": "throw_in", "team": None, "frame": 100,
              "start": 50, "end": 250, "clock": "00:04"},
             {"id": "corner-50", "group": "set_pieces", "kind": "corner", "team": "team_b", "frame": 50,
              "start": 0, "end": 200, "clock": "00:02"}]
    exp = {"corner-900": {"keep": True, "in": 850, "out": 5000}}
    plan = X.plan(clips, exp, "team_a")
    assert [p[3] for p in plan] == ["Set pieces/Throw-ins/01_throw_in_00m04s.mp4",
                                    "Set pieces/Corners/01_corner_00m36s.mp4",
                                    "Passes into the final third/Passes/01_pass_00m20s.mp4"]
    assert plan[1][1:3] == (850, 1100)  # trimmed start, end clamped to the clip
    exp["pass-500"] = {"keep": False}
    assert len(X.plan(clips, exp, "team_a")) == 2
    assert [p[0]["id"] for p in X.plan(clips, exp, "team_a", group="final_third")] == []


# ---------------------------------------------------------------- app + export end to end
def write_clips(root):
    clips = [{"id": "corner-100", "group": "set_pieces", "kind": "corner", "team": "team_a", "frame": 100,
              "start": 60, "end": 160, "clock": "00:04", "x": 105.0, "y": 68.0, "how": "stoppage, players"},
             {"id": "shot-300", "group": "chances", "kind": "shot", "team": "team_a", "frame": 300,
              "start": 250, "end": 340, "clock": "00:12", "player_name": "9 Haaland",
              "published": {"minute": "1'", "type": "shot", "outcome": "saved"}},
             {"id": "throw_in-200", "group": "set_pieces", "kind": "throw_in", "team": "team_b", "frame": 200,
              "start": 170, "end": 260, "clock": "00:08"}]
    d = {"match_id": MID, "fps": FPS, "teams": {"team_a": "Man City", "team_b": "Tottenham"},
         "groups": {g: list(k) for g, k in D.GROUPS.items()}, "clips": clips}
    (root / ".tmp" / MID / "clips.json").write_text(json.dumps(d))


@pytest.fixture
def root(tmp_path):
    r = build_synthetic(tmp_path)
    write_clips(r)
    return r


@pytest.fixture
def client(root):
    from fastapi.testclient import TestClient

    from app.jobs import JobRunner
    from app.server import create_app
    return TestClient(create_app(root, runner=JobRunner(root, nice_heavy=False), warm=False))


def test_clips_api_groups_keep_drop_and_trim(client, root):
    base = f"/api/matches/{MID}/clips"
    s = client.get(base, params={"team": "team_a"}).json()
    assert [g["title"] for g in s["groups"]] == ["Set pieces", "Shots & key chances", "Passes into the final third"]
    sp = s["groups"][0]["kinds"]
    assert [k["title"] for k in sp] == ["Goal kicks", "Throw-ins", "Corners", "Free kicks"]  # penalties: none
    assert [c["id"] for c in sp[2]["clips"]] == ["corner-100"] and not sp[1]["clips"]  # team_b's throw-in not here
    assert s["width"] == 640 and s["zip_exists"] is False
    r = client.put(f"{base}/corner-100/export", json={"trim_in": 80, "trim_out": 9999}).json()
    assert (r["keep"], r["in"], r["out"], r["trimmed"]) == (True, 80, 160, True)
    assert client.put(f"{base}/shot-300/export", json={"keep": False}).json()["keep"] is False
    assert client.put(f"{base}/nope-1/export", json={"keep": False}).status_code == 409
    saved = json.loads((root / "config" / f"{MID}_clips_export.json").read_text())["clips"]
    assert saved["corner-100"]["in"] == 80 and saved["shot-300"]["keep"] is False
    r = client.put(f"{base}/export", json={"team": "team_a", "keep": False, "group": "set_pieces"}).json()
    assert r["changed"] == 1
    s = client.get(base, params={"team": "team_a"}).json()
    assert s["groups"][0]["kinds"][2]["clips"][0]["export"]["keep"] is False


def test_clips_api_says_what_to_run_when_there_are_none(client, root):
    (root / ".tmp" / MID / "clips.json").unlink()
    r = client.get(f"/api/matches/{MID}/clips")
    assert r.status_code == 409 and "Detect clips" in r.json()["detail"]


def test_stages_and_downloads_are_registered():
    from app import catalog, matches
    assert catalog.get_stage("b_clips").tool == "detect_clips.py"
    assert catalog.get_stage("b_clips_export").tool == "export_clips.py"
    assert "event_clips_zip" in matches.DOWNLOADABLE
    assert matches.artifact_paths("/r", "m", "team_b")["event_clips_zip"].name == "m_team_b_clips.zip"


def test_export_writes_a_zip_with_folders_per_heading(root):
    X.save_export(X.export_path(MID, root / "config"), {"shot-300": {"keep": False}})
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "export_clips.py"), "--match-id", MID, "--team", "team_a"],
                       cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    z = zipfile.ZipFile(root / "outputs" / "videos" / f"{MID}_team_a_clips.zip")
    names = z.namelist()
    assert names == ["Set pieces/Corners/01_corner_00m04s.mp4", "clips.csv"]
    csv = z.read("clips.csv").decode()
    assert "Corners,Man City,00:04" in csv
