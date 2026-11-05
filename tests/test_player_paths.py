"""Issue #16: per-player full-match paths, heatmaps and stats (tools/player_paths.py, Players tab)."""
import numpy as np
import pandas as pd

from player_paths import (attacking_x, heatmap, path, player_index, player_rows, runs, sprint_count, stats)

FPS = 25


def _tracks(points, tid=1):
    """points: [(frame, x, y)] -> tracks rows for one track id."""
    f, x, y = zip(*points)
    return pd.DataFrame({"frame": f, "track_id": tid, "pitch_x": x, "pitch_y": y})


def _walk(f0, n, x0, y0, vx=0.0, vy=0.0, tid=1):
    return _tracks([(f0 + i, x0 + vx * i / FPS, y0 + vy * i / FPS) for i in range(n)], tid)


def test_player_index_keeps_both_teams_with_tracks():
    pm = {"players": {
        "a2": {"team": "team_a", "number": 2, "name": "Walker", "track_ids": [2], "name_source": "official lineup",
               "tracklets": 54, "fragile_links": 18},
        "a31": {"team": "team_a", "number": 31, "name": "Ederson", "role": "GK", "track_ids": [1]},
        "b10": {"team": "team_b", "number": 10, "name": "Kane", "track_ids": [110], "anchors": 2,
                "name_source": "manual (Lines tab anchors)"},
        "ref": {"team": "referee", "track_ids": [900]},
        "none": {"team": "team_a", "number": 9, "track_ids": []}}}
    idx = player_index(pm)
    assert [p["id"] for p in idx] == ["a31", "a2", "b10"]   # GK first, then by number; no ref / empty
    assert idx[2]["manual"] and idx[2]["anchors"] == 2 and not idx[1]["manual"]
    assert (idx[1]["tracklets"], idx[1]["fragile_links"]) == (54, 18)


def test_distance_speed_and_gaps():
    # 10 s at 5 m/s, a 3 s tracking gap (the player "teleports" 40 m), then 10 s at 3 m/s
    t = pd.concat([_walk(0, 10 * FPS + 1, 10, 30, vx=5), _walk(13 * FPS, 10 * FPS + 1, 100, 30, vx=-3)])
    r = runs(player_rows(t, [1]), FPS)
    assert r["run"].nunique() == 2
    s = stats(r, FPS, window_frames=23 * FPS)
    # 50 + 30 m; nothing counted across the gap (the 0.5 s smoothing trims ~1 m off each run's ends)
    assert 76 < s["distance_m"] <= 80
    assert abs(s["tracked_min"] - 502 / FPS / 60) < 0.05
    assert 47 < s["bands_m"]["run"] <= 50 and 27 < s["bands_m"]["jog"] <= 30
    assert 4.9 <= s["top_speed"] <= 5.1
    assert s["sprints"] == 0
    assert s["per_90_m"] is None                   # under a minute tracked is too short to scale


def test_glitch_steps_dropped_and_two_tracks_merged():
    # two track ids for one player on different frames (chain fragments), plus a 30 m glitch frame
    a = _walk(0, 50, 0, 0, vx=4)
    b = _walk(50, 50, 8, 0, vx=4, tid=7)
    b.loc[b.index[25], "pitch_x"] += 30
    rows = player_rows(pd.concat([a, b]), [1, 7])
    assert len(rows) == 100 and rows["frame"].is_monotonic_increasing
    s = stats(runs(rows, FPS, smooth_s=0.0), FPS, 100)
    assert abs(s["distance_m"] - 4 * 99 / FPS) < 0.5   # the glitch's two jumps are > 12 m/s and dropped


def test_window_and_sprints():
    # jog, a 2 s sprint at 8 m/s, jog, a 0.4 s burst (too short), jog
    seg = [(0, 3), (2 * FPS, 8), (5 * FPS, 3), (int(0.4 * FPS), 8), (3 * FPS, 3)]
    pts, f, x = [], 0, 0.0
    for n, v in seg:
        for _ in range(n):
            pts.append((f, x, 34.0))
            f += 1
            x += v / FPS
    r = runs(player_rows(_tracks(pts), [1]), FPS, smooth_s=0.0)
    assert sprint_count(r, FPS) == 1
    early = player_rows(_tracks(pts), [1], start=0, end=FPS)
    assert early["frame"].max() == FPS and len(early) == FPS + 1


def test_heatmap_counts_seconds_and_path_samples_runs():
    t = pd.concat([_walk(0, 4 * FPS, 1.0, 1.0), _walk(10 * FPS, 2 * FPS, 104.0, 67.0)])
    r = runs(player_rows(t, [1]), FPS)
    hm = heatmap(r, FPS, 105, 68, cell=5)
    assert (hm["nx"], hm["ny"]) == (21, 14)
    secs = np.array(hm["secs"])
    assert secs[0, 0] == 4.0 and secs[13, 20] == 2.0 and secs.sum() == 6.0 and hm["max"] == 4.0
    p = path(r, FPS, hz=2)
    assert len(p) == 2 and p[0][1][0] - p[0][0][0] == 12            # every 12 frames (~2 Hz) ...
    assert [q[0] for q in p[0]][-2:] == [96, 99] and p[1][-1][0] == 299  # ... plus each run's last frame


def test_attacking_direction_from_keepers():
    players = [{"team": "team_a", "role": "GK", "track_ids": [1]}, {"team": "team_a", "role": None, "track_ids": [2]},
               {"team": "team_b", "role": "GK", "track_ids": [101]}]
    t = pd.concat([_walk(0, 10, 100, 34), _walk(0, 10, 50, 34, tid=2), _walk(0, 10, 4, 34, tid=101)])
    assert attacking_x(t, players, 105) == {"team_a": 0.0, "team_b": 105.0}


def test_players_api(tmp_path):
    import json
    import sys
    from pathlib import Path

    from fastapi.testclient import TestClient
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from test_lines import MID, build_synthetic

    from app.jobs import JobRunner
    from app.server import create_app
    root = build_synthetic(tmp_path)
    c = TestClient(create_app(root, runner=JobRunner(root, nice_heavy=False)))
    s = c.get(f"/api/matches/{MID}/players")
    assert s.status_code == 200, s.text
    s = s.json()
    assert s["players"] and s["pitch"] == [105.0, 68.0] and set(s["teams"]) == {"team_a", "team_b"}
    pid = s["players"][0]["id"]
    d = c.get(f"/api/matches/{MID}/players/{pid}?start={s['start']}&end={s['start'] + 50}").json()
    assert d["end"] == s["start"] + 50 and "track_ids" not in d["player"]
    assert d["stats"]["tracked_min"] >= 0 and d["heatmap"]["nx"] == 42 and d["team_shape"]
    assert c.get(f"/api/matches/{MID}/players/nobody").status_code == 404
    assert c.get(f"/api/matches/{MID}/players/{pid}?start=50&end=10").status_code == 400
    # a re-run of track_identities rewrites player_map.json: the next call sees the new names
    pm_path = root / ".tmp" / MID / "player_map.json"
    pm = json.loads(pm_path.read_text())
    pm["players"][pid]["name"] = "Renamed"
    pm_path.write_text(json.dumps(pm))
    import os
    os.utime(pm_path, (os.path.getmtime(pm_path) + 5,) * 2)
    assert c.get(f"/api/matches/{MID}/players/{pid}").json()["player"]["name"] == "Renamed"
