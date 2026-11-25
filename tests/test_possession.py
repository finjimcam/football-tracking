import json
import sys

import numpy as np
import pandas as pd

import compute_possession


def write_match(tmp_path, match_id="m"):
    """300 frames: ball at player 1 (team_a) for 0-99, at player 2 (team_b)
    for 100-199, then no ball at all (should be held as team_b)."""
    d = tmp_path / match_id
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"match_id": match_id, "fps": 25.0, "frame_count": 300}))
    (d / "team_map.json").write_text(json.dumps({"1": "team_a", "2": "team_b"}))
    rows = []
    for f in range(300):
        for tid, x in ((1, 100.0), (2, 400.0)):
            rows.append(dict(frame=f, t_s=f / 25, track_id=tid, **{"class": "player"}, x1=x - 5, y1=80, x2=x + 5,
                             y2=100, foot_x=x, foot_y=100.0, conf=0.9))
        if f < 200:
            bx = 105.0 if f < 100 else 395.0
            rows.append(dict(frame=f, t_s=f / 25, track_id=-1, **{"class": "ball"}, x1=bx - 3, y1=94, x2=bx + 3,
                             y2=100, foot_x=bx, foot_y=100.0, conf=0.5))
    pd.DataFrame(rows).to_parquet(d / "tracks.parquet", index=False)
    return match_id


def test_timeline_hysteresis_and_hold(tmp_path, monkeypatch):
    mid = write_match(tmp_path)
    monkeypatch.setattr(sys, "argv", ["compute_possession.py", "--match-id", mid, "--tmp-dir", str(tmp_path),
                                      "--config-dir", str(tmp_path / "no_config")])
    compute_possession.main()
    tl = pd.read_parquet(tmp_path / mid / "possession_timeline.parquet").set_index("frame")["state"]
    res = json.loads((tmp_path / mid / "possession.json").read_text())
    assert tl[50] == "team_a"
    assert tl[105] == "team_a"   # hysteresis: 0.4 s before switching
    assert tl[150] == "team_b"
    assert tl[250] == "team_b"   # no ball for 2 s: possession is held, not lost
    assert res["possession_switches"] == 1
    assert not res["calibrated"] and res["unit"] == "px"
    assert np.isclose(res["team_a_time_pct"] + res["team_b_time_pct"], 100.0)
