"""Calibration clips (issue #46): clip picking, naming players, accepting guesses."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from calibration_clips import periods_of, pick_clips  # noqa: E402
from test_lines import MID, build_synthetic  # noqa: E402


def test_pick_clips_three_per_period_on_the_busiest_moments():
    fps, lo = 25.0, 0
    counts = np.full(20000, 10.0)
    counts[400:600] = 22          # busy stretch near the start of half 1
    counts[4800:5000] = 3         # the exact middle of half 1 is a close-up: avoid it
    counts[5300:5500] = 21
    periods = [("first_half", "1st half", 0, 9999), ("second_half", "2nd half", 10000, 19999)]
    clips = pick_clips(counts, lo, periods, fps, clip_s=5, search_s=45, edge_s=15)
    assert [c["id"] for c in clips] == ["first_half-start", "first_half-middle", "first_half-end",
                                        "second_half-start", "second_half-middle", "second_half-end"]
    assert all(c["end"] - c["start"] + 1 == 125 for c in clips)
    assert 400 <= clips[0]["start"] and clips[0]["end"] < 600 and clips[0]["players_avg"] == 22
    assert 5300 <= clips[1]["start"] and clips[1]["end"] < 5500
    assert all(ps <= c["start"] and c["end"] <= pe for c in clips
               for k, _, ps, pe in periods if c["period"] == k)


def test_periods_fall_back_to_the_analysis_window():
    assert periods_of({"periods": {"second_half": {"start_frame": 9, "end_frame": 20},
                                   "first_half": {"start_frame": 1, "end_frame": 5}}})[0][0] == "first_half"
    assert periods_of({"analysis_window": {"start_frame": 3, "end_frame": 99}}) == [("match", "Match", 3, 99)]


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from app.jobs import JobRunner
    from app.server import create_app
    root = build_synthetic(tmp_path)
    ref = {"lineups": {"team_a": {"lines": [[{"number": 1, "name": "Ed Keeper"}],
                                            [{"number": n, "name": f"Player {n}"} for n in range(2, 12)]]},
                       "team_b": {"lines": [[{"number": 1, "name": "Other Keeper"}]]}}}
    (root / "config" / f"{MID}_reference.json").write_text(json.dumps(ref))
    pm = json.loads((root / ".tmp" / MID / "player_map.json").read_text())
    for k, v in pm["players"].items():  # the tracker's guesses: number = track id
        v["number"] = v["track_ids"][0]
    (root / ".tmp" / MID / "player_map.json").write_text(json.dumps(pm))
    c = TestClient(create_app(root, runner=JobRunner(root, nice_heavy=False)))
    c.root = root
    return c


def test_clip_naming_and_accepting_guesses(client):
    s = client.get(f"/api/matches/{MID}/calibration").json()
    assert len(s["clips"]) == 3 and s["lineups"]["team_a"]["players"][0]["name"] == "Ed Keeper"
    cid = s["clips"][0]["id"]
    clip = client.get(f"/api/matches/{MID}/calibration/{cid}").json()
    f = str(clip["start"])
    rows = clip["frames"][f]
    assert {r[0] for r in rows} >= {1, 3, 4, 5} and all(r[5] == "team_a" for r in rows)
    by_tid = {r[0]: r for r in rows}
    assert by_tid[3][6] == 3  # the guess is shown
    # the tracker guessed 3 for this box; the analyst says it's No. 4
    r = client.post(f"/api/matches/{MID}/calibration/{cid}/names",
                    json={"items": [{"frame": int(f), "box": by_tid[3][1:5], "team": "team_a", "number": 4}]})
    assert r.status_code == 200 and r.json()["named"]["team_a"] == [4]
    # naming No. 4 again on another box moves the name (one box per player per clip)
    r = client.post(f"/api/matches/{MID}/calibration/{cid}/names",
                    json={"items": [{"frame": int(f), "box": by_tid[4][1:5], "team": "team_a", "number": 4}]}).json()
    assert len(r["anchors"]) == 1 and r["anchors"][0]["track_id"] == 4 and r["anchors"][0]["clip"] == cid
    anchors = json.loads((client.root / "config" / f"{MID}_identity_anchors.json").read_text())["anchors"]
    assert anchors[0]["source"] == "calibration" and anchors[0]["number"] == 4
    # not in the lineup / outside the clip
    assert client.post(f"/api/matches/{MID}/calibration/{cid}/names",
                       json={"items": [{"frame": int(f), "box": by_tid[5][1:5], "team": "team_a", "number": 77}]}).status_code == 400
    assert client.post(f"/api/matches/{MID}/calibration/{cid}/names",
                       json={"items": [{"frame": clip["end"] + 50, "box": by_tid[5][1:5], "team": "team_a",
                                        "number": 2}]}).status_code == 400
    # remove it
    aid = r["anchors"][0]["id"]
    r = client.delete(f"/api/matches/{MID}/calibration/{cid}/names/{aid}").json()
    assert r["anchors"] == [] and r["named"]["team_a"] == []
    s = client.get(f"/api/matches/{MID}/calibration").json()
    assert s["anchors_total"] == 0 and s["run_chain"][0] == "b_identities"
