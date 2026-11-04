"""Web line picker (app/lines.py + its API) on a small synthetic broadcast match:
one City out-of-possession phase, a back four + a midfield three, and one
defender whose track fragments (id 2 -> 22) so the chain has to hand off."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MID = "synthetic_bc_1234abcd"
FPS = 25
N = 500


def player_xy(pitch_x, pitch_y):
    fx, fy = 40 + pitch_y * 8, 340 - pitch_x * 3
    return fx, fy


@pytest.fixture
def root(tmp_path):
    return build_synthetic(tmp_path)


def build_synthetic(tmp_path):
    """Write the synthetic match under tmp_path (also used for manual UI checks)."""
    tmp_path = Path(tmp_path)
    mdir = tmp_path / ".tmp" / MID
    mdir.mkdir(parents=True)
    (tmp_path / "config").mkdir()
    (tmp_path / "videos").mkdir()
    video = tmp_path / "videos" / "synthetic_bc.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 360))
    for i in range(N):
        w.write(np.full((360, 640, 3), (40, 120 + i % 50, 40), np.uint8))
    w.release()
    (mdir / "meta.json").write_text(json.dumps({
        "match_id": MID, "video": str(video), "fps": FPS, "frame_count": N, "duration_s": N / FPS,
        "width": 640, "height": 360, "config": {}}))
    (tmp_path / "config" / f"{MID}_match.json").write_text(json.dumps({
        "team_a_name": "Man City", "team_b_name": "Tottenham", "colors": {"team_a": "#6CABDD"},
        "clock_sync": {"kickoff_frame": 0, "fps": FPS}}))

    # back four at ~20 m from their goal, midfield three at ~35 m, two forwards, GK
    layout = {1: (20, 10), 2: (21, 25), 3: (20, 42), 4: (19, 58),
              5: (35, 15), 6: (36, 34), 7: (35, 52), 8: (55, 25), 9: (56, 45), 11: (3, 34)}
    rows = []
    for f in range(N):
        for tid, (px, py) in layout.items():
            px2 = px + 2 * np.sin(f / 40)
            t = tid
            if tid == 2 and f >= 200:
                t = 22  # fragment: same player, new track id
                if f < 205:
                    continue  # 5-frame drop-out before the hand-off
            fx, fy = player_xy(px2, py)
            rows.append((f, f / FPS, t, "player", fx - 6, fy - 24, fx + 6, fy, fx, fy, 0.9, px2, py, "team_a"))
    tracks = pd.DataFrame(rows, columns=["frame", "t_s", "track_id", "class", "x1", "y1", "x2", "y2",
                                         "foot_x", "foot_y", "conf", "pitch_x", "pitch_y", "kit"])
    tracks.to_parquet(mdir / "tracks.parquet", index=False)
    pm = {"players": {f"p{t}": {"team": "team_a", "track_ids": [t] + ([22] if t == 2 else []),
                                "role": "GK" if t == 11 else "OUT"} for t in layout},
          "ignored_tracks": []}
    (mdir / "player_map.json").write_text(json.dumps(pm))
    state = ["team_a" if (f < 100 or f >= 300) else "team_b" for f in range(N)]
    pd.DataFrame({"frame": range(N), "state": state}).to_parquet(mdir / "possession_timeline.parquet")
    return tmp_path


@pytest.fixture
def client(root):
    from fastapi.testclient import TestClient

    from app.jobs import JobRunner
    from app.server import create_app
    return TestClient(create_app(root, runner=JobRunner(root, nice_heavy=False)))


def test_summary_lists_the_phase(client):
    r = client.get(f"/api/matches/{MID}/lines?team=team_a")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["team_name"] == "Man City" and d["opponent_name"] == "Tottenham"
    assert len(d["phases"]) == 1
    ph = d["phases"][0]
    assert (ph["start"], ph["end"]) == (100 - int(1.5 * FPS), 299 + FPS)
    assert ph["status"] == "todo"


def test_phase_prefills_auto_guess_and_hands_off(client):
    start = client.get(f"/api/matches/{MID}/lines").json()["phases"][0]["start"]
    d = client.get(f"/api/matches/{MID}/lines/{start}").json()
    assert d["source"] == "auto"
    lines = sorted(sl["line"] for sl in d["slots"])
    assert lines == ["back"] * 4 + ["mid"] * 3
    assert {sl["anchors"][0][1] for sl in d["slots"] if sl["line"] == "back"} == {1, 2, 3, 4}
    # the chain on player 2 hands off to track 22 after the drop-out (filled frames in between)
    ch = next(c for c, sl in zip(d["chains"], d["slots"]) if sl["anchors"][0][1] == 2)
    assert ch["pts"]["210"][0] == 22
    assert ch["pts"]["202"][4] == "f"
    assert any(p[4] == "l" for p in ch["pts"].values())
    assert str(start + 10) in d["rows"]


def test_cycle_relink_save_roundtrip(client, root):
    base = f"/api/matches/{MID}/lines"
    start = client.get(base).json()["phases"][0]["start"]
    d = client.get(f"{base}/{start}").json()
    slots = d["slots"]
    f = start + 20
    # cycle a forward: none -> back -> mid -> none
    r = client.post(f"{base}/{start}/edit", json={"slots": slots, "action": "cycle", "frame": f, "tid": 8}).json()
    assert len(r["slots"]) == 8 and r["slots"][-1]["line"] == "back"
    r = client.post(f"{base}/{start}/edit", json={"slots": r["slots"], "action": "cycle", "frame": f, "tid": 8}).json()
    assert r["slots"][-1]["line"] == "mid"
    r = client.post(f"{base}/{start}/edit", json={"slots": r["slots"], "action": "cycle", "frame": f, "tid": 8}).json()
    assert len(r["slots"]) == 7
    # re-link with every chain detected -> message, no change
    r = client.post(f"{base}/{start}/edit", json={"slots": r["slots"], "action": "relink", "frame": f, "tid": 9}).json()
    assert "nothing to re-link" in r["msg"]
    # bad input
    bad = client.post(f"{base}/{start}/edit", json={"slots": [], "action": "cycle", "frame": f, "tid": 999})
    assert bad.status_code == 400
    # save -> CLI-format file, status picked
    r = client.put(f"{base}/{start}", json={"slots": r["slots"], "keyframe": d["keyframe"]})
    assert r.json()["status"] == "picked"
    saved = json.loads((root / "config" / f"{MID}_team_a_out_of_possession_shape_selection.json").read_text())
    entry = saved["phases"][str(start)]
    assert entry["keyframe"] == d["keyframe"] and len(entry["slots"]) == 7
    again = client.get(f"{base}/{start}").json()
    assert again["source"] == "saved" and again["status"] == "picked"
    # skip
    client.put(f"{base}/{start}", json={"skip": True})
    assert client.get(base).json()["phases"][0]["status"] == "skipped"


def test_saved_selection_feeds_the_renderer(root, client):
    """What the web picker saves is exactly what cut_phase_clips --shape-selection reads."""
    sys.path.insert(0, str(ROOT / "tools"))
    from cut_phase_clips import entry_slots, load_selection
    base = f"/api/matches/{MID}/lines"
    start = client.get(base).json()["phases"][0]["start"]
    d = client.get(f"{base}/{start}").json()
    client.put(f"{base}/{start}", json={"slots": d["slots"], "keyframe": d["keyframe"]})
    sel = load_selection(root / "config" / f"{MID}_team_a_out_of_possession_shape_selection.json")
    assert entry_slots(sel[start]) == d["slots"]


def test_frame_jpeg(client):
    r = client.get(f"/api/matches/{MID}/frame/120.jpg?w=320")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 320
    assert client.get(f"/api/matches/{MID}/frame/121.jpg").status_code == 200  # sequential read
    assert client.get(f"/api/matches/{MID}/frame/99999.jpg").status_code == 404


def test_lines_need_tracking(client, root):
    (root / ".tmp" / MID / "possession_timeline.parquet").unlink()
    r = client.get(f"/api/matches/{MID}/lines")
    assert r.status_code == 409 and "possession" in r.json()["detail"]
