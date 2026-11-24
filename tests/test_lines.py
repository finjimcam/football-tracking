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


def build_synthetic(tmp_path, collide=False):
    """Write the synthetic match under tmp_path (also used for manual UI checks).
    collide=True: defenders 1 and 3 run across each other, meeting at frame
    200, and the tracker swaps their ids there (id 1 continues on player 3's
    body and vice versa) - the case the picker's swap fix exists for."""
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
            if collide and tid in (1, 3):
                # 1 runs y 10 -> 42, 3 runs y 42 -> 10; they meet at f=200
                u = min(max((f - 100) / 200, 0), 1)
                py = 10 + 32 * u if tid == 1 else 42 - 32 * u
                if f >= 200:
                    t = 3 if tid == 1 else 1  # tracker id swap at the collision
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


def test_swap_fix_after_collision(tmp_path):
    from app.lines import MatchLines
    root = build_synthetic(tmp_path, collide=True)
    ml = MatchLines(root, MID, "team_a")
    s, e = ml.segs[0][0], ml.segs[0][1]
    kf = 150
    slots = [{"line": "back", "anchors": [[kf, 1]]}, {"line": "back", "anchors": [[kf, 3]]}]
    chains = ml._chains(slots, s, e)
    assert chains[0]["pts"][260][0] == 1  # chain on player "1" followed id 1 onto the wrong body

    # player 1's body carries id 3 after the swap: tell chain 0 it belongs on track 3 at frame 260
    r = ml.edit(s, slots, "swap", frame=260, tid=3, chain=0)
    assert "swap point" in r["msg"] and "other picked chain" in r["msg"]
    cut_frames = {a[0] for sl in r["slots"] for a in sl["anchors"] if len(a) > 2}
    assert len(cut_frames) == 1 and abs(next(iter(cut_frames)) - 200) <= 2
    c0, c1 = r["chains"][0]["pts"], r["chains"][1]["pts"]
    assert c0[150][0] == 1 and c0[190][0] == 1    # before the collision: unchanged
    assert c0[210][0] == 3 and c0[300][0] == 3    # after: follows the right body
    assert c1[190][0] == 3 and c1[210][0] == 1    # partner swapped back too

    # same frame, same click again: nothing to do
    again = ml.edit(s, r["slots"], "swap", frame=260, tid=3, chain=0)
    assert "already on this player" in again["msg"]

    # renderer agrees: cut anchors survive a save / load through the CLI format
    sys.path.insert(0, str(ROOT / "tools"))
    from cut_phase_clips import build_chains, entry_slots, load_selection
    ml.save(s, r["slots"], keyframe=kf)
    sel = load_selection(ml.sel_path)
    re_slots = entry_slots(sel[s])
    assert re_slots == r["slots"]
    rendered = build_chains(ml.rows_idx, re_slots, s, e, ml.fps, ml.H_at)
    assert rendered[0]["pts"][300][0] == 3


def test_cut_anchor_semantics():
    sys.path.insert(0, str(ROOT / "tools"))
    from cut_phase_clips import norm_anchors
    assert norm_anchors([[50, 2, 1], [10, 1], [50, 9]]) == [[10, 1], [50, 2, 1]]
    assert norm_anchors([(10, 1), (5, 2)]) == [[5, 2], [10, 1]]


def test_chains_never_share_a_track():
    """Two picked defenders both lose their tracks at frame 50; one fresh track
    (id 9) appears next to defender A. Without exclusivity both chains hand
    off onto it and draw two markers on one player."""
    sys.path.insert(0, str(ROOT / "tools"))
    from types import SimpleNamespace as R

    from cut_phase_clips import build_chains

    def row(tid, px, py):
        return R(track_id=tid, pitch_x=px, pitch_y=py, foot_x=px * 10, foot_y=py * 10)

    rows_idx = {}
    for f in range(0, 101):
        cur = {}
        if f < 50:
            cur[1] = row(1, 20.0, 10.0)
            cur[2] = row(2, 20.0, 12.0)
        elif f >= 52:
            cur[9] = row(9, 20.2, 10.5)   # A again, under a fresh id
        rows_idx[f] = cur
    slots = [{"line": "back", "anchors": [[10, 1]]}, {"line": "back", "anchors": [[10, 2]]}]
    chains = build_chains(rows_idx, slots, 0, 100, 25, None)
    on9 = [i for i, ch in enumerate(chains) if ch["raw"].get(80, (None,))[0] == 9]
    assert on9 == [0]  # only A's chain (closer) takes track 9
    assert chains[1]["raw"].get(80) is None
    assert build_chains(rows_idx, slots, 0, 100, 25, None, max_rounds=0)[1]["raw"][80][0] == 9  # the old bug


# ---- auto-fix unrealistic jumps (issue #44) ----------------------------------
def _jump_world(teleport=True, a_reappears=True, decoy_on_chain=True):
    """Defender A (id 1) jogs along y=10 at 3 m/s. Unlabelled B (id 5) stands
    10 m away. At frame 50 the tracker gives id 1 to B's body (a 10 m
    teleport) and A carries on under a fresh id 7. A second picked defender
    (id 2) stands right where A is heading, so the nearest track that ISN'T
    on a chain has to be chosen, not just the nearest."""
    sys.path.insert(0, str(ROOT / "tools"))
    from types import SimpleNamespace as R

    def row(tid, px, py):
        return R(track_id=tid, pitch_x=px, pitch_y=py, foot_x=px * 10, foot_y=py * 10)

    rows_idx = {}
    for f in range(0, 151):
        ax = 20.0 + 3.0 * f / 25
        cur = {}
        if decoy_on_chain:
            cur[2] = row(2, 26.3, 10.2)  # stands where A is heading
        if not teleport or f < 50:
            cur[1] = row(1, ax, 10.0)
            cur[5] = row(5, 30.0, 18.0)
        else:
            cur[1] = row(1, 30.0, 18.0)  # id 1 now on B's body
            if a_reappears and f >= 52:
                cur[7] = row(7, ax + 0.5, 10.4)  # A again, fresh id, a touch off the line
        rows_idx[f] = cur
    slots = [{"line": "back", "anchors": [[0, 1]]}]
    if decoy_on_chain:
        slots.append({"line": "back", "anchors": [[0, 2]]})
    return rows_idx, slots


def test_autofix_relabels_teleport_onto_nearest_unlabelled():
    rows_idx, slots = _jump_world()
    from cut_phase_clips import auto_fix_jumps, build_chains, find_jump
    before = build_chains(rows_idx, slots, 0, 150, 25, None)
    assert before[0]["raw"][100][0] == 1 and before[0]["raw"][100][2] == 18.0  # chain went with id 1 onto B
    a, b, d = find_jump(before[0]["raw"], slots[0]["anchors"], 25)
    assert b == 50 and d > 8.5
    new, fixes, unresolved, chains = auto_fix_jumps(rows_idx, slots, 0, 150, 25, None)
    assert unresolved == []
    assert [(x["chain"], x["frame"], x["from_tid"], x["to_tid"]) for x in fixes] == [(0, 50, 1, 7)]
    assert new[0]["anchors"] == [[0, 1], [50, 7, 1]]  # analyst's label kept, cut added
    assert new[1] == slots[1]                          # the other chain is untouched
    assert chains[0]["raw"][49][0] == 1 and chains[0]["raw"][100][0] == 7
    assert 50 not in chains[0]["raw"] and chains[0]["pts"][50][5] == "filled"  # gap filled until A is seen
    assert find_jump(chains[0]["raw"], new[0]["anchors"], 25) is None


def test_autofix_flags_jump_with_nobody_in_reach():
    rows_idx, slots = _jump_world(a_reappears=False, decoy_on_chain=False)
    from cut_phase_clips import auto_fix_jumps
    new, fixes, unresolved, _ = auto_fix_jumps(rows_idx, slots, 0, 150, 25, None)
    assert fixes == [] and new == slots
    assert [(u["chain"], u["frame"]) for u in unresolved] == [(0, 50)]


def test_autofix_trusts_the_analysts_own_anchor_and_normal_running():
    from cut_phase_clips import auto_fix_jumps, find_jump
    rows_idx, slots = _jump_world(decoy_on_chain=False)
    slots[0]["anchors"].append([50, 1])  # analyst says: id 1 at frame 50 IS this player
    _, fixes, unresolved, _ = auto_fix_jumps(rows_idx, slots, 0, 150, 25, None)
    assert fixes == [] and unresolved == []
    # a sprinter at 9 m/s with +-0.4 m pitch jitter is never a jump
    rng = np.random.default_rng(0)
    raw = {f: (1, 9.0 * f / 25 + rng.uniform(-0.4, 0.4), 10 + rng.uniform(-0.4, 0.4), 0, 0, False)
           for f in range(0, 300) if f % 17}  # with short drop-outs
    assert find_jump(raw, [[0, 1]], 25) is None


def test_autofix_api(client):
    r = client.get(f"/api/matches/{MID}/lines").json()
    s = r["phases"][0]["start"]
    bad = client.post(f"/api/matches/{MID}/lines/{s}/edit", json={"team": "team_a", "slots": [], "action": "autofix"})
    assert bad.status_code == 400 and "Label the players" in bad.json()["detail"]
    ph = client.get(f"/api/matches/{MID}/lines/{s}").json()
    r = client.post(f"/api/matches/{MID}/lines/{s}/edit",
                    json={"team": "team_a", "slots": ph["slots"], "action": "autofix"}).json()
    assert r["fixes"] == [] and r["unresolved"] == [] and "No unrealistic jumps" in r["msg"]
    assert r["slots"] == ph["slots"]
