"""Issue #16: manual identity anchors (tools/identity_anchors.py)."""
import pandas as pd

from identity_anchors import apply, best_box, iou, load_anchors, resolve, save_anchors


def _names():
    # role-inferred: chain 2 = No. 2 Walker, 3 = No. 3 Dias, 5 = No. 17 De Bruyne, 7 = No. 8 Gundogan
    return {1: {"name": "Ederson", "number": 31, "role": "GK"},
            2: {"name": "Walker", "number": 2, "role": "line1-slot1"},
            3: {"name": "Dias", "number": 3, "role": "line1-slot2"},
            5: {"name": "De Bruyne", "number": 17, "role": "line2-slot1"},
            7: {"name": "Gundogan", "number": 8, "role": "line2-slot3"}}


def _tl():
    return pd.DataFrame({
        "tracklet": [10, 11, 12, 20, 21, 30, 40],
        "identity": [5, 5, 5, 7, 7, -1, 3],
        "start":    [0, 200, 500, 0, 300, 250, 0],
        "end":      [150, 450, 900, 280, 900, 380, 900],
        "link": ["linked"] * 5 + ["", "linked"],
        "team": ["team_a"] * 5 + ["", "team_a"]})


def test_iou_and_best_box():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    rows = pd.DataFrame({"x1": [0, 100], "y1": [0, 0], "x2": [10, 110], "y2": [20, 20]})
    assert best_box(rows, [101, 1, 111, 21]) == 1
    assert best_box(rows, [50, 50, 60, 60]) is None


def test_resolve_by_box_then_by_stored_tracklet():
    p = pd.DataFrame({"frame": [100, 100], "tracklet": [11, 30], "x1": [0, 50], "y1": [0, 0],
                      "x2": [10, 60], "y2": [30, 30]})
    anchors = [{"frame": 100, "box": [50, 0, 60, 30]},                     # box hits tracklet 30
               {"frame": 100, "box": [200, 0, 210, 30], "tracklet": 12},   # no box match -> stored id
               {"frame": 999, "box": [0, 0, 10, 30], "tracklet": 77}]      # nothing -> unresolved
    got = [t for _, t in resolve(anchors, p, known_tracklets=[10, 11, 12, 30])]
    assert got == [30, 12, None]


def test_anchor_names_a_chain_and_swaps_the_displaced_name():
    tl, names = _tl(), _names()
    # the analyst says chain 7 (role-inferred Gundogan) is De Bruyne, twice
    resolved = [({"team": "team_a", "number": 17, "name": "De Bruyne"}, 20),
                ({"team": "team_a", "number": 17, "name": "De Bruyne"}, 21)]
    rep = apply(tl, names, resolved)
    assert (names[7]["number"], names[7]["name"], names[7]["name_source"]) == (17, "De Bruyne", "manual (Lines tab anchors)")
    assert (names[5]["number"], names[5]["name"]) == (8, "Gundogan")  # took the displaced name
    assert names[7]["role"] == "line2-slot3"  # the chain keeps its own place in the shape
    assert rep["named_chains"] == 1 and rep["moved"] == 0
    assert list(tl["identity"]) == [5, 5, 5, 7, 7, -1, 3]  # nothing moved


def test_unidentified_tracklet_moves_in_and_evicts_overlap():
    tl, names = _tl(), _names()
    # tracklet 30 (frames 250-380, unidentified) is No. 17; chain 5 has tracklet 11 at 200-450
    rep = apply(tl, names, [({"team": "team_a", "number": 17}, 30)])
    ident = dict(zip(tl["tracklet"], tl["identity"]))
    assert ident[30] == 5 and ident[11] == -1 and ident[10] == 5 and ident[12] == 5
    assert (rep["moved"], rep["evicted"]) == (1, 1)
    assert tl.set_index("tracklet").loc[30, "link"] == "anchor"
    assert tl.set_index("tracklet").loc[11, "link"] == "evicted_by_anchor"
    assert tl.set_index("tracklet").loc[30, "team"] == "team_a"


def test_unresolved_and_conflicting_anchors_are_reported():
    tl, names = _tl(), _names()
    resolved = [({"team": "team_a", "number": 2}, 40), ({"team": "team_a", "number": 2}, 40),  # chain 3 is No. 2
                ({"team": "team_a", "number": 3}, 40),                                       # ...and No. 3 once
                ({"team": "team_a", "number": 2}, None)]
    rep = apply(tl, names, resolved)
    assert rep["unresolved"] == 1 and len(rep["conflicts"]) == 1
    assert names[3]["number"] == 2 and names[2]["number"] == 3


def test_save_and_load(tmp_path):
    p = tmp_path / "m_identity_anchors.json"
    assert load_anchors(p) == []
    save_anchors(p, [{"id": "x", "team": "team_a", "number": 7, "frame": 5, "box": [0, 0, 1, 1]}])
    assert load_anchors(p)[0]["number"] == 7


# ---------- Lines tab API ----------
def test_identify_api_click_drag_and_delete(tmp_path):
    import json
    import sys
    from pathlib import Path

    from fastapi.testclient import TestClient
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from test_lines import MID, build_synthetic

    from app.jobs import JobRunner
    from app.server import create_app
    root = build_synthetic(tmp_path)
    (root / "config" / f"{MID}_reference.json").write_text(json.dumps({"lineups": {"team_a": {"lines": [
        [{"name": "Ederson", "number": 31}], [{"name": "Walker", "number": 2}, {"name": "Dias", "number": 3}]]}}}))
    tracks = pd.read_parquet(root / ".tmp" / MID / "tracks.parquet")
    tracks["tracklet"] = tracks["track_id"] + 500  # the synthetic match has no tracklet column
    tracks.to_parquet(root / ".tmp" / MID / "tracks.parquet", index=False)
    c = TestClient(create_app(root, runner=JobRunner(root, nice_heavy=False)))
    base = f"/api/matches/{MID}/lines"
    s = c.get(base).json()
    assert [p["number"] for p in s["lineup"]] == [31, 2, 3] and s["lineup"][0]["gk"]
    assert s["names"] == {} or isinstance(s["names"], dict)
    assert s["anchors"] == []

    r = tracks[(tracks["frame"] == 120) & (tracks["track_id"] == 3)].iloc[0]
    box = [float(r.x1), float(r.y1), float(r.x2), float(r.y2)]
    res = c.post(f"{base}/identify", json={"team": "team_a", "frame": 120, "box": box, "number": 2})
    assert res.status_code == 200, res.text
    a = res.json()["anchor"]
    assert (a["number"], a["name"], a["track_id"], a["tracklet"]) == (2, "Walker", 3, 503)
    # a rough dragged box round the same player replaces that anchor rather than adding one
    rough = [box[0] - 8, box[1] - 15, box[2] + 8, box[3] + 4]
    res = c.post(f"{base}/identify", json={"team": "team_a", "frame": 120, "box": rough, "number": 3})
    assert res.status_code == 200, res.text
    anchors = res.json()["anchors"]
    assert len(anchors) == 1 and anchors[0]["number"] == 3 and anchors[0]["pending"] is True
    stored = json.loads((root / "config" / f"{MID}_identity_anchors.json").read_text())["anchors"]
    assert stored[0]["box"] == [round(v, 1) for v in box]   # the detection's own box is what's kept
    # empty box / unknown number -> 400
    assert c.post(f"{base}/identify", json={"team": "team_a", "frame": 120, "box": [0, 0, 5, 5], "number": 2}).status_code == 400
    assert c.post(f"{base}/identify", json={"team": "team_a", "frame": 120, "box": box, "number": 99}).status_code == 400
    assert c.delete(f"{base}/identify/{anchors[0]['id']}?team=team_a").json()["anchors"] == []
    assert c.delete(f"{base}/identify/nope?team=team_a").status_code == 400
