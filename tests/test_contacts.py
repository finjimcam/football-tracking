"""Issue #34: cutting tracklets at same-team contacts lets the global linker
undo a tracker id swap, and the contact-swap proxy spots swaps."""
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from contact_swaps import evaluate  # noqa: E402
from track_identities import link_flow, split_at_contacts, summarize  # noqa: E402

FPS = 25


def crossing(n=100, swap=True):
    """Two team_a players run across each other on the same line, meeting at
    frame 50: A from y=20 to y=40, B from y=40 to y=20. With swap=True the
    tracker's tracklet ids exchange bodies at the meeting point."""
    rows = []
    for f in range(n):
        u = f / (n - 1)
        for body, y in (("A", 20 + 20 * u), ("B", 40 - 20 * u)):
            tl = 0 if body == "A" else 1
            if swap and f >= 50:  # closest approach is the 49/50 tie; idxmin takes 49
                tl = 1 - tl
            rows.append({"frame": f, "tracklet": tl, "pitch_x": 30.0, "pitch_y": y, "kit": "team_a",
                         "conf": 0.9, "body": body})
    return pd.DataFrame(rows)


def args(**kw):
    base = dict(vmax=8.5, w_role=1.5, c_link=0.5, reward=0.5, reward_per_s=1.0, edge_s=20.0, cost_late=8.0,
                max_gap_s=300.0, max_succ=12, overlap_frames=3, reach_m=3.0, role_tau_s=5.0, role_sigma_m=8.0,
                contact_role=0.6, w_vel=1.0)
    base.update(kw)
    return Namespace(**base)


def test_split_cuts_both_tracklets_at_closest_approach():
    p = crossing()
    tl = summarize(p, FPS, 105, 68)
    p2, nxt, n_cuts, flags = split_at_contacts(p, tl, 2, contact_m=1.0, min_side=15)
    assert n_cuts == 2 and nxt == 4
    # pieces before the cut got new ids and end at the contact; originals keep the tail
    assert flags[2] == (False, True) and flags[3] == (False, True)
    assert flags[0] == (True, False) and flags[1] == (True, False)
    cut = p2[p2["tracklet"].isin([2, 3])]["frame"].max()
    assert abs(cut - 50) <= 1
    # every piece is one body except across the swap: pieces now = pure bodies
    purity = p2.groupby("tracklet")["body"].nunique()
    assert (purity == 1).all()


def test_linker_reattaches_the_right_bodies():
    p = crossing()
    tl = summarize(p, FPS, 105, 68)
    p2, _, _, flags = split_at_contacts(p, tl, 2, contact_m=1.0, min_side=15)
    T = summarize(p2, FPS, 105, 68)
    T["start_contact"] = [flags.get(t, (False, False))[0] for t in T["tracklet"]]
    T["end_contact"] = [flags.get(t, (False, False))[1] for t in T["tracklet"]]
    T["role_x"], T["role_y"] = np.nan, np.nan
    assign, links = link_flow(T, FPS, args(), (0, 99), n_paths=2)
    body = p2.groupby("tracklet")["body"].first()
    chains = {}
    for t, k in assign.items():
        chains.setdefault(k, set()).add(body[t])
    assert sorted(len(v) for v in chains.values()) == [1, 1], chains  # each chain = one body


def test_contact_swap_proxy_flags_a_swap():
    rows = []
    # 4 team_a outfielders (ids 2-5) so the centroid is meaningful. 2 (left) and
    # 3 (right) converge on the same spot at frame 200, then go back to their own
    # sides - but the tracker swaps their ids at the contact, so id 2 ends up on
    # the right. (A full crossing with a swap looks like a bounce to any
    # role-based check; collisions where both keep their roles are the target.)
    for f in range(400):
        for tid, (x, y) in {2: (20, 10), 3: (20, 30), 4: (40, 20), 5: (40, 50)}.items():
            if tid in (2, 3):
                u = 1 - min(abs(f - 200) / 50, 1)  # 0 -> 1 at f=200 -> 0
                y = 10 + 10 * u if tid == 2 else 30 - 10 * u
            t = tid
            if f >= 200 and tid in (2, 3):
                t = 3 if tid == 2 else 2
            rows.append({"frame": f, "track_id": t, "class": "player", "pitch_x": float(x), "pitch_y": float(y)})
    tracks = pd.DataFrame(rows)
    res = evaluate(tracks, FPS)
    assert res["team_a"]["contacts"] >= 1 and res["team_a"]["suspected_swaps"] == 1


def test_reference_scoring_matches_own_scoring():
    """Scoring a run against its own contact set gives the same numbers."""
    from contact_swaps import reference_contacts
    rows = []
    for f in range(400):
        for tid, (x, y) in {2: (20, 10), 3: (20, 30), 4: (40, 20), 5: (40, 50)}.items():
            if tid in (2, 3):
                u = 1 - min(abs(f - 200) / 50, 1)
                y = 10 + 10 * u if tid == 2 else 30 - 10 * u
            t = (3 if tid == 2 else 2) if (f >= 200 and tid in (2, 3)) else tid
            rows.append({"frame": f, "track_id": t, "class": "player", "pitch_x": float(x), "pitch_y": float(y)})
    tracks = pd.DataFrame(rows)
    own = evaluate(tracks, FPS)["team_a"]
    ref = evaluate(tracks, FPS, reference=reference_contacts(tracks))["team_a"]
    assert (ref["contacts"], ref["evaluated"], ref["suspected_swaps"], ref["unlinked"]) == \
        (own["contacts"], own["evaluated"], own["suspected_swaps"], 0)
    # an unidentified variant: ids 2/3 replaced by fragments -> unlinked, not evaluated
    frag = tracks.assign(track_id=tracks["track_id"].where(~tracks["track_id"].isin([2, 3]), 10001))
    r = evaluate(frag, FPS, reference=reference_contacts(tracks))["team_a"]
    assert r["unlinked"] == own["contacts"] and r["evaluated"] == 0
