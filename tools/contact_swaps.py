"""Count likely identity swaps at player contacts -> .tmp/<id>/contact_swaps.json.

Issue #34. When two same-team players collide (feet within --contact-m), the
tracker can come out of it with their identities exchanged: the "LB" chain
continues on the centre-back. No single link check catches this, because
both chains stay smooth. What does change is each identity's place in the
team shape (offset from the visible teammates' centroid) before vs after the
contact. A swap shows as "A before looks like B after, and B before looks
like A after".

For every contact episode between two identified outfielders of one team:
  pre  = mean offset over [-pre_s, -gap_s] before the closest approach
  post = mean offset over [+gap_s, +post_s] after it
  straight = |preA - postA| + |preB - postB|
  crossed  = |preA - postB| + |preB - postA|
  suspected swap if crossed + margin < straight
The margin keeps it to swaps that are clear. Players who legitimately trade
places do so over seconds, not at one contact, so they rarely trip it.

This is a proxy with no ground truth. Use it to compare tracking variants on
the same footage, not as an absolute error rate.

Usage:
    python tools/contact_swaps.py --match-id <id> [--tmp-dir .tmp]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import load_meta, load_tracks, match_dir, write_json

TEAM_IDS = {"team_a": range(2, 12), "team_b": range(102, 112)}  # outfield identities


def contact_episodes(pos, ids, contact_m, join_frames=3):
    """[(a, b, closest_frame, min_dist)] per run of frames where identities a
    and b are within contact_m (runs closer than join_frames are merged)."""
    near = {}
    for f, g in pos.groupby(level=0):
        tid = g["track_id"].to_numpy()
        keep = np.isin(tid, ids)
        if keep.sum() < 2:
            continue
        tid = tid[keep]
        xy = g[["pitch_x", "pitch_y"]].to_numpy()[keep]
        d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        ii, jj = np.nonzero(np.triu(d < contact_m, 1))
        for i, j in zip(ii, jj):
            a, b = sorted((int(tid[i]), int(tid[j])))
            near.setdefault((a, b), []).append((int(f), float(d[i, j])))
    eps = []
    for (a, b), fl in near.items():
        run = [fl[0]]
        for item in fl[1:] + [None]:
            if item is not None and item[0] - run[-1][0] <= join_frames:
                run.append(item)
                continue
            fc, dmin = min(run, key=lambda x: x[1])
            eps.append((a, b, fc, dmin))
            if item is not None:
                run = [item]
    return sorted(eps, key=lambda e: e[2])


def evaluate(tracks, fps, contact_m=1.0, pre_s=3.0, post_s=3.0, gap_s=0.5, margin_m=2.0, min_frames=15):
    out = {}
    pl = tracks[(tracks["class"] == "player") & tracks["pitch_x"].notna()]
    for team, rng in TEAM_IDS.items():
        ids = np.array(list(rng))
        t = pl[pl["track_id"].isin(ids)][["frame", "track_id", "pitch_x", "pitch_y"]].copy()
        cen = t.groupby("frame")[["pitch_x", "pitch_y"]].transform("mean")
        cnt = t.groupby("frame")["track_id"].transform("size")
        t["ox"], t["oy"] = t["pitch_x"] - cen["pitch_x"], t["pitch_y"] - cen["pitch_y"]
        t = t[cnt >= 4]  # an offset from 2-3 visible players means little
        pos = t.set_index("frame")
        by_id = {k: g for k, g in t.groupby("track_id")}
        eps = contact_episodes(pos, ids, contact_m)
        n_eval, swaps = 0, []
        g_pre, g_post = int(gap_s * fps), int(pre_s * fps)
        g_post2 = int(post_s * fps)
        for a, b, fc, dmin in eps:
            def mean_off(k, f0, f1):
                g = by_id.get(k)
                if g is None:
                    return None
                s = g[(g["frame"] >= f0) & (g["frame"] <= f1)]
                return s[["ox", "oy"]].mean().to_numpy() if len(s) >= min_frames else None
            pa, pb = mean_off(a, fc - g_post, fc - g_pre), mean_off(b, fc - g_post, fc - g_pre)
            qa, qb = mean_off(a, fc + g_pre, fc + g_post2), mean_off(b, fc + g_pre, fc + g_post2)
            if any(v is None for v in (pa, pb, qa, qb)):
                continue
            n_eval += 1
            straight = np.linalg.norm(pa - qa) + np.linalg.norm(pb - qb)
            crossed = np.linalg.norm(pa - qb) + np.linalg.norm(pb - qa)
            if crossed + margin_m < straight:
                swaps.append({"a": a, "b": b, "frame": fc, "min_dist_m": round(dmin, 2),
                              "straight_m": round(float(straight), 1), "crossed_m": round(float(crossed), 1)})
        out[team] = {"contacts": len(eps), "evaluated": n_eval, "suspected_swaps": len(swaps),
                     "swap_rate": round(len(swaps) / max(n_eval, 1), 4), "swaps": swaps}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--contact-m", type=float, default=1.0)
    ap.add_argument("--margin-m", type=float, default=2.0)
    ap.add_argument("--tmp-dir", default=".tmp")
    a = ap.parse_args()
    meta = load_meta(a.match_id, a.tmp_dir)
    mdir = match_dir(a.match_id, a.tmp_dir)
    tracks = load_tracks(a.match_id, a.tmp_dir, Path(a.tmp_dir).parent / "config",  # with hand fixes (#42)
                         columns=["frame", "track_id", "class", "pitch_x", "pitch_y"])
    res = evaluate(tracks, meta["fps"], contact_m=a.contact_m, margin_m=a.margin_m)
    write_json(mdir / "contact_swaps.json", res)
    for team, r in res.items():
        print(f"{team}: {r['contacts']} same-team contacts, {r['evaluated']} evaluable, "
              f"{r['suspected_swaps']} suspected swaps ({r['swap_rate']:.1%})")
    print(json.dumps({t: {k: v for k, v in r.items() if k != "swaps"} for t, r in res.items()}))


if __name__ == "__main__":
    main()
