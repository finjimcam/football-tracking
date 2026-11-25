"""Calibration clips (issue #46): the few short moments where the analyst names
the players, so everything else about identities can run automatically.

Per period of the match (config/<id>_match.json "periods", else the analysis
window) it picks three 5-second clips: near the start, the middle and the end.
Around each of those points it searches +-45 s for the 5 s with the most
players in view (both teams, every frame of the clip counted), so each clip
shows as many players as possible to name.

The names are saved as identity anchors (config/<id>_identity_anchors.json,
tools/identity_anchors.py) tagged with the clip, and track_identities.py
applies them on its next run.

    python tools/calibration_clips.py --match-id <id>      # list the clips
"""
import argparse

import numpy as np
import pandas as pd

from _common import clock_label, load_meta, match_config, match_dir

PLAYER_KITS = ("team_a", "team_b", "gk_a", "gk_b")
SPOTS = (("start", "Start"), ("middle", "Middle"), ("end", "End"))
PERIOD_LABEL = {"first_half": "1st half", "second_half": "2nd half",
                "extra_time_first_half": "ET 1st half", "extra_time_second_half": "ET 2nd half"}


def periods_of(cfg, lo=None, hi=None):
    """[(key, label, start_frame, end_frame)] from the match config, in order."""
    out = []
    for key, p in (cfg.get("periods") or {}).items():
        if isinstance(p, dict) and p.get("start_frame") is not None and p.get("end_frame") is not None:
            out.append((key, PERIOD_LABEL.get(key, key.replace("_", " ")), int(p["start_frame"]), int(p["end_frame"])))
    if not out:
        w = cfg.get("analysis_window") or {}
        s, e = w.get("start_frame", lo), w.get("end_frame", hi)
        if s is not None and e is not None:
            out.append(("match", "Match", int(s), int(e)))
    return sorted(out, key=lambda r: r[2])


def players_per_frame(tracks, lo, hi):
    """Array over frames lo..hi: how many players (outfield or keeper, either
    team) are detected on each frame."""
    t = tracks[(tracks["class"] == "player") & tracks["frame"].between(lo, hi)]
    if "kit" in t.columns:
        t = t[t["kit"].isin(PLAYER_KITS)]
    counts = np.zeros(hi - lo + 1, dtype=float)
    vc = t["frame"].value_counts()
    counts[vc.index.to_numpy() - lo] = vc.to_numpy()
    return counts


def pick_clips(counts, lo, periods, fps, clip_s=5.0, search_s=45.0, edge_s=15.0, step=5):
    """Three clips per period. counts[i] = players on frame lo + i.
    Returns [{"id", "period", "spot", "label", "start", "end"}] (inclusive frames).
    Each clip is the clip_s window with the most players (sum over its
    frames, so a moment where everyone stays in shot wins) within +-search_s
    of its spot: edge_s after the period starts, the middle, edge_s before it
    ends."""
    n = int(round(clip_s * fps))
    csum = np.concatenate([[0.0], np.cumsum(counts)])
    out = []
    for key, plabel, ps, pe in periods:
        if pe - ps + 1 < n:
            continue
        mid = (ps + pe) // 2 - n // 2
        nominal = {"start": ps + int(edge_s * fps), "middle": mid, "end": pe - int(edge_s * fps) - n + 1}
        for spot, slabel in SPOTS:
            c = min(max(nominal[spot], ps), pe - n + 1)
            a = max(ps, c - int(search_s * fps))
            b = min(pe - n + 1, c + int(search_s * fps))
            best, score = c, -1.0
            for s in range(a, b + 1, step):
                i, j = s - lo, s - lo + n
                if i < 0 or j > len(counts):
                    continue
                v = csum[j] - csum[i]
                # ties go to the start closest to the nominal spot
                if v > score + 1e-9 or (abs(v - score) <= 1e-9 and abs(s - c) < abs(best - c)):
                    best, score = s, v
            out.append({"id": f"{key}-{spot}", "period": key, "spot": spot, "label": f"{plabel} · {slabel}",
                        "start": int(best), "end": int(best + n - 1),
                        "players_avg": round(score / n, 1) if score >= 0 else 0.0})
    return out


def clips_for_match(match_id, tmp_dir=".tmp", config_dir="config", tracks=None, **kw):
    meta = load_meta(match_id, tmp_dir)
    fps = float(meta["fps"])
    cfg = match_config(match_id, tmp_dir, config_dir)
    if tracks is None:
        tracks = pd.read_parquet(match_dir(match_id, tmp_dir) / "tracks.parquet", columns=["frame", "class", "kit"])
    p = tracks[tracks["class"] == "player"]
    lo, hi = int(p["frame"].min()), int(p["frame"].max())
    periods = [(k, lab, max(s, lo), min(e, hi)) for k, lab, s, e in periods_of(cfg, lo, hi) if e >= lo and s <= hi]
    clips = pick_clips(players_per_frame(tracks, lo, hi), lo, periods, fps, **kw)
    for c in clips:
        c["clock"] = clock_label(c["start"], cfg, fps) or ""
    return clips


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--tmp-dir", default=".tmp")
    ap.add_argument("--config-dir", default="config")
    a = ap.parse_args()
    for c in clips_for_match(a.match_id, a.tmp_dir, a.config_dir):
        print(f"{c['id']:<22} {c['label']:<18} frames {c['start']}-{c['end']}  {c['clock']:>6}  "
              f"~{c['players_avg']} players in view")


if __name__ == "__main__":
    main()
