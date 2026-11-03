"""Clean the raw ball detections into one ball position per frame ->
rewrites the ball rows of tracks.parquet (+ .tmp/<id>/ball_track.parquet).

Stage 4 of workflows/track_identities.md. The detector keeps every
`sports ball` box down to conf 0.05 (a 4-8 px ball on a tactical camera is
rarely confident), so the raw rows mix the match ball with false positives:
spare balls on the plinths round the perimeter, white boots, Spurs shirts.
Calibration and motion separate them:

  1. Project each box's bottom-centre to pitch metres; drop anything more
     than --margin-m outside the pitch lines (the spare balls live there).
  2. Link candidates frame-to-frame into ball tracklets (gate allows a
     30+ m/s shot, widening while the ball is unseen for <= --max-miss frames).
  3. Per frame keep the candidate on the strongest tracklet (sum of
     confidences): an isolated weak blip loses to a ball seen moving
     consistently. Tracklets of one weak detection are discarded.
  4. Interpolate gaps of <= --interp-frames in pitch space (rows flagged
     interp=True, conf 0) so possession isn't starved by short dropouts.

Pitch positions assume the ball is on the ground; a lofted ball projects
long. That's accepted: possession only needs the ball's ground position
near a player's feet, and lofted balls are by definition between players.

Usage:
    python tools/track_ball.py --match-id <id>
"""
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from _common import apply_calibration, load_calibration, load_meta, match_dir, pitch_dims

INF = 1e9


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--margin-m", type=float, default=1.0, help="Allowed distance outside the pitch lines")
    parser.add_argument("--gate", type=float, default=2.0, help="Base association gate, metres")
    parser.add_argument("--gate-step", type=float, default=1.3, help="Gate growth per unseen frame (~32 m/s)")
    parser.add_argument("--max-miss", type=int, default=10)
    parser.add_argument("--min-track-conf", type=float, default=0.4,
                        help="Drop ball tracklets whose summed confidence is below this")
    parser.add_argument("--interp-frames", type=int, default=12, help="Max gap to interpolate (frames)")
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--tmp-dir", default=".tmp")
    a = parser.parse_args()

    meta = load_meta(a.match_id, a.tmp_dir)
    mdir = match_dir(a.match_id, a.tmp_dir)
    calib = load_calibration(a.match_id, a.config_dir, a.calibration)
    if calib is None:
        raise SystemExit("track_ball.py needs a calibration (tools/calibrate_auto.py)")
    L, W = pitch_dims(calib)
    tracks = pd.read_parquet(mdir / "tracks.parquet")
    raw = tracks[tracks["class"] == "ball"].drop(columns=["pitch_x", "pitch_y"], errors="ignore")
    if "interp" in raw.columns:  # re-run: start again from real detections only
        raw = raw[~raw["interp"].astype("boolean").fillna(False).to_numpy(dtype=bool)]
    raw, _ = apply_calibration(raw.reset_index(drop=True), calib)
    m = a.margin_m
    on = raw["pitch_x"].between(-m, L + m) & raw["pitch_y"].between(-m, W + m)
    print(f"[{a.match_id}] {len(raw)} raw ball detections in {raw['frame'].nunique()} frames; "
          f"{int((~on).sum())} project off the pitch (spare balls etc.) -> dropped")
    cand = raw[on].sort_values(["frame", "conf"], ascending=[True, False])
    cand = cand.groupby("frame").head(5).reset_index(drop=True)

    # 2. ball tracklets
    frames = cand["frame"].to_numpy()
    xs, ys, cf = cand["pitch_x"].to_numpy(), cand["pitch_y"].to_numpy(), cand["conf"].to_numpy()
    tid = np.full(len(cand), -1)
    active = []  # [id, x, y, vx, vy, last_frame, miss]
    nid = 0
    fps = meta["fps"]
    bounds = np.flatnonzero(np.diff(np.r_[-1, frames, frames[-1] + 1]))
    for s, e in zip(bounds[:-1], bounds[1:]):
        f = frames[s]
        rows = np.arange(s, e)
        if active:
            dt = np.array([f - t[5] for t in active])
            px = np.array([t[1] + t[3] * d for t, d in zip(active, dt)])
            py = np.array([t[2] + t[4] * d for t, d in zip(active, dt)])
            D = np.hypot(px[:, None] - xs[rows][None], py[:, None] - ys[rows][None])
            gate = a.gate + a.gate_step * (dt - 1)[:, None]
            C = np.where(D <= gate, D, INF)
            ri, ci = linear_sum_assignment(C)
            used = set()
            for i, j in zip(ri, ci):
                if C[i, j] >= INF:
                    continue
                t, r = active[i], rows[j]
                d = max(f - t[5], 1)
                t[3], t[4] = 0.5 * t[3] + 0.5 * (xs[r] - t[1]) / d, 0.5 * t[4] + 0.5 * (ys[r] - t[2]) / d
                t[1], t[2], t[5], t[6] = xs[r], ys[r], f, 0
                tid[r] = t[0]
                used.add(j)
            for j, r in enumerate(rows):
                if j not in used:
                    active.append([nid, xs[r], ys[r], 0.0, 0.0, f, 0])
                    tid[r] = nid
                    nid += 1
        else:
            for r in rows:
                active.append([nid, xs[r], ys[r], 0.0, 0.0, f, 0])
                tid[r] = nid
                nid += 1
        active = [t for t in active if f - t[5] <= a.max_miss]
    cand["btl"] = tid

    # 3. strongest tracklet per frame
    strength = cand.groupby("btl")["conf"].sum()
    cand["strength"] = cand["btl"].map(strength)
    cand = cand[cand["strength"] >= a.min_track_conf]
    best = cand.sort_values(["frame", "strength", "conf"], ascending=[True, False, False]).groupby("frame").head(1)
    best = best.assign(interp=False)

    # 4. interpolate short gaps in pitch space, re-project to the image
    f_arr = best["frame"].to_numpy()
    fill = []
    npz = np.load(calib["frame_homographies"]) if "frame_homographies" in calib else None
    H_of = dict(zip(npz["frames"].tolist(), npz["H"])) if npz is not None else {}
    for (f1, x1, y1), (f2, x2, y2) in zip(best[["frame", "pitch_x", "pitch_y"]].to_numpy()[:-1],
                                          best[["frame", "pitch_x", "pitch_y"]].to_numpy()[1:]):
        gap = int(f2 - f1)
        if 1 < gap <= a.interp_frames:
            for k in range(1, gap):
                f = int(f1 + k)
                w = k / gap
                px, py = x1 + w * (x2 - x1), y1 + w * (y2 - y1)
                Hi = np.linalg.inv(H_of[f]) if f in H_of else None
                if Hi is None:
                    continue
                q = Hi @ np.array([px, py, 1.0])
                u, v = q[0] / q[2], q[1] / q[2]
                fill.append({"frame": f, "t_s": f / fps, "track_id": -1, "class": "ball",
                             "x1": u - 3, "y1": v - 6, "x2": u + 3, "y2": v, "foot_x": u, "foot_y": v,
                             "conf": 0.0, "pitch_x": px, "pitch_y": py, "interp": True})
    ball = pd.concat([best.drop(columns=["btl", "strength"]), pd.DataFrame(fill)], ignore_index=True)
    ball = ball.sort_values("frame").reset_index(drop=True)
    ball.to_parquet(mdir / "ball_track.parquet", index=False)

    players = tracks[tracks["class"] != "ball"]
    keep = [c for c in players.columns]
    out = pd.concat([players, ball.reindex(columns=keep + ["interp"])], ignore_index=True)
    out = out.sort_values(["frame", "class", "track_id"]).reset_index(drop=True)
    out.to_parquet(mdir / "tracks.parquet", index=False)

    window = (int(tracks["frame"].min()), int(tracks["frame"].max()))
    n = window[1] - window[0] + 1
    det_rate = best["frame"].nunique() / n
    print(f"  ball tracklets kept: {cand['btl'].nunique()}; ball located in {det_rate:.1%} of frames "
          f"(detected), {ball['frame'].nunique() / n:.1%} with <= {a.interp_frames}-frame interpolation")
    print(f"Wrote {mdir / 'ball_track.parquet'} and replaced the ball rows of tracks.parquet")


if __name__ == "__main__":
    main()
