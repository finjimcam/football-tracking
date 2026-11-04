"""Build a YOLO dataset (person + ball) from a tracked match's own output -> <out>/dataset.yaml.

Issue #7: fine-tune the detector for the ball without hand labels. The cleaned
pipeline output is the teacher:
  person  detections.parquet person boxes with conf >= --person-conf
  ball    ball_track.parquet rows (track_ball.py: linked into tracklets, static
          clutter dropped), plus interpolated positions inside gaps of
          <= --interp-gap frames, where the detector missed a ball that the
          track says was there. Those boxes take the size of the nearest
          real detection.
Static clutter (penalty spots, spare balls) is not labelled, so the model
learns it as background.

Frames are sampled every --step frames from --train window and every
--val-step from --val window. Keep the two windows apart, so validation is
footage the model never trained on. Images are full-size JPEGs.

Usage:
    python tools/make_ball_dataset.py --match-id <id> --out /srv/football-data/tmp-exp/ballds \\
        --train 1151:52000 --val 53000:71434
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from _common import load_meta, match_dir


def window(s):
    a, b = (int(v) for v in s.split(":"))
    return a, b


def ball_labels(ball, interp_gap):
    """{frame: (cx, cy, w, h)} from a ball track: detected rows, plus interpolated rows
    inside short gaps (sized like the nearest detection)."""
    ball = ball.sort_values("frame").drop_duplicates("frame")
    det = ball[~ball["interp"].astype(bool)]
    out = {}
    w = (det["x2"] - det["x1"]).to_numpy()
    h = (det["y2"] - det["y1"]).to_numpy()
    df = det["frame"].to_numpy()
    for f, x1, y1, x2, y2 in det[["frame", "x1", "y1", "x2", "y2"]].itertuples(index=False):
        out[int(f)] = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    if interp_gap > 0 and len(df):
        gaps = np.diff(df)
        ok = set()
        for a, g in zip(df[:-1], gaps):
            if 1 < g <= interp_gap + 1:
                ok.update(range(int(a) + 1, int(a + g)))
        for r in ball[ball["interp"].astype(bool) & ball["frame"].isin(ok)].itertuples():
            j = min(max(np.searchsorted(df, r.frame), 0), len(df) - 1)
            bw, bh = float(w[j]), float(h[j])
            out[int(r.frame)] = (r.foot_x, r.foot_y - bh / 2, bw, bh)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=window, required=True, help="first:last frame")
    ap.add_argument("--val", type=window, required=True, help="first:last frame (keep apart from --train)")
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--val-step", type=int, default=30)
    ap.add_argument("--person-conf", type=float, default=0.4)
    ap.add_argument("--interp-gap", type=int, default=4)
    ap.add_argument("--ball-track", default=None, help="ball_track.parquet (default: the match's)")
    ap.add_argument("--tmp-dir", default=".tmp")
    a = ap.parse_args()
    if not (a.train[1] < a.val[0] or a.val[1] < a.train[0]):
        raise SystemExit("--train and --val windows overlap")

    meta = load_meta(a.match_id, a.tmp_dir)
    mdir = match_dir(a.match_id, a.tmp_dir)
    W, H = meta["width"], meta["height"]
    frames = {"train": set(range(a.train[0], a.train[1] + 1, a.step)),
              "val": set(range(a.val[0], a.val[1] + 1, a.val_step))}
    want = frames["train"] | frames["val"]
    det = pd.read_parquet(mdir / "detections.parquet", columns=["frame", "class", "x1", "y1", "x2", "y2", "conf"])
    det = det[(det["class"] == "player") & (det["conf"] >= a.person_conf) & det["frame"].isin(want)]
    persons = {f: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in det.groupby("frame")}
    ball = pd.read_parquet(a.ball_track or (mdir / "ball_track.parquet"))
    balls = ball_labels(ball, a.interp_gap)

    out = Path(a.out)
    for split in frames:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(meta["video"]))
    lo, hi = min(want), max(want)
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    stats = {s: {"images": 0, "persons": 0, "balls": 0} for s in frames}
    f = lo
    while f <= hi:
        ok, img = cap.read()
        if not ok:
            break
        if f in want:
            split = "train" if f in frames["train"] else "val"
            lines = []
            for x1, y1, x2, y2 in persons.get(f, []):
                x1, x2 = max(0.0, x1), min(float(W), x2)
                y1, y2 = max(0.0, y1), min(float(H), y2)
                if x2 - x1 > 1 and y2 - y1 > 1:
                    lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            b = balls.get(f)
            if b is not None and 0 <= b[0] < W and 0 <= b[1] < H:
                cx, cy, bw, bh = b
                lines.append(f"1 {cx / W:.6f} {cy / H:.6f} {max(bw, 3) / W:.6f} {max(bh, 3) / H:.6f}")
                stats[split]["balls"] += 1
            stats[split]["persons"] += len(persons.get(f, []))
            stats[split]["images"] += 1
            name = f"{a.match_id}_{f:07d}"
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        f += 1
    cap.release()
    (out / "dataset.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
                                      "names:\n  0: person\n  1: ball\n")
    for s, v in stats.items():
        print(f"  {s}: {v['images']} images, {v['persons']} person boxes, {v['balls']} ball boxes "
              f"({v['balls'] / max(v['images'], 1):.0%} of images)")
    print(f"Wrote {out / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
