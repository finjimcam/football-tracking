"""Tracking tab (issue #42): every identity on the video and a top-down pitch,
a lane per player showing where tracking holds / is predicted / is lost, the
problems worth checking, and hand fixes saved to config/<id>_track_edits.json
(see tools/track_edits.py, which also applies them for every other tool).

Per player and frame the status is one of
  seen       a tracked box
  hand       a box you drew, or interpolated from one
  predicted  a short gap (<= PRED_S) between two boxes: interpolated, dashed
  lost       no box, but the last-known spot is inside the camera's view
  off        no box and the last-known spot is outside the view
"""
import json
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import track_edits as te  # noqa: E402
from _common import clock_label, load_calibration, load_meta, load_player_map, match_config  # noqa: E402
from identity_anchors import anchors_path, iou, load_anchors, save_anchors  # noqa: E402

from app.lines import ANCHOR_LOCK, team_lineup  # noqa: E402

OFF, LOST, PRED, SEEN, HAND = 0, 1, 2, 3, 4
CODES = "olpsh"
PRED_S = 1.0          # gaps up to this long are drawn as predicted
LOST_MIN_S = 3.0      # a lost run this long is a problem worth checking
SWAP_NEAR_S = 3.0     # a swap fix this close to a flagged swap resolves it
MAX_CHUNK = 750       # frames per /frames request (30 s at 25 fps)
KIT_CODE = {"team_a": "a", "team_b": "b", "gk_a": "a", "gk_b": "b", "official": "r"}
AUTO_END_M = 3.0      # elimination: a track starts/ends within this (+ running distance) of where he vanished/reappears
BOX_GAP_S = 300.0     # box search: gaps up to this long get a rectangle
BOX_PAD_M = 3.0       # ... the rectangle from his last box to his next one, padded by this
BOX_COL_D = 2.5       # ... an unread-kit box counts as his colours this close to his team's
BOX_CONF = 0.3        # ... boxes fainter than this are not searched
BOX_OFF_M = 2.0       # ... nor boxes further than this outside the lines (boards, bench, stands)
BOX_EDGE_PX = 4       # ... nor boxes cut off at the edge of the picture (colours unreadable)
BOX_REF_SHARE = 0.3   # ... nor boxes of a track that reads as the referee this often
ELIM_MIN_S = 1.0      # elimination: orphan tracks at least this long
ELIM_KIT = 0.8        # ... whose readable kit votes are at least this much one team
ELIM_GAP_S = 60.0     # ... go to a player lost for at most this long around them
ELIM_EDGE_PX = 4      # ... and are not mostly cut off at the edge of the picture
ELIM_CONF = 0.4       # ... nor faint detections (median confidence below this)
ELIM_CROWD = 0.4      # ... nor mostly tangled with another box (IoU > 0.3 in this share of frames)
ELIM_OFF_M = 1.0      # ... nor off the pitch (median spot this far outside the lines)
LAST_MIN_S = 0.5      # last ones left: runs at least this long
LAST_CONF = (0.1, 0.2, 0.3)  # ... faintest box accepted with 1 / 2 / 3 outfield players of a team missing
LAST_END_M = (6.0, 4.5, 3.0)  # ... and the slack (m) on top of running distance (a half-hidden box's spot is off)
HOP_MAX_S = 20.0      # label hops: a stretch between a jump away and a jump back at most this long
AUTO_DUP = 0.6        # box this much inside a labelled player's box: a second detection of him
AUTO_VMAX = 9.0       # m/s: a line faster than this means one end is wrong
AUTO_ROUNDS = 6       # rounds of box search / elimination / label hops (+ last ones left)
STRICT_ROUNDS = 2     # ... the first rounds without the looser last-ones-left step
COLOUR = ["shirt_L", "shirt_a", "shirt_b", "shorts_L", "shorts_a", "shorts_b"]
REF_D, REF_RATIO, REF_MIN_S = 2.5, 0.5, 0.5   # referee look-alike rows (see _ref_runs)
COLS = ["frame", "track_id", "class", "x1", "y1", "x2", "y2", "pitch_x", "pitch_y", "tracklet", "kit", "conf"]


class TrackingError(ValueError):
    pass


def _r(v, n=1):
    return None if v is None or not np.isfinite(v) else round(float(v), n)


class MatchTracking:
    """Everything the Tracking tab needs for one match, loaded once; fixes
    are re-applied in memory after every change."""

    def __init__(self, root, match_id):
        self.root, self.match_id = Path(root), match_id
        self.tmp_dir, self.config_dir = self.root / ".tmp", self.root / "config"
        self.mdir = self.tmp_dir / match_id
        if not (self.mdir / "tracks.parquet").exists():
            raise TrackingError("Needs tracks.parquet: run the pipeline down to 'Track identities' first")
        self.meta = load_meta(match_id, self.tmp_dir)
        self.fps = float(self.meta["fps"])
        self.W, self.H = int(self.meta["width"]), int(self.meta["height"])
        self.cfg = match_config(match_id, self.tmp_dir, self.config_dir)
        raw = pd.read_parquet(self.mdir / "tracks.parquet")
        self.base = raw[[c for c in COLS if c in raw.columns]].copy()
        if "tracklet" not in self.base.columns:
            self.base["tracklet"] = np.nan
        self.base_fp = te.fingerprint(self.base)
        p = self.base[self.base["class"] == "player"]
        self.lo, self.hi = int(p["frame"].min()), int(p["frame"].max())
        self.N = self.hi - self.lo + 1
        self._load_players()
        self._load_camera()
        self._load_ball()
        self.swaps = self._load_swaps()
        self.edits_path = te.edits_path(match_id, self.config_dir)
        self.anchor_path = anchors_path(match_id, self.config_dir)
        self.tracks_mtime = (self.mdir / "tracks.parquet").stat().st_mtime
        self.lock = threading.Lock()
        self.edits_mtime = None
        self._rebuild()

    # ---- static inputs -------------------------------------------------
    def _load_players(self):
        pm = load_player_map(self.match_id, self.tmp_dir) or {"players": {}}
        colors = self.cfg.get("colors") or {}
        self.team_color = {"team_a": colors.get("team_a") or "#6CABDD", "team_b": colors.get("team_b") or "#F0F0F0"}
        self.players, self.key_of = [], {}
        for key, v in pm.get("players", {}).items():
            ids = [int(t) for t in v.get("track_ids", [])]
            if not ids:
                continue
            num = v.get("number")
            self.players.append({"key": key, "tid": ids[0], "ids": ids, "team": v.get("team"),
                                 "number": num, "name": v.get("name") or (f"#{num}" if num else key),
                                 "role": v.get("role")})
            for t in ids:
                self.key_of[t] = key
        order = {"team_a": 0, "team_b": 1}
        self.players.sort(key=lambda q: (order.get(q["team"], 2), q.get("role") != "GK",
                                         q["number"] if isinstance(q["number"], int) else 999, q["key"]))
        self.by_key = {q["key"]: q for q in self.players}

    def _load_camera(self):
        """Dense per-frame pitch->image homographies for the window (NaN where
        the calibration has none), to tell 'lost' from 'off camera'."""
        self.Hinv = None
        try:
            calib = load_calibration(self.match_id, self.config_dir)
        except SystemExit:
            calib = None
        self.H_img2pitch = None
        if calib is None:
            return
        dense = np.full((self.N, 3, 3), np.nan)
        if "frame_H" in calib:
            fi = np.asarray(calib["frame_index"], int)
            ok = (fi >= self.lo) & (fi <= self.hi)
            Hs = np.asarray(calib["frame_H"], float)[ok]
            dense[fi[ok] - self.lo] = Hs
        elif "homography" in calib:
            dense[:] = np.asarray(calib["homography"], float)
        self.H_img2pitch = dense
        good = np.isfinite(dense).all(axis=(1, 2))
        inv = np.full_like(dense, np.nan)
        if good.any():
            inv[good] = np.linalg.inv(dense[good])
        self.Hinv = inv

    def _load_ball(self):
        bp = self.mdir / "ball_track.parquet"
        if bp.exists():
            b = pd.read_parquet(bp)
            cx = b["foot_x"] if "foot_x" in b.columns else (b["x1"] + b["x2"]) / 2
            cy = b["foot_y"] if "foot_y" in b.columns else b["y2"]
            interp = b["interp"] if "interp" in b.columns else False
            ball = pd.DataFrame({"frame": b["frame"].astype(int), "x": cx, "y": cy,
                                 "px": b.get("pitch_x"), "py": b.get("pitch_y"), "interp": interp})
        else:
            b = self.base[self.base["class"] == "ball"]
            ball = pd.DataFrame({"frame": b["frame"].astype(int), "x": (b["x1"] + b["x2"]) / 2, "y": b["y2"],
                                 "px": b["pitch_x"], "py": b["pitch_y"], "interp": False})
        ball = ball.sort_values("frame").drop_duplicates("frame")
        self.ball = {int(r.frame): [_r(r.x), _r(r.y), _r(r.px, 2), _r(r.py, 2), bool(r.interp)]
                     for r in ball.itertuples()}

    def _load_swaps(self):
        path = self.mdir / "contact_swaps.json"
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            return []
        out = []
        for team, r in data.items():
            if not isinstance(r, dict):
                continue
            for s in r.get("swaps", []):
                a, b, f = int(s["a"]), int(s["b"]), int(s["frame"])
                out.append({"key": f"swap:{f}:{a}:{b}", "kind": "swap", "frame": f, "end": f, "ids": [a, b],
                            "dist_m": s.get("min_dist_m"), "team": team})
        return out

    # ---- edits + derived state -------------------------------------------
    def _rebuild(self):
        data = te.load_edits(self.edits_path)
        self.edits_mtime = self.edits_path.stat().st_mtime if self.edits_path.exists() else None
        self.data = data
        live = [e for e in data["edits"] if e.get("kind") != "ok"]
        self.stale = bool(live) and data.get("base") not in (None, self.base_fp)
        df = self.base if (not live or self.stale) else te.apply_edits(self.base, live, self.fps)
        if "source" not in df.columns:
            df = df.assign(source="auto")
        df = df[df["class"] == "player"].sort_values("frame", kind="stable").reset_index(drop=True)
        self.df = df
        self.df_frames = df["frame"].to_numpy()
        self._index_identities()
        self.problems = self._problems()

    def _index_identities(self):
        df = self.df
        ids = df["track_id"].to_numpy()
        self.ident = {}
        self.status = {}
        pos = np.arange(self.N)
        src_hand = df["source"].isin(["hand", "hand_fill"]).to_numpy()
        for q in self.players:
            m = np.isin(ids, q["ids"])
            g = df[m].drop_duplicates("frame")
            fr = g["frame"].to_numpy() - self.lo
            vals = g[["x1", "y1", "x2", "y2", "pitch_x", "pitch_y"]].to_numpy(float)
            hand = src_hand[m][~df[m]["frame"].duplicated().to_numpy()]
            tids = g["track_id"].to_numpy()
            self.ident[q["key"]] = (fr, vals, hand, tids)
            st = np.zeros(self.N, np.uint8)
            if len(fr):
                i = np.searchsorted(fr, pos, side="right") - 1
                j = np.searchsorted(fr, pos, side="left")
                has_p, has_n = i >= 0, j < len(fr)
                pf = np.where(has_p, fr[np.clip(i, 0, None)], -10**9)
                nf = np.where(has_n, fr[np.clip(j, None, len(fr) - 1)], 10**9)
                seen = pf == pos
                st[seen] = SEEN
                hidx = np.clip(i, 0, None)
                st[seen & hand[hidx]] = HAND
                pred = ~seen & has_p & has_n & ((nf - pf) <= int(PRED_S * self.fps))
                st[pred] = PRED
                rest = ~seen & ~pred
                if rest.any():
                    # last-known (else next-known) spot, projected with that frame's camera
                    k = np.where(has_p, np.clip(i, 0, None), np.clip(j, None, len(fr) - 1))[rest]
                    xy = vals[k, 4:6]
                    st[rest] = np.where(self._in_view(pos[rest], xy), LOST, OFF)
            self.status[q["key"]] = st

    def _in_view(self, offs, xy, margin=0.02):
        if self.Hinv is None:
            return np.ones(len(offs), bool)
        Hi = self.Hinv[offs]
        p = np.einsum("nij,nj->ni", Hi, np.c_[xy, np.ones(len(xy))])
        with np.errstate(invalid="ignore", divide="ignore"):
            u, v = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]
        mx, my = self.W * margin, self.H * margin
        inside = (u >= -mx) & (u <= self.W + mx) & (v >= -my) & (v <= self.H + my) & (p[:, 2] > 0)
        return np.where(np.isfinite(u), inside, True)  # no camera model: assume visible

    def _problems(self):
        done = {}
        for e in self.data["edits"]:
            if e.get("problem"):
                done[e["problem"]] = e.get("id")
        near = int(SWAP_NEAR_S * self.fps)
        out = []
        for s in self.swaps:
            a, b = s["ids"]
            fixed = done.get(s["key"])
            if fixed is None:
                for e in self.data["edits"]:
                    pair = {e.get("a"), e.get("b")} if e["kind"] == "swap" else {e.get("tid"), e.get("to")}
                    if e["kind"] in ("swap", "assign") and pair == {a, b} and abs(e["frame"] - s["frame"]) <= near:
                        fixed = e.get("id")
                        break
            ka, kb = self.key_of.get(a), self.key_of.get(b)
            out.append({**s, "keys": [k for k in (ka, kb) if k], "fixed": fixed,
                        "text": f"Possible swap: {self._who(ka, a)} and {self._who(kb, b)}"})
        min_run = int(LOST_MIN_S * self.fps)
        for q in self.players:
            st = self.status[q["key"]]
            lost = np.r_[False, st == LOST, False].astype(np.int8)
            d = np.diff(lost)
            starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
            for s0, e0 in zip(starts, ends):
                if e0 - s0 < min_run:
                    continue
                f0, f1 = int(s0 + self.lo), int(e0 - 1 + self.lo)
                key = f"lost:{q['tid']}:{f0}"
                out.append({"key": key, "kind": "lost", "frame": f0, "end": f1, "ids": [q["tid"]],
                            "keys": [q["key"]], "fixed": done.get(key),
                            "text": f"{self._who(q['key'], q['tid'])} lost for {(f1 - f0 + 1) / self.fps:.0f} s"})
        # problems your fixes made disappear (a lost run filled by hand boxes):
        # keep them in the list as fixed, with the text saved on the fix
        have = {p["key"] for p in out}
        for e in self.data["edits"]:
            k = e.get("problem")
            if not k or k in have:
                continue
            have.add(k)
            parts = k.split(":")
            kind = parts[0] if parts[0] in ("swap", "lost") else "lost"
            f = int(parts[1]) if kind == "swap" and len(parts) > 1 and parts[1].isdigit() else \
                int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else int(e.get("frame") or self.lo)
            tids = [int(x) for x in (parts[2:4] if kind == "swap" else parts[1:2]) if x.lstrip("-").isdigit()]
            out.append({"key": k, "kind": kind, "frame": f, "end": f, "ids": tids,
                        "keys": [self.key_of[t] for t in tids if t in self.key_of], "fixed": e.get("id"),
                        "text": e.get("note") or k})
        out.sort(key=lambda p: (p["frame"], p["key"]))
        return out

    def _who(self, key, tid):
        q = self.by_key.get(key)
        if not q:
            return f"track {tid}"
        return f"#{q['number']} {q['name']}" if q.get("number") is not None else q["name"]

    def _refresh_if_changed(self):
        m = self.edits_path.stat().st_mtime if self.edits_path.exists() else None
        if m != self.edits_mtime:
            self._rebuild()

    # ---- API ---------------------------------------------------------------
    def summary(self):
        with self.lock:
            self._refresh_if_changed()
            n_open = sum(1 for p in self.problems if p["fixed"] is None)
            return {
                "match_id": self.match_id, "fps": self.fps, "width": self.W, "height": self.H,
                "start": self.lo, "end": self.hi, "calibrated": self.Hinv is not None,
                "pitch": [float(self.cfg.get("pitch_length") or 105.0), float(self.cfg.get("pitch_width") or 68.0)],
                "teams": {t: {"name": self.cfg.get(f"{t}_name") or t, "color": self.team_color[t]}
                          for t in ("team_a", "team_b")},
                "players": self.players,
                "problems": [{**p, "clock": clock_label(p["frame"], self.cfg, self.fps)} for p in self.problems],
                "open_problems": n_open, "stale": self.stale,
                "edits": [{**{k: v for k, v in e.items() if k != "items"},
                           **({"n_items": len(e["items"])} if "items" in e else {}),
                           "clock": clock_label(e["frame"], self.cfg, self.fps) if e.get("frame") is not None
                           else None} for e in self.data["edits"]],
                "edits_file": self.edits_path.name,
                "retired": len(self.data.get("retired") or []),
                "anchors": self.anchors(),
                "clocks": {f: clock_label(f, self.cfg, self.fps)
                           for f in range(self.lo - self.lo % int(self.fps), self.hi + 1, int(self.fps) * 60)},
            }

    def lanes(self, start, end, bins):
        """Per player: one status code per bin over [start, end] (the most
        common status in the bin; any 'hand' wins so your fixes always show)."""
        start, end = max(self.lo, int(start)), min(self.hi, int(end))
        if end < start:
            raise TrackingError("end must be after start")
        bins = max(1, min(int(bins), 4000, end - start + 1))
        edges = np.linspace(start - self.lo, end - self.lo + 1, bins + 1).astype(int)
        with self.lock:
            self._refresh_if_changed()
            out = {}
            for q in self.players:
                st = self.status[q["key"]][start - self.lo:end - self.lo + 1]
                onehot = np.stack([(st == k) for k in range(5)], axis=1).astype(np.int32)
                counts = np.add.reduceat(onehot, edges[:-1] - edges[0], axis=0)
                code = counts.argmax(axis=1)
                code[counts[:, HAND] > 0] = HAND
                out[q["key"]] = "".join(CODES[c] for c in code)
        return {"start": start, "end": end, "bins": bins, "lanes": out}

    def frames(self, start, end):
        """Per frame in [start, end]: players [key, tid, status, x1, y1, x2,
        y2, pitch_x, pitch_y] (box null when lost / off; pitch = last-known
        spot), other boxes [tid, kit, x1, y1, x2, y2, pitch_x, pitch_y] and
        the ball [x, y, pitch_x, pitch_y, interpolated]."""
        start, end = max(self.lo, int(start)), min(self.hi, int(end))
        if end < start:
            raise TrackingError("No frames there")
        if end - start + 1 > MAX_CHUNK:
            end = start + MAX_CHUNK - 1
        with self.lock:
            self._refresh_if_changed()
            n = end - start + 1
            out = [{"p": [], "o": [], "b": self.ball.get(f)} for f in range(start, end + 1)]
            for q in self.players:
                fr, vals, _hand, tids = self.ident[q["key"]]
                st = self.status[q["key"]][start - self.lo:end - self.lo + 1]
                if not len(fr):
                    continue
                offs = np.arange(start, end + 1) - self.lo
                i = np.clip(np.searchsorted(fr, offs, side="right") - 1, 0, len(fr) - 1)
                j = np.clip(np.searchsorted(fr, offs, side="left"), 0, len(fr) - 1)
                for k in range(n):
                    s = int(st[k])
                    if s in (SEEN, HAND):
                        v = vals[i[k]]
                        tid = int(tids[i[k]])
                    elif s == PRED:
                        a, b = i[k], j[k]
                        w = (offs[k] - fr[a]) / max(fr[b] - fr[a], 1)
                        v = vals[a] * (1 - w) + vals[b] * w
                        tid = int(tids[a])
                    else:
                        idx = i[k] if fr[i[k]] <= offs[k] else j[k]
                        v = np.r_[np.full(4, np.nan), vals[idx, 4:6]]
                        tid = int(tids[idx])
                    out[k]["p"].append([q["key"], tid, CODES[s], _r(v[0]), _r(v[1]), _r(v[2]), _r(v[3]),
                                        _r(v[4], 2), _r(v[5], 2)])
            a, b = np.searchsorted(self.df_frames, [start, end + 1])
            g = self.df.iloc[a:b]
            other = g[~g["track_id"].isin(list(self.key_of))]
            for r in other.itertuples():
                out[int(r.frame) - start]["o"].append([int(r.track_id), KIT_CODE.get(r.kit, "u"), _r(r.x1), _r(r.y1),
                                                       _r(r.x2), _r(r.y2), _r(r.pitch_x, 2), _r(r.pitch_y, 2)])
            views = self._views(start, end)
            for k in range(n):
                out[k]["c"] = clock_label(start + k, self.cfg, self.fps)
                out[k]["v"] = views[k]
        return {"start": start, "end": end, "frames": out}

    def _views(self, start, end):
        """The camera's view on the pitch per frame: the image's corners
        projected (clamped, since the top edge can sit above the horizon)."""
        n = end - start + 1
        if self.H_img2pitch is None:
            return [None] * n
        Hs = self.H_img2pitch[start - self.lo:end - self.lo + 1]
        c = np.array([[0, 0, 1], [self.W, 0, 1], [self.W, self.H, 1], [0, self.H, 1]], float)
        p = np.einsum("nij,kj->nki", Hs, c)
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.where(p[..., 2] > 1e-9, p[..., 2], np.nan)
            x = np.clip(p[..., 0] / w, -40, 145)
            y = np.clip(p[..., 1] / w, -40, 108)
        # a corner above the horizon: push it to the far edge of the clamp box
        y = np.where(np.isnan(y), -40, y)
        x = np.where(np.isnan(x), np.where(np.arange(4) % 3 == 0, -40, 145), x)
        out = []
        for k in range(n):
            if not np.isfinite(Hs[k]).all():
                out.append(None)
            else:
                out.append([[round(float(x[k, i]), 1), round(float(y[k, i]), 1)] for i in range(4)])
        return out

    def tracklet_at(self, tid, frame):
        g = self.df[(self.df_frames == frame) & (self.df["track_id"].to_numpy() == tid)]
        if g.empty:
            return None
        v = g["tracklet"].iloc[0]
        return None if pd.isna(v) else int(v)

    def _pitch_at(self, frame, x, y):
        if self.H_img2pitch is None:
            raise TrackingError("This match has no pitch calibration: can't place a hand box")
        Hm = self.H_img2pitch[frame - self.lo]
        if not np.isfinite(Hm).all():
            raise TrackingError("No camera calibration on this frame: try a frame nearby")
        p = Hm @ np.array([x, y, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    # ---- names -> identity anchors (the re-link uses them) ----------------
    def person(self, key):
        """(team, number, name) of a player key, the number looked up in the
        lineup by name when the tracking didn't have one."""
        q = self.by_key.get(key)
        if not q or q.get("team") not in ("team_a", "team_b"):
            return None
        num = q.get("number")
        if num is None:
            lu = team_lineup(self.config_dir, self.match_id, q["team"])
            hit = next((x for x in lu if x.get("name") == q.get("name") and x.get("number") is not None), None)
            num = hit["number"] if hit else None
        return q["team"], (None if num is None else int(num)), q.get("name")

    def _named_boxes(self, frame, tids=None):
        """Detected (not hand-drawn) boxes of labelled players on `frame`, after
        the fixes: [(key, row)]. tids limits it to those track ids."""
        a, b = np.searchsorted(self.df_frames, [frame, frame + 1])
        g = self.df.iloc[a:b]
        out = []
        for r in g.itertuples():
            tid = int(r.track_id)
            if tid not in self.key_of or (tids is not None and tid not in tids):
                continue
            if getattr(r, "source", "auto") != "auto":
                continue
            out.append((self.key_of[tid], r))
        return out

    def _write_anchors(self, edit_id, frame, named):
        """One anchor per named box, tagged with the edit, replacing older
        anchors on the same box or for the same player on that frame."""
        new, skipped = [], []
        for key, r in named:
            who = self.person(key)
            if not who or who[1] is None:
                skipped.append(key)
                continue
            team, num, name = who
            tl = getattr(r, "tracklet", None)
            new.append({"id": uuid.uuid4().hex[:10], "team": team, "number": num, "name": name,
                        "frame": int(frame), "box": [round(float(v), 1) for v in (r.x1, r.y1, r.x2, r.y2)],
                        "track_id": int(r.track_id),
                        "tracklet": int(tl) if tl is not None and pd.notna(tl) and tl >= 0 else None,
                        "source": "tracking", "edit": int(edit_id), "created": time.time()})
        if not new:
            return 0, skipped
        with ANCHOR_LOCK:
            anchors = load_anchors(self.anchor_path)
            for a in new:
                anchors = [b for b in anchors if not (int(b["frame"]) == a["frame"] and (
                    iou(b.get("box") or [0, 0, 0, 0], a["box"]) > 0.7
                    or (b.get("team") == a["team"] and b.get("number") is not None and int(b["number"]) == a["number"])))]
                anchors.append(a)
            save_anchors(self.anchor_path, anchors)
        return len(new), skipped

    def _drop_anchors(self, edit_id):
        with ANCHOR_LOCK:
            anchors = load_anchors(self.anchor_path)
            keep = [a for a in anchors if not (a.get("source") == "tracking" and a.get("edit") == int(edit_id))]
            if len(keep) != len(anchors):
                save_anchors(self.anchor_path, keep)

    def anchors(self):
        """Every anchor (calibration clips, Lines tab, this tab) in the tracked
        window, and how many are newer than the tracking (waiting for a re-link)."""
        out = []
        for a in load_anchors(self.anchor_path):
            f = int(a.get("frame", -1))
            if self.lo <= f <= self.hi and a.get("box"):
                out.append({"frame": f, "box": a["box"], "team": a.get("team"), "number": a.get("number"),
                            "name": a.get("name"), "source": a.get("source") or "lines", "edit": a.get("edit"),
                            "pending": float(a.get("created") or 0) > self.tracks_mtime})
        return out

    def add_edit(self, edit):
        """Validate, fill in what the server knows (tracklet, pitch spot),
        save, re-apply. Returns the saved edit."""
        if not isinstance(edit, dict):
            raise TrackingError("An edit must be an object")
        edit = dict(edit)
        f = edit.get("frame")
        if f is not None:
            try:
                f = int(f)
            except (TypeError, ValueError):
                raise TrackingError("frame must be a whole number") from None
            if not self.lo <= f <= self.hi:
                raise TrackingError("That frame is outside the tracked window")
        with self.lock:
            self._refresh_if_changed()
            kind = edit.get("kind")
            if kind in ("assign", "split", "not_player") and edit.get("tracklet") is None and f is not None:
                tl = self.tracklet_at(int(edit.get("tid", -1)), f)
                if tl is not None:
                    edit["tracklet"] = tl
            if kind == "box" and f is not None and edit.get("pitch_x") is None:
                for k in ("x1", "y1", "x2", "y2"):
                    if edit.get(k) is None:
                        raise TrackingError(f"'{k}' is required")
                x1, x2 = sorted((float(edit["x1"]), float(edit["x2"])))
                y1, y2 = sorted((float(edit["y1"]), float(edit["y2"])))
                edit.update(x1=x1, x2=x2, y1=y1, y2=y2)
                edit["pitch_x"], edit["pitch_y"] = self._pitch_at(f, (x1 + x2) / 2, y2)
            if kind in ("swap",):
                known = {t for q in self.players for t in q["ids"]}
                if edit.get("a") not in known or edit.get("b") not in known:
                    raise TrackingError("Swap needs two identified players")
            if kind in ("assign", "box") and edit.get("to" if kind == "assign" else "tid") not in self.key_of:
                raise TrackingError("Pick one of the identified players")
            if self.stale:
                raise TrackingError("Your saved fixes were made against different tracks (tracking was re-run). "
                                    f"Check or remove {self.edits_path.name} first")
            if kind in ("box", "delete", "split") and edit.get("tid") in self.key_of:
                who = self.person(self.key_of[edit["tid"]])
                if who:
                    edit["player"] = {"team": who[0], "number": who[1], "name": who[2]}
            if kind == "lock" and f is not None and not self._named_boxes(f):
                raise TrackingError("No tracked, labelled boxes on this frame to lock")
            try:
                data = te.load_edits(self.edits_path)
                data["match_id"] = self.match_id
                saved = te.add_edit(data, edit, base=self.base_fp)
            except te.EditError as e:
                raise TrackingError(str(e)) from None
            te.save_edits(self.edits_path, data)
            self._rebuild()
            if kind in te.NAMING and f is not None:
                if kind == "swap":
                    tids = {saved["a"], saved["b"]}
                elif kind == "assign":
                    tids = {saved["to"]} | ({saved["tid"]} if saved["tid"] < te.CHAIN_MAX else set())
                else:
                    tids = None
                n, skipped = self._write_anchors(saved["id"], f, self._named_boxes(f, tids))
                saved["anchored"] = n
                if skipped:
                    saved["unanchored"] = [self._who(k, None) for k in skipped]
                if n:
                    data = te.load_edits(self.edits_path)
                    for e in data["edits"]:
                        if e.get("id") == saved["id"]:
                            e["anchored"] = n
                    te.save_edits(self.edits_path, data)
                    self._rebuild()
            return saved

    # ---- Auto-fix ------------------------------------------------------------
    def _colours(self, df):
        """Shirt + shorts Lab medians per row of df (NaN where unreadable or
        hand-drawn), from tracklets_cache.parquet, matched on frame + box."""
        if getattr(self, "_colour_tab", None) is None:
            path = self.mdir / "tracklets_cache.parquet"
            if not path.exists():
                return None
            c = pd.read_parquet(path, columns=["frame", "x1", "y1", *COLOUR, "shirt_n", "shorts_n"])
            c = c.drop_duplicates(["frame", "x1", "y1"])
            bad = (c["shirt_n"] < 4) | (c["shorts_n"] < 4)
            c.loc[bad, COLOUR] = np.nan
            self._colour_tab = c.drop(columns=["shirt_n", "shorts_n"])
        m = df[["frame", "x1", "y1"]].merge(self._colour_tab, on=["frame", "x1", "y1"], how="left")
        return m[COLOUR].to_numpy(float)

    @staticmethod
    def _profile(X):
        X = X[np.isfinite(X).all(axis=1)]
        if len(X) < 20:
            return None
        mu = np.median(X, axis=0)
        sd = np.maximum(1.4826 * np.median(np.abs(X - mu), axis=0), 8.0)
        return mu, sd

    @staticmethod
    def _dist(X, prof):
        mu, sd = prof
        with np.errstate(invalid="ignore"):
            return np.sqrt((((X - mu) / sd) ** 2).mean(axis=1))

    def _ref_runs(self, df, X):
        """Runs of a player's rows that look like the referee(s) you marked
        'not a player': closer to that shirt + shorts colour than to the
        player's own kit, for at least REF_MIN_S. Returns (items, ref profile)."""
        ids = df["track_id"].to_numpy()
        marked = {te.NEW_ID_BASE + int(e["id"]) for e in self.data["edits"] if e.get("kind") == "not_player"}
        ref = self._profile(X[np.isin(ids, list(marked))]) if marked else None
        if ref is None:
            return [], None
        kit = df["kit"].to_numpy() if "kit" in df.columns else np.full(len(df), None)
        fr = df["frame"].to_numpy()
        d_ref = self._dist(X, ref)
        items, min_run = [], int(REF_MIN_S * self.fps)
        own = {}
        for q in self.players:
            gk = q.get("role") == "GK"
            k = ("gk_a" if q["team"] == "team_a" else "gk_b") if gk else q["team"]
            if k not in own:
                mates = [t for x in self.players if x["team"] == q["team"] and (x.get("role") == "GK") == gk for t in x["ids"]]
                own[k] = self._profile(X[np.isin(ids, mates) & (kit == k)])
            prof = own[k]
            if prof is None:
                continue
            m = np.flatnonzero(np.isin(ids, q["ids"]))
            if not len(m):
                continue
            m = m[np.argsort(fr[m], kind="stable")]
            dr, do = d_ref[m], self._dist(X[m], prof)
            readable = np.isfinite(dr) & np.isfinite(do)
            like = readable & (dr < REF_D) & (dr < REF_RATIO * do)
            # smooth over the readable rows: 15 in a row, >= 60% referee-like
            r_idx = np.flatnonzero(readable)
            if len(r_idx) < 15:
                continue
            sm = pd.Series(like[r_idx].astype(float)).rolling(15, center=True, min_periods=8).mean().to_numpy() >= 0.6
            flag = np.zeros(len(m), bool)
            flag[r_idx] = sm
            # unreadable rows between two flagged readable rows go with them
            fl = pd.Series(np.where(readable, flag.astype(float), np.nan)).ffill().to_numpy()
            bl = pd.Series(np.where(readable, flag.astype(float), np.nan)).bfill().to_numpy()
            flag = (fl == 1) & (bl == 1)
            d = np.diff(np.r_[0, flag.astype(np.int8), 0])
            for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                rows = m[a:b]
                f0, f1 = int(fr[rows[0]]), int(fr[rows[-1]])
                if f1 - f0 + 1 < min_run:
                    continue
                for tid in np.unique(ids[rows]):
                    items.append([int(tid), f0, f1])
        return items, ref

    @staticmethod
    def _overlap(df, rows, others, chunk=200_000):
        """For each row in `rows` (positions): the most of its box that any box
        in `others` (positions; itself excluded) covers on the same frame, and
        the best IoU with one of them."""
        fr = df["frame"].to_numpy()
        B = df[["x1", "y1", "x2", "y2"]].to_numpy(float)
        oi = others[np.argsort(fr[others], kind="stable")]
        of = fr[oi]
        cover, iou = np.zeros(len(rows)), np.zeros(len(rows))
        for c0 in range(0, len(rows), chunk):
            ci = rows[c0:c0 + chunk]
            lo, hi = np.searchsorted(of, fr[ci], "left"), np.searchsorted(of, fr[ci], "right")
            n = hi - lo
            if not n.sum():
                continue
            rep = np.repeat(np.arange(len(ci)), n)
            other = oi[np.repeat(lo, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))]
            keep = other != ci[rep]
            rep, other = rep[keep], other[keep]
            a, b = B[ci[rep]], B[other]
            iw = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
            ih = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
            inter = iw * ih
            aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
            ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            with np.errstate(invalid="ignore", divide="ignore"):
                cv = np.where(aa > 0, inter / aa, 0)
                io = np.where(aa + ab - inter > 0, inter / (aa + ab - inter), 0)
            np.maximum.at(cover[c0:c0 + chunk], rep, cv)
            np.maximum.at(iou[c0:c0 + chunk], rep, io)
        return cover, iou

    def _dups(self, df, cand, known):
        """Candidate rows whose box mostly overlaps a labelled player's box on
        the same frame (a second detection of someone already tracked)."""
        ci = np.flatnonzero(cand)
        cover, _ = self._overlap(df, ci, np.flatnonzero(known))
        dup = np.zeros(len(df), bool)
        dup[ci[cover > AUTO_DUP]] = True
        return dup

    def _free_boxes(self, df, X, ref, min_conf=BOX_CONF):
        """The boxes Auto-fix may give to a missing player: nobody has them
        (not labelled, not a second box on someone labelled, not the referee),
        on the pitch, not fainter than min_conf, not cut off at the edge of
        the picture. Returns (mask over df rows, colour_ok(rows, group))."""
        ids = df["track_id"].to_numpy()
        px, py = df["pitch_x"].to_numpy(float), df["pitch_y"].to_numpy(float)
        kit = df["kit"].to_numpy() if "kit" in df.columns else np.full(len(df), "unknown")
        known = np.isin(ids, list(self.key_of))
        cand = ~known & np.isfinite(px) & np.isfinite(py)
        if ref is not None:
            dr = self._dist(X, ref)
            cand &= ~(np.isfinite(dr) & (dr < REF_D))
        L = float(self.cfg.get("pitch_length") or 105.0)
        Wd = float(self.cfg.get("pitch_width") or 68.0)
        cand &= (px >= -BOX_OFF_M) & (px <= L + BOX_OFF_M) & (py >= -BOX_OFF_M) & (py <= Wd + BOX_OFF_M)
        if "conf" in df.columns:
            cand &= df["conf"].to_numpy(float) >= min_conf
        x1, y1 = df["x1"].to_numpy(float), df["y1"].to_numpy(float)
        x2, y2 = df["x2"].to_numpy(float), df["y2"].to_numpy(float)
        e = BOX_EDGE_PX
        cand &= (x1 > e) & (y1 > e) & (x2 < self.W - e) & (y2 < self.H - e)
        tv = pd.DataFrame({"t": ids, "k": kit})
        tv = tv[tv["k"] != "unknown"].groupby("t")["k"].agg(lambda k: (k == "official").mean())
        refish = tv.index[tv >= BOX_REF_SHARE].to_numpy()
        cand &= ~np.isin(ids, refish)
        if cand.any():
            cand &= ~self._dups(df, cand, known)

        # colours: a box with an unread kit counts if it looks like his team's kit
        own = {}
        if X is not None and np.isfinite(X).any():
            for c in ("team_a", "team_b", "gk_a", "gk_b"):
                own[c] = self._profile(X[known & (kit == c)])

        def colour_ok(rows, grp):
            ok = kit[rows] == grp
            unk = kit[rows] == "unknown"
            prof = own.get(grp)
            if prof is not None and unk.any():
                d = self._dist(X[rows], prof)
                others = [own[c] for c in own if c != grp and own[c] is not None]
                if ref is not None:
                    others.append(ref)
                closer = np.ones(len(rows), bool)
                for o in others:
                    closer &= d < self._dist(X[rows], o)
                ok |= unk & np.isfinite(d) & (d < BOX_COL_D) & closer
            return ok

        return cand, colour_ok

    def _outfield(self, df, team):
        """A team's outfield players: [(q, frames, x, y, tids)] with one
        labelled spot per frame, plus a per-frame mask of who has a box and
        where the whole team is in play (each of them seen before and after),
        so a substitute coming on is never taken for anyone."""
        ids = df["track_id"].to_numpy()
        fr = df["frame"].to_numpy()
        px, py = df["pitch_x"].to_numpy(float), df["pitch_y"].to_numpy(float)
        outf = [q for q in self.players if q["team"] == team and q.get("role") != "GK"]
        have = np.zeros((len(outf), self.N), bool)
        inplay = np.ones(self.N, bool)
        info = []
        for i, q in enumerate(outf):
            m = np.isin(ids, q["ids"]) & np.isfinite(px)
            g = pd.DataFrame({"f": fr[m], "x": px[m], "y": py[m], "t": ids[m]}).drop_duplicates("f").sort_values("f")
            f = g["f"].to_numpy()
            info.append((q, f, g["x"].to_numpy(), g["y"].to_numpy(), g["t"].to_numpy()))
            if not len(f):
                inplay[:] = False
                continue
            have[i, f - self.lo] = True
            span = np.zeros(self.N, bool)
            span[f[0] - self.lo:f[-1] - self.lo + 1] = True
            inplay &= span
        return info, have, inplay

    def _reach(self, f, x, y, fq, qx, qy, tol=AUTO_END_M):
        """Could a player seen on frames f (spots x, y) be at (qx, qy) on
        frames fq, which fall between two of his boxes? Within running
        distance (+ tol metres, per query or one for all) of the box before
        and of the box after."""
        j = np.searchsorted(f, fq)
        ok = (j > 0) & (j < len(f))
        j = np.clip(j, 1, max(len(f) - 1, 1))
        dt_a, dt_b = (fq - f[j - 1]) / self.fps, (f[j] - fq) / self.fps
        near_a = np.hypot(qx - x[j - 1], qy - y[j - 1]) <= tol + AUTO_VMAX * dt_a
        near_b = np.hypot(qx - x[j], qy - y[j]) <= tol + AUTO_VMAX * dt_b
        return ok & near_a & near_b

    def _last_player(self, df, X, ref):
        """The last ones left, frame by frame. While one to three outfield
        players of a team have no box (and the whole team is in play), a free
        box in that team's colours (see _free_boxes) is his when he is the only
        missing player who could have run there from his last box and on to
        his next one, and it is the only such box he could reach on that
        frame. The fewer players missing, the fainter a box may be
        (LAST_CONF: half-hidden players are faint) and the further its spot
        may be off (LAST_END_M: a cut-off box's feet are not where his are). Runs of at least
        LAST_MIN_S; goalkeepers are left out. Returns [[tid, f0, f1, to], ...]."""
        ids = df["track_id"].to_numpy()
        fr = df["frame"].to_numpy()
        px, py = df["pitch_x"].to_numpy(float), df["pitch_y"].to_numpy(float)
        conf = df["conf"].to_numpy(float) if "conf" in df.columns else np.ones(len(df))
        cand, colour_ok = self._free_boxes(df, X, ref, min_conf=min(LAST_CONF))
        if not cand.any():
            return []
        ci = np.flatnonzero(cand)
        floor = np.asarray(LAST_CONF, float)
        picks_r, picks_k = [], []
        for team in ("team_a", "team_b"):
            info, have, inplay = self._outfield(df, team)
            if len(info) < 2:
                continue
            nmiss = (~have).sum(axis=0)
            frame_ok = inplay & (nmiss >= 1) & (nmiss <= len(floor))
            if not frame_ok.any():
                continue
            rows = ci[colour_ok(ci, team)]
            rf = fr[rows] - self.lo
            rows, rf = rows[frame_ok[rf]], rf[frame_ok[rf]]
            sel = conf[rows] >= floor[np.clip(nmiss[rf] - 1, 0, len(floor) - 1)]
            rows, rf = rows[sel], rf[sel]
            if not len(rows):
                continue
            tol = np.asarray(LAST_END_M, float)[np.clip(nmiss[rf] - 1, 0, len(LAST_END_M) - 1)]
            reach = np.zeros((len(info), len(rows)), bool)
            for i, (_q, f, x, y, _t) in enumerate(info):
                s = np.flatnonzero(~have[i, rf])
                if len(s) and len(f):
                    reach[i, s] = self._reach(f, x, y, fr[rows[s]], px[rows[s]], py[rows[s]], tol[s])
            by_n = reach.sum(axis=0)              # how many missing players could be this box
            for i, (q, *_rest) in enumerate(info):
                s = np.flatnonzero(reach[i])
                if not len(s):
                    continue
                per_f = np.bincount(rf[s], minlength=self.N)
                good = s[(per_f[rf[s]] == 1) & (by_n[s] == 1)]
                picks_r.append(rows[good])
                picks_k.append(np.full(len(good), q["key"], dtype=object))
        if not picks_r:
            return []
        rows = np.concatenate(picks_r)
        keys = np.concatenate(picks_k)
        if not len(rows):
            return []
        # rank of each row within its own track, so a run never skips one of its rows
        rank = df.groupby("track_id").cumcount().to_numpy()
        t, f, rk = ids[rows], fr[rows], rank[rows]
        o = np.lexsort((f, t, keys))
        keys, t, f, rk = keys[o], t[o], f[o], rk[o]
        gap = int(PRED_S * self.fps)
        brk = np.r_[True, (keys[1:] != keys[:-1]) | (t[1:] != t[:-1]) | (rk[1:] - rk[:-1] != 1)
                    | (f[1:] - f[:-1] > gap)]
        starts = np.flatnonzero(brk)
        ends = np.r_[starts[1:], len(t)] - 1
        min_n = int(round(LAST_MIN_S * self.fps))
        return [[int(t[s0]), int(f[s0]), int(f[e0]), int(self.by_key[keys[s0]]["tid"])]
                for s0, e0 in zip(starts, ends) if f[e0] - f[s0] + 1 >= min_n]

    def _hop_back(self, df, moved=()):
        """Label hops. When a player's label jumps further than anyone can run
        and later jumps back to where he was heading, it sat on someone else
        in between. If exactly one team-mate could be that someone (no box of
        his own meanwhile, within running distance of his last box at the
        start and of his next box at the end), the stretch is his; the player
        it came from is then missing there and the last-ones-left step can
        find him. A stretch that overlaps one already moved for either of the
        two players (`moved`, earlier rounds) is left, so nothing bounces back
        and forth. Returns [(from key, to key, f0, f1), ...]."""
        claims = []
        for team in ("team_a", "team_b"):
            info, _have, _inplay = self._outfield(df, team)
            for i, (q, f, x, y, t) in enumerate(info):
                if len(f) < 3:
                    continue
                dt = np.diff(f) / self.fps
                jump = np.flatnonzero(np.hypot(np.diff(x), np.diff(y)) > AUTO_END_M + AUTO_VMAX * dt)
                done = -1
                for a in jump:
                    if a <= done:
                        continue
                    for b in jump[jump > a]:
                        if f[b] - f[a + 1] > HOP_MAX_S * self.fps:
                            break
                        back = (np.hypot(x[b + 1] - x[a], y[b + 1] - y[a])
                                <= AUTO_END_M + AUTO_VMAX * (f[b + 1] - f[a]) / self.fps)
                        if not back:
                            continue
                        s0, s1 = a + 1, b
                        done = b
                        if f[s1] - f[s0] + 1 < LAST_MIN_S * self.fps:
                            break
                        who = []
                        for k, (q2, f2, x2, y2, _t2) in enumerate(info):
                            if k == i or not len(f2):
                                continue
                            lo, hi = np.searchsorted(f2, f[s0]), np.searchsorted(f2, f[s1], "right")
                            if hi > lo or lo == 0 or hi == len(f2):
                                continue   # has a box meanwhile, or not on yet / gone off
                            if self._reach(f2, x2, y2, np.array([f[s0], f[s1]]),
                                           np.array([x[s0], x[s1]]), np.array([y[s0], y[s1]])).all():
                                who.append(q2)
                        if len(who) == 1:
                            claims.append((q["key"], who[0]["key"], int(f[s0]), int(f[s1])))
                        break
        # two stretches handed to the same player at the same time: neither is sure
        claims = [c for c in claims
                  if not any(d is not c and d[1] == c[1] and d[2] <= c[3] and c[2] <= d[3] for d in claims)]
        return [c for c in claims
                if not any({c[0], c[1]} & {m[0], m[1]} and m[2] <= c[3] and c[2] <= m[3] for m in moved)]

    def _apply_hops(self, df, claims):
        """Move each claimed stretch to its player. The saved items name the
        rows by the track ids they had before this auto-fix (column _t0), in
        runs that never skip a row of that track, so replaying the edit moves
        exactly these rows even when earlier items in it relabelled them."""
        ids, fr, t0 = df["track_id"].to_numpy(), df["frame"].to_numpy(), df["_t0"].to_numpy()
        rank0 = df.groupby("_t0").cumcount().to_numpy()
        tcol = df.columns.get_loc("track_id")
        items = []
        for fk, tk, f0, f1 in claims:
            rows = np.flatnonzero(np.isin(ids, self.by_key[fk]["ids"]) & (fr >= f0) & (fr <= f1))
            if not len(rows):
                continue
            to = int(self.by_key[tk]["tid"])
            df.iloc[rows, tcol] = to
            r = rows[np.lexsort((fr[rows], t0[rows]))]
            brk = np.r_[True, (t0[r][1:] != t0[r][:-1]) | (rank0[r][1:] - rank0[r][:-1] != 1)]
            s = np.flatnonzero(brk)
            e = np.r_[s[1:], len(r)] - 1
            items += [[int(t0[r[a]]), int(fr[r[a]]), int(fr[r[b]]), to] for a, b in zip(s, e)]
        return df, items

    def _fill_boxes(self, df, X, ref):
        """Box search, frame by frame. While a player is missing, he is somewhere
        in the rectangle between his last box and his next one. Every box on
        that frame that nobody has (not labelled, not a second box on someone
        labelled, not the referee), in his colours and inside his rectangle,
        is his, unless it is also inside the rectangle of a team-mate missing
        at the same time (then it could be either of them and is left). When
        more than one box qualifies, the one nearest the line from his last
        box to his next one. Returns [[tid, f0, f1, to], ...]."""
        ids = df["track_id"].to_numpy()
        fr = df["frame"].to_numpy()
        px, py = df["pitch_x"].to_numpy(float), df["pitch_y"].to_numpy(float)
        cand, colour_ok = self._free_boxes(df, X, ref)
        if not cand.any():
            return []
        ci = np.flatnonzero(cand)
        ci = ci[np.argsort(fr[ci], kind="stable")]
        cfr = fr[ci]
        lo_f = self.lo
        n_f = self.N
        maxg = int(BOX_GAP_S * self.fps)
        INF = np.inf

        def group_of(q):
            gk = q.get("role") == "GK"
            return ("gk_a" if q["team"] == "team_a" else "gk_b") if gk else q["team"]

        # each player's rectangle per frame: NaN where he has a box (or is not
        # on the pitch yet / any more), +-inf where he is missing too long to bound
        rect, line = {}, {}
        for q in self.players:
            m = np.isin(ids, q["ids"]) & np.isfinite(px)
            g = pd.DataFrame({"f": fr[m], "x": px[m], "y": py[m]}).drop_duplicates("f").sort_values("f")
            f, x, y = g["f"].to_numpy(), g["x"].to_numpy(), g["y"].to_numpy()
            R = np.full((n_f, 4), np.nan)
            Lx = np.full(n_f, np.nan)
            Ly = np.full(n_f, np.nan)
            if not len(f):
                continue
            for k in np.flatnonzero(np.diff(f) > 1):
                fa, fb = int(f[k]), int(f[k + 1])
                sl = slice(fa + 1 - lo_f, fb - lo_f)
                if fb - fa > maxg:
                    R[sl] = (-INF, INF, -INF, INF)
                    continue
                R[sl] = (min(x[k], x[k + 1]) - BOX_PAD_M, max(x[k], x[k + 1]) + BOX_PAD_M,
                         min(y[k], y[k + 1]) - BOX_PAD_M, max(y[k], y[k + 1]) + BOX_PAD_M)
                w = (np.arange(fa + 1, fb) - fa) / (fb - fa)
                Lx[sl] = x[k] + w * (x[k + 1] - x[k])
                Ly[sl] = y[k] + w * (y[k + 1] - y[k])
            rect[q["key"]], line[q["key"]] = R, (Lx, Ly)

        groups = {}
        for q in self.players:
            groups.setdefault(group_of(q), []).append(q["key"])
        pick = {}      # row -> (dist, key)
        for q in self.players:
            key, grp = q["key"], group_of(q)
            R = rect[key]
            bounded = np.isfinite(R[:, 0])
            if not bounded.any():
                continue
            fi = np.flatnonzero(bounded) + lo_f
            a, b = np.searchsorted(cfr, fi[0]), np.searchsorted(cfr, fi[-1], "right")
            rows = ci[a:b]
            rf = fr[rows] - lo_f
            r = R[rf]
            inside = (bounded[rf] & (px[rows] >= r[:, 0]) & (px[rows] <= r[:, 1])
                      & (py[rows] >= r[:, 2]) & (py[rows] <= r[:, 3]))
            rows, rf = rows[inside], rf[inside]
            if not len(rows):
                continue
            ok = colour_ok(rows, grp)
            rows, rf = rows[ok], rf[ok]
            # not inside a missing team-mate's rectangle too
            for mate in groups[grp]:
                if mate == key or not len(rows):
                    continue
                M2 = rect[mate][rf]
                with np.errstate(invalid="ignore"):
                    also = ((px[rows] >= M2[:, 0]) & (px[rows] <= M2[:, 1])
                            & (py[rows] >= M2[:, 2]) & (py[rows] <= M2[:, 3]))
                rows, rf = rows[~also], rf[~also]
            if not len(rows):
                continue
            Lx, Ly = line[key]
            d = np.hypot(px[rows] - Lx[rf], py[rows] - Ly[rf])
            # one box per frame for him: the nearest the line
            o = np.lexsort((d, rf))
            first = np.r_[True, rf[o][1:] != rf[o][:-1]]
            for r_, d_ in zip(rows[o][first], d[o][first]):
                if r_ not in pick or d_ < pick[r_][0]:
                    pick[int(r_)] = (float(d_), key)
        if not pick:
            return []
        rows = np.fromiter(pick, int)
        keys = np.array([pick[r][1] for r in rows])
        t = ids[rows]
        f = fr[rows]
        o = np.lexsort((f, t, keys))
        rows, keys, t, f = rows[o], keys[o], t[o], f[o]
        items = []
        # contiguous runs of the same box track going to the same player
        brk = np.r_[True, (keys[1:] != keys[:-1]) | (t[1:] != t[:-1]) | (f[1:] - f[:-1] > 1)]
        starts = np.flatnonzero(brk)
        ends = np.r_[starts[1:], len(rows)] - 1
        for s0, e0 in zip(starts, ends):
            items.append([int(t[s0]), int(f[s0]), int(f[e0]), int(self.by_key[keys[s0]]["tid"])])
        return items

    def _eliminate(self, df, X, ref):
        """Process of elimination. An unlabelled track with a clear team kit,
        during which exactly one player of that team (same role: outfield or
        goalkeeper) is unaccounted for, and which he could reach from where he
        vanished and leave in time to where he reappears, is him. Skipped when
        another orphan track at the same time could also be him.
        Returns [[tid, f0, f1, to], ...]."""
        ids = df["track_id"].to_numpy()
        fr = df["frame"].to_numpy()
        px, py = df["pitch_x"].to_numpy(float), df["pitch_y"].to_numpy(float)
        kit = df["kit"].to_numpy() if "kit" in df.columns else np.full(len(df), "unknown")
        known = np.isin(ids, list(self.key_of))
        cand = ~known & np.isfinite(px) & np.isfinite(py)
        if ref is not None:
            dr = self._dist(X, ref)
            cand &= ~(np.isfinite(dr) & (dr < REF_D))
        cand &= ~self._dups(df, cand, known)
        if not cand.any():
            return []
        x1, x2 = df["x1"].to_numpy(float), df["x2"].to_numpy(float)
        edge = (x1 <= ELIM_EDGE_PX) | (x2 >= self.W - ELIM_EDGE_PX)
        conf = df["conf"].to_numpy(float) if "conf" in df.columns else np.ones(len(df))
        u = pd.DataFrame({"t": ids[cand], "f": fr[cand], "k": kit[cand], "x": px[cand], "y": py[cand],
                          "edge": edge[cand], "conf": conf[cand]})
        votes = u.groupby(["t", "k"]).size().unstack(fill_value=0)
        for c in ("team_a", "team_b", "gk_a", "gk_b", "official"):
            if c not in votes.columns:
                votes[c] = 0
        rd = votes[["team_a", "team_b", "gk_a", "gk_b", "official"]].sum(axis=1)
        side = pd.Series(None, index=votes.index, dtype=object)
        for c in ("team_a", "team_b", "gk_a", "gk_b"):
            side[(rd >= 10) & (votes[c] >= ELIM_KIT * rd)] = c
        # the kit has to hold all along: in each third of the track, not just overall
        u = u.sort_values(["t", "f"], kind="stable")
        u["third"] = (u.groupby("t").cumcount() * 3 // u.groupby("t")["f"].transform("size")).astype(int)
        tv = u[u["k"].isin(["team_a", "team_b", "gk_a", "gk_b", "official"])].groupby(["t", "third", "k"]).size()
        tv = tv.unstack(fill_value=0)
        tsum = tv.sum(axis=1)
        steady = {}
        for c in ("team_a", "team_b", "gk_a", "gk_b"):
            if c in tv.columns:
                okc = ((tv[c] >= ELIM_KIT * tsum) & (tsum >= 3)).groupby(level=0).agg(["sum", "size"])
                steady[c] = set(okc.index[(okc["sum"] == 3) & (okc["size"] == 3)].tolist())
        edge_share = u.groupby("t")["edge"].mean()
        med = u.groupby("t")[["conf", "x", "y"]].median()
        L = float(self.cfg.get("pitch_length") or 105.0)
        Wd = float(self.cfg.get("pitch_width") or 68.0)
        first = u.sort_values("f").groupby("t").first()
        last = u.sort_values("f").groupby("t").last()
        n = u.groupby("t").size()
        all_rows = pd.Series(ids).value_counts()
        pieces = []
        for t, c in side.dropna().items():
            if n[t] < ELIM_MIN_S * self.fps or all_rows[t] > 1.25 * n[t]:
                continue  # short, or much of it is a referee / a second box on someone
            if t not in steady.get(c, ()) or edge_share[t] > 0.5:
                continue  # changes kit part-way (two people), or cut off at the edge
            mx, my, mc = med.at[t, "x"], med.at[t, "y"], med.at[t, "conf"]
            if mc < ELIM_CONF or not (-ELIM_OFF_M <= mx <= L + ELIM_OFF_M and -ELIM_OFF_M <= my <= Wd + ELIM_OFF_M):
                continue  # faint, or off the pitch (boards, bench, crowd)
            pieces.append((int(t), c, int(first.at[t, "f"]), int(last.at[t, "f"]),
                           float(first.at[t, "x"]), float(first.at[t, "y"]),
                           float(last.at[t, "x"]), float(last.at[t, "y"])))
        if not pieces:
            return []
        keep_t = np.array([p_[0] for p_ in pieces])
        prow = np.flatnonzero(cand & np.isin(ids, keep_t))
        _, piou = self._overlap(df, prow, np.arange(len(df)))
        crowd = pd.Series(piou > 0.3).groupby(ids[prow]).mean()
        pieces = [p_ for p_ in pieces if crowd.get(p_[0], 0) <= ELIM_CROWD]
        # each player's labelled frames + positions
        seen = {}
        for q in self.players:
            m = np.isin(ids, q["ids"]) & np.isfinite(px)
            o = np.argsort(fr[m], kind="stable")
            seen[q["key"]] = (fr[m][o], px[m][o], py[m][o])
        group = {}
        for q in self.players:
            gk = q.get("role") == "GK"
            c = ("gk_a" if q["team"] == "team_a" else "gk_b") if gk else q["team"]
            group.setdefault(c, []).append(q["key"])

        def fits(key, f0, f1, x0, y0, x1, y1):
            sf, sx, sy = seen[key]
            if not len(sf):
                return False
            lo, hi = np.searchsorted(sf, f0), np.searchsorted(sf, f1, "right")
            if hi > lo or lo == 0 or hi == len(sf):
                return False  # seen meanwhile, or not on the pitch before / after
            fa, fb = sf[lo - 1], sf[hi]
            if (fb - fa) > ELIM_GAP_S * self.fps:
                return False
            if np.hypot(x0 - sx[lo - 1], y0 - sy[lo - 1]) > AUTO_END_M + AUTO_VMAX * (f0 - fa) / self.fps:
                return False
            return np.hypot(x1 - sx[hi], y1 - sy[hi]) <= AUTO_END_M + AUTO_VMAX * (fb - f1) / self.fps

        who = [[k for k in group.get(p[1], []) if fits(k, *p[2:])] for p in pieces]
        items = []
        for i, p in enumerate(pieces):
            if len(who[i]) != 1:
                continue
            key = who[i][0]
            if any(j != i and key in who[j] and pieces[j][2] <= p[3] and p[2] <= pieces[j][3]
                   for j in range(len(pieces))):
                continue  # another orphan at the same time could be him too
            items.append([p[0], p[2], p[3], int(self.by_key[key]["tid"])])
        return items

    def autofix(self):
        """Go through the problems and fix what can be fixed without you:
        1. referee look-alikes in players' tracks (needs one box marked 'not a
           player' to learn the referee's colours) are ruled out;
        2. box search: while a player is missing, the free boxes in his colours
           inside the rectangle between his last box and his next one are his
           (see _fill_boxes);
        3. elimination: a whole orphan track goes to the only player of that
           team who could be it (see _eliminate);
        0. label hops: a label that jumps away and back sat on a missing
           team-mate in between (see _hop_back);
        4. last ones left: with 1-3 outfield players of a team missing, a box
           in their colours only one of them could reach is his, and fainter
           boxes count the fewer are missing (see _last_player).
        Saved as one or two edits tagged with a batch id: undo removes them."""
        t0 = time.time()
        with self.lock:
            self._refresh_if_changed()
            if self.stale:
                raise TrackingError("Your saved fixes were made against different tracks: check them first")
            # a re-run replaces the last auto-fix instead of stacking on it
            data = te.load_edits(self.edits_path)
            if any(e.get("kind") in te.AUTO for e in data["edits"]):
                data["edits"] = [e for e in data["edits"] if e.get("kind") not in te.AUTO]
                if not data["edits"]:
                    data["base"] = None
                te.save_edits(self.edits_path, data)
                self._rebuild()
            lost_before = sum(1 for p in self.problems if p["kind"] == "lost" and p["fixed"] is None)
            lost_s_before = self._lost_seconds()
            df = self.df.reset_index(drop=True)
            X = self._colours(df)
            if X is None:
                X = np.full((len(df), len(COLOUR)), np.nan)
            ref_items, ref = self._ref_runs(df, X)
            batch = uuid.uuid4().hex[:8]
            data["match_id"] = self.match_id
            report = {"batch": batch, "ref_runs": len(ref_items), "ref_learned": ref is not None,
                      "ref_seconds": round(sum(b - a + 1 for _t, a, b in ref_items) / self.fps)}
            fill, n_box, n_elim, n_last, n_hop = [], 0, 0, 0, 0
            try:
                if ref_items:
                    e = te.add_edit(data, {"kind": "autoref", "frame": min(i[1] for i in ref_items),
                                           "items": ref_items[:999], "batch": batch,
                                           "note": f"Auto-fix: {len(ref_items)} referee look-alike runs taken off players"},
                                    base=self.base_fp)
                    df = te._autoref(df, e, e["id"])
                # the track ids the saved edit is replayed against (see _apply_hops)
                df = df.assign(_t0=df["track_id"].to_numpy())
                moved = []
                for rnd in range(AUTO_ROUNDS):
                    line = self._fill_boxes(df, X, ref)
                    df = te._autofill(df, {"items": line}) if line else df
                    elim = self._eliminate(df, X, ref)
                    df = te._autofill(df, {"items": elim}) if elim else df
                    # label hops show once box search has filled in where the
                    # player really was; put them back before the last ones left
                    claims = self._hop_back(df, moved)
                    df, hop = self._apply_hops(df, claims) if claims else (df, [])
                    moved += claims
                    # the looser last-ones-left step trusts every label it sees, so
                    # it waits until the stricter steps have had STRICT_ROUNDS
                    # rounds (or nothing left) to fill players in and put hops back
                    last = []
                    strict_done = not line and not elim and not hop
                    if strict_done or rnd >= STRICT_ROUNDS:
                        last = self._last_player(df, X, ref)
                        df = te._autofill(df, {"items": last}) if last else df
                    if strict_done and not last:
                        break
                    fill += line + elim + hop + last
                    n_box += len(line)
                    n_elim += len(elim)
                    n_hop += len(claims)
                    n_last += len(last)
                if fill:
                    te.add_edit(data, {"kind": "autofill", "frame": min(i[1] for i in fill),
                                       "items": fill[:te.MAX_ITEMS], "batch": batch,
                                       "note": f"Auto-fix: {n_hop} label jumps put back, "
                                               f"{n_box} runs of boxes found in a missing player's rectangle, "
                                               f"{n_elim} tracks named by elimination, {n_last} runs for the last player left"},
                                base=self.base_fp)
            except te.EditError as e:
                raise TrackingError(str(e)) from None
            report["fill_pieces"] = len(fill)
            report["box_pieces"], report["elim_pieces"], report["last_pieces"] = n_box, n_elim, n_last
            report["hop_pieces"] = n_hop
            if ref_items or fill:
                te.save_edits(self.edits_path, data)
                self._rebuild()
            report["lost_before"] = lost_before
            report["lost_after"] = sum(1 for p in self.problems if p["kind"] == "lost" and p["fixed"] is None)
            report["lost_s_before"] = lost_s_before
            report["lost_s_after"] = self._lost_seconds()
            report["took_s"] = round(time.time() - t0, 1)
            return report

    def _lost_seconds(self):
        return round(sum(int((st == LOST).sum()) for st in self.status.values()) / self.fps)

    def undo_autofix(self):
        with self.lock:
            data = te.load_edits(self.edits_path)
            n = len(data["edits"])
            data["edits"] = [e for e in data["edits"] if e.get("kind") not in te.AUTO]
            if len(data["edits"]) == n:
                raise TrackingError("No auto-fixes to undo")
            if not data["edits"]:
                data["base"] = None
            te.save_edits(self.edits_path, data)
            self._rebuild()
            return {"removed": n - len(data["edits"])}

    def remove_edit(self, edit_id):
        with self.lock:
            data = te.load_edits(self.edits_path)
            try:
                te.remove_edit(data, edit_id)
            except te.EditError as e:
                raise TrackingError(str(e)) from None
            te.save_edits(self.edits_path, data)
            self._drop_anchors(edit_id)
            self._rebuild()
            return {"removed": int(edit_id)}


class TrackingCache:
    """MatchTracking per match, rebuilt when the tracking artifacts change."""

    def __init__(self, root, size=1):
        self.root, self.size = Path(root), size
        self.items = OrderedDict()
        self.lock = threading.Lock()

    def _stamp(self, match_id):
        paths = [self.root / ".tmp" / match_id / n for n in
                 ("tracks.parquet", "player_map.json", "ball_track.parquet", "contact_swaps.json")]
        paths.append(self.root / "config" / f"{match_id}_calibration.json")
        return tuple(p.stat().st_mtime if p.exists() else 0 for p in paths)

    def get(self, match_id):
        stamp = self._stamp(match_id)
        with self.lock:
            hit = self.items.get(match_id)
            if hit and hit[0] == stamp:
                self.items.move_to_end(match_id)
                return hit[1]
        mt = MatchTracking(self.root, match_id)  # slow: outside the lock
        with self.lock:
            self.items[match_id] = (stamp, mt)
            self.items.move_to_end(match_id)
            while len(self.items) > self.size:
                self.items.popitem(last=False)
        return mt
