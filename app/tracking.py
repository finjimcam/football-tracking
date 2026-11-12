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
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import track_edits as te  # noqa: E402
from _common import clock_label, load_calibration, load_meta, load_player_map, match_config  # noqa: E402

OFF, LOST, PRED, SEEN, HAND = 0, 1, 2, 3, 4
CODES = "olpsh"
PRED_S = 1.0          # gaps up to this long are drawn as predicted
LOST_MIN_S = 3.0      # a lost run this long is a problem worth checking
SWAP_NEAR_S = 3.0     # a swap fix this close to a flagged swap resolves it
MAX_CHUNK = 750       # frames per /frames request (30 s at 25 fps)
KIT_CODE = {"team_a": "a", "team_b": "b", "gk_a": "a", "gk_b": "b", "official": "r"}
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
                "edits": [{**e, "clock": clock_label(e["frame"], self.cfg, self.fps) if e.get("frame") is not None
                           else None} for e in self.data["edits"]],
                "edits_file": self.edits_path.name,
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
            try:
                data = te.load_edits(self.edits_path)
                data["match_id"] = self.match_id
                saved = te.add_edit(data, edit, base=self.base_fp)
            except te.EditError as e:
                raise TrackingError(str(e)) from None
            te.save_edits(self.edits_path, data)
            self._rebuild()
            return saved

    def remove_edit(self, edit_id):
        with self.lock:
            data = te.load_edits(self.edits_path)
            try:
                te.remove_edit(data, edit_id)
            except te.EditError as e:
                raise TrackingError(str(e)) from None
            te.save_edits(self.edits_path, data)
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
