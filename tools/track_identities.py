"""Pitch-space tracking with persistent player identities ->
tracks.parquet, team_map.json, player_map.json, identity_links.parquet.

Stage 3 of workflows/track_identities.md. Reads only cached artifacts
(detections.parquet from detect_frames.py, frame_homographies.npz from
calibrate_auto.py, config/<id>_match.json + _reference.json), so a full half
re-runs in minutes - tune and re-run freely.

1. Drop duplicate person boxes (yolo26's NMS-free head emits near-identical
   pairs for ~8% of boxes; each used to split a tracklet), then project
   every foot point to pitch metres with that frame's homography. Frames
   whose calibration lost lock are skipped.
2. Kit colour per detection (kit_colour(), from detect_frames.py's Lab
   features) -> team via the match config's "kits" map.
3. Tracklets: frame-to-frame association in pitch space (velocity
   prediction, Hungarian, gate widens while coasting, kit mismatch costs).
   A tracklet ends when the player leaves the camera's view or is missed
   for > --max-miss frames, so each tracklet is short-term reliable.
4. Identity linking - the "same player all half" step - solved GLOBALLY per
   team as a min-cost flow (tracklet-linking formulation of Zhang, Li &
   Nevatia 2008): every tracklet is a node with a reward for being covered,
   edges join a tracklet's end to plausible successors, and exactly 10
   units of flow = exactly 10 disjoint outfield identity chains (valid
   because the window has no substitutions or red cards). An edge exists
   only if the gap is physically runnable (<= --vmax m/s); its cost is the
   distance from the predicted position (uncertainty grows with the gap)
   plus, increasingly for long gaps, the change in the tracklet's place in
   the team shape (offset from the visible teammates' centroid). Global
   solving matters: the earlier greedy version made a choice at every
   crossing and its mistakes cascaded.
5. Stability check: the flow is re-solved with perturbed parameters; a link
   that doesn't survive every variant is flagged 'fragile' (verify_tracking.py
   lists them as the moments to eyeball).
6. Names: identities are matched to the official lineup (formation lines
   in config/<id>_reference.json, listed right-to-left from each team's
   perspective) by average depth and width in the team shape. These names
   are ROLE-INFERRED; verify_tracking.py checks them against Opta events.
   Manual anchors from the Stopframe Lines tab (config/<id>_identity_anchors.json,
   tools/identity_anchors.py, issue #16) override them: a chain is named
   after the player its anchors point at, and an anchored tracklet is moved
   into that player's chain.
7. With names (calibration clips / anchors, issue #46) the linking itself is
   different: one chain per PLAYER, solved together as an integer programme
   (link_anchored) in which each player's chain must pass through every
   tracklet named as him, and pays for straying from his place in the team
   shape as read off those named tracklets. The plain flow keeps a player
   right for seconds but, over the minutes between two named clips, swaps
   most of them (0-3 of 10 kept the same chain clip to clip); voting names
   onto its chains afterwards couldn't fix that. Then: a tracklet named as two
   players is cut between the names; loose team tracklets go to the one
   off-screen player who can reach them and fits the shape
   (assign_by_elimination); and the part of a duplicate/neighbour tracklet
   that bridges a player's gap becomes his (fill_partial).

Output track_id scheme: team_a players 1-11 (1 = GK), team_b 101-111
(101 = GK), everything else 10000 + tracklet. Officials and people living
off the pitch are listed in player_map.json's ignored_tracks, so every stats
tool drops them.

Usage:
    python tools/track_identities.py --match-id <id>
"""
import argparse
from collections import Counter, deque

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from _common import (apply_calibration, load_calibration, load_meta, match_config, match_dir,
                     pitch_dims, read_json, reference_path, write_json)
from identity_anchors import anchors_path, load_anchors  # noqa: E402
from identity_anchors import apply as apply_anchors  # noqa: E402
from identity_anchors import resolve as resolve_anchors  # noqa: E402
from identity_anchors import split_conflicts, team_pins  # noqa: E402

TEAMS = ("team_a", "team_b")
KIT_LABELS = ("team_a", "team_b", "gk_a", "gk_b", "official")
INF = 1e9


# --------------------------------------------------------------------------
# 1-2. Duplicates and kit colour per detection
# --------------------------------------------------------------------------
def suppress_duplicates(p, iou_thresh=0.6):
    """Drop near-identical person boxes in the same frame (keep the most
    confident). yolo26's NMS-free head emits them for ~8% of boxes; each one
    used to start a second tracklet that then stole the player's next
    detection and killed the original - the biggest single source of
    tracklet fragmentation on this footage."""
    keep = np.ones(len(p), bool)
    order = np.lexsort((-p["conf"].to_numpy(), p["frame"].to_numpy()))
    fr = p["frame"].to_numpy()[order]
    boxes = p[["x1", "y1", "x2", "y2"]].to_numpy()[order]
    cuts = np.flatnonzero(np.diff(fr)) + 1
    for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(fr)]):
        if e - s < 2:
            continue
        b = boxes[s:e]
        x1 = np.maximum(b[:, None, 0], b[None, :, 0])
        y1 = np.maximum(b[:, None, 1], b[None, :, 1])
        x2 = np.minimum(b[:, None, 2], b[None, :, 2])
        y2 = np.minimum(b[:, None, 3], b[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iou = inter / (area[:, None] + area[None, :] - inter + 1e-9)
        alive = np.ones(e - s, bool)
        for i in range(e - s):  # rows are conf-descending within the frame
            if alive[i]:
                alive[i + 1:] &= iou[i, i + 1:] <= iou_thresh
        keep[order[s:e]] = alive
    return p[keep].reset_index(drop=True)


def kit_colour(det):
    """Colour class per detection from detect_frames.py's shirt Lab median.
    Thresholds were set on this broadcast's crops (see the workflow):
    sky = City shirts, white = Spurs shirts, orange = Lloris, dark =
    officials/staff. 'green' = torso reads as grass (shirt_n ~ 0): Ederson's
    all-green kit - but also any box too small to read, so it is only a weak
    vote, combined with position at tracklet level."""
    L, a, b = det["shirt_L"].to_numpy(), det["shirt_a"].to_numpy(), det["shirt_b"].to_numpy()
    n = det["shirt_n"].to_numpy()
    h = (det["y2"] - det["y1"]).to_numpy()
    w = (det["x2"] - det["x1"]).to_numpy()
    band = (np.clip(np.floor(0.45 * h) - np.floor(0.15 * h), 1, None)
            * np.clip(np.floor(0.8 * w) - np.floor(0.2 * w), 1, None))
    grass_frac = 1 - n / band
    out = np.full(len(det), "unknown", dtype=object)
    green = (n == 0) | (grass_frac > 0.85)
    out[green] = "green"
    ok = (~green) & (n >= 4) & np.isfinite(L)
    with np.errstate(invalid="ignore"):
        out[ok & (L < 105)] = "dark"
        out[ok & (L >= 105) & (b < 126.5) & (a < 135)] = "sky"
        out[ok & (L >= 175) & (b >= 129) & (a < 135)] = "white"
        out[ok & (a > 138) & (b > 140)] = "orange"
    return out


# --------------------------------------------------------------------------
# 3. Tracklets
# --------------------------------------------------------------------------
class Track:
    __slots__ = ("tid", "x", "y", "vx", "vy", "last", "miss", "hist", "votes")

    def __init__(self, tid, f, x, y, kit):
        self.tid, self.x, self.y, self.vx, self.vy = tid, x, y, 0.0, 0.0
        self.last, self.miss = f, 0
        self.hist = deque([(f, x, y)], maxlen=10)
        self.votes = Counter([kit])

    def kit(self):
        team = {k: v for k, v in self.votes.items() if k in KIT_LABELS}
        if not team or sum(team.values()) < 4:
            return None
        k, v = max(team.items(), key=lambda kv: kv[1])
        return k if v >= 0.6 * sum(team.values()) else None


def build_tracklets(frames, xs, ys, conf, kit, world_to_img, img_wh, ok_frames, fps, a):
    """Returns (tracklet id per detection row (-1 = not tracked),
    {tracklet: end reason 'missed' | 'left_view' | 'open'})."""
    order = np.argsort(frames, kind="stable")
    fr_sorted = frames[order]
    f0, f1 = int(fr_sorted[0]), int(fr_sorted[-1])
    starts = np.searchsorted(fr_sorted, np.arange(f0, f1 + 2))
    assign = np.full(len(frames), -1, dtype=np.int64)
    active, next_id, ended = [], 0, {}
    w_img, h_img = img_wh
    for f in range(f0, f1 + 1):
        rows = order[starts[f - f0]:starts[f - f0 + 1]]
        good_frame = ok_frames(f)
        if not good_frame:
            rows = rows[:0]
        matched_t, matched_r = set(), set()
        if active and len(rows):
            dt = np.array([(f - t.last) / fps for t in active])
            px = np.array([t.x + t.vx * d for t, d in zip(active, dt)])
            py = np.array([t.y + t.vy * d for t, d in zip(active, dt)])
            D = np.hypot(px[:, None] - xs[rows][None], py[:, None] - ys[rows][None])
            gates = np.minimum(a.gate + a.gate_step * np.array([t.miss for t in active]), a.gate_max)
            C = D.copy()
            rk = kit[rows]
            for i, t in enumerate(active):
                tk = t.kit()
                if tk is not None:
                    C[i] += np.where(np.isin(rk, KIT_LABELS) & (rk != tk), a.kit_penalty, 0.0)
            C[D > gates[:, None]] = INF
            ri, ci = linear_sum_assignment(C)
            for i, j in zip(ri, ci):
                if C[i, j] >= INF:
                    continue
                t, r = active[i], rows[j]
                t.hist.append((f, xs[r], ys[r]))
                f_old, x_old, y_old = t.hist[0]
                span = (f - f_old) / fps
                if len(t.hist) >= 5 and span > 0:
                    t.vx, t.vy = (xs[r] - x_old) / span, (ys[r] - y_old) / span
                t.x, t.y, t.last, t.miss = xs[r], ys[r], f, 0
                t.votes[kit[r]] += 1
                assign[r] = t.tid
                matched_t.add(i)
                matched_r.add(j)
        survivors = []
        for i, t in enumerate(active):
            if i in matched_t:
                survivors.append(t)
                continue
            t.miss += 1
            if t.miss > a.max_miss:
                ended[t.tid] = "missed"
                continue
            if good_frame:  # left the camera's view? (predicted position projected into the image)
                d = (f - t.last) / fps
                q = world_to_img(f) @ np.array([t.x + t.vx * d, t.y + t.vy * d, 1.0])
                u, v = (q[0] / q[2], q[1] / q[2]) if q[2] > 0 else (-1e9, -1e9)
                if u < -8 or u > w_img + 8 or v < -8 or v > h_img + 8:
                    ended[t.tid] = "left_view"
                    continue
            survivors.append(t)
        active = survivors
        dup_m = getattr(a, "dup_m", 0.0)
        taken = [rows[j] for j in matched_r] if dup_m > 0 else []
        for j, r in enumerate(rows):
            if j in matched_r or conf[r] < a.new_conf:
                continue
            # issue #46: a second box on a player who's already tracked this
            # frame (a duplicate the IoU filter missed) used to start a new
            # tracklet; the two then shared the player, and the old one
            # drifted onto a neighbour. Not a new person: leave it untracked.
            if taken and np.min(np.hypot(xs[taken] - xs[r], ys[taken] - ys[r])) < dup_m:
                continue
            t = Track(next_id, f, xs[r], ys[r], kit[r])
            assign[r] = next_id
            next_id += 1
            active.append(t)
    for t in active:
        ended[t.tid] = "open"
    return assign, ended


def summarize(df, fps, L, W):
    """One row per tracklet: span, entry/exit state, kit decision, positions."""
    out = []
    for tl, g in df[df["tracklet"] >= 0].groupby("tracklet", sort=True):
        g = g.sort_values("frame")
        f = g["frame"].to_numpy()
        xy = g[["pitch_x", "pitch_y"]].to_numpy()
        k = min(5, len(g))

        def vel(sl):
            ff, pp = f[sl], xy[sl]
            if len(ff) < 3 or ff[-1] == ff[0]:
                return (0.0, 0.0)
            return tuple((pp[-1] - pp[0]) / ((ff[-1] - ff[0]) / fps))

        votes = Counter(g["kit"])
        off = ((xy[:, 0] < -1) | (xy[:, 0] > L + 1) | (xy[:, 1] < -1) | (xy[:, 1] > W + 1)).mean()
        vin, vout = vel(slice(0, min(10, len(g)))), vel(slice(max(0, len(g) - 10), len(g)))
        out.append({
            "tracklet": int(tl), "start": int(f[0]), "end": int(f[-1]), "n": len(g),
            "entry_x": xy[:k, 0].mean(), "entry_y": xy[:k, 1].mean(),
            "exit_x": xy[-k:, 0].mean(), "exit_y": xy[-k:, 1].mean(),
            "vin_x": vin[0], "vin_y": vin[1], "vout_x": vout[0], "vout_y": vout[1],
            "med_x": float(np.median(xy[:, 0])), "med_y": float(np.median(xy[:, 1])),
            "offpitch": float(off), "conf": float(g["conf"].mean()),
            **{f"v_{c}": votes.get(c, 0) for c in ("team_a", "team_b", "gk_a", "gk_b", "official", "unknown")},
        })
    tl = pd.DataFrame(out)
    tl["cls"] = [decide_class(r) for r in tl.itertuples()]
    return tl


def decide_class(r):
    """Tracklet class from its detections' kit votes + where it lives. A team
    needs team votes on >= 50% of ALL its detections (the referee's dark
    green kit is mostly unreadable, with a few 'white' votes when he stands
    near Spurs players - judged on readable votes alone he became a Spurs
    player) and >= 75% purity between the two teams."""
    if r.offpitch > 0.6:
        return "off_pitch"
    readable = r.v_team_a + r.v_team_b + r.v_gk_b + r.v_official
    teams = r.v_team_a + r.v_team_b
    if readable >= 3:
        if r.v_official >= 0.4 * readable or (r.v_official >= 0.1 * r.n and teams < 0.5 * r.n):
            return "official"
        if r.v_gk_b >= 0.4 * readable:
            return "gk_b"
        if teams >= 0.5 * r.n:
            if max(r.v_team_a, r.v_team_b) >= 0.75 * teams:
                return "team_a" if r.v_team_a >= r.v_team_b else "team_b"
            return "mixed"
    if r.v_gk_a >= 0.6 * r.n:
        return "gk_a?"  # unreadable/green - resolved with position in link_goalkeepers()
    return "unknown"


KIT_FEATURES = ["shirt_L", "shirt_a", "shirt_b", "shorts_L", "shorts_a", "shorts_b", "box_h"]


def relabel_teams(p, tl, lo=0.25, hi=0.75):
    """Self-trained team colour model: gradient-boosted trees on shirt +
    shorts colour + box height, trained on detections from 'pure' tracklets
    (>= 90% one team over >= 40 detections), then used to relabel every
    readable outfield detection. Smooths the hard thresholds of kit_colour(),
    which flicker for distant players whose sky blue washes towards white.
    Detections between lo and hi probability become 'unknown'."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    pure = tl[(tl["n"] >= 40) & tl["cls"].isin(TEAMS)]
    purity = np.maximum(pure["v_team_a"], pure["v_team_b"]) / (pure["v_team_a"] + pure["v_team_b"]).clip(lower=1)
    pure = pure[purity >= 0.9]
    label = dict(zip(pure["tracklet"], (pure["cls"] == "team_a").astype(int)))
    X = p.assign(box_h=p["y2"] - p["y1"])[KIT_FEATURES]
    train = p["tracklet"].isin(list(label)) & (p["shirt_n"] >= 4)
    y = p.loc[train, "tracklet"].map(label)
    if y.nunique() < 2:
        return p["kit"].to_numpy(), None
    idx = np.flatnonzero(train.to_numpy())
    if len(idx) > 200_000:
        idx = np.random.default_rng(0).choice(idx, 200_000, replace=False)
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31, random_state=0)
    clf.fit(X.iloc[idx], y.loc[p.index[idx]])
    prob = clf.predict_proba(X)[:, 1]
    kit = p["kit"].to_numpy().copy()
    outfield = np.isin(kit, ["team_a", "team_b", "unknown"]) & (p["shirt_n"].to_numpy() >= 4)
    kit[outfield] = np.where(prob[outfield] > hi, "team_a", np.where(prob[outfield] < lo, "team_b", "unknown"))
    return kit, prob


def split_by_kit(p, next_id, window=15, min_run=15):
    """Cut tracklets where the smoothed team signal flips for a sustained
    run (>= min_run frames each side): the tracker followed one player into
    a crossing and came out on an opponent. Returns the next free id."""
    s_all = np.select([p["kit"] == "team_a", p["kit"] == "team_b"], [1.0, -1.0], 0.0)
    p = p.assign(_s=s_all)
    new_ids = p["tracklet"].to_numpy().copy()
    pos = {i: k for k, i in enumerate(p.index)}
    kernel = np.ones(window) / window
    n_cuts = 0
    for t, g in p[p["tracklet"] >= 0].groupby("tracklet"):
        s = g["_s"].to_numpy()
        if len(g) < 2 * min_run or not ((s > 0).any() and (s < 0).any()):
            continue
        g = g.sort_values("frame")
        r = np.convolve(g["_s"].to_numpy(), kernel, mode="same")
        lab = np.where(r > 0.5, 1, np.where(r < -0.5, -1, 0))
        runs, start = [], 0
        for k in range(1, len(lab) + 1):
            if k == len(lab) or lab[k] != lab[start]:
                if lab[start] != 0 and k - start >= min_run:
                    runs.append((start, k - 1, lab[start]))
                start = k
        cuts = [(a[1] + b[0]) // 2 + 1 for a, b in zip(runs[:-1], runs[1:]) if a[2] != b[2]]
        if not cuts:
            continue
        rows = [pos[i] for i in g.index]
        for c0, c1 in zip(cuts, cuts[1:] + [len(g)]):
            new_ids[rows[c0:c1]] = next_id
            next_id += 1
            n_cuts += 1
    p["tracklet"] = new_ids
    return p.drop(columns="_s"), next_id, n_cuts


def _flicker(tl):
    t = tl[tl["cls"].isin(TEAMS + ("mixed",))]
    if t.empty:
        return 0.0
    minority = np.minimum(t["v_team_a"], t["v_team_b"]) / (t["v_team_a"] + t["v_team_b"]).clip(lower=1)
    return float((minority > 0.2).mean())


def contact_pairs(df, key, contact_m):
    """Same-team pairs closer than contact_m, per frame: DataFrame
    [frame, a, b, d] with a < b (values of `key`). Vectorised self-join on
    (frame, team) - a python loop over 70k frames took minutes."""
    q = df[["frame", "team", key, "pitch_x", "pitch_y"]]
    m = q.merge(q, on=["frame", "team"], suffixes=("_a", "_b"))
    m = m[m[f"{key}_a"] < m[f"{key}_b"]]
    d = np.hypot(m["pitch_x_a"] - m["pitch_x_b"], m["pitch_y_a"] - m["pitch_y_b"])
    m = m.assign(d=d)[d < contact_m]
    return m.rename(columns={f"{key}_a": "a", f"{key}_b": "b"})[["frame", "a", "b", "d"]]


def episodes(pairs, join_frames=3):
    """Collapse per-frame contact pairs into episodes: one row per run of
    frames (gaps <= join_frames) per pair, at its closest approach."""
    if pairs.empty:
        return pairs.assign(ep=[])
    pairs = pairs.sort_values(["a", "b", "frame"])
    new = (pairs["a"].diff().ne(0) | pairs["b"].diff().ne(0) | pairs["frame"].diff().gt(join_frames))
    pairs = pairs.assign(ep=new.cumsum())
    return pairs.loc[pairs.groupby("ep")["d"].idxmin()]


def split_at_contacts(p, tl, next_id, contact_m, min_side=15):
    """Issue #34: cut both same-team tracklets at the closest approach of every
    contact (feet within contact_m). Inside a contact the frame-to-frame
    tracker has no good way to tell two teammates apart and sometimes comes
    out on the wrong one; cutting hands that decision to the global linker,
    which weighs velocity continuity and each piece's place in the team shape.
    The piece before the cut gets a new id and ends 'contact'; the last piece
    keeps the original id. Cuts that would leave a side shorter than
    min_side detections are skipped. Returns (p, next_id, n_cuts,
    {tracklet: (starts_at_contact, ends_at_contact)})."""
    team_of = dict(zip(tl["tracklet"], tl["cls"]))
    q = p[p["tracklet"] >= 0].assign(team=lambda d: d["tracklet"].map(team_of))
    q = q[q["team"].isin(TEAMS)]
    ep = episodes(contact_pairs(q, "tracklet", contact_m))
    if ep.empty:
        return p, next_id, 0, {}
    frames_of = {t: np.sort(g.to_numpy()) for t, g in q.groupby("tracklet")["frame"]}
    cuts = {}
    for r in ep.itertuples():
        ok = True
        for t in (r.a, r.b):
            fr = frames_of[t]
            k = np.searchsorted(fr, r.frame, side="right")
            if k < min_side or len(fr) - k < min_side:
                ok = False
        if ok:
            for t in (r.a, r.b):
                cuts.setdefault(int(t), set()).add(int(r.frame))
    new_ids = p["tracklet"].to_numpy().copy()
    frames = p["frame"].to_numpy()
    flags, n_cuts = {}, 0
    for t, cs in cuts.items():
        rows = np.flatnonzero(new_ids == t)
        bounds = sorted(cs)
        # pieces: (-inf, c0], (c0, c1], ..., (c_last, inf) -> last keeps id t
        lo = -np.inf
        for k, c in enumerate(bounds):
            sel = rows[(frames[rows] > lo) & (frames[rows] <= c)]
            new_ids[sel] = next_id
            flags[next_id] = (k > 0, True)
            next_id += 1
            n_cuts += 1
            lo = c
        flags[t] = (True, flags.get(t, (False, False))[1])
    p = p.assign(tracklet=new_ids)
    return p, next_id, n_cuts, flags


def split_at_handovers(p, tl, next_id, fps, handover_m=1.5, dup_m=0.8, coast_max=12, min_head=3):
    """Issue #46: cut tracklet i where another same-team tracklet j starts on
    i's player - i is coasting there (its last box, <= coast_max frames
    ago, is within handover_m + 5 m/s * gap of j's first box) or i's box on
    that frame is a duplicate of j's (< dup_m). The tracker often starts the
    new tracklet while the old one coasts, and the old one then grabs a
    neighbour's box, so the two overlap and the linker (which allows only a
    few frames of overlap) can't chain them: the player's label waits and
    then jumps to someone far away. After the cut the flow can chain i's
    head -> j, and i's tail links on its own merits (it can still re-join i's
    head if that's cheaper). Returns (p, next_id, n_cuts, {tail tracklet: the
    tracklet it was cut from})."""
    team_of = dict(zip(tl["tracklet"], tl["cls"]))
    q = p[p["tracklet"] >= 0]
    q = q[q["tracklet"].map(team_of).isin(TEAMS)]
    fr_of, xy_of = {}, {}
    for t, g in q.groupby("tracklet"):
        g = g.sort_values("frame")
        fr_of[t], xy_of[t] = g["frame"].to_numpy(), g[["pitch_x", "pitch_y"]].to_numpy()
    T = tl[tl["tracklet"].isin(fr_of.keys())].sort_values("start")
    starts, ends, ids = T["start"].to_numpy(), T["end"].to_numpy(), T["tracklet"].to_numpy()
    cls = T["cls"].to_numpy()
    cuts = {}
    for k in range(len(T)):
        j, s_j = ids[k], starts[k]
        e_j = xy_of[j][0]
        alive = np.flatnonzero((starts < s_j) & (ends >= s_j) & (cls == cls[k]))
        best, bd = None, np.inf
        for m in alive:
            i = ids[m]
            fr = fr_of[i]
            n_head = np.searchsorted(fr, s_j, side="left")  # boxes before s_j
            if n_head < min_head:
                continue
            fa = fr[n_head - 1]
            gap = s_j - fa
            if gap > coast_max:
                continue
            d = float(np.hypot(*(xy_of[i][n_head - 1] - e_j)))
            if d > handover_m + 5.0 * gap / fps:
                continue
            if n_head < len(fr) and fr[n_head] == s_j:  # i has a box on s_j too: only a duplicate counts
                if float(np.hypot(*(xy_of[i][n_head] - e_j))) >= dup_m:
                    continue
            if d < bd:
                best, bd = i, d
        if best is not None:
            cuts.setdefault(int(best), set()).add(int(s_j))
    new_ids = p["tracklet"].to_numpy().copy()
    frames = p["frame"].to_numpy()
    tails, n_cuts = {}, 0
    for t, cs in cuts.items():
        rows = np.flatnonzero(new_ids == t)
        for c in sorted(cs, reverse=True):  # last cut first: each tail is [c, next cut)
            sel = rows[(frames[rows] >= c) & (new_ids[rows] == t)]
            if not len(sel):
                continue
            new_ids[sel] = next_id
            tails[next_id] = t
            next_id += 1
            n_cuts += 1
    return p.assign(tracklet=new_ids), next_id, n_cuts, tails


def role_vectors(p, tl, team, min_visible=4, frame="centroid"):
    """Each tracklet's place in the team shape, which survives a long
    off-screen gap far better than its absolute position does.
    frame='centroid': offset from the centroid of the team's visible outfield
    players (both axes). frame='lane': depth offset as above, but the lateral
    coordinate is the absolute pitch-width position (the player's channel) -
    a ball-following camera mostly shows the full width, so channels stay
    comparable across views while a visible-centroid y shifts with framing."""
    ids = tl.loc[tl["cls"] == team, "tracklet"]
    rows = p[p["tracklet"].isin(ids)]
    g = rows.groupby("frame")
    cx, cy, cnt = g["pitch_x"].transform("mean"), g["pitch_y"].transform("mean"), g["pitch_x"].transform("size")
    ry = rows["pitch_y"] if frame == "lane" else rows["pitch_y"] - cy
    rel = rows.assign(rx=rows["pitch_x"] - cx, ry=ry)[cnt >= min_visible]
    return rel.groupby("tracklet")[["rx", "ry"]].median()


# --------------------------------------------------------------------------
# 4. Global identity linking (min-cost flow)
# --------------------------------------------------------------------------
def link_edges(T, fps, a, window, vmax=None, w_role=None, c_link=None):
    """The linking graph shared by link_flow and link_anchored. T must be
    sorted by start. Returns (I, J, cost, gap_s, dist_m) over candidate links
    i -> j plus per-node reward, start cost and end cost."""
    vmax = a.vmax if vmax is None else vmax
    w_role = a.w_role if w_role is None else w_role
    c_link = a.c_link if c_link is None else c_link
    n = len(T)
    st, en = T["start"].to_numpy(), T["end"].to_numpy()
    ent, ext = T[["entry_x", "entry_y"]].to_numpy(), T[["exit_x", "exit_y"]].to_numpy()
    vout = T[["vout_x", "vout_y"]].to_numpy()
    vin = T[["vin_x", "vin_y"]].to_numpy()
    sc = T["start_contact"].to_numpy(bool) if "start_contact" in T else np.zeros(n, bool)
    ec = T["end_contact"].to_numpy(bool) if "end_contact" in T else np.zeros(n, bool)
    role = T[["role_x", "role_y"]].to_numpy()
    has_role = np.isfinite(role).all(axis=1)
    I, J, C, G, D = [], [], [], [], []
    for i in range(n):
        lo = np.searchsorted(st, en[i] - a.overlap_frames, side="left")
        hi = np.searchsorted(st, en[i] + a.max_gap_s * fps, side="right")
        j = np.arange(lo, hi)
        j = j[st[j] > st[i]]
        if not len(j):
            continue
        gap = np.maximum((st[j] - en[i]) / fps, 0.0)
        disp = np.hypot(ent[j, 0] - ext[i, 0], ent[j, 1] - ext[i, 1])
        feas = disp <= vmax * gap + a.reach_m
        j, gap, disp = j[feas], gap[feas], disp[feas]
        if not len(j):
            continue
        pred = ext[i] + vout[i] * np.minimum(gap, 0.5)[:, None]
        d = np.hypot(ent[j, 0] - pred[:, 0], ent[j, 1] - pred[:, 1])
        sig = 0.8 + 1.5 * np.minimum(gap, 10.0)
        lam = 1 - np.exp(-gap / a.role_tau_s)
        if ec[i]:  # issue #34: at a contact cut, place in the shape matters even with no gap
            lam = np.where(sc[j], np.maximum(lam, a.contact_role), lam)
        rr = np.where(has_role[i] & has_role[j],
                      np.hypot(role[j, 0] - role[i, 0], role[j, 1] - role[i, 1]) / a.role_sigma_m, 1.0)
        cost = d / sig + w_role * lam * rr + c_link
        if a.w_vel:  # velocity continuity across short gaps (a runner keeps running)
            dv = np.hypot(vin[j, 0] - vout[i, 0], vin[j, 1] - vout[i, 1])
            cost = cost + np.where(gap < 1.0, a.w_vel * dv / 3.0, 0.0)
        short = gap < 2.0
        keep = short.copy()
        long_idx = np.flatnonzero(~short)
        if len(long_idx):
            keep[long_idx[np.argsort(cost[long_idx])[:a.max_succ]]] = True
        I.append(np.full(int(keep.sum()), i))
        J.append(j[keep])
        C.append(cost[keep])
        G.append(gap[keep])
        D.append(disp[keep])
    cat = (lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt))
    # Reward = seconds of detected play covered (+ a small constant). An
    # earlier count-based reward (3 + 0.1/s, capped) made the flow prefer
    # chaining several short, sometimes wrong pieces over one long pure
    # tracklet (a 2.8-minute Spurs tracklet was dropped that way).
    reward = a.reward + a.reward_per_s * T["n"].to_numpy() / fps
    c_in = np.where(st - window[0] <= a.edge_s * fps, 0.0, a.cost_late)
    c_out = np.where(window[1] - en <= a.edge_s * fps, 0.0, a.cost_late)
    return (cat(I, int), cat(J, int), cat(C, float), cat(G, float), cat(D, float), reward, c_in, c_out)


def link_flow(T, fps, a, window, n_paths=10, vmax=None, w_role=None, c_link=None):
    """Exactly n_paths disjoint tracklet chains of minimum total cost.
    Returns ({tracklet: path index}, [(from, to, cost, gap_s, dist_m)])."""
    import networkx as nx

    T = T.sort_values("start").reset_index(drop=True)
    I, J, C, Gs, Ds, reward, c_in, c_out = link_edges(T, fps, a, window, vmax, w_role, c_link)
    scale = 100
    G = nx.DiGraph()
    G.add_node("S", demand=-n_paths)
    G.add_node("T", demand=n_paths)
    for i in range(len(T)):
        G.add_edge(("i", i), ("o", i), capacity=1, weight=-int(round(scale * reward[i])))
        G.add_edge("S", ("i", i), capacity=1, weight=int(scale * c_in[i]))
        G.add_edge(("o", i), "T", capacity=1, weight=int(scale * c_out[i]))
    meta = {}
    for i, j, cc, gg, dd in zip(I, J, C, Gs, Ds):
        G.add_edge(("o", int(i)), ("i", int(j)), capacity=1, weight=int(round(scale * cc)))
        meta[(int(i), int(j))] = (float(cc), float(gg), float(dd))
    flow = nx.min_cost_flow(G)
    assign, links = {}, []
    k = 0
    for node, f in flow["S"].items():
        if f <= 0:
            continue
        cur = node
        while cur != "T":
            i = cur[1]
            assign[int(T["tracklet"].iat[i])] = k
            nxt = next(v for v, fl in flow[("o", i)].items() if fl > 0)
            if nxt != "T":
                c, g_, d_ = meta[(i, nxt[1])]
                links.append((int(T["tracklet"].iat[i]), int(T["tracklet"].iat[nxt[1]]), c, g_, d_))
            cur = nxt
        k += 1
    return assign, links


def shape_costs(T, pins, K, fps, w_shape=1.0, sigma_m=6.0, cap_s=20.0):
    """(n, K) cost of giving each tracklet to each player, from the team shape
    (issue #46). A player's place in the shape (role_x/role_y: offset from the
    visible teammates' centroid) is read off the tracklets the analyst named
    and interpolated in time between them; a tracklet far from a player's
    place costs more the longer it is. Shifted so each tracklet's best player
    costs 0 (the cover reward is untouched). Zero for players with no pins."""
    n = len(T)
    role = T[["role_x", "role_y"]].to_numpy(float)
    mid = ((T["start"] + T["end"]) / 2).to_numpy(float)
    dur = np.minimum(T["n"].to_numpy(float) / fps, cap_s)
    nc = np.zeros((n, K))
    pts = {}
    for i, k in pins.items():
        if np.isfinite(role[i]).all():
            pts.setdefault(k, []).append((mid[i], role[i]))
    for k, P in pts.items():
        P.sort(key=lambda z: z[0])
        tt = np.array([z[0] for z in P])
        rr = np.array([z[1] for z in P])
        dd = np.hypot(role[:, 0] - np.interp(mid, tt, rr[:, 0]), role[:, 1] - np.interp(mid, tt, rr[:, 1])) / sigma_m
        nc[:, k] = np.where(np.isfinite(dd), w_shape * dur * np.minimum(dd, 3.0) ** 2, 0.0)
    return nc - nc.min(axis=1, keepdims=True)


def link_anchored(T, fps, a, window, pins, K, time_limit=900.0):
    """K disjoint chains, one per PLAYER, through the tracklets the analyst
    named (issue #46). pins = {tracklet: player index}. A multi-commodity
    flow solved as an integer programme: each player's chain must pass
    through every tracklet named as him, and pays the shape cost
    (shape_costs) for every tracklet it takes, on top of link_flow's link
    costs and rewards. Names therefore steer the linking itself, instead of
    being voted onto chains afterwards - a chain the plain flow swaps midway
    between two named clips comes out the right way round.
    Returns ({tracklet: player index}, [(from, to, cost, gap_s, dist_m)]),
    or None when the solver finds no solution."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    T = T.sort_values("start").reset_index(drop=True)
    node_of = {int(t): i for i, t in enumerate(T["tracklet"])}
    pin_ix = {node_of[int(t)]: int(k) for t, k in pins.items() if int(t) in node_of}
    I, J, C, Gs, Ds, reward, c_in, c_out = link_edges(T, fps, a, window)
    # make sure consecutive tracklets named as one player can always be joined
    st, en = T["start"].to_numpy(), T["end"].to_numpy()
    by_k = {}
    for i, k in pin_ix.items():
        by_k.setdefault(k, []).append(i)
    have = set(zip(I.tolist(), J.tolist()))
    extra = []
    for k, ii in by_k.items():
        ii.sort(key=lambda i: st[i])
        for u, v in zip(ii, ii[1:]):
            if st[v] > en[u] - a.overlap_frames and (u, v) not in have:
                extra.append((u, v))
    if extra:
        I = np.r_[I, [u for u, v in extra]]
        J = np.r_[J, [v for u, v in extra]]
        C = np.r_[C, np.full(len(extra), 2 * a.cost_late)]
        Gs = np.r_[Gs, [(st[v] - en[u]) / fps for u, v in extra]]
        Ds = np.r_[Ds, np.full(len(extra), np.nan)]
    n, E = len(T), len(I)
    nc = shape_costs(T, pin_ix, K, fps, a.w_shape, a.shape_sigma_m) if a.w_shape > 0 else np.zeros((n, K))
    nv = E + 2 * n  # per player: links, starts, ends
    c = np.zeros(K * nv)
    for k in range(K):
        c[k * nv:k * nv + E] = C - reward[J] + nc[J, k]
        c[k * nv + E:k * nv + E + n] = c_in - reward + nc[:, k]
        c[k * nv + E + n:(k + 1) * nv] = c_out
    ub = np.ones(K * nv)
    for i, k in pin_ix.items():  # a named tracklet is closed to everyone else
        for kk in range(K):
            if kk != k:
                ub[kk * nv + E + i] = 0
                ub[kk * nv + np.flatnonzero(J == i)] = 0
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    e = np.arange(E)
    nn = np.arange(n)
    for k in range(K):
        b = k * nv
        rows += [r + J, r + I, r + nn, r + nn]
        cols += [b + e, b + e, b + E + nn, b + E + n + nn]
        vals += [np.ones(E), -np.ones(E), np.ones(n), -np.ones(n)]
        lo.append(np.zeros(n))
        hi.append(np.zeros(n))
        r += n
        rows.append(np.full(n, r))
        cols.append(b + E + nn)
        vals.append(np.ones(n))
        lo.append([1.0])
        hi.append([1.0])
        r += 1
    for k in range(K):  # each tracklet is one player at most
        b = k * nv
        rows += [r + J, r + nn]
        cols += [b + e, b + E + nn]
        vals += [np.ones(E), np.ones(n)]
    lo.append(np.zeros(n))
    hi.append(np.ones(n))
    r += n
    for i, k in pin_ix.items():
        m = np.flatnonzero(J == i)
        rows.append(np.full(len(m) + 1, r))
        cols.append(np.r_[k * nv + m, k * nv + E + i])
        vals.append(np.ones(len(m) + 1))
        lo.append([1.0])
        hi.append([1.0])
        r += 1
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, K * nv)).tocsr()
    res = milp(c, constraints=LinearConstraint(A, np.concatenate(lo), np.concatenate(hi)),
               integrality=np.ones(K * nv), bounds=Bounds(0, ub),
               options={"time_limit": time_limit, "disp": False, "mip_rel_gap": 1e-3})
    if res.x is None:
        return None
    x = res.x.reshape(K, nv)
    assign, links = {}, []
    tid = T["tracklet"].to_numpy()
    for k in range(K):
        for i in np.flatnonzero(x[k, E:E + n] > 0.5):
            assign[int(tid[i])] = k
        for q in np.flatnonzero(x[k, :E] > 0.5):
            assign[int(tid[I[q]])] = k
            assign[int(tid[J[q]])] = k
            links.append((int(tid[I[q]]), int(tid[J[q]]), float(C[q]), float(Gs[q]), float(Ds[q])))
    return assign, links


def link_goalkeepers(tl, gk_cls, goal_x, W=68.0, depth_m=22.0):
    """All of a keeper-kit's tracklets in that team's goal zone -> the GK.
    The zone (<= depth_m from the goal line, inside the penalty-area width
    + 8 m) matters for an 'unreadable' kit: the referee's dark-green kit also
    reads as grass, and with a looser 32 m zone he became Ederson whenever he
    drifted towards City's box."""
    cand = tl[(tl["cls"].isin([gk_cls, gk_cls + "?"])) & (np.abs(tl["med_x"] - goal_x) < depth_m)
              & tl["med_y"].between(W / 2 - 28, W / 2 + 28)]
    cand = cand.sort_values("n", ascending=False)
    chosen, spans = [], []
    for r in cand.itertuples():
        if any(not (r.end < s or r.start > e) for s, e in spans):
            continue  # overlaps a longer GK tracklet: someone else in similar colours
        chosen.append(int(r.tracklet))
        spans.append((r.start, r.end))
    return chosen


def fill_gaps(tl, fps, max_gap_s=3.0, margin=0.5, rounds=6):
    """Give short / unreadable tracklets to the identity whose timeline has a
    gap they fit into, continuous at BOTH ends: "we know who it is before and
    after, so it's him in between". The neighbours must be within max_gap_s
    on each side - a stitch for blips inside a player's run, not a
    re-identification. Repeats so a gap holding several short pieces fills
    piece by piece. When two players fit, the closer fit wins if it is clearly
    closer (by margin of the summed normalised distances), else neither."""
    total = 0
    for _ in range(rounds):
        seqs = {}
        for k, g in tl[tl["identity"] >= 0].sort_values("start").groupby("identity"):
            seqs[k] = (g["start"].to_numpy(), g["end"].to_numpy(),
                       g[["entry_x", "entry_y"]].to_numpy(), g[["exit_x", "exit_y"]].to_numpy(),
                       g["team"].iloc[0])
        pending = tl[(tl["identity"] < 0) & tl["cls"].isin(["team_a", "team_b", "unknown", "gk_a?", "mixed"])]
        updates = {}
        for r in pending.itertuples():
            fits = []
            for k, (st, en, ent, ext, team) in seqs.items():
                if r.cls in TEAMS and team != r.cls:
                    continue
                i = np.searchsorted(st, r.start)
                if (i > 0 and en[i - 1] >= r.start) or (i < len(st) and st[i] <= r.end):
                    continue  # overlaps one of k's tracklets
                if i == 0 or i == len(st):
                    continue  # must sit between two of k's tracklets
                g1, g2 = (r.start - en[i - 1]) / fps, (st[i] - r.end) / fps
                if g1 > max_gap_s or g2 > max_gap_s:
                    continue
                d1 = np.hypot(r.entry_x - ext[i - 1, 0], r.entry_y - ext[i - 1, 1])
                d2 = np.hypot(ent[i, 0] - r.exit_x, ent[i, 1] - r.exit_y)
                t1, t2 = 1.5 + 4.0 * g1, 1.5 + 4.0 * g2
                if d1 <= t1 and d2 <= t2:
                    fits.append((d1 / t1 + d2 / t2, k, team))
            fits.sort(key=lambda z: z[0])
            if len(fits) == 1 or (len(fits) > 1 and fits[1][0] - fits[0][0] >= margin):
                updates[r.tracklet] = fits[0][1:]
        # one piece per identity gap per round (two pieces can't both fill the same blip)
        seen, upd = set(), {}
        for t, (k, team) in sorted(updates.items(), key=lambda kv: int(tl.loc[tl["tracklet"] == kv[0], "start"].iat[0])):
            r0 = tl.loc[tl["tracklet"] == t].iloc[0]
            key = (k, int(np.searchsorted(seqs[k][0], r0["start"])))
            if key in seen:
                continue
            seen.add(key)
            upd[t] = (k, team)
        if not upd:
            break
        m = tl["tracklet"].isin(list(upd))
        tl.loc[m, "identity"] = tl.loc[m, "tracklet"].map(lambda t: upd[t][0])
        tl.loc[m, "team"] = tl.loc[m, "tracklet"].map(lambda t: upd[t][1])
        tl.loc[m, "link"] = "gap_fill"
        total += len(upd)
    return total


def assign_by_elimination(tl, roles, fps, a, ids, window_s=120.0, max_shape=1.5, margin=1.0):
    """A team tracklet left out of every chain is one of the players who
    are NOT on screen at that moment (issue #46): of those, the ones who can
    reach it in time from where they were last seen and get to where they
    are seen next, and whose place in the team shape (median role of their
    own tracklets within +-window_s) matches it. Taken when one candidate
    is left, or the best is clearly better (by margin, in shape sigmas).
    Longest tracklets first, so each assignment narrows the next.
    ids: identity numbers of the team's outfield chains. Returns the count."""
    sig = a.shape_sigma_m
    rx, ry = roles["rx"].to_dict(), roles["ry"].to_dict()
    chains = {}
    for k in ids:
        g = tl[tl["identity"] == k].sort_values("start")
        chains[k] = [list(g["start"]), list(g["end"]), list(zip(g["entry_x"], g["entry_y"])),
                     list(zip(g["exit_x"], g["exit_y"])),
                     [((s_ + e_) / 2, rx.get(t, np.nan), ry.get(t, np.nan))
                      for t, s_, e_ in zip(g["tracklet"], g["start"], g["end"])]]
    team = tl.loc[tl["identity"].isin(ids), "team"].iloc[0] if tl["identity"].isin(ids).any() else None
    if team is None:
        return 0
    pend = tl[(tl["identity"] < 0) & (tl["cls"] == team)].sort_values("n", ascending=False)
    done = 0
    for r in pend.itertuples():
        ru = np.array([rx.get(r.tracklet, np.nan), ry.get(r.tracklet, np.nan)])
        mid = (r.start + r.end) / 2
        cands = []
        for k, (st, en, ent, ext, rl) in chains.items():
            i = int(np.searchsorted(st, r.start))
            if (i > 0 and en[i - 1] > r.start - a.overlap_frames) or (i < len(st) and st[i] < r.end + a.overlap_frames):
                continue  # he's on screen then
            ok = True
            if i > 0:
                gap = (r.start - en[i - 1]) / fps
                ok &= np.hypot(r.entry_x - ext[i - 1][0], r.entry_y - ext[i - 1][1]) <= a.vmax * gap + a.reach_m
            if i < len(st):
                gap = (st[i] - r.end) / fps
                ok &= np.hypot(ent[i][0] - r.exit_x, ent[i][1] - r.exit_y) <= a.vmax * gap + a.reach_m
            if not ok:
                continue
            near = np.array([(x, y) for m_, x, y in rl if abs(m_ - mid) <= window_s * fps and np.isfinite(x)])
            if np.isfinite(ru).all() and len(near):
                d = float(np.hypot(*(ru - np.median(near, axis=0)))) / sig
            else:
                d = np.nan
            cands.append((d, k, i))
        if not cands:
            continue
        shaped = sorted((c for c in cands if np.isfinite(c[0])), key=lambda c: c[0])
        if len(cands) == 1 and (not shaped or shaped[0][0] <= max_shape):
            pick = cands[0]
        elif shaped and shaped[0][0] <= max_shape and len(shaped) == len(cands) and (
                len(shaped) == 1 or shaped[1][0] - shaped[0][0] >= margin):
            pick = shaped[0]
        else:
            continue
        _, k, i = pick
        st, en, ent, ext, rl = chains[k]
        st.insert(i, r.start)
        en.insert(i, r.end)
        ent.insert(i, (r.entry_x, r.entry_y))
        ext.insert(i, (r.exit_x, r.exit_y))
        rl.insert(i, (mid, ru[0], ru[1]))
        tl.loc[tl["tracklet"] == r.tracklet, ["identity", "team", "link"]] = [k, team, "elimination"]
        done += 1
    return done


def fill_partial(p, tl, fps, L, W, max_gap_s=10.0, v_m_s=7.0, slack_m=3.0, min_rows=5, rounds=3):
    """"We know it's him before and after, so it's him in between" for a box
    whose tracklet doesn't fit the gap whole (issue #46): a duplicate or
    neighbour tracklet that started while his own was still running, or
    runs on past his next one, leaves him unlabelled for seconds while an
    unlabelled box sits on his path. The part of that tracklet INSIDE the gap
    becomes a new tracklet of his, when its first box is within running
    reach of where he was last seen and its last box within reach of where
    he is seen next. Lowest-cost pieces first; a gap takes one piece per
    round. Returns (p, tl, pieces)."""
    pend_cls = {"team_a": ("team_a", "unknown", "mixed"), "team_b": ("team_b", "unknown", "mixed")}
    total = 0
    for _ in range(rounds):
        free = tl[tl["identity"] < 0]
        if free.empty:
            break
        fst, fen = free["start"].to_numpy(), free["end"].to_numpy()
        fid, fcls = free["tracklet"].to_numpy(), free["cls"].to_numpy()
        rows_of = {t: g for t, g in p[p["tracklet"].isin(set(fid))].groupby("tracklet")}
        offers = []
        for k, g in tl[tl["identity"] >= 0].sort_values("start").groupby("identity"):
            team = g["team"].iloc[0]
            ok_cls = pend_cls.get(team, ())
            st, en = g["start"].to_numpy(), g["end"].to_numpy()
            ext, ent = g[["exit_x", "exit_y"]].to_numpy(), g[["entry_x", "entry_y"]].to_numpy()
            reach_end = np.maximum.accumulate(en)
            for i in range(len(g) - 1):
                a0, b0 = int(reach_end[i]), int(st[i + 1])
                if b0 - a0 < 2 or (b0 - a0) > max_gap_s * fps:
                    continue
                cand = np.flatnonzero((fst < b0) & (fen > a0))
                for c in cand:
                    if fcls[c] not in ok_cls:
                        continue
                    r = rows_of.get(fid[c])
                    if r is None:
                        continue
                    r = r[(r["frame"] > a0) & (r["frame"] < b0)]
                    if len(r) < min_rows:
                        continue
                    r = r.sort_values("frame")
                    f1, f2 = int(r["frame"].iat[0]), int(r["frame"].iat[-1])
                    d1 = np.hypot(r["pitch_x"].iat[0] - ext[i, 0], r["pitch_y"].iat[0] - ext[i, 1])
                    d2 = np.hypot(r["pitch_x"].iat[-1] - ent[i + 1, 0], r["pitch_y"].iat[-1] - ent[i + 1, 1])
                    t1, t2 = slack_m + v_m_s * (f1 - a0) / fps, slack_m + v_m_s * (b0 - f2) / fps
                    if d1 <= t1 and d2 <= t2:
                        offers.append((d1 / t1 + d2 / t2, int(k), team, (int(k), i), int(fid[c]), f1, f2))
        if not offers:
            break
        offers.sort(key=lambda o: o[0])
        used_gap, used = set(), {}
        take = []
        for cost, k, team, gk, u, f1, f2 in offers:
            if gk in used_gap or any(not (f2 < a_ or f1 > b_) for a_, b_ in used.get(u, [])):
                continue
            used_gap.add(gk)
            used.setdefault(u, []).append((f1, f2))
            take.append((k, team, u, f1, f2))
        if not take:
            break
        nxt = int(max(p["tracklet"].max(), tl["tracklet"].max())) + 1
        trk = p["tracklet"].to_numpy().copy()
        fr = p["frame"].to_numpy()
        new_ident, touched = {}, set()
        for k, team, u, f1, f2 in take:
            m = (trk == u) & (fr >= f1) & (fr <= f2)
            trk[m] = nxt
            new_ident[nxt] = (k, team)
            touched.update((u, nxt))
            nxt += 1
        # the rest of a cut tracklet: split into its before / after parts
        for u in {t[2] for t in take}:
            rows = np.flatnonzero(trk == u)
            if not len(rows):
                continue
            fu = fr[rows]
            cuts = sorted(f1 for k, team, uu, f1, f2 in take if uu == u)
            seg = np.searchsorted(cuts, fu, side="right")
            for sgi in np.unique(seg)[1:]:
                trk[rows[seg == sgi]] = nxt
                touched.add(nxt)
                nxt += 1
        p = p.assign(tracklet=trk)
        keep = tl[~tl["tracklet"].isin(touched)]
        old = tl.set_index("tracklet")
        fresh = summarize(p[p["tracklet"].isin(touched)], fps, L, W)
        for col in tl.columns:
            if col not in fresh.columns:
                fresh[col] = np.nan if tl[col].dtype.kind == "f" else ("" if tl[col].dtype == object else -1)
        fresh["identity"] = [new_ident.get(t, (-1, ""))[0] for t in fresh["tracklet"]]
        fresh["team"] = [new_ident.get(t, (-1, ""))[1] for t in fresh["tracklet"]]
        fresh["link"] = ["gap_piece" if t in new_ident else "" for t in fresh["tracklet"]]
        if "ended" in tl.columns:
            fresh["ended"] = [old["ended"].get(t, "gap_piece") for t in fresh["tracklet"]]
        tl = pd.concat([keep, fresh[tl.columns]], ignore_index=True)
        total += len(take)
    return p, tl, total


# --------------------------------------------------------------------------
# 6. Role-inferred names
# --------------------------------------------------------------------------
def infer_names(det, ref):
    """Match each team's outfield identities to the official formation lines
    by average depth (towards the opponent's goal) and width (team's right
    = first in each line). Returns {track_id: info}."""
    out = {}
    for team, base in (("team_a", 0), ("team_b", 100)):
        lines = ref["lineups"][team]["lines"]
        gk = lines[0][0]
        out[base + 1] = {"name": gk["name"], "number": gk["number"], "role": "GK", "line": 0, "slot": 0}
        ids = list(range(base + 2, base + 12))
        rows = det[det["track_id"].isin(ids)]
        if rows.empty:
            continue
        gk_rows = det[det["track_id"] == base + 1]
        direction = 1.0 if (gk_rows["pitch_x"].median() < 52.5 if len(gk_rows) else True) else -1.0
        # Facing +x the team's right hand points to +y (near touchline) on the
        # TV-view map, so 'right' = +dy for a team attacking +x, -dy otherwise.
        g = rows.groupby("frame")
        cx, cy, cnt = g["pitch_x"].transform("mean"), g["pitch_y"].transform("mean"), g["pitch_x"].transform("size")
        rel = rows.assign(dx=(rows["pitch_x"] - cx) * direction, dy=(rows["pitch_y"] - cy) * direction)[cnt >= 5]
        stats = rel.groupby("track_id")[["dx", "dy"]].median()
        order_depth = stats.sort_values("dx").index.tolist()  # deepest first
        k0 = 0
        for li, line in enumerate(lines[1:], start=1):
            members = order_depth[k0:k0 + len(line)]
            k0 += len(line)
            right_to_left = stats.loc[members].sort_values("dy", ascending=False).index.tolist()
            for slot, (tid, person) in enumerate(zip(right_to_left, line)):
                out[int(tid)] = {"name": person["name"], "number": person["number"],
                                 "role": f"line{li}-slot{slot + 1}", "line": li, "slot": slot + 1,
                                 "depth_m": round(float(stats.loc[tid, "dx"]), 1),
                                 "width_m": round(float(stats.loc[tid, "dy"]), 1)}
    return out


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--min-conf", type=float, default=0.15, help="Ignore weaker person boxes")
    parser.add_argument("--new-conf", type=float, default=0.35, help="Min confidence to start a tracklet")
    parser.add_argument("--gate", type=float, default=1.2, help="Association gate, metres")
    parser.add_argument("--gate-step", type=float, default=0.3, help="Gate growth per missed frame, metres")
    parser.add_argument("--gate-max", type=float, default=3.0)
    parser.add_argument("--kit-penalty", type=float, default=1.0,
                        help="Metres added for a kit mismatch (pass 1, rule-based kits)")
    parser.add_argument("--kit-penalty-2", type=float, default=2.5,
                        help="Kit mismatch cost in pass 2, once kits come from the learned colour model")
    parser.add_argument("--max-miss", type=int, default=12, help="Frames a tracklet may coast")
    parser.add_argument("--vmax", type=float, default=8.5, help="Max average speed across a gap, m/s")
    parser.add_argument("--reach-m", type=float, default=3.0, help="Slack on the reachability test, metres")
    parser.add_argument("--w-role", type=float, default=1.5, help="Weight of the team-shape term")
    parser.add_argument("--role-tau-s", type=float, default=5.0,
                        help="Gap length over which the role term takes over from position")
    parser.add_argument("--role-sigma-m", type=float, default=8.0)
    parser.add_argument("--role-frame", default="centroid", choices=["centroid", "lane"],
                        help="Team-shape descriptor: offsets from the visible centroid, or depth offset + "
                             "absolute lateral lane")
    parser.add_argument("--c-link", type=float, default=0.5, help="Fixed cost per link (prefers fewer, longer chains)")
    parser.add_argument("--reward", type=float, default=0.5, help="Constant reward for covering a tracklet")
    parser.add_argument("--reward-per-s", type=float, default=1.0,
                        help="Reward per second of detected play a chain covers (dominates link costs)")
    parser.add_argument("--edge-s", type=float, default=20.0,
                        help="Chains may start/end freely this close to the window edges")
    parser.add_argument("--cost-late", type=float, default=8.0, help="Cost of a chain starting/ending mid-window")
    parser.add_argument("--max-gap-s", type=float, default=300.0,
                        help="Longest off-screen gap a link may bridge (a centre-back can stay off a "
                             "ball-following camera for minutes during a long attacking spell)")
    parser.add_argument("--max-succ", type=int, default=12, help="Long-gap successor candidates kept per tracklet")
    parser.add_argument("--overlap-frames", type=int, default=3)
    parser.add_argument("--dup-m", type=float, default=0.8,
                        help="An unmatched box this close (m) to one already tracked on the frame is a duplicate: "
                             "it doesn't start a tracklet (issue #46). 0 = off")
    parser.add_argument("--handover-m", type=float, default=1.5,
                        help="Cut a coasting tracklet where another starts on its player within this distance "
                             "(issue #46). 0 = off")
    parser.add_argument("--min-len", type=int, default=8, help="Shorter tracklets skip the flow (gap-filled later)")
    parser.add_argument("--contact-m", type=float, default=0.0,
                        help="Cut same-team tracklets where two come within this many metres, so the "
                             "linker (not the frame-to-frame tracker) decides who is who after a "
                             "collision (issue #34). 0 = off")
    parser.add_argument("--contact-min-side", type=int, default=15,
                        help="Skip a contact cut that would leave a piece shorter than this (detections)")
    parser.add_argument("--contact-role", type=float, default=0.6,
                        help="Weight of the team-shape term on links across a contact cut (0-1)")
    parser.add_argument("--w-vel", type=float, default=0.0,
                        help="Cost per 3 m/s of velocity change across links with gaps < 1 s")
    parser.add_argument("--no-stability", action="store_true", help="Skip the perturbed re-solves")
    parser.add_argument("--no-anchor-solve", dest="anchor_solve", action="store_false",
                        help="Apply names after linking (majority vote per chain) instead of linking through them")
    parser.add_argument("--w-shape", type=float, default=1.0,
                        help="Anchored linking: weight of a tracklet's distance from a named player's place in "
                             "the team shape (per second, capped at 20 s)")
    parser.add_argument("--shape-sigma-m", type=float, default=6.0)
    parser.add_argument("--anchor-time-limit", type=float, default=900.0, help="Seconds per team for the solver")
    parser.add_argument("--reuse-tracklets", action="store_true",
                        help="Reuse tracklets_cache.parquet from the last run: re-run only linking + outputs "
                             "(for tuning the linker in minutes instead of ~15)")
    parser.add_argument("--calibration", default=None, help="Default: config/<match_id>_calibration.json")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--tmp-dir", default=".tmp")
    a = parser.parse_args()

    meta = load_meta(a.match_id, a.tmp_dir)
    fps = meta["fps"]
    cfg = match_config(a.match_id, a.tmp_dir, a.config_dir)
    ref = read_json(reference_path(a.match_id, a.config_dir))
    kits = cfg.get("kits")
    if not kits:
        raise SystemExit("match config has no 'kits' map (colour class -> team_a/team_b/gk_a/gk_b/official)")
    mdir = match_dir(a.match_id, a.tmp_dir)
    calib = load_calibration(a.match_id, a.config_dir, a.calibration)
    if calib is None or "frame_H" not in calib:
        raise SystemExit("needs a dynamic calibration - run tools/calibrate_auto.py first")
    L, W = pitch_dims(calib)
    npz = np.load(calib["frame_homographies"])
    status = dict(zip(npz["frames"].tolist(), npz["status"].tolist())) if "status" in npz else {}
    w2i = {int(f): np.linalg.inv(H) for f, H in zip(npz["frames"], npz["H"])}
    f_lo, f_hi = int(npz["frames"][0]), int(npz["frames"][-1])

    det = pd.read_parquet(mdir / "detections.parquet")
    det = det[(det["frame"] >= f_lo) & (det["frame"] <= f_hi)].reset_index(drop=True)
    balls = det[det["class"] == "ball"].copy()
    cache_p, cache_tl = mdir / "tracklets_cache.parquet", mdir / "tracklets_summary.parquet"
    if a.reuse_tracklets and cache_p.exists() and cache_tl.exists():
        # Linking experiments: skip the ~10-minute tracking passes.
        p, tl = pd.read_parquet(cache_p), pd.read_parquet(cache_tl)
        print(f"[{a.match_id}] reusing cached tracklets ({len(tl)} tracklets, {len(p)} detections)")
    else:
        p = det[(det["class"] == "player") & (det["conf"] >= a.min_conf)].reset_index(drop=True)
        n_before = len(p)
        p = suppress_duplicates(p)
        print(f"[{a.match_id}] dropped {n_before - len(p)} duplicate person boxes "
              f"({(n_before - len(p)) / n_before:.1%})")
        p, _ = apply_calibration(p, calib)
        colour = kit_colour(p)
        p["kit"] = [kits.get(c, "unknown") for c in colour]
        p = p[p["pitch_x"].between(-8, L + 8) & p["pitch_y"].between(-8, W + 8)].reset_index(drop=True)
        print(f"  {len(p)} person detections near the pitch over frames {f_lo}-{f_hi}; kit mix: "
              + ", ".join(f"{k} {v / len(p):.0%}" for k, v in Counter(p['kit']).most_common()))

        # 3. tracklets: pass 1 on the rule-based kits, learn a team colour model
        # from its pure tracklets, relabel, then re-track with the cleaner kits.
        def track(kit_penalty):
            a.kit_penalty = kit_penalty
            return build_tracklets(
                p["frame"].to_numpy(), p["pitch_x"].to_numpy(), p["pitch_y"].to_numpy(), p["conf"].to_numpy(),
                p["kit"].to_numpy(), lambda f: w2i[f], (meta["width"], meta["height"]),
                lambda f: status.get(f, 0) != 3, fps, a)

        p["tracklet"], _ = track(a.kit_penalty)
        tl0 = summarize(p, fps, L, W)
        flicker0 = _flicker(tl0)
        p["kit"], prob = relabel_teams(p, tl0)
        p["p_team_a"] = prob if prob is not None else np.nan
        p["tracklet"], end_reason = track(a.kit_penalty_2)
        p, _, n_cuts = split_by_kit(p, int(p["tracklet"].max()) + 1)
        tl = summarize(p, fps, L, W)
        tl["ended"] = tl["tracklet"].map(end_reason).fillna("kit_split")
        print(f"  team colour model relabelled detections: tracklets with >20% minority-team votes "
              f"{flicker0:.0%} -> {_flicker(tl):.0%}; {n_cuts} tracklet(s) cut at a team flip")
        p.to_parquet(cache_p, index=False)
        tl.to_parquet(cache_tl, index=False)
    if a.handover_m > 0:
        ended = dict(zip(tl["tracklet"], tl["ended"]))
        p, _, n_ho, tails = split_at_handovers(p, tl, int(p["tracklet"].max()) + 1, fps, a.handover_m, a.dup_m)
        cut = set(tails.values())
        tl = summarize(p, fps, L, W)
        # a cut tracklet's head now ends at the handover; its tail keeps the old end reason
        tl["ended"] = ["handover" if t in cut else ended.get(tails.get(t, t), "handover") for t in tl["tracklet"]]
        print(f"  cut {n_ho} tracklet(s) where another tracklet took over their player (frame context, issue #46)")
    if a.contact_m > 0:
        ended = dict(zip(tl["tracklet"], tl["ended"]))
        p, _, n_cc, flags = split_at_contacts(p, tl, int(p["tracklet"].max()) + 1, a.contact_m,
                                              a.contact_min_side)
        tl = summarize(p, fps, L, W)
        tl["ended"] = [("contact" if flags.get(t, (False, False))[1] and t not in ended else ended.get(t, "contact"))
                       for t in tl["tracklet"]]
        tl["start_contact"] = [flags.get(t, (False, False))[0] for t in tl["tracklet"]]
        tl["end_contact"] = [flags.get(t, (False, False))[1] and t not in ended for t in tl["tracklet"]]
        print(f"  cut {n_cc} tracklet piece(s) at same-team contacts (< {a.contact_m} m)")
    team_tl = tl[tl["cls"].isin(TEAMS)]
    print(f"  {len(tl)} tracklets; classes: " + ", ".join(f"{k} {v}" for k, v in tl["cls"].value_counts().items()))
    print(f"  team tracklets: median {team_tl['n'].median():.0f} frames; ended by "
          + ", ".join(f"{k} {v}" for k, v in team_tl["ended"].value_counts().items()))

    # 3b. the analyst's names (calibration clips / Lines tab anchors): a tracklet
    # named as two players is cut between the names, before any linking.
    anchors = load_anchors(anchors_path(a.match_id, a.config_dir))
    resolved = resolve_anchors(anchors, p, tl["tracklet"]) if anchors else []
    if resolved:
        p, n_ac, ac_tails = split_conflicts(p, resolved, int(p["tracklet"].max()) + 1)
        if n_ac:
            keep = tl.set_index("tracklet")
            tl = summarize(p, fps, L, W)
            for col in ("ended", "start_contact", "end_contact"):
                if col in keep.columns:
                    old = keep[col].to_dict()
                    tl[col] = [old.get(ac_tails.get(t, t)) if t not in set(ac_tails.values()) or col != "ended"
                               else "anchor_split" for t in tl["tracklet"]]
            resolved = resolve_anchors(anchors, p, tl["tracklet"])
            print(f"  cut {n_ac} tracklet(s) the analyst named as two different players")

    # 4. keepers, then global outfield linking per team
    tl["identity"], tl["team"], tl["link"] = -1, "", ""
    tl["link_cost"], tl["link_gap_s"], tl["link_dist_m"], tl["stability"] = np.nan, np.nan, np.nan, np.nan
    gk_b = tl[tl["cls"] == "gk_b"]
    goal_b = L if (gk_b["med_x"] * gk_b["n"]).sum() / max(gk_b["n"].sum(), 1) > L / 2 else 0.0
    goal_a = L - goal_b
    for team, cls, goal, gid in (("team_a", "gk_a", goal_a, 1), ("team_b", "gk_b", goal_b, 101)):
        ids = link_goalkeepers(tl, cls, goal, W)
        tl.loc[tl["tracklet"].isin(ids), ["identity", "team", "link"]] = [gid, team, "goalkeeper"]
        print(f"  {team} keeper ({cls}, goal x={goal:.0f}): {len(ids)} tracklets, "
              f"{tl.loc[tl['tracklet'].isin(ids), 'n'].sum()} detections")

    span = dict(zip(tl["tracklet"].astype(int), zip(tl["start"].astype(int), tl["end"].astype(int))))
    lineup = {}
    for team in TEAMS:
        lines = ((ref.get("lineups") or {}).get(team) or {}).get("lines") or []
        lineup[team] = ([q for q in lines[0]][:1] if lines else [],
                        [q for line in lines[1:] for q in line][:10])
    # keeper anchors join the keeper's chain directly
    for team, gid in (("team_a", 1), ("team_b", 101)):
        gk_nums = [q["number"] for q in lineup[team][0]]
        gpins, _ = team_pins(resolved, team, gk_nums, span)
        for t in gpins:
            s0, e0 = span[t]
            over = (tl["identity"] == gid) & (tl["start"] <= e0) & (tl["end"] >= s0) & (tl["tracklet"] != t)
            tl.loc[over, ["identity", "link"]] = [-1, "evicted_by_anchor"]
            tl.loc[tl["tracklet"] == t, ["identity", "team", "link"]] = [gid, team, "anchor"]

    stability, anchored = {}, {}
    jobs, team_roles = {}, {}
    for team, base in (("team_a", 2), ("team_b", 102)):
        numbers = [q["number"] for q in lineup[team][1]]
        pins, dropped = team_pins(resolved, team, numbers, span) if numbers else ({}, 0)
        other = set(team_pins(resolved, "team_b" if team == "team_a" else "team_a",
                              [q["number"] for q in lineup["team_b" if team == "team_a" else "team_a"][1]],
                              span)[0])
        pins = {t: k for t, k in pins.items() if tl.loc[tl["tracklet"] == t, "identity"].iat[0] < 0}
        roles = role_vectors(p, tl, team, frame=a.role_frame)
        sel = tl[(((tl["cls"] == team) & (tl["n"] >= a.min_len)) | tl["tracklet"].isin(list(pins)))
                 & (tl["identity"] < 0) & ~tl["tracklet"].isin(list(other))].copy()
        sel["role_x"] = sel["tracklet"].map(roles["rx"])
        sel["role_y"] = sel["tracklet"].map(roles["ry"])
        jobs[team] = (base, sel, pins, dropped, numbers)
        team_roles[team] = roles
    solved = {}
    todo = {t: j for t, j in jobs.items() if j[2] and a.anchor_solve}
    if todo:
        # one integer programme per team (~1-3 min each), both at once
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=len(todo)) as ex:
            fut = {team: ex.submit(link_anchored, sel, fps, a, (f_lo, f_hi), pins, len(numbers),
                                   a.anchor_time_limit)
                   for team, (base, sel, pins, dropped, numbers) in todo.items()}
            for team, fu in fut.items():
                try:
                    solved[team] = fu.result()
                except Exception as exc:  # noqa: BLE001 - fall back to the plain flow
                    print(f"  {team}: anchored solve failed ({exc}); falling back to plain linking")
                    solved[team] = None
    for team, (base, sel, pins, dropped, numbers) in jobs.items():
        variants = []
        if solved.get(team):
            assign, links = solved[team]
            stab = {}
            anchored[team] = {"pins": len(pins), "dropped": dropped, "numbers": numbers,
                              "named": sorted({numbers[k] for k in pins.values()})}
            link_kind = "anchored"
        else:
            assign, links = link_flow(sel, fps, a, (f_lo, f_hi))
            stab = {(i, j): 0 for i, j, *_ in links}
            variants = [] if a.no_stability else [
                {"vmax": a.vmax * 0.85}, {"w_role": a.w_role * 0.5},
                {"w_role": a.w_role * 1.5, "c_link": a.c_link * 1.5}]
            for v in variants:
                alt, _ = link_flow(sel, fps, a, (f_lo, f_hi), **v)
                for i, j in stab:
                    stab[(i, j)] += int(i in alt and alt.get(i) == alt.get(j))
            link_kind = "linked"
        incoming = {j: (i, c, g_, d_) for i, j, c, g_, d_ in links}
        upd = {}
        for t_id, k in assign.items():
            i, c, g_, d_ = incoming.get(t_id, (None, np.nan, np.nan, np.nan))
            s_ = stab[(i, t_id)] / len(variants) if (i is not None and variants) else np.nan
            kind = "anchor" if t_id in pins and team in anchored else link_kind
            upd[t_id] = (base + k, team, kind, c, g_, d_, s_)
        m = tl["tracklet"].isin(list(upd))
        for col, ix in (("identity", 0), ("team", 1), ("link", 2), ("link_cost", 3), ("link_gap_s", 4),
                        ("link_dist_m", 5), ("stability", 6)):
            tl.loc[m, col] = tl.loc[m, "tracklet"].map(lambda t, ix=ix: upd[t][ix])
        n_links = len(links)
        if team in anchored:
            stability[team] = None
            r = anchored[team]
            print(f"  {team}: {len(assign)}/{len(sel)} tracklets in {len(set(assign.values()))} player chains, "
                  f"{n_links} links, solved through {r['pins']} named tracklet(s) of {len(r['named'])} players "
                  f"+ team shape (issue #46)" + (f"; {dropped} conflicting name(s) ignored" if dropped else ""))
            continue
        stable = sum(1 for v in stab.values() if variants and v == len(variants))
        stability[team] = round(stable / max(n_links, 1), 4) if variants else None
        print(f"  {team}: {len(assign)}/{len(sel)} tracklets in {len(set(assign.values()))} identity chains, "
              f"{n_links} links" + (f", {stability[team]:.1%} stable under all {len(variants)} perturbations"
                                     if variants else ""))

    filled = fill_gaps(tl, fps)
    print(f"  gap-filled {filled} short/unreadable tracklets")
    for team in anchored:
        base = jobs[team][0]
        n_el = assign_by_elimination(tl, team_roles[team], fps, a, list(range(base, base + len(jobs[team][4]))))
        print(f"  {team}: {n_el} unlinked tracklet(s) given to the one player who could be there (elimination + shape)")
    if anchored:
        filled = fill_gaps(tl, fps)
        print(f"  gap-filled {filled} more")
    p, tl, n_pc = fill_partial(p, tl, fps, L, W)
    print(f"  {n_pc} piece(s) of other tracklets given to the player whose gap they bridge")

    # 4b. names (role-inferred), then the analyst's manual anchors (issue #16)
    prov = p.assign(track_id=p["tracklet"].map(dict(zip(tl["tracklet"], tl["identity"]))).fillna(-1).astype(int))
    names = infer_names(prov[prov["track_id"] >= 0], ref)
    anchor_report = None
    for team, r in anchored.items():
        # chain k IS lineup player k for the players the analyst named; the
        # unnamed chains take the role-inferred names left over
        base = 2 if team == "team_a" else 102
        outfield = lineup[team][1]
        named = set(r["named"])
        left = [q for q in outfield if q["number"] not in named]
        guess = {tid: names.get(tid, {}) for tid in range(base, base + len(outfield))}
        for k, q in enumerate(outfield):
            tid = base + k
            info = dict(guess[tid])
            if q["number"] in named:
                n_a = sum(1 for kk in jobs[team][2].values() if kk == k)
                info.update(name=q["name"], number=q["number"], anchors=n_a,
                            name_source="calibration names (anchored linking)")
                names[tid] = info
        free = [tid for k, tid in enumerate(range(base, base + len(outfield))) if outfield[k]["number"] not in named]
        pool = list(left)
        for tid in free:
            g = guess[tid].get("number")
            q = next((x for x in pool if x["number"] == g), None) or (pool[0] if pool else None)
            if q is None:
                continue
            pool.remove(q)
            info = dict(guess[tid])
            info.update(name=q["name"], number=q["number"],
                        name_source="official lineup, role-inferred (not named in calibration)")
            names[tid] = info
    rest = [(x, t) for x, t in resolved if x.get("team") not in anchored]
    if rest:
        anchor_report = apply_anchors(tl, names, rest)
        r = anchor_report
        print(f"  manual anchors: {r['anchors']} ({r['unresolved']} unresolved) -> {r['named_chains']} chains named, "
              f"{r['moved']} tracklets moved onto their player, {r['evicted']} overlapping ones dropped"
              + (f"; {len(r['conflicts'])} conflicting votes" if r["conflicts"] else ""))
    if anchored:
        anchor_report = {**(anchor_report or {}), "anchored_linking": {
            t: {k: v for k, v in r.items() if k != "numbers"} for t, r in anchored.items()},
            "anchors": len(anchors), "unresolved": sum(1 for _, t in resolved if t is None)}

    # 5. tracks.parquet in the pipeline schema
    ident = dict(zip(tl["tracklet"], tl["identity"]))
    cls_of = dict(zip(tl["tracklet"], tl["cls"]))
    p["identity"] = p["tracklet"].map(ident).fillna(-1).astype(int)
    p["track_id"] = np.where(p["identity"] >= 0, p["identity"],
                             np.where(p["tracklet"] >= 0, 10000 + p["tracklet"], -2)).astype(int)
    p = p[p["track_id"] != -2]
    p = p.sort_values("conf", ascending=False)
    p = p[~(p["identity"].ge(0) & p.duplicated(["frame", "identity"]))]  # one row per identity per frame
    balls["track_id"] = -1
    cols = ["frame", "t_s", "track_id", "class", "x1", "y1", "x2", "y2", "foot_x", "foot_y", "conf"]
    tracks = pd.concat([p[cols + ["pitch_x", "pitch_y", "tracklet", "kit", "p_team_a"]], balls[cols]],
                       ignore_index=True)
    tracks = tracks.sort_values(["frame", "class", "track_id"]).reset_index(drop=True)
    tracks.to_parquet(mdir / "tracks.parquet", index=False)

    # 6. player_map / team_map (names from step 4b)
    n_frames = f_hi - f_lo + 1
    players, t2p = {}, {}
    for team, base in (("team_a", 0), ("team_b", 100)):
        letter = "a" if team == "team_a" else "b"
        for k in range(1, 12):
            tid = base + k
            rows = p[p["track_id"] == tid]
            info = names.get(tid, {})
            key = f"{letter}{info.get('number', k)}"
            if key in players:
                key = f"{letter}id{tid}"
            t2p[str(tid)] = key
            sub = tl[tl["identity"] == tid]
            players[key] = {
                "team": team, "number": info.get("number", k), "name": info.get("name"),
                "role": info.get("role"),
                "name_source": info.get("name_source") or (
                    "kit colour (goalkeeper)" if info.get("line", 1) == 0
                    else "official lineup, role-inferred from average position"),
                "anchors": info.get("anchors", 0),
                "depth_m": info.get("depth_m"), "width_m": info.get("width_m"),
                "track_ids": [tid], "detections": int(len(rows)),
                "first_frame": int(rows["frame"].min()) if len(rows) else None,
                "last_frame": int(rows["frame"].max()) if len(rows) else None,
                "coverage": round(rows["frame"].nunique() / n_frames, 3),
                "tracklets": int(len(sub)),
                "fragile_links": int((sub["stability"] < 1).sum()),
            }
    others = p[p["identity"] < 0].groupby("track_id").agg(n=("frame", "size"), tl=("tracklet", "first"))
    # Officials and people off the pitch are excluded from all stats; orange
    # tracklets that aren't Lloris are stewards/ball kids in orange.
    # Unreadable ('gk_a?') and unlinked team tracklets stay *unidentified*.
    ignored = [{"track_id": int(t), "reason": cls_of.get(int(r.tl)), "detections": int(r.n)}
               for t, r in others.iterrows() if cls_of.get(int(r.tl)) in ("official", "off_pitch", "gk_b")]
    ign_ids = {i["track_id"] for i in ignored}
    pm = {
        "match_id": a.match_id, "unit": "m",
        "team_names": {"team_a": cfg.get("team_a_name", "team_a"), "team_b": cfg.get("team_b_name", "team_b")},
        "team_map_alignment": "aligned",
        "method": "tools/track_identities.py (automatic, global min-cost-flow linking; names role-inferred)",
        "window": {"start_frame": f_lo, "end_frame": f_hi},
        "track_to_player": t2p, "players": players,
        "unidentified_tracks": [int(t) for t in others.index if int(t) not in ign_ids],
        "ignored_tracks": ignored,
        "identified_detection_fraction": round(float((p["identity"] >= 0).sum() / max(len(p), 1)), 3),
        "link_stability": stability,
        "manual_anchors": anchor_report,
    }
    write_json(mdir / "player_map.json", pm)
    team_map = {str(t): ("team_a" if 1 <= t <= 11 else "team_b" if 101 <= t <= 111 else "unknown")
                for t in p["track_id"].unique()}
    write_json(mdir / "team_map.json", team_map)
    tl.to_parquet(mdir / "identity_links.parquet", index=False)

    print(f"Wrote tracks.parquet, player_map.json, team_map.json, identity_links.parquet to {mdir}")
    print(f"  identified {pm['identified_detection_fraction']:.0%} of on-pitch person detections; "
          f"{len(ignored)} ignored tracks (officials/off-pitch), {len(pm['unidentified_tracks'])} unidentified")
    print(f"{'player':>8} {'name':<16} {'role':<13} {'coverage':>8} {'tracklets':>9} {'fragile':>7}")
    for key, pl in players.items():
        print(f"{key:>8} {str(pl['name']):<16} {str(pl['role']):<13} {pl['coverage']:>8.0%} "
              f"{pl['tracklets']:>9} {pl['fragile_links']:>7}")
    print("Next: python tools/track_ball.py --match-id " + a.match_id)


if __name__ == "__main__":
    main()
