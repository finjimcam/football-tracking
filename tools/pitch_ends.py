"""Which end each team attacks, per period, and turning a whole match round so it never changes
(issue #63).

The teams change ends at half time. Pitch positions are in the camera's frame, so over a whole
match a team attacks one way in the first half and the other way in the second, and anything
summed over the match (heatmaps, average positions, team shape) smears across both ends.

orient() turns every period where the keepers have swapped ends through 180 degrees
(x -> L - x, y -> W - y), so each team attacks the way it did in the first period all match.
Rotating rather than mirroring x keeps a left winger on the left of the attack.

Each period's ends come from where its keepers stand (else the team's mean x), as
detect_events.attacking_goal does. Periods are config "periods" (start_frame / end_frame);
a match without them is one period and is left as it is.
"""
import numpy as np

TEAMS = ("team_a", "team_b")


def period_ranges(cfg, lo, hi):
    """[(start_frame, end_frame)] of the match's periods, in time order (else one range)."""
    per = sorted((int(p["start_frame"]), int(p["end_frame"])) for p in ((cfg or {}).get("periods") or {}).values()
                 if p.get("start_frame") is not None and p.get("end_frame") is not None)
    return per or [(int(lo), int(hi))]


def team_ids(pmap):
    """({team: [track ids]}, {team: [keeper track ids]}) from player_map.json."""
    ids, gk = {t: [] for t in TEAMS}, {t: [] for t in TEAMS}
    for p in (pmap or {}).get("players", {}).values():
        if p.get("team") not in TEAMS:
            continue
        tids = [int(t) for t in p.get("track_ids", [])]
        ids[p["team"]] += tids
        if p.get("role") == "GK":
            gk[p["team"]] += tids
    return ids, gk


def attacking_x(tracks, pmap, L):
    """{team: x of the goal it attacks} over `tracks` (frame, track_id, pitch_x), from where its
    keeper stands, else the team's median x. None when neither team is seen."""
    ids, gk = team_ids(pmap)
    own = {}
    for team in TEAMS:
        for pool in (gk[team], ids[team]):
            xs = tracks.loc[tracks["track_id"].isin(pool), "pitch_x"].dropna()
            if len(xs):
                own[team] = float(xs.median())
                break
    if not own:
        return None
    out = {t: (float(L) if x < L / 2 else 0.0) for t, x in own.items()}
    if len(out) == 1 or out["team_a"] == out["team_b"]:  # one team only, or both "at one end"
        t = "team_a" if "team_a" in out else "team_b"
        out = {t: out[t], ("team_b" if t == "team_a" else "team_a"): float(L) - out[t]}
    return out


def period_ends(tracks, pmap, cfg, L):
    """[(start, end, {team: attacking x} or None)] per period."""
    lo, hi = int(tracks["frame"].min()), int(tracks["frame"].max())
    out = []
    for s, e in period_ranges(cfg, lo, hi):
        t = tracks[(tracks["frame"] >= s) & (tracks["frame"] <= e)]
        out.append((s, e, attacking_x(t, pmap, L) if len(t) else None))
    return out


def flipped_ranges(tracks, pmap, cfg, L):
    """[(start, end)] of the periods whose ends are swapped relative to the first period with any
    players in it. Frames between periods are left as they are."""
    ends = period_ends(tracks, pmap, cfg, L)
    ref = next((g for _, _, g in ends if g), None)
    if ref is None:
        return []
    return [(s, e) for s, e, g in ends if g and g["team_a"] != ref["team_a"]]


def orient(df, flips, L, W, x="pitch_x", y="pitch_y"):
    """Copy of df with (x, y) turned 180 degrees in the frames of `flips` (flipped_ranges)."""
    if not flips or df.empty:
        return df
    f = df["frame"].to_numpy()
    m = np.zeros(len(df), bool)
    for s, e in flips:
        m |= (f >= s) & (f <= e)
    if not m.any():
        return df
    out = df.copy()
    out.loc[m, x] = L - out.loc[m, x]
    out.loc[m, y] = W - out.loc[m, y]
    return out
