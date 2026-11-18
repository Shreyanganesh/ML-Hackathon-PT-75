"""Candidate generation and feature engineering.

For every GPS point we collect up to K road arcs within a radius (the
candidates). Each (point, candidate) pair becomes one row; the classifier
learns P(candidate is the true arc | features). Features describe:

  geometry   : perpendicular distance, rank and relative distance among the
               point's candidates, projection position along the arc
  direction  : agreement between the GPS heading and the arc direction
               (separates the two directions of a two-way road)
  context    : the same distance / heading checks for neighbouring points in
               a +-w window (smooths single noisy fixes)
  kinematics : speed, time gaps, turning
  topology   : arc length, intersection degree, two-way flag
"""
import numpy as np
import pandas as pd

from .config import CAND, FEAT
from .network import RoadNetwork
from .geo import point_segment

FEATURE_COLS = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio_min", "n_cand", "softmin_p",
    "head_cos", "head_abs_cos", "moving",
    "win_mean_dist", "win_max_dist", "win_rank", "win_softmin_p", "win_head_cos",
    "speed", "dt_prev", "dt_next", "turn",
    "t_proj", "at_endpoint", "arc_len", "out_deg_to", "in_deg_from", "has_twin", "fallback",
]


def _headings(px, py, t, lag, min_speed):
    n = len(px)
    i0 = np.clip(np.arange(n) - lag, 0, n - 1)
    i1 = np.clip(np.arange(n) + lag, 0, n - 1)
    dx, dy = px[i1] - px[i0], py[i1] - py[i0]
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    speed = np.hypot(dx, dy) / dt
    head = np.arctan2(dy, dx)
    moving = (speed >= min_speed).astype(np.int8)
    return head, speed, moving


def build_trip_table(trip, route=None, net=None):
    """Return (DataFrame of candidate rows, per-point info dict, network).

    The route argument is optional and only used to attach labels; feature
    values never depend on it.
    """
    if net is None:
        net = RoadNetwork(trip, margin_m=CAND["bbox_margin_m"])
    tr = trip.track
    px, py = net.proj.to_xy(tr[:, 0], tr[:, 1])
    t = tr[:, 2]
    n = len(px)
    head, speed, moving = _headings(px, py, t, FEAT["heading_lag"], FEAT["min_speed_for_heading"])
    dt_prev = np.r_[0.0, np.diff(t)]
    dt_next = np.r_[np.diff(t), 0.0]
    turn = np.abs(np.angle(np.exp(1j * (np.r_[head[1:], head[-1]] - np.r_[head[0], head[:-1]]))))

    cands = net.candidates(px, py, CAND["radius_m"], CAND["max_candidates"])
    lens = np.array([len(c[0]) for c in cands])
    pt = np.repeat(np.arange(n), lens)
    loc = np.concatenate([c[0] for c in cands]).astype(int)
    dist = np.concatenate([c[1] for c in cands])
    tproj = np.concatenate([c[2] for c in cands])
    fb = np.repeat(np.array([c[3] for c in cands], dtype=np.int8), lens)
    rank = np.concatenate([np.arange(k) for k in lens])
    first = np.r_[0, np.cumsum(lens)[:-1]]
    dmin = np.repeat(dist[first] if len(dist) else np.zeros(0), lens)
    sig = FEAT["emission_sigma_m"]

    def _softmin(d):
        e = np.exp(-0.5 * (d / sig) ** 2)
        s = np.add.reduceat(e, first) if len(e) else e
        return e / np.maximum(np.repeat(s, lens), 1e-300)

    # heading agreement: cos(angle between GPS heading and arc bearing)
    hc = np.cos(head[pt] - net.bearing[loc]) * moving[pt]

    # context window
    w = FEAT["window"]
    wd = np.zeros((2 * w + 1, len(pt)))
    wh = np.zeros_like(wd)
    for k, o in enumerate(range(-w, w + 1)):
        j = np.clip(pt + o, 0, n - 1)
        d, _ = point_segment(px[j], py[j], net.ax[loc], net.ay[loc], net.bx[loc], net.by[loc])
        wd[k] = d
        wh[k] = np.cos(head[j] - net.bearing[loc]) * moving[j]
    win_mean = wd.mean(0)
    # rank of the window-mean distance among the point's candidates
    df = pd.DataFrame({"pt": pt, "win_mean_dist": win_mean})
    win_rank = df.groupby("pt")["win_mean_dist"].rank(method="first").to_numpy() - 1

    tab = pd.DataFrame({
        "trip": trip.trip_id,
        "pt": pt,
        "arc": net.arc_ids[loc],
        "loc": loc,
        "dist": dist,
        "dist_rank": rank,
        "dist_minus_min": dist - dmin,
        "dist_ratio_min": dist / (dmin + 1.0),
        "n_cand": np.repeat(lens, lens),
        "softmin_p": _softmin(dist),
        "head_cos": hc,
        "head_abs_cos": np.abs(hc),
        "moving": moving[pt],
        "win_mean_dist": win_mean,
        "win_max_dist": wd.max(0),
        "win_rank": win_rank,
        "win_softmin_p": _softmin(win_mean),
        "win_head_cos": wh.mean(0),
        "speed": speed[pt],
        "dt_prev": dt_prev[pt],
        "dt_next": dt_next[pt],
        "turn": turn[pt],
        "t_proj": tproj,
        "at_endpoint": ((tproj <= 1e-6) | (tproj >= 1 - 1e-6)).astype(np.int8),
        "arc_len": net.length[loc],
        "out_deg_to": net.out_deg_to[loc],
        "in_deg_from": net.in_deg_from[loc],
        "has_twin": net.has_twin[loc],
        "fallback": fb,
    })
    info = {"px": px, "py": py, "t": t, "n": n, "lens": lens}
    if route is not None:
        from .labels import align_points_to_route
        lab, ldist, _ = align_points_to_route(px, py, net, route)
        info["label_arc"] = lab
        info["label_dist"] = ldist
        tab["label"] = (tab["arc"].to_numpy() == lab[pt]).astype(np.int8)
    return tab, info, net
