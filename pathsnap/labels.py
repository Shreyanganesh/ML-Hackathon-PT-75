"""Turning the reference arc sequence (.route) into one label per GPS point.

The .route file lists arcs in travel order but does not say which GPS point
belongs to which arc. We solve a monotone alignment: every point is assigned a
position j in the route, positions never go backwards in time, and the total
point-to-arc distance is minimised. This is a dynamic programme

    C[i, j] = d(i, j) + min_{k <= j} C[i-1, k]

solved in O(N * M). Using positions (not arc ids) keeps loops in the route
correct when an arc is driven twice.
"""
import numpy as np
from .geo import point_segment


def align_points_to_route(px, py, net, route):
    """Return (label_arc_ids, label_dist_m, route_pos) for every GPS point.

    Labels are original arc ids. Points whose label arc is not in the kept
    network window get label -1.
    """
    N = len(px)
    loc = np.array([net.local_of.get(int(a), -1) for a in route])
    ok = loc >= 0
    route_ok = np.asarray(route)[ok]
    loc = loc[ok]
    M = len(loc)
    if M == 0 or N == 0:
        return np.full(N, -1), np.full(N, np.inf), np.full(N, -1)
    D = np.empty((N, M), dtype=np.float32)
    step = max(1, 4_000_000 // M)
    for s in range(0, N, step):
        e = min(N, s + step)
        d, _ = point_segment(px[s:e, None], py[s:e, None], net.ax[loc][None], net.ay[loc][None],
                             net.bx[loc][None], net.by[loc][None])
        D[s:e] = d
    C = np.empty_like(D)
    C[0] = D[0]
    for i in range(1, N):
        C[i] = D[i] + np.minimum.accumulate(C[i - 1])
    pos = np.empty(N, dtype=np.int64)
    pos[-1] = int(np.argmin(C[-1]))
    for i in range(N - 1, 0, -1):
        j = pos[i]
        pos[i - 1] = int(np.argmin(C[i - 1, : j + 1]))
    lab = route_ok[pos]
    dist = D[np.arange(N), pos].astype(float)
    return lab, dist, pos
