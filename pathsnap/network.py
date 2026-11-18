"""Per-trip road network: projected geometry, spatial index and shortest paths."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .geo import LocalProjection, point_segment


class RoadNetwork:
    """Holds the arcs of one trip that lie near the GPS track.

    Arc ids are always the original 0-based row indices of the .arcs file, so
    the candidate-to-road-segment correspondence is preserved end to end.
    """

    def __init__(self, trip, margin_m=300.0, sample_step_m=15.0):
        self.proj = LocalProjection.for_track(trip.track)
        nx, ny = self.proj.to_xy(trip.nodes[:, 0], trip.nodes[:, 1])
        self.node_xy = np.c_[nx, ny]
        px, py = self.proj.to_xy(trip.track[:, 0], trip.track[:, 1])
        x0, x1 = px.min() - margin_m, px.max() + margin_m
        y0, y1 = py.min() - margin_m, py.max() + margin_m

        arcs = trip.arcs
        self.n_arcs_total = len(arcs)
        valid = (arcs[:, 0] >= 0) & (arcs[:, 1] >= 0) & (arcs[:, 0] < len(nx)) & (arcs[:, 1] < len(nx))
        a = self.node_xy[np.clip(arcs[:, 0], 0, len(nx) - 1)]
        b = self.node_xy[np.clip(arcs[:, 1], 0, len(nx) - 1)]
        inside = ((np.maximum(a[:, 0], b[:, 0]) >= x0) & (np.minimum(a[:, 0], b[:, 0]) <= x1) &
                  (np.maximum(a[:, 1], b[:, 1]) >= y0) & (np.minimum(a[:, 1], b[:, 1]) <= y1))
        keep = np.where(valid & inside)[0]
        self.arc_ids = keep                       # original ids of kept arcs
        self.local_of = {int(g): i for i, g in enumerate(keep)}
        self.frm = arcs[keep, 0]
        self.to = arcs[keep, 1]
        self.ax, self.ay = a[keep, 0], a[keep, 1]
        self.bx, self.by = b[keep, 0], b[keep, 1]
        self.length = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.bearing = np.arctan2(self.by - self.ay, self.bx - self.ax)

        # degrees in the full network (intersection complexity)
        V = len(nx)
        outdeg = np.bincount(arcs[valid, 0], minlength=V)
        indeg = np.bincount(arcs[valid, 1], minlength=V)
        self.out_deg_to = outdeg[self.to]
        self.in_deg_from = indeg[self.frm]
        # does the opposite-direction arc exist (two-way road)?
        pair = set(zip(arcs[valid, 0].tolist(), arcs[valid, 1].tolist()))
        self.has_twin = np.array([(int(t), int(f)) in pair for f, t in zip(self.frm, self.to)], dtype=np.int8)

        # spatial index over points sampled along each arc
        n_s = np.maximum(np.ceil(self.length / sample_step_m).astype(int), 1) + 1
        idx = np.repeat(np.arange(len(keep)), n_s)
        frac = np.concatenate([np.linspace(0, 1, k) for k in n_s]) if len(keep) else np.zeros(0)
        sx = self.ax[idx] + frac * (self.bx - self.ax)[idx]
        sy = self.ay[idx] + frac * (self.by - self.ay)[idx]
        self.sample_arc = idx
        self.sample_step = sample_step_m
        self.tree = cKDTree(np.c_[sx, sy]) if len(idx) else None

        # directed graph over the kept arcs only, with compact node numbering
        nodes_used, inv = np.unique(np.r_[self.frm, self.to], return_inverse=True)
        self.node_global = nodes_used
        self.ufrm = inv[: len(keep)]
        self.uto = inv[len(keep):]
        Vl = len(nodes_used)
        self.n_nodes = Vl
        w = self.length + 1e-3
        # keep the shortest arc when two arcs join the same node pair
        pair_key = self.ufrm.astype(np.int64) * max(Vl, 1) + self.uto
        uk, first_idx = np.unique(pair_key[np.argsort(w, kind="stable")], return_index=True)
        best = np.argsort(w, kind="stable")[first_idx]
        self.graph = csr_matrix((w[best], (self.ufrm[best], self.uto[best])), shape=(Vl, Vl))
        self.arc_of_pair = {}
        for i in np.argsort(w)[::-1]:
            self.arc_of_pair[(int(self.ufrm[i]), int(self.uto[i]))] = int(i)
        self._sp_cache = {}

    # ------------------------------------------------------------------
    def candidates(self, px, py, radius, k_max):
        """Return list (per point) of (local_arc_idx array, dist array, t array, fallback flag)."""
        out = []
        if self.tree is None:
            return [(np.zeros(0, int), np.zeros(0), np.zeros(0), True) for _ in px]
        pts = np.c_[px, py]
        balls = self.tree.query_ball_point(pts, r=radius + self.sample_step)
        for i, lst in enumerate(balls):
            fallback = False
            if lst:
                arcs = np.unique(self.sample_arc[lst])
            else:
                arcs = np.zeros(0, int)
            if len(arcs):
                d, t = point_segment(px[i], py[i], self.ax[arcs], self.ay[arcs], self.bx[arcs], self.by[arcs])
                m = d <= radius
                arcs, d, t = arcs[m], d[m], t[m]
            if len(arcs) == 0:
                # missing-candidate handling: take the nearest arcs whatever the distance
                fallback = True
                _, nn = self.tree.query(pts[i], k=min(40, len(self.sample_arc)))
                arcs = np.unique(self.sample_arc[np.atleast_1d(nn)])
                d, t = point_segment(px[i], py[i], self.ax[arcs], self.ay[arcs], self.bx[arcs], self.by[arcs])
            order = np.argsort(d, kind="stable")[:k_max]
            out.append((arcs[order], d[order], t[order], fallback))
        return out

    # ------------------------------------------------------------------
    def sp_from(self, unode, limit):
        """Network distances (metres) from a compact node, as a dict node->dist,
        searching no further than limit metres. Cached per (node, limit)."""
        key = (int(unode), float(limit))
        hit = self._sp_cache.get(key)
        if hit is None:
            dist = dijkstra(self.graph, directed=True, indices=int(unode), limit=limit)
            nz = np.where(np.isfinite(dist))[0]
            hit = dict(zip(nz.tolist(), dist[nz].tolist()))
            self._sp_cache[key] = hit
        return hit

    def path_arcs(self, unode_src, unode_dst, limit):
        """Local arc indices of the shortest path between two compact nodes, or None."""
        if unode_src == unode_dst:
            return []
        dist, pred = dijkstra(self.graph, directed=True, indices=int(unode_src), limit=limit,
                              return_predecessors=True)
        if not np.isfinite(dist[unode_dst]):
            return None
        nodes = [int(unode_dst)]
        while nodes[-1] != unode_src:
            nodes.append(int(pred[nodes[-1]]))
        nodes = nodes[::-1]
        return [self.arc_of_pair[(u, v)] for u, v in zip(nodes[:-1], nodes[1:])]
