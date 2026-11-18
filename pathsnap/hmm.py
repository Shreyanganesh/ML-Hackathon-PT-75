"""HMM map matching decoded with Viterbi.

Hidden state at GPS step i : one candidate road arc of point i (the candidate
                              list from features.py, so every state keeps its
                              original arc id).
Emission  log b_i(s)        : w * log P_clf(s is true arc)
                              + (1 - w) * log N(dist_i(s); 0, sigma)
Transition log a(s -> s')   : -| route_dist(s, s') - gps_dist(i-1, i) | / beta_i
                              (Newson & Krumm style), where route_dist is the
                              driving distance through the directed network
                              from the projection of point i-1 on s to the
                              projection of point i on s'. beta grows with the
                              time gap so long GPS gaps are not over-penalised.
                              Staying on one arc costs the distance moved along
                              it (backward steps up to back_tol are accepted as
                              projection jitter). Unreachable pairs get log prob -inf.
Breaks                      : if no state at step i is reachable from any
                              state at i-1 (disconnected segments, missing
                              network, very long gap) the chain is split and a
                              new chain starts at i from emissions only.

Viterbi recurrence:
  delta_i(s') = log b_i(s') + max_s [ delta_{i-1}(s) + log a(s -> s') ]
"""
import numpy as np

DEFAULT_PARAMS = {
    "w_clf": 0.7,          # weight of classifier log-probability in the emission
    "sigma": 10.0,         # metres, Gaussian distance term in the emission
    "beta": 5.0,           # metres, transition scale for a 1 s step
    "beta_gap_power": 0.5, # beta_i = beta * dt^power
    "back_tol": 15.0,      # metres of backwards movement on the same arc tolerated (GPS jitter)
    "search_extra": 400.0, # Dijkstra search radius = 3 * gps_dist + search_extra
    "backward_as_standstill": False,  # ablation: count small backward steps as 0 m moved
}


def _emission(prob, dist, p):
    lp = np.log(np.clip(prob, 1e-6, 1.0))
    lg = -0.5 * (dist / p["sigma"]) ** 2
    return p["w_clf"] * lp + (1 - p["w_clf"]) * lg


def viterbi_decode(net, px, py, t, cand_loc, cand_t, cand_dist, cand_prob, params=None):
    """Decode one trip.

    cand_* : lists (one entry per GPS point) of arrays over that point's candidates.
    Returns (state_index_per_point, candidate lists used, n_breaks).
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    n = len(px)
    cand_loc, cand_t, cand_dist = list(cand_loc), list(cand_t), list(cand_dist)
    back = [None] * n
    em = [_emission(cand_prob[i], cand_dist[i], p) if len(cand_loc[i]) else np.zeros(0) for i in range(n)]
    delta = em[0].copy()
    chain_starts = [0]
    chain_end_delta = []
    L = net.length

    def new_chain(i, d0):
        chain_end_delta.append(delta)
        chain_starts.append(i)
        return d0

    for i in range(1, n):
        a, b = cand_loc[i - 1], cand_loc[i]
        if len(b) == 0:
            # no candidate for this point: carry the previous states forward unchanged
            cand_loc[i], cand_t[i], cand_dist[i] = a, cand_t[i - 1], cand_dist[i - 1]
            back[i] = np.arange(len(a))
            continue
        if len(a) == 0:
            delta = new_chain(i, em[i].copy())
            continue
        gc = float(np.hypot(px[i] - px[i - 1], py[i] - py[i - 1]))
        dt = max(float(t[i] - t[i - 1]), 1.0)
        beta = p["beta"] * dt ** p["beta_gap_power"]
        limit = 3.0 * gc + p["search_extra"]
        ta, tb = cand_t[i - 1], cand_t[i]
        R = np.full((len(a), len(b)), np.inf)
        for u in range(len(a)):
            au = a[u]
            rest = (1.0 - ta[u]) * L[au]
            sp = None
            for v in range(len(b)):
                bv = b[v]
                if au == bv:
                    delta_m = (tb[v] - ta[u]) * L[au]
                    if delta_m >= -p["back_tol"]:
                        # forward move, or a small backward step caused by projection jitter:
                        # the route distance is the distance moved along the arc
                        if delta_m < 0 and p["backward_as_standstill"]:
                            R[u, v] = 0.0
                        else:
                            R[u, v] = abs(delta_m)
                        continue
                if sp is None:
                    sp = net.sp_from(net.uto[au], limit)
                d = sp.get(int(net.ufrm[bv]))
                if d is not None:
                    R[u, v] = rest + d + tb[v] * L[bv]
        logA = -np.abs(R - gc) / beta
        score = delta[:, None] + logA
        arg = np.argmax(score, axis=0)
        mx = score[arg, np.arange(len(b))]
        if not np.any(np.isfinite(mx)):
            # consecutive points only reach disconnected segments: split the chain
            delta = new_chain(i, em[i].copy())
            continue
        delta = mx + em[i]
        back[i] = arg
    chain_end_delta.append(delta)

    best = np.zeros(n, dtype=int)
    ends = chain_starts[1:] + [n]
    for s, e, d in zip(chain_starts, ends, chain_end_delta):
        j = int(np.argmax(d)) if len(d) else 0
        best[e - 1] = j
        for i in range(e - 1, s, -1):
            j = int(back[i][j])
            best[i - 1] = j
    return best, cand_loc, len(chain_starts) - 1
