"""End-to-end prediction for one trip: features -> classifier -> HMM/Viterbi -> route.

predict_trip() never sees the reference route. score_trip() is the separate,
server-side scoring step that compares a prediction with the reference.
"""
import time
import numpy as np

from .features import build_trip_table, FEATURE_COLS
from .hmm import viterbi_decode
from .labels import align_points_to_route
from .metrics import (route_continuity, route_set_scores, length_weighted_route_scores)


def _group(tab, n, col):
    lens = np.bincount(tab["pt"].to_numpy(), minlength=n)
    splits = np.cumsum(lens)[:-1]
    return np.split(tab[col].to_numpy(), splits)


class BoosterModel:
    """predict_proba wrapper around a native LightGBM Booster file."""

    def __init__(self, path):
        import lightgbm as lgb
        # Read the text ourselves and drop Windows line endings: git on Windows can
        # convert this file to CRLF on checkout, which LightGBM's parser rejects.
        with open(path, "rb") as f:
            text = f.read().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        self.booster = lgb.Booster(model_str=text)

    def predict_proba(self, X):
        p = self.booster.predict(X)
        return np.c_[1 - p, p]


def load_final_model(models_dir):
    """Prefer the native LightGBM text model; fall back to the joblib pickle."""
    from pathlib import Path
    txt = Path(models_dir) / "final_model_lgbm.txt"
    if txt.exists():
        return BoosterModel(txt)
    import joblib
    return joblib.load(Path(models_dir) / "final_model.joblib")


def classify(tab, model):
    X = tab[FEATURE_COLS].to_numpy(float)
    return model.predict_proba(X)[:, 1]


def assemble_route(net, loc_seq, limit=3000.0):
    """Collapse the per-point arc sequence into a connected arc route, filling
    each jump between consecutive distinct arcs with the network shortest path."""
    route = []
    prev = None
    for a in loc_seq:
        a = int(a)
        if a == prev:
            continue
        if prev is not None and net.uto[prev] != net.ufrm[a]:
            gap = net.path_arcs(int(net.uto[prev]), int(net.ufrm[a]), limit)
            if gap:
                route.extend(gap)
        route.append(a)
        prev = a
    # remove immediate duplicates created by filling
    out = [route[0]] if route else []
    for a in route[1:]:
        if a != out[-1]:
            out.append(a)
    return out


def predict_trip(trip, model, hmm_params=None, use_hmm=True):
    """Full prediction for one trip. Uses only the GPS track and road network."""
    t0 = time.time()
    tab, info, net = build_trip_table(trip, route=None)
    t_feat = time.time() - t0
    tab["prob"] = classify(tab, model)
    t_clf = time.time() - t0 - t_feat
    out = decode_table(trip.trip_id, tab, info, net, hmm_params, use_hmm)
    out["timing_s"] = {"features": t_feat, "classifier": t_clf, "hmm": out.pop("hmm_s"),
                       "total": time.time() - t0}
    return out


def decode_table(trip_id, tab, info, net, hmm_params=None, use_hmm=True):
    """Classifier-only and HMM decoding from a candidate table with a prob column."""
    t0 = time.time()
    n = info["n"]
    loc_l = _group(tab, n, "loc")
    t_l = _group(tab, n, "t_proj")
    d_l = _group(tab, n, "dist")
    p_l = _group(tab, n, "prob")
    clf_loc = np.array([l[int(np.argmax(p))] if len(l) else -1 for l, p in zip(loc_l, p_l)])
    breaks = 0
    if use_hmm:
        best, used_loc, breaks = viterbi_decode(net, info["px"], info["py"], info["t"],
                                                loc_l, t_l, d_l, p_l, hmm_params)
        hmm_loc = np.array([used_loc[i][best[i]] if len(used_loc[i]) else -1 for i in range(n)])
    else:
        hmm_loc = clf_loc
    route_loc = assemble_route(net, hmm_loc[hmm_loc >= 0])
    return {
        "trip_id": trip_id,
        "net": net,
        "n_points": n,
        "n_candidate_rows": len(tab),
        "clf_arcs": net.arc_ids[np.clip(clf_loc, 0, None)],
        "hmm_arcs": net.arc_ids[np.clip(hmm_loc, 0, None)],
        "route_arcs": net.arc_ids[route_loc] if route_loc else np.zeros(0, int),
        "route_loc": route_loc,
        "hmm_breaks": breaks,
        "hmm_s": time.time() - t0,
    }


def route_geometry(net, route_loc):
    """List of [lat, lon] polylines (one per connected run of arcs)."""
    inv = net.proj
    lines, cur = [], []
    prev = None
    for a in route_loc:
        ax, ay, bx, by = net.ax[a], net.ay[a], net.bx[a], net.by[a]
        A = (ay / inv.ky + inv.lat0, ax / inv.kx + inv.lon0)
        B = (by / inv.ky + inv.lat0, bx / inv.kx + inv.lon0)
        if prev is not None and net.uto[prev] != net.ufrm[a]:
            lines.append(cur)
            cur = []
        if not cur:
            cur.append([float(A[0]), float(A[1])])
        cur.append([float(B[0]), float(B[1])])
        prev = a
    if cur:
        lines.append(cur)
    return lines


def score_trip(pred, trip, route):
    """Server-side scoring of a prediction against the reference route."""
    net = pred["net"]
    px, py = net.proj.to_xy(trip.track[:, 0], trip.track[:, 1])
    lab, ldist, _ = align_points_to_route(px, py, net, route)
    arc_len = {int(g): float(net.length[i]) for i, g in enumerate(net.arc_ids)}
    rp, rr, rf = route_set_scores(pred["route_arcs"], route)
    lp, lr, lf = length_weighted_route_scores(pred["route_arcs"], route, arc_len)
    return {
        "point_accuracy_hmm": float(np.mean(pred["hmm_arcs"] == lab)),
        "point_accuracy_classifier_only": float(np.mean(pred["clf_arcs"] == lab)),
        "route_arc_precision": rp, "route_arc_recall": rr, "route_arc_f1": rf,
        "route_length_precision": lp, "route_length_recall": lr, "route_length_f1": lf,
        "route_continuity_hmm": route_continuity(pred["hmm_arcs"], trip.arcs),
        "route_continuity_classifier_only": route_continuity(pred["clf_arcs"], trip.arcs),
        "n_reference_arcs": int(len(route)), "n_predicted_route_arcs": int(len(pred["route_arcs"])),
    }
