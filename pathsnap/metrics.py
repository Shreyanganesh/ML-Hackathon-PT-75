"""Evaluation metrics shared by training scripts, decoder experiments and the API."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score

# Trips whose reference route sits far from the GPS track for the whole trip
# (median point-to-route distance above this) are treated as label defects.
LABEL_DEFECT_MEDIAN_M = 50.0


def row_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "row_accuracy": accuracy_score(y, yhat),
        "macro_precision": precision_score(y, yhat, average="macro", zero_division=0),
        "macro_recall": recall_score(y, yhat, average="macro", zero_division=0),
        "macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
        "log_loss": log_loss(y, pc, labels=[0, 1]),
    }


def top1_predictions(tab, prob_col="prob"):
    """Per point, the candidate arc with the highest score. Returns DataFrame trip, pt, pred_arc."""
    idx = tab.groupby(["trip", "pt"], sort=False)[prob_col].idxmax()
    out = tab.loc[idx, ["trip", "pt", "arc"]].rename(columns={"arc": "pred_arc"})
    return out


def top1_accuracy(points, pred):
    """points: trip, pt, label_arc (all points). pred: trip, pt, pred_arc.

    Points that got no candidate count as errors.
    """
    m = points[["trip", "pt", "label_arc"]].merge(pred, on=["trip", "pt"], how="left")
    m["pred_arc"] = m["pred_arc"].fillna(-2)
    return float((m["pred_arc"].to_numpy() == m["label_arc"].to_numpy()).mean())


def route_continuity(pred_arcs, arcs):
    """Share of consecutive point pairs whose predicted arcs are the same arc
    or directly connected (head of the first = tail of the second)."""
    a = np.asarray(pred_arcs)
    if len(a) < 2:
        return 1.0
    p, q = a[:-1], a[1:]
    same = p == q
    conn = arcs[p, 1] == arcs[q, 0]
    return float(np.mean(same | conn))


def route_set_scores(pred_route, ref_route):
    """Arc-level precision / recall / F1 between the set of predicted route arcs
    and the set of reference route arcs."""
    P, R = set(map(int, pred_route)), set(map(int, ref_route))
    if not P or not R:
        return 0.0, 0.0, 0.0
    tp = len(P & R)
    prec, rec = tp / len(P), tp / len(R)
    f1 = 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def length_weighted_route_scores(pred_route, ref_route, arc_len):
    """Length-weighted precision/recall: metres of predicted route that are on
    the reference route, and metres of reference route that were recovered."""
    P, R = set(map(int, pred_route)), set(map(int, ref_route))
    lp = sum(arc_len.get(a, 0.0) for a in P)
    lr = sum(arc_len.get(a, 0.0) for a in R)
    lt = sum(arc_len.get(a, 0.0) for a in P & R)
    prec = lt / lp if lp else 0.0
    rec = lt / lr if lr else 0.0
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def defect_trips(points):
    med = points.groupby("trip")["label_dist"].median()
    return sorted(med[med > LABEL_DEFECT_MEDIAN_M].index.tolist())
