"""Step 3: train and compare candidate classifiers.

Protocol
  * 5-fold GroupKFold over the 80 training trips (a trip never appears in two
    folds). Trips flagged as reference-route defects are excluded from fitting.
  * Each model is refitted on all usable training trips and evaluated once on
    the 10 student-testing trips.
  * Metrics: row-level macro precision / recall / F1 and log loss, observation
    level Top-1 accuracy, route continuity of the classifier-only sequence,
    training and inference time.

Outputs (results/):
  model_cv_folds.csv, model_comparison.csv, confusion_<model>.png,
  loss_curves_<model>.csv / .png, models/experiments/<model>.joblib

Run:  python scripts/03_train_models.py            (all models)
      python scripts/03_train_models.py LightGBM   (selected models)
"""
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from pathsnap.config import CACHE_DIR, RESULTS_DIR, MODELS_DIR, SEED, CFG, TRAIN_DIR, STUDENT_TEST_DIR
from pathsnap.features import FEATURE_COLS
from pathsnap.metrics import (row_metrics, top1_predictions, top1_accuracy, route_continuity,
                              defect_trips)
from pathsnap.model_zoo import MODELS
from pathsnap.data_io import load_trip

EXP_DIR = MODELS_DIR / "experiments"
EXP_DIR.mkdir(parents=True, exist_ok=True)


def load_partitions():
    tr = joblib.load(CACHE_DIR / "train_candidates.joblib")
    trp = joblib.load(CACHE_DIR / "train_points.joblib")
    te = joblib.load(CACHE_DIR / "student_test_candidates.joblib")
    tep = joblib.load(CACHE_DIR / "student_test_points.joblib")
    bad = defect_trips(trp)
    return tr, trp, te, tep, bad


_ARCS = {}


def arcs_of(part_dir, tid):
    if (part_dir, tid) not in _ARCS:
        _ARCS[(part_dir, tid)] = load_trip(part_dir, tid).arcs
    return _ARCS[(part_dir, tid)]


def continuity(tab, part_dir):
    pred = top1_predictions(tab)
    vals = []
    for tid, g in pred.groupby("trip"):
        g = g.sort_values("pt")
        vals.append(route_continuity(g["pred_arc"].to_numpy(), arcs_of(str(part_dir), tid)))
    return float(np.mean(vals)), pred


def fit_model(name, X, y):
    family, conf, factory, cap = MODELS[name]
    m = factory()
    if cap is not None and len(X) > cap:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(X), cap, replace=False)
        X, y = X[idx], y[idx]
    t0 = time.time()
    m.fit(X, y)
    return m, time.time() - t0


def predict(m, X):
    t0 = time.time()
    p = m.predict_proba(X)[:, 1]
    return p, time.time() - t0


def run(names):
    tr, trp, te, tep, bad = load_partitions()
    print("label-defect trips excluded from fitting:", bad)
    trf = tr[~tr["trip"].isin(bad)].reset_index(drop=True)
    trpf = trp[~trp["trip"].isin(bad)].reset_index(drop=True)
    X, y, groups = trf[FEATURE_COLS].to_numpy(float), trf["label"].to_numpy(), trf["trip"].to_numpy()
    Xte, yte = te[FEATURE_COLS].to_numpy(float), te["label"].to_numpy()
    gkf = GroupKFold(n_splits=CFG["cv_folds"])
    folds = list(gkf.split(X, y, groups))

    fold_rows, comp_rows = [], []
    for name in names:
        family, conf, _, _ = MODELS[name]
        print(f"\n=== {name}", flush=True)
        for k, (a, b) in enumerate(folds):
            m, ttrain = fit_model(name, X[a], y[a])
            p, tinf = predict(m, X[b])
            rm = row_metrics(y[b], p)
            vt = trf.iloc[b][["trip", "pt", "arc"]].assign(prob=p)
            vp = trpf[trpf["trip"].isin(set(groups[b]))]
            acc = top1_accuracy(vp, top1_predictions(vt))
            fold_rows.append({"model": name, "fold": k, **rm, "top1_accuracy": acc,
                              "train_s": ttrain, "infer_s": tinf})
            print(f"  fold {k}: macroF1={rm['macro_f1']:.4f} top1={acc:.4f} logloss={rm['log_loss']:.4f} "
                  f"({ttrain:.1f}s)", flush=True)
        fdf = pd.DataFrame([r for r in fold_rows if r["model"] == name])

        m, ttrain = fit_model(name, X, y)
        p, tinf = predict(m, Xte)
        rm = row_metrics(yte, p)
        tt = te[["trip", "pt", "arc"]].assign(prob=p)
        cont, pred = continuity(tt, STUDENT_TEST_DIR)
        acc = top1_accuracy(tep, pred)
        # training-set Top-1 (refit model on its own data) for over/underfitting evidence
        ptr, _ = predict(m, X)
        acc_tr = top1_accuracy(trpf, top1_predictions(trf[["trip", "pt", "arc"]].assign(prob=ptr)))
        comp_rows.append({
            "model": name, "family": family, "config": conf,
            "cv_macro_f1_mean": fdf["macro_f1"].mean(), "cv_macro_f1_std": fdf["macro_f1"].std(),
            "cv_top1_mean": fdf["top1_accuracy"].mean(), "cv_top1_std": fdf["top1_accuracy"].std(),
            "cv_log_loss_mean": fdf["log_loss"].mean(),
            "train_top1": acc_tr,
            "test_top1_accuracy": acc, "test_macro_f1": rm["macro_f1"],
            "test_macro_precision": rm["macro_precision"], "test_macro_recall": rm["macro_recall"],
            "test_log_loss": rm["log_loss"], "test_route_continuity": cont,
            "train_time_s": ttrain, "test_inference_s": tinf,
            "inference_ms_per_point": 1000 * tinf / len(tep),
        })
        print(f"  TEST top1={acc:.4f} macroF1={rm['macro_f1']:.4f} logloss={rm['log_loss']:.4f} "
              f"continuity={cont:.4f} train_top1={acc_tr:.4f}", flush=True)

        cm = confusion_matrix(yte, (p >= 0.5).astype(int))
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ConfusionMatrixDisplay(cm, display_labels=["not route arc", "route arc"]).plot(ax=ax, colorbar=False, values_format="d")
        ax.set_title(f"{name} (student test, rows)")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"confusion_{name}.png", dpi=130)
        plt.close(fig)
        try:
            joblib.dump(m, EXP_DIR / f"{name}.joblib", compress=3)
        except Exception as e:  # pragma: no cover
            print("could not save", name, e)

        # merge into existing files so models can be run separately
        _merge_csv(RESULTS_DIR / "model_cv_folds.csv", fdf, "model", name)
        _merge_csv(RESULTS_DIR / "model_comparison.csv", pd.DataFrame([comp_rows[-1]]), "model", name)


def _merge_csv(path, df, key, val):
    if path.exists():
        old = pd.read_csv(path)
        old = old[old[key] != val]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


def learning_curves():
    """Epoch-wise training/validation log loss for the models that have one
    (MLP epochs, boosting rounds). Validation = 16 held-out training trips."""
    from pathsnap.model_zoo import EpochMLP, lgbm
    from xgboost import XGBClassifier
    tr, trp, te, tep, bad = load_partitions()
    tr = tr[~tr["trip"].isin(bad)]
    trips = np.array(sorted(tr["trip"].unique()))
    rng = np.random.RandomState(SEED)
    val_trips = set(rng.choice(trips, size=len(trips) // 5, replace=False))
    a = ~tr["trip"].isin(val_trips)
    X, y = tr.loc[a, FEATURE_COLS].to_numpy(float), tr.loc[a, "label"].to_numpy()
    Xv, yv = tr.loc[~a, FEATURE_COLS].to_numpy(float), tr.loc[~a, "label"].to_numpy()

    mlp = EpochMLP().fit(X, y, Xv, yv)
    h = pd.DataFrame(mlp.history_)
    h.to_csv(RESULTS_DIR / "loss_curves_MLP.csv", index=False)
    _plot_curve(h["epoch"], h["train_loss"], h["val_loss"], "MLP", "epoch")

    lg = lgbm()
    lg.fit(X, y, eval_set=[(X, y), (Xv, yv)], eval_metric="binary_logloss")
    r = lg.evals_result_
    h = pd.DataFrame({"iteration": np.arange(1, len(r["training"]["binary_logloss"]) + 1),
                      "train_loss": r["training"]["binary_logloss"], "val_loss": r["valid_1"]["binary_logloss"]})
    h.to_csv(RESULTS_DIR / "loss_curves_LightGBM.csv", index=False)
    _plot_curve(h["iteration"], h["train_loss"], h["val_loss"], "LightGBM", "boosting round")

    xg = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8,
                       tree_method="hist", n_jobs=2, random_state=SEED, eval_metric="logloss")
    xg.fit(X, y, eval_set=[(X, y), (Xv, yv)], verbose=False)
    r = xg.evals_result()
    h = pd.DataFrame({"iteration": np.arange(1, len(r["validation_0"]["logloss"]) + 1),
                      "train_loss": r["validation_0"]["logloss"], "val_loss": r["validation_1"]["logloss"]})
    h.to_csv(RESULTS_DIR / "loss_curves_XGBoost.csv", index=False)
    _plot_curve(h["iteration"], h["train_loss"], h["val_loss"], "XGBoost", "boosting round")
    with open(RESULTS_DIR / "loss_curve_split.json", "w") as f:
        json.dump({"validation_trips": sorted(val_trips)}, f, indent=1)


def _plot_curve(x, a, b, name, xlabel):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(x, a, label="training log loss")
    ax.plot(x, b, label="validation log loss (held-out train trips)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("log loss")
    ax.set_title(f"{name} training history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"loss_curves_{name}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["curves"]:
        learning_curves()
    else:
        run(args or list(MODELS.keys()))
