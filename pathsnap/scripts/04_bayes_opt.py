"""Step 4: Bayesian hyperparameter optimisation of the selected model (LightGBM).

Search  : Optuna TPE sampler (Bayesian, tree-structured Parzen estimator),
          median pruner, seed fixed.
Objective: mean observation-level Top-1 accuracy over a 3-fold GroupKFold on the
          training trips (trip-grouped, label-defect trips excluded). Top-1 is
          the quantity the decoder consumes and the API reports.
After the search the best configuration is evaluated with exactly the same
5-fold protocol and student-test evaluation as the untuned model, then refitted
on all training trips and exported as models/final_model.joblib.

Outputs: results/optuna_trials.csv, results/optuna_history.png,
         results/optuna_param_importance.png, results/tuned_vs_untuned.csv,
         results/oof_train_probs.joblib (out-of-fold probabilities for decoder tuning),
         models/final_model.joblib, models/model_config.json

Run:  python scripts/04_bayes_opt.py [n_trials]
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

from pathsnap.config import CACHE_DIR, RESULTS_DIR, MODELS_DIR, SEED, CFG, STUDENT_TEST_DIR
from pathsnap.features import FEATURE_COLS
from pathsnap.metrics import row_metrics, top1_predictions, top1_accuracy, defect_trips, route_continuity
from pathsnap.model_zoo import lgbm
from pathsnap.data_io import load_trip

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 30

SPACE = {
    "n_estimators": ("int", 150, 900),
    "learning_rate": ("float_log", 0.01, 0.2),
    "num_leaves": ("int_log", 15, 255),
    "min_child_samples": ("int_log", 10, 300),
    "subsample": ("float", 0.5, 1.0),
    "colsample_bytree": ("float", 0.4, 1.0),
    "reg_lambda": ("float_log", 1e-3, 20.0),
    "reg_alpha": ("float_log", 1e-3, 10.0),
}


def suggest(trial):
    out = {}
    for k, (kind, lo, hi) in SPACE.items():
        if kind == "int":
            out[k] = trial.suggest_int(k, lo, hi)
        elif kind == "int_log":
            out[k] = trial.suggest_int(k, lo, hi, log=True)
        elif kind == "float":
            out[k] = trial.suggest_float(k, lo, hi)
        else:
            out[k] = trial.suggest_float(k, lo, hi, log=True)
    return out


def main():
    tr = joblib.load(CACHE_DIR / "train_candidates.joblib")
    trp = joblib.load(CACHE_DIR / "train_points.joblib")
    te = joblib.load(CACHE_DIR / "student_test_candidates.joblib")
    tep = joblib.load(CACHE_DIR / "student_test_points.joblib")
    bad = defect_trips(trp)
    tr = tr[~tr["trip"].isin(bad)].reset_index(drop=True)
    trp = trp[~trp["trip"].isin(bad)].reset_index(drop=True)
    X, y, g = tr[FEATURE_COLS].to_numpy(float), tr["label"].to_numpy(), tr["trip"].to_numpy()
    Xte, yte = te[FEATURE_COLS].to_numpy(float), te["label"].to_numpy()

    folds3 = list(GroupKFold(n_splits=3).split(X, y, g))

    def objective(trial):
        params = suggest(trial)
        accs = []
        for k, (a, b) in enumerate(folds3):
            m = lgbm(**params).fit(X[a], y[a])
            p = m.predict_proba(X[b])[:, 1]
            vt = tr.iloc[b][["trip", "pt", "arc"]].assign(prob=p)
            acc = top1_accuracy(trp[trp["trip"].isin(set(g[b]))], top1_predictions(vt))
            accs.append(acc)
            trial.report(float(np.mean(accs)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(accs))

    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=0),
                                study_name="pathsnap_lightgbm")
    # trial 0 = the untuned configuration, so the search starts from the baseline
    study.enqueue_trial({"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 40,
                         "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 1e-3})
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    search_s = time.time() - t0
    df = study.trials_dataframe()
    df.to_csv(RESULTS_DIR / "optuna_trials.csv", index=False)
    with open(RESULTS_DIR / "optuna_search_space.json", "w") as f:
        json.dump({"sampler": "TPE", "n_trials": N_TRIALS, "objective": "mean Top-1 accuracy, 3-fold GroupKFold",
                   "space": SPACE, "seed": SEED, "search_seconds": search_s,
                   "best_value": study.best_value, "best_params": study.best_params}, f, indent=2)
    print("best", study.best_value, study.best_params, f"search {search_s/60:.1f} min")

    done = df[df["state"] == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(done["number"], done["value"], s=14, label="completed trial")
    ax.plot(done["number"], done["value"].cummax(), c="tab:red", label="best so far")
    ax.set_xlabel("trial"); ax.set_ylabel("CV Top-1 accuracy (3-fold)")
    ax.set_title("Bayesian optimisation history (TPE)"); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "optuna_history.png", dpi=130); plt.close(fig)
    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(list(imp.keys())[::-1], list(imp.values())[::-1])
        ax.set_xlabel("importance (fANOVA)"); ax.set_title("Hyperparameter importance")
        fig.tight_layout(); fig.savefig(RESULTS_DIR / "optuna_param_importance.png", dpi=130); plt.close(fig)
    except Exception as e:
        print("importance failed", e)

    # ---------- tuned vs untuned with identical protocol ----------
    folds5 = list(GroupKFold(n_splits=CFG["cv_folds"]).split(X, y, g))
    rows = []
    oof = np.zeros(len(X))
    for label, params in (("untuned", {}), ("tuned", study.best_params)):
        f1s, accs, lls = [], [], []
        for a, b in folds5:
            m = lgbm(**params).fit(X[a], y[a])
            p = m.predict_proba(X[b])[:, 1]
            if label == "tuned":
                oof[b] = p
            rm = row_metrics(y[b], p)
            vt = tr.iloc[b][["trip", "pt", "arc"]].assign(prob=p)
            accs.append(top1_accuracy(trp[trp["trip"].isin(set(g[b]))], top1_predictions(vt)))
            f1s.append(rm["macro_f1"]); lls.append(rm["log_loss"])
        t1 = time.time()
        m = lgbm(**params).fit(X, y)
        ttrain = time.time() - t1
        t1 = time.time()
        p = m.predict_proba(Xte)[:, 1]
        tinf = time.time() - t1
        rm = row_metrics(yte, p)
        pred = top1_predictions(te[["trip", "pt", "arc"]].assign(prob=p))
        cont = np.mean([route_continuity(gp.sort_values("pt")["pred_arc"].to_numpy(),
                                         load_trip(STUDENT_TEST_DIR, tid).arcs) for tid, gp in pred.groupby("trip")])
        rows.append({"model": f"LightGBM_{label}", "params": json.dumps(params),
                     "cv_macro_f1_mean": np.mean(f1s), "cv_macro_f1_std": np.std(f1s, ddof=1),
                     "cv_top1_mean": np.mean(accs), "cv_log_loss_mean": np.mean(lls),
                     "test_top1_accuracy": top1_accuracy(tep, pred), "test_macro_f1": rm["macro_f1"],
                     "test_macro_precision": rm["macro_precision"], "test_macro_recall": rm["macro_recall"],
                     "test_log_loss": rm["log_loss"], "test_route_continuity": cont,
                     "train_time_s": ttrain, "test_inference_s": tinf,
                     "inference_ms_per_point": 1000 * tinf / len(tep)})
        print(rows[-1])
        if label == "tuned":
            joblib.dump(m, MODELS_DIR / "final_model.joblib", compress=3)
            # native LightGBM text format: loads on any LightGBM 4.x without pickle version issues
            m.booster_.save_model(str(MODELS_DIR / "final_model_lgbm.txt"))
            with open(MODELS_DIR / "model_config.json", "w") as f:
                json.dump({"model": "LightGBM (LGBMClassifier)", "params": {**lgbm(**params).get_params()},
                           "feature_columns": FEATURE_COLS, "candidates": CFG["candidates"],
                           "features": CFG["features"], "seed": SEED,
                           "trained_on": "train partition minus label-defect trips " + ",".join(bad)},
                          f, indent=2, default=str)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "tuned_vs_untuned.csv", index=False)
    joblib.dump(tr[["trip", "pt", "arc"]].assign(prob=oof), RESULTS_DIR / "oof_train_probs.joblib", compress=3)


if __name__ == "__main__":
    main()
