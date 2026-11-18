"""Step 5: HMM / Viterbi route reconstruction experiments.

1. Decoder parameters (emission weight, sigma, beta, gap power, back tolerance)
   are tuned with Optuna TPE on training trips, using out-of-fold classifier
   probabilities from step 4 so the decoder never sees in-sample scores.
2. The final classifier + decoder is evaluated on the 10 student-testing trips
   and compared with classifier-only predictions and the untuned decoder.
3. Folium maps of every student-testing trip are written to maps/.

Outputs: results/hmm_param_search.csv, models/hmm_config.json,
         results/hmm_vs_classifier_per_trip.csv, results/hmm_vs_classifier_summary.csv,
         results/hmm_vs_classifier.png, maps/pred_student_test_<id>.html

Run:  python scripts/05_hmm_decoder.py [n_trials]
      python scripts/05_hmm_decoder.py ablation   (backward steps counted as standing still;
                                                   writes results/hmm_ablation_*.csv only)
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

from pathsnap.config import TRAIN_DIR, STUDENT_TEST_DIR, RESULTS_DIR, MODELS_DIR, MAPS_DIR, SEED, CACHE_DIR
from pathsnap.data_io import list_trips, load_trip, load_route
from pathsnap.features import build_trip_table
from pathsnap.hmm import DEFAULT_PARAMS
from pathsnap.pipeline import decode_table, classify, score_trip, load_final_model
from pathsnap.metrics import defect_trips, route_continuity
from pathsnap.maps import prediction_map

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 25
N_TUNE_TRIPS = 30


def prep_train_trips():
    oof = joblib.load(RESULTS_DIR / "oof_train_probs.joblib")
    bad = set(defect_trips(joblib.load(CACHE_DIR / "train_points.joblib")))
    trips = [t for t in list_trips(TRAIN_DIR) if t not in bad]
    rng = np.random.RandomState(SEED)
    trips = sorted(rng.choice(trips, size=min(N_TUNE_TRIPS, len(trips)), replace=False).tolist())
    data = []
    for tid in trips:
        trip = load_trip(TRAIN_DIR, tid)
        route = load_route(TRAIN_DIR, tid)
        tab, info, net = build_trip_table(trip, route)
        o = oof[oof["trip"] == tid][["pt", "arc", "prob"]]
        tab = tab.merge(o, on=["pt", "arc"], how="left")
        tab["prob"] = tab["prob"].fillna(0.0)
        data.append((tid, tab, info, net))
    return data


def pooled_acc(data, params):
    correct = total = 0
    for tid, tab, info, net in data:
        out = decode_table(tid, tab, info, net, params)
        correct += int(np.sum(out["hmm_arcs"] == info["label_arc"]))
        total += info["n"]
    return correct / total


ABLATION = len(sys.argv) > 1 and sys.argv[1] == "ablation"
FIXED = {"backward_as_standstill": True} if ABLATION else {}
TAG = "hmm_ablation_" if ABLATION else "hmm_"


def tune(data):
    def objective(trial):
        params = {
            "w_clf": trial.suggest_float("w_clf", 0.2, 1.0),
            "sigma": trial.suggest_float("sigma", 4.0, 30.0, log=True),
            "beta": trial.suggest_float("beta", 1.0, 30.0, log=True),
            "beta_gap_power": trial.suggest_float("beta_gap_power", 0.0, 1.0),
            "back_tol": trial.suggest_float("back_tol", 2.0, 40.0),
            **FIXED,
        }
        return pooled_acc(data, params)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=6))
    study.enqueue_trial({k: DEFAULT_PARAMS[k] for k in ("w_clf", "sigma", "beta", "beta_gap_power", "back_tol")})
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS)
    df = study.trials_dataframe()
    df.to_csv(RESULTS_DIR / f"{TAG}param_search.csv", index=False)
    best = {**DEFAULT_PARAMS, **study.best_params, **FIXED}
    cfg = {"params": best, "default_params": DEFAULT_PARAMS, "objective": "pooled point Top-1 accuracy",
           "tuning_trips": [d[0] for d in data], "probabilities": "out-of-fold LightGBM (tuned)",
           "trials": N_TRIALS, "search_seconds": time.time() - t0,
           "default_value": float(df.loc[0, "value"]), "best_value": study.best_value}
    with open((RESULTS_DIR / "hmm_ablation_config.json") if ABLATION else (MODELS_DIR / "hmm_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("decoder best", study.best_value, best)
    return best


def evaluate(best):
    model = load_final_model(MODELS_DIR)
    rows = []
    for tid in list_trips(STUDENT_TEST_DIR):
        trip = load_trip(STUDENT_TEST_DIR, tid)
        route = load_route(STUDENT_TEST_DIR, tid)
        tab, info, net = build_trip_table(trip, None)
        t0 = time.time()
        tab["prob"] = classify(tab, model)
        tclf = time.time() - t0
        for name, params in (("hmm_default", {**DEFAULT_PARAMS, **FIXED}), ("hmm_tuned", best)):
            out = decode_table(tid, tab, info, net, params)
            s = score_trip(out, trip, route)
            rows.append({"trip": tid, "decoder": name, "n_points": info["n"], **s,
                         "hmm_breaks": out["hmm_breaks"], "classifier_s": tclf, "decode_s": out["hmm_s"]})
            if name == "hmm_tuned" and not ABLATION:
                prediction_map(trip, out, MAPS_DIR / f"pred_student_test_{tid}.html", route=route,
                               title=f"Student-test trip {tid}: point acc {s['point_accuracy_hmm']:.3f}")
        print(tid, rows[-2]["point_accuracy_classifier_only"], rows[-2]["point_accuracy_hmm"],
              rows[-1]["point_accuracy_hmm"], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"{TAG}vs_classifier_per_trip.csv", index=False)

    w = df["n_points"]
    summ = []
    t = df[df.decoder == "hmm_tuned"]
    d = df[df.decoder == "hmm_default"]
    def pooled(frame, col):
        return float((frame[col] * frame["n_points"]).sum() / frame["n_points"].sum())
    summ.append({"method": "Classifier only (argmax per point)",
                 "pooled_point_accuracy": pooled(t, "point_accuracy_classifier_only"),
                 "mean_trip_point_accuracy": t["point_accuracy_classifier_only"].mean(),
                 "route_continuity": t["route_continuity_classifier_only"].mean(),
                 "route_arc_f1": np.nan, "route_length_f1": np.nan})
    for nm, fr in (("Classifier + HMM/Viterbi (default params)", d), ("Classifier + HMM/Viterbi (tuned params)", t)):
        summ.append({"method": nm, "pooled_point_accuracy": pooled(fr, "point_accuracy_hmm"),
                     "mean_trip_point_accuracy": fr["point_accuracy_hmm"].mean(),
                     "route_continuity": fr["route_continuity_hmm"].mean(),
                     "route_arc_f1": fr["route_arc_f1"].mean(), "route_length_f1": fr["route_length_f1"].mean()})
    sdf = pd.DataFrame(summ)
    sdf.to_csv(RESULTS_DIR / f"{TAG}vs_classifier_summary.csv", index=False)
    if ABLATION:
        return
    print(sdf.to_string())

    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(t))
    ax.bar(x - 0.2, t["point_accuracy_classifier_only"], 0.4, label="classifier only")
    ax.bar(x + 0.2, t["point_accuracy_hmm"], 0.4, label="classifier + HMM (tuned)")
    ax.set_xticks(x, t["trip"], rotation=45)
    ax.set_ylim(min(0.7, t["point_accuracy_classifier_only"].min() - 0.02), 1.0)
    ax.set_ylabel("point Top-1 accuracy"); ax.set_title("Student-test trips: effect of HMM decoding")
    ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "hmm_vs_classifier.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    if ABLATION:
        N_TRIALS = 25
        evaluate(tune(prep_train_trips()))
    elif len(sys.argv) > 1 and sys.argv[1] == "eval":
        with open(MODELS_DIR / "hmm_config.json") as f:
            evaluate(json.load(f)["params"])
    else:
        best = tune(prep_train_trips())
        evaluate(best)
