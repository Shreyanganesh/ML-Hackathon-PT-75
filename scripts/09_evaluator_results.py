"""Step 9: final held-out results on the evaluator test set (reporting only).

Runs the deployed pipeline (tuned LightGBM + tuned HMM, exactly what the API
serves) on every trip in PATHSNAP_EVAL_DIR and writes the per-trip scores.
Nothing here trains, tunes or selects anything; the evaluator trips are only
predicted and then scored against their reference routes.

Run:  set PATHSNAP_EVAL_DIR to the evaluator_test folder, then
      python scripts/09_evaluator_results.py
Output: results/evaluator_test_results.csv, results/evaluator_test_summary.csv
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from pathsnap.config import EVAL_DIR, MODELS_DIR, RESULTS_DIR
from pathsnap.data_io import list_trips, load_trip, load_route
from pathsnap.pipeline import predict_trip, score_trip, load_final_model

model = load_final_model(MODELS_DIR)
hmm_params = json.load(open(MODELS_DIR / "hmm_config.json"))["params"]
rows = []
for tid in list_trips(EVAL_DIR):
    trip = load_trip(EVAL_DIR, tid)
    t0 = time.time()
    pred = predict_trip(trip, model, hmm_params)
    elapsed = time.time() - t0
    route = load_route(EVAL_DIR, tid)          # read only after prediction, for scoring
    s = score_trip(pred, trip, route)
    rows.append({"trip": tid, "n_points": pred["n_points"], **s,
                 "hmm_breaks": pred["hmm_breaks"], "prediction_s": round(elapsed, 3)})
    print(tid, pred["n_points"], round(s["point_accuracy_classifier_only"], 4), round(s["point_accuracy_hmm"], 4),
          round(s["route_length_f1"], 4), flush=True)

df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / "evaluator_test_results.csv", index=False)
n = df["n_points"].sum()
summ = pd.DataFrame([{
    "trips": len(df), "gps_points": int(n),
    "pooled_accuracy_classifier_only": (df.point_accuracy_classifier_only * df.n_points).sum() / n,
    "pooled_accuracy_hmm": (df.point_accuracy_hmm * df.n_points).sum() / n,
    "mean_route_arc_f1": df.route_arc_f1.mean(), "mean_route_length_f1": df.route_length_f1.mean(),
    "mean_route_continuity_hmm": df.route_continuity_hmm.mean(),
    "trips_improved_by_hmm": int((df.point_accuracy_hmm > df.point_accuracy_classifier_only).sum()),
    "max_prediction_s": df.prediction_s.max(),
}])
summ.to_csv(RESULTS_DIR / "evaluator_test_summary.csv", index=False)
print(summ.T)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 3.8))
x = np.arange(len(df))
ax.bar(x - 0.2, df["point_accuracy_classifier_only"], 0.4, label="classifier only")
ax.bar(x + 0.2, df["point_accuracy_hmm"], 0.4, label="classifier + HMM (tuned)")
ax.set_xticks(x, df["trip"], rotation=45)
ax.set_ylim(0.75, 1.0)
ax.set_ylabel("point Top-1 accuracy")
ax.set_title("Evaluator test trips: classifier only vs classifier + HMM")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "evaluator_test_hmm_vs_classifier.png", dpi=130)
plt.close(fig)
