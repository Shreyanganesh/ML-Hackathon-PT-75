"""Step 6: comparison plots from the recorded experiment CSVs (no retraining).

Reads results/model_comparison.csv, model_cv_folds.csv, tuned_vs_untuned.csv
and writes the comparison figures used in the report.

Run:  python scripts/06_plots.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathsnap.config import RESULTS_DIR

comp = pd.read_csv(RESULTS_DIR / "model_comparison.csv").sort_values("cv_top1_mean", ascending=False)
folds = pd.read_csv(RESULTS_DIR / "model_cv_folds.csv")
names = comp["model"].tolist()
x = np.arange(len(names))

# 1. CV macro F1 with fold spread
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.bar(x, comp["cv_macro_f1_mean"], yerr=comp["cv_macro_f1_std"], capsize=4, color="#4c78a8")
for i, n in enumerate(names):
    f = folds[folds.model == n]["macro_f1"]
    ax.scatter(np.full(len(f), i), f, s=10, c="k", zorder=3)
ax.set_xticks(x, names, rotation=30, ha="right")
ax.set_ylim(comp["cv_macro_f1_mean"].min() - 0.08, 1.0)
ax.set_ylabel("macro F1 (candidate rows)")
ax.set_title("5-fold GroupKFold CV macro F1 (bars = mean, dots = folds)")
fig.tight_layout(); fig.savefig(RESULTS_DIR / "cmp_cv_macro_f1.png", dpi=130); plt.close(fig)

# 2. student-test Top-1 and macro F1
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.bar(x - 0.2, comp["test_top1_accuracy"], 0.4, label="Top-1 accuracy (points)", color="#4c78a8")
ax.bar(x + 0.2, comp["test_macro_f1"], 0.4, label="macro F1 (candidate rows)", color="#f58518")
ax.set_xticks(x, names, rotation=30, ha="right")
ax.set_ylim(0.8, 1.0)
ax.set_title("Student-testing partition (10 trips), classifier only")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(RESULTS_DIR / "cmp_test_top1_f1.png", dpi=130); plt.close(fig)

# 3. log loss
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.bar(x - 0.2, comp["cv_log_loss_mean"], 0.4, label="CV mean", color="#54a24b")
ax.bar(x + 0.2, comp["test_log_loss"], 0.4, label="student test", color="#e45756")
ax.set_xticks(x, names, rotation=30, ha="right")
ax.set_ylabel("log loss (lower is better)"); ax.set_yscale("log")
ax.set_title("Probability quality"); ax.legend()
fig.tight_layout(); fig.savefig(RESULTS_DIR / "cmp_log_loss.png", dpi=130); plt.close(fig)

# 4. accuracy vs inference cost
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.scatter(comp["inference_ms_per_point"], comp["test_top1_accuracy"], s=40)
for _, r in comp.iterrows():
    ax.annotate(r["model"], (r["inference_ms_per_point"], r["test_top1_accuracy"]), fontsize=8,
                xytext=(4, 3), textcoords="offset points")
ax.set_xscale("log"); ax.set_xlabel("inference time per GPS point (ms, log)")
ax.set_ylabel("student-test Top-1"); ax.set_title("Accuracy vs inference cost")
fig.tight_layout(); fig.savefig(RESULTS_DIR / "cmp_accuracy_vs_time.png", dpi=130); plt.close(fig)

# 5. tuned vs untuned
tv = RESULTS_DIR / "tuned_vs_untuned.csv"
if tv.exists():
    t = pd.read_csv(tv)
    cols = [("cv_macro_f1_mean", "CV macro F1"), ("cv_top1_mean", "CV Top-1"),
            ("test_top1_accuracy", "test Top-1"), ("test_macro_f1", "test macro F1")]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    xx = np.arange(len(cols))
    for k, (_, r) in enumerate(t.iterrows()):
        ax.bar(xx + (k - 0.5) * 0.38, [r[c] for c, _ in cols], 0.38, label=r["model"])
    ax.set_xticks(xx, [l for _, l in cols]); ax.set_ylim(0.85, 0.97)
    ax.set_title("LightGBM before and after Bayesian optimisation"); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "cmp_tuned_vs_untuned.png", dpi=130); plt.close(fig)

cols = ["model", "family", "config", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_top1_mean", "train_top1",
        "test_top1_accuracy", "test_macro_f1", "test_macro_precision", "test_macro_recall", "test_log_loss",
        "test_route_continuity", "train_time_s", "inference_ms_per_point"]
comp[cols].round(4).to_csv(RESULTS_DIR / "model_comparison_table.csv", index=False)
print(comp[cols].round(4).to_string())
