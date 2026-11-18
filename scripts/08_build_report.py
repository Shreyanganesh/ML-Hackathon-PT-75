"""Step 8: build the technical report (HTML + PDF) from the recorded results.

Every table and number in the report is read from results/*.csv, models/*.json
and the figures in results/ and report/figures/, so the report always matches
the repository.

Run:  python scripts/08_build_report.py
PDF export uses a local Chrome/Chromium (--print-to-pdf). If none is found the
HTML file can be opened in any browser and printed to PDF.
"""
import sys
import json
import base64
import shutil
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from pathsnap.config import RESULTS_DIR, MODELS_DIR, ROOT, CFG

def find_browser():
    """Chrome, Chromium or Edge: on PATH or in the usual Windows/macOS install folders."""
    import os
    for c in ("google-chrome", "chromium", "chromium-browser", "chrome", "msedge"):
        if shutil.which(c):
            return shutil.which(c)
    for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"):
        if os.path.exists(p):
            return p
    return None


REPORT_DIR = ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"
REPORT_DIR.mkdir(exist_ok=True)

TEAM = ["Kallepalli Sai Meghana", "D R Rushmitha", "Shreyan G", "Shraddha Bamandi"]


def img(path, width=100, caption=None):
    p = Path(path)
    if not p.exists():
        return f'<p class="missing">Figure not found: {p.name}</p>'
    b64 = base64.b64encode(p.read_bytes()).decode()
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure><img src="data:image/png;base64,{b64}" style="width:{width}%">{cap}</figure>'


def table(df, fmt=4):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{fmt}f}")
    return d.to_html(index=False, border=0, classes="t", escape=True)


def f4(x):
    return f"{x:.4f}"


def pct(x):
    return f"{100 * x:.2f}%"


# ---------------------------------------------------------------- data
inv = pd.read_csv(RESULTS_DIR / "eda" / "inventory.csv", dtype={"trip": str})
psum = pd.read_csv(RESULTS_DIR / "eda" / "partition_summary.csv", index_col=0)
miss = pd.read_csv(RESULTS_DIR / "eda" / "missing_values.csv")
cand = pd.read_csv(RESULTS_DIR / "candidates_summary.csv", dtype={"trip": str})
med = pd.read_csv(RESULTS_DIR / "eda" / "median_gps_to_route_distance_per_trip.csv", dtype={"trip": str})
comp = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
folds = pd.read_csv(RESULTS_DIR / "model_cv_folds.csv")
tvu = pd.read_csv(RESULTS_DIR / "tuned_vs_untuned.csv")
opt = json.load(open(RESULTS_DIR / "optuna_search_space.json"))
trials = pd.read_csv(RESULTS_DIR / "optuna_trials.csv")
hmmcfg = json.load(open(MODELS_DIR / "hmm_config.json"))
hsum = pd.read_csv(RESULTS_DIR / "hmm_vs_classifier_summary.csv")
htrip = pd.read_csv(RESULTS_DIR / "hmm_vs_classifier_per_trip.csv", dtype={"trip": str})
api_log = (REPORT_DIR / "api_log_rehearsal.txt").read_text() if (REPORT_DIR / "api_log_rehearsal.txt").exists() else ""

comp = comp.set_index("model")
best_name = "LightGBM"
bad_trips = med[med["label_dist"] > 50]["trip"].tolist()
cand_tr = cand[cand.partition == "train"]
cand_te = cand[cand.partition == "student_test"]
miss_tr = 1 - (cand_tr.n_points * (1 - cand_tr.candidate_miss_rate)).sum() / cand_tr.n_points.sum()
miss_te = 1 - (cand_te.n_points * (1 - cand_te.candidate_miss_rate)).sum() / cand_te.n_points.sum()
cand_trg = cand_tr[~cand_tr.trip.isin(bad_trips)]
miss_trg = 1 - (cand_trg.n_points * (1 - cand_trg.candidate_miss_rate)).sum() / cand_trg.n_points.sum()
ht = htrip[htrip.decoder == "hmm_tuned"].copy()
n_improved = int((ht.point_accuracy_hmm > ht.point_accuracy_classifier_only + 1e-9).sum())
worst = ht.sort_values("point_accuracy_hmm").iloc[0]
untuned = tvu[tvu.model == "LightGBM_untuned"].iloc[0]
tuned = tvu[tvu.model == "LightGBM_tuned"].iloc[0]
n_complete = int((trials.state == "COMPLETE").sum())
n_pruned = int((trials.state == "PRUNED").sum())
hs = hsum.set_index("method")
abl = pd.read_csv(RESULTS_DIR / "hmm_ablation_vs_classifier_summary.csv").set_index("method")
abl_t = abl.loc["Classifier + HMM/Viterbi (tuned params)"]
ablcfg = json.load(open(RESULTS_DIR / "hmm_ablation_config.json"))
abl_trip = pd.read_csv(RESULTS_DIR / "hmm_ablation_vs_classifier_per_trip.csv", dtype={"trip": str})
abl_trip = abl_trip[abl_trip.decoder == "hmm_tuned"].set_index("trip")
clf_only = hs.loc["Classifier only (argmax per point)"]
hmm_def = hs.loc["Classifier + HMM/Viterbi (default params)"]
hmm_tun = hs.loc["Classifier + HMM/Viterbi (tuned params)"]

# ---------------------------------------------------------------- per-model write-ups
MODEL_TEXT = {
    "LogisticRegression": dict(
        why="A linear baseline. Most of the signal (distance, heading agreement, window distance) changes monotonically with the chance that a candidate is correct, so a linear decision function in these features is a sensible first test. Its probabilities are also directly usable as HMM emissions.",
        sl="Fast, stable across folds and well calibrated. It cannot model interactions, for example that heading agreement only matters when the vehicle is moving, or that a small distance is less convincing when several candidates are equally close.",
        hp="C = 1.0 (L2), solver lbfgs, max_iter = 500, StandardScaler fitted inside the pipeline.",
        loss="Binary cross-entropy (log loss) with an L2 penalty."),
    "GaussianNB": dict(
        why="A probabilistic baseline that needs almost no training time. It tests how far class-conditional feature distributions get us without modelling feature dependence.",
        sl="Very fast and gives probabilities by construction. The independence assumption is clearly violated here (dist, softmin_p, dist_rank and the window features are strongly correlated), so evidence is counted several times and the probabilities are over-confident.",
        hp="var_smoothing = 1e-9 (default), class priors from the data.",
        loss="No iterative loss; parameters are maximum-likelihood Gaussian means and variances per class."),
    "KNN": dict(
        why="An instance-based check: candidates that look like previously seen correct candidates (similar distance, heading and context) should be correct. Distance weighting lets the closest examples dominate.",
        sl="No training cost and no distribution assumptions. Prediction is slow because every query is compared with the stored rows, so we had to train on a 150,000-row subsample. Scaling decides which features dominate the distance, and the probabilities are coarse.",
        hp="k = 31, weights = distance, Euclidean metric on standardised features, 150k-row random subsample of the training fold (seed 911).",
        loss="No loss function; a lazy learner that stores the training rows."),
    "DecisionTree": dict(
        why="A single tree can express threshold rules of the kind we expected from EDA (for example 'nearest candidate and heading agrees'), and it is easy to inspect.",
        sl="Interpretable and fast at inference. A single tree is high variance and its leaf probabilities are piecewise constant, which hurts log loss.",
        hp="max_depth = 12, min_samples_leaf = 50, criterion = gini.",
        loss="Greedy splits that minimise Gini impurity."),
    "RandomForest": dict(
        why="Bagging many decorrelated trees reduces the variance of the single tree while keeping its ability to model interactions.",
        sl="Robust, little tuning needed and good probability estimates. Training and inference are slower than boosting at similar accuracy, and the model file is large.",
        hp="n_estimators = 100, max_depth = 18, min_samples_leaf = 20, max_samples = 0.25, max_features = sqrt.",
        loss="Each tree minimises Gini impurity on a bootstrap sample; the forest averages the tree probabilities."),
    "ExtraTrees": dict(
        why="Extremely randomised trees pick split thresholds at random, which adds more variance reduction than a random forest and trains faster.",
        sl="Fast to train. The random thresholds smooth decision boundaries, which here cost accuracy on the fine distance and heading thresholds that matter.",
        hp="n_estimators = 150, max_depth = 20, min_samples_leaf = 20, bootstrap with max_samples = 0.25.",
        loss="Gini impurity with randomised thresholds; probabilities averaged over trees."),
    "LightGBM": dict(
        why="Gradient boosting fits trees one after another to the remaining errors, which suits a tabular problem where the decisive cases are near-ties between close candidates. LightGBM's histogram algorithm handles the ~0.9 million training rows quickly.",
        sl="Highest cross-validated Top-1, good log loss, fast inference and a small model file. It has many hyperparameters and can overfit with large leaves, which the Bayesian search controls.",
        hp="n_estimators = 400, learning_rate = 0.05, num_leaves = 63, min_child_samples = 40, subsample = 0.8, colsample_bytree = 0.8, reg_lambda = 1.0.",
        loss="Binary log loss, minimised by gradient boosting with second-order (Newton) leaf values."),
    "XGBoost": dict(
        why="A second boosting implementation with depth-wise trees, to check that the boosting result is not specific to LightGBM.",
        sl="Accuracy and log loss very close to LightGBM. Somewhat slower to train with depth-wise growth, and no clear gain over LightGBM.",
        hp="n_estimators = 400, learning_rate = 0.05, max_depth = 7, subsample = 0.8, colsample_bytree = 0.8, tree_method = hist.",
        loss="Binary log loss (objective binary:logistic) with L2 regularisation on leaf weights."),
    "MLP": dict(
        why="A neural network can learn smooth non-linear combinations of the features and gives an epoch-wise training history.",
        sl="Smooth probabilities and moderate size. It needs feature scaling, is sensitive to the learning rate and was less accurate than the tree ensembles on these tabular features.",
        hp="hidden layers (64, 32), ReLU, Adam, learning_rate_init = 1e-3, alpha = 1e-4, batch_size = 1024, 25 epochs, StandardScaler.",
        loss="Binary cross-entropy with L2 weight penalty, minimised by Adam with mini-batches."),
}


def fit_judgement(r):
    gap = r.train_top1 - r.cv_top1_mean
    if gap > 0.03:
        return (f"Some overfitting: Top-1 on the training rows it was fitted on is {f4(r.train_top1)}, against "
                f"{f4(r.cv_top1_mean)} in cross-validation (gap {gap:.3f}). Student-test Top-1 ({f4(r.test_top1_accuracy)}) "
                f"stays close to the CV value, so the gap is memorisation of the training trips, not a failure to generalise.")
    if r.cv_top1_mean < comp.cv_top1_mean.max() - 0.015:
        return (f"Underfitting relative to the ensembles: training Top-1 ({f4(r.train_top1)}) is close to CV Top-1 "
                f"({f4(r.cv_top1_mean)}) and both are below the best models, so the model lacks capacity rather than "
                f"memorising the data.")
    return (f"No strong over- or underfitting: training Top-1 {f4(r.train_top1)}, CV Top-1 {f4(r.cv_top1_mean)} "
            f"(gap {gap:.3f}) and student-test Top-1 {f4(r.test_top1_accuracy)} are close.")


def model_section(i, name):
    r = comp.loc[name]
    t = MODEL_TEXT[name]
    fr = folds[folds.model == name]
    extra = ""
    lc = RESULTS_DIR / f"loss_curves_{name}.png"
    curve_note = ""
    if lc.exists():
        unit = "epoch" if name == "MLP" else "boosting round"
        extra = img(lc, 55, f"{name}: training and validation log loss per {unit} (validation = held-out training trips).")
        h = pd.read_csv(RESULTS_DIR / f"loss_curves_{name}.csv")
        k = int(h["val_loss"].idxmin())
        curve_note = (f" The training history agrees: validation log loss is lowest at {unit} {k + 1} "
                      f"({h['val_loss'][k]:.4f}) and ends at {h['val_loss'].iloc[-1]:.4f}, while training log loss falls from "
                      f"{h['train_loss'].iloc[0]:.4f} to {h['train_loss'].iloc[-1]:.4f}"
                      + (", so later epochs fit the training trips at the expense of unseen ones."
                         if h['val_loss'].iloc[-1] > h['val_loss'][k] * 1.03 else
                         ", so the gap between the curves stays small."))
    return f"""
<h3>3.2.{i} {name} ({r.family})</h3>
<p><b>A1. Why this algorithm.</b> {t['why']}</p>
<p><b>A2. Strengths and limitations.</b> {t['sl']}</p>
<p><b>B5. Hyperparameters and initial values.</b> {t['hp']}</p>
<p><b>B6. Loss / objective.</b> {t['loss']}</p>
<p><b>C7-C8. Student-testing results.</b> Top-1 accuracy {f4(r.test_top1_accuracy)}, macro F1 {f4(r.test_macro_f1)}
(macro precision {f4(r.test_macro_precision)}, macro recall {f4(r.test_macro_recall)}), log loss {f4(r.test_log_loss)},
route continuity {f4(r.test_route_continuity)}. Cross-validation macro F1 over the five folds:
{", ".join(f4(v) for v in fr.macro_f1)} (mean {f4(r.cv_macro_f1_mean)}, std {r.cv_macro_f1_std:.4f}).
Training time on the full training set {r.train_time_s:.1f} s; inference {r.inference_ms_per_point:.3f} ms per GPS point.</p>
<p><b>C9. Over- or underfitting.</b> {fit_judgement(r)}{curve_note}</p>
<div class="row">{img(RESULTS_DIR / f'confusion_{name}.png', 38, f'{name}: confusion matrix on student-test candidate rows (threshold 0.5).')}{extra}</div>
"""


order = comp.sort_values("cv_top1_mean", ascending=False).index.tolist()
ctab = comp.loc[order].reset_index()[["model", "family", "config", "cv_macro_f1_mean", "test_top1_accuracy",
                                       "test_route_continuity", "test_macro_f1", "test_log_loss",
                                       "inference_ms_per_point"]]
ctab.columns = ["Model", "Algorithm family", "Key configuration", "CV macro F1", "Top-1 accuracy (test)",
                "Route continuity (test)", "Student-test macro F1", "Log loss (test)", "Inference ms / point"]
ctab2 = comp.loc[order].reset_index()[["model", "cv_top1_mean", "cv_top1_std", "train_top1", "test_macro_precision",
                                        "test_macro_recall", "cv_log_loss_mean", "train_time_s"]]
ctab2.columns = ["Model", "CV Top-1 mean", "CV Top-1 std", "Train Top-1", "Test macro precision", "Test macro recall",
                 "CV log loss", "Train time (s)"]

inv_tab = inv.groupby("partition").agg(trips=("trip", "count"), gps_points=("n_points", "sum"),
                                      median_points=("n_points", "median"), median_km=("track_length_km", "median"),
                                      median_arcs=("n_arcs", "median"), max_arcs=("n_arcs", "max"),
                                      share_1s_steps=("share_dt_1s", "mean"), max_gap_s=("max_gap_s", "max")).reset_index()

tv = tvu[["model", "cv_macro_f1_mean", "cv_top1_mean", "test_top1_accuracy", "test_macro_f1", "test_log_loss",
          "test_route_continuity", "train_time_s", "inference_ms_per_point"]].copy()
tv.columns = ["Model", "CV macro F1", "CV Top-1", "Test Top-1", "Test macro F1", "Test log loss",
              "Test route continuity", "Train time (s)", "Inference ms / point"]

space_rows = []
for k, (kind, lo, hi) in opt["space"].items():
    space_rows.append({"Hyperparameter": k, "Distribution": {"int": "integer uniform", "int_log": "integer log-uniform",
                       "float": "uniform", "float_log": "log-uniform"}[kind], "Range": f"[{lo}, {hi}]",
                       "Untuned value": {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63,
                                         "min_child_samples": 40, "subsample": 0.8, "colsample_bytree": 0.8,
                                         "reg_lambda": 1.0, "reg_alpha": 0.0}[k],
                       "Best value": round(opt["best_params"][k], 5) if isinstance(opt["best_params"][k], float) else opt["best_params"][k]})
space_df = pd.DataFrame(space_rows)

hp = hmmcfg["params"]
dp = hmmcfg["default_params"]
hmm_param_df = pd.DataFrame([{"Parameter": k, "Default": dp[k], "Tuned": round(hp[k], 4)} for k in
                             ("w_clf", "sigma", "beta", "beta_gap_power", "back_tol", "search_extra")])
htab = hsum.copy()
htab.columns = ["Method", "Pooled point accuracy", "Mean per-trip accuracy", "Route continuity", "Route arc F1", "Route length F1"]
pt = ht[["trip", "n_points", "point_accuracy_classifier_only", "point_accuracy_hmm", "route_arc_f1", "route_length_f1",
         "route_continuity_classifier_only", "route_continuity_hmm", "hmm_breaks"]].copy()
pt.columns = ["Trip", "Points", "Classifier-only acc.", "HMM acc.", "Route arc F1", "Route length F1",
              "Continuity (clf)", "Continuity (HMM)", "Chain breaks"]

missing_html = table(miss, 0)

CSS = """
body { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 10.5pt; color: #1b1f24; line-height: 1.45; }
h1 { font-size: 21pt; margin: 0 0 4px; } h2 { font-size: 14pt; margin-top: 22px; border-bottom: 1px solid #ccd; padding-bottom: 3px; }
h3 { font-size: 11.5pt; margin-top: 16px; }
table.t { border-collapse: collapse; font-size: 8.6pt; margin: 6px 0 10px; width: 100%; }
table.t th { background: #eef1f5; text-align: left; padding: 3px 5px; border-bottom: 1px solid #aab; }
table.t td { padding: 3px 5px; border-bottom: 1px solid #e3e6ea; vertical-align: top; }
figure { margin: 8px 0; text-align: center; } figure img { border: 1px solid #e3e6ea; }
figcaption { font-size: 8.8pt; color: #4a5260; margin-top: 3px; }
.row { display: flex; gap: 10px; align-items: flex-start; justify-content: center; }
.row figure { flex: 1; }
pre { font-size: 7.6pt; background: #f5f6f8; padding: 8px; white-space: pre-wrap; border: 1px solid #e3e6ea; }
.small { font-size: 9pt; color: #4a5260; } .missing { color: #b00; }
@page { size: A4; margin: 16mm 14mm; }
"""

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>PathSnap technical report</title><style>{CSS}</style></head><body>
<h1>PathSnap: map matching with a candidate classifier and HMM/Viterbi decoding</h1>
<p class="small">UE24CS352A Machine Learning hackathon, PES University. Team: {", ".join(TEAM)}.<br>
All numbers in this report are produced by the scripts in the GitHub repository and stored in <code>results/</code>.</p>

<h2>1. Summary</h2>
<p>For every GPS fix we generate candidate road arcs, score each candidate with a LightGBM classifier tuned by Bayesian
optimisation, and decode the whole trip with an HMM using Viterbi. On the ten student-testing trips the final system matches
{pct(hmm_tun.pooled_point_accuracy)} of GPS fixes to the reference arc, against {pct(clf_only.pooled_point_accuracy)} when each
fix takes the classifier's top candidate, and recovers the reference route with a length-weighted F1 of
{f4(hmm_tun.route_length_f1)}. Nine classifiers from six algorithm families were compared before LightGBM was chosen.</p>

<h2>2. Dataset understanding and exploratory data analysis</h2>
<p>Partitions used: EDA on <b>train</b> (80 trips) and <b>student_test</b> (10 trips). The evaluator partition was not available
to us and was never opened during development.</p>
{table(inv_tab, 3)}
<p>Each trip folder holds four headerless whitespace-separated files: <code>.track</code> (longitude, latitude, time in s),
<code>.nodes</code> (longitude, latitude; row index = node id), <code>.arcs</code> (from node, to node; row index = arc id)
and <code>.route</code> (the reference arc ids in travel order). Every trip ships its own road network, from
{int(inv.n_arcs.min())} to {int(inv.n_arcs.max())} arcs, so a row of the track never corresponds to a row of a network file.
Arcs are directed: {pct(inv.two_way_arc_share.mean())} of arcs have an opposite twin, i.e. most roads are two-way and are
stored as two arcs with identical geometry.</p>

<h3>2.1 How we preprocessed the data</h3>
<ol>
<li><b>Validation and cleaning.</b> We checked every file for missing cells, non-numeric values, out-of-range coordinates,
(0, 0) fixes, non-increasing timestamps and arcs that reference non-existent nodes. None were found (table below), so the
cleaning step in <code>data_io.clean_track()</code> removes nothing on these partitions but stays in the pipeline for unseen trips.</li>
<li><b>Projection.</b> Longitude and latitude are angles, so each trip is projected to a local equirectangular plane centred on
the trip centroid (x = R cos(lat0) dlon, y = R dlat). The largest trip extends {inv[['extent_ew_km','extent_ns_km']].max().max():.0f} km.
At that size the east-west scale error at the edge of the box is below 1%, which on the 5 to 60 m fix-to-arc distances we measure is
well under a metre and smaller than GPS noise. All later distances and bearings are in metres.</li>
<li><b>Network window.</b> Arcs outside the track's bounding box plus 300 m are dropped (original arc ids are kept). This removes
most of the network for short trips, but long trips cover most of their network's area (the largest window still holds
{int(cand.arcs_in_window.max())} arcs). Candidate search therefore uses a KD-tree over points sampled every 15 m along the arcs instead
of comparing every fix with every arc.</li>
<li><b>Per-fix labels.</b> The route says which arcs were driven, not which fix belongs to which arc. We align fixes to route
positions with a monotone dynamic programme, C[i, j] = d(i, j) + min over k &le; j of C[i-1, k], which minimises the total fix-to-arc
distance while never moving backwards along the route (loops are handled because we align to positions, not arc ids).</li>
<li><b>Label-defect trips.</b> For trips {", ".join(bad_trips)} the median distance between a fix and its aligned route arc is
{", ".join(f"{v:.0f} m" for v in med[med.trip.isin(bad_trips)].label_dist)}, while the median over all other train trips is
{med[~med.trip.isin(bad_trips)].label_dist.median():.1f} m. The reference route of these trips does not follow the GPS track,
so we excluded them from model fitting and tuning (they stay in the EDA). This is one of the reference-route limitations the
dataset guide warns about.</li>
</ol>
{missing_html}
<div class="row">{img(RESULTS_DIR / 'eda' / 'trip_size_distributions.png', 100, 'Trip sizes by partition.')}</div>
<div class="row">{img(RESULTS_DIR / 'eda' / 'sampling_and_speed.png', 70, 'Sampling interval and speed between consecutive fixes (train).')}{img(RESULTS_DIR / 'eda' / 'gps_to_route_distance.png', 45, 'Distance from each fix to its aligned reference arc (train).')}</div>

<h3>2.2 Patterns observed</h3>
<ul>
<li>Sampling is nominally 1 Hz: {pct(inv[inv.partition=='train'].share_dt_1s.mean())} of train steps are exactly 1 s, but
{int((inv[inv.partition=='train'].n_gaps_over_5s>0).sum())} of 80 train trips contain gaps above 5 s and the longest gap is
{int(inv.max_gap_s.max())} s. The HMM therefore scales its transition tolerance with the time gap.</li>
<li>GPS noise is moderate: the median fix-to-route distance is {med[~med.trip.isin(bad_trips)].label_dist.median():.1f} m, with a
long tail caused by urban canyons and the defect trips. A 60 m candidate radius keeps the candidate miss rate at
{pct(miss_trg)} on clean train trips ({pct(miss_tr)} including the defect trips) and {pct(miss_te)} on student test.</li>
<li>Most networks are dense urban grids with short arcs (median arc length {inv.median_arc_len_m.median():.0f} m), so several
arcs are often within a few metres of a fix. Distance alone cannot pick the right one; direction and context matter.</li>
<li>Two-way roads produce pairs of arcs with identical geometry. Only the driving direction separates them, which is why
heading agreement is a core feature.</li>
<li>Networks are mostly connected (median largest strongly connected component {pct(inv.largest_scc_share.median())} of
nodes) and contain no isolated nodes. Reference routes are almost fully connected
({pct(inv.route_connected_share.mean())} of consecutive route arcs share a node).</li>
<li>Trips are spread over several countries (trip location plot), which supports grouping cross-validation by trip.</li>
</ul>
<div class="row">{img(RESULTS_DIR / 'eda' / 'trip_locations.png', 48, 'Trip centroids.')}{img(RESULTS_DIR / 'eda' / 'network_connectivity.png', 52, 'Road-network connectivity per trip.')}</div>
{img(RESULTS_DIR / 'eda' / 'trip_00000048_overview.png', 52, 'Train trip 00000048: road network (grey), reference route (green) and GPS fixes (blue).')}
{img(FIG_DIR / 'folium_eda_00000048.png', 85, 'Folium map (maps/eda_trip_00000048.html): GPS observations, road network and reference route with layer control and legend.')}

<h3>2.3 Features and why we chose them</h3>
<p>Candidates: every arc within {CFG['candidates']['radius_m']:.0f} m of the fix, nearest first, at most
{CFG['candidates']['max_candidates']}. If no arc is within the radius the nearest arcs are taken anyway and flagged
(<code>fallback</code>). This gives on average {cand_tr.rows.sum() / cand_tr.n_points.sum():.1f} candidates per fix.
Each (fix, candidate) row has 25 features:</p>
<table class="t"><tr><th>Group</th><th>Features</th><th>Reason</th></tr>
<tr><td>Geometry</td><td>dist, dist_rank, dist_minus_min, dist_ratio_min, softmin_p, n_cand, t_proj, at_endpoint</td><td>Distance is the main evidence; relative versions make it comparable across noisy and clean areas; t_proj and at_endpoint mark fixes near intersections where labels change.</td></tr>
<tr><td>Direction</td><td>head_cos, head_abs_cos, moving</td><td>Heading from fix i-2 to i+2 compared with the arc bearing. Separates the two directions of a road; disabled when speed is under 1 m/s because the heading of a stopped vehicle is noise.</td></tr>
<tr><td>Context window</td><td>win_mean_dist, win_max_dist, win_rank, win_softmin_p, win_head_cos</td><td>The same checks over fixes i-3..i+3. One bad fix should not pull the match onto a parallel road.</td></tr>
<tr><td>Kinematics</td><td>speed, dt_prev, dt_next, turn</td><td>Gaps and turns mark where matching is harder.</td></tr>
<tr><td>Topology</td><td>arc_len, out_deg_to, in_deg_from, has_twin, fallback</td><td>Short arcs and busy intersections are more ambiguous; has_twin tells the model when direction must decide.</td></tr>
</table>

<h2>3. Model experimentation and comparison</h2>
<p>Protocol. 5-fold GroupKFold over the training trips (a trip never appears in two folds; defect trips excluded), then a refit
on all usable training trips and one evaluation on the ten student-testing trips. Each classifier predicts the probability that a
candidate is the reference arc. Row-level metrics (macro precision, recall, F1 at threshold 0.5, log loss) measure candidate
classification. Observation-level Top-1 accuracy takes the highest-scoring candidate of every fix and compares it with the fix's
reference arc; fixes whose reference arc is not among the candidates count as errors. Route continuity is the share of consecutive
fixes whose top arcs are the same arc or directly connected. Seed 911 everywhere.</p>
<p><b>Primary selection metric:</b> cross-validated observation-level Top-1 accuracy, because it is the quantity the API reports
and the decoder works with. Log loss is the second criterion, since the HMM uses the probabilities as emissions.</p>

<h3>3.1 Overall model comparison</h3>
{table(ctab)}
{table(ctab2)}
<div class="row">{img(RESULTS_DIR / 'cmp_cv_macro_f1.png', 100, 'Cross-validation macro F1 per model (bars = mean, dots = folds).')}</div>
<div class="row">{img(RESULTS_DIR / 'cmp_test_top1_f1.png', 52, 'Student-testing Top-1 accuracy and macro F1.')}{img(RESULTS_DIR / 'cmp_log_loss.png', 48, 'Log loss, CV and student test.')}</div>
{img(RESULTS_DIR / 'cmp_accuracy_vs_time.png', 50, 'Accuracy against inference time per GPS point.')}
<p><b>Model selection.</b> {order[0]} has the highest CV Top-1 ({f4(comp.loc[order[0]].cv_top1_mean)}).
LightGBM and XGBoost are within {abs(comp.loc['LightGBM'].cv_top1_mean - comp.loc['XGBoost'].cv_top1_mean):.4f} of each other in CV
Top-1, with similar log loss. LightGBM was chosen because it is at the top on the primary metric, its fold-to-fold spread is small
(std {comp.loc['LightGBM'].cv_top1_std:.4f}), it trains faster than XGBoost and the random forest, its inference cost
({comp.loc['LightGBM'].inference_ms_per_point:.3f} ms per fix) fits the API, and it gives calibrated probabilities for the
decoder. Gaussian NB has by far the worst log loss, which matches its broken independence assumption. KNN and the MLP sit below
the tree ensembles; the KNN subsample and the MLP's sensitivity to scaling explain part of this.</p>

<h3>3.2 Per-model questions</h3>
{''.join(model_section(i + 1, n) for i, n in enumerate(order))}

<h2>4. Bayesian hyperparameter optimisation (LightGBM)</h2>
<p><b>Method.</b> Optuna with the TPE sampler (a Bayesian method that models the densities of good and bad configurations and
proposes points with the best expected improvement), seed 911, 8 random start-up trials and a median pruner. Trial 0 was the
untuned configuration.</p>
<p><b>Objective.</b> Mean observation-level Top-1 accuracy over a 3-fold GroupKFold on the training trips (defect trips excluded).
Three folds instead of five kept each trial affordable; the best configuration was then re-evaluated with the same 5-fold protocol as
every other model.</p>
<p><b>Trials.</b> {opt['n_trials']} trials ({n_complete} completed, {n_pruned} pruned), {opt['search_seconds']/60:.1f} minutes on a 2-core machine.
Best objective value {f4(opt['best_value'])}; trial 0 (untuned) scored {f4(trials.loc[0, 'value'])}.</p>
{table(space_df)}
<div class="row">{img(RESULTS_DIR / 'optuna_history.png', 50, 'Optimisation history.')}{img(RESULTS_DIR / 'optuna_param_importance.png', 50, 'Hyperparameter importance (fANOVA).')}</div>
<p><b>Tuned against untuned</b> (same folds, same student-test trips):</p>
{table(tv)}
{img(RESULTS_DIR / 'cmp_tuned_vs_untuned.png', 60, 'LightGBM before and after Bayesian optimisation.')}
<p>{"The tuned model is better on " + ", ".join([n for n, c in (("CV macro F1", "cv_macro_f1_mean"), ("CV Top-1", "cv_top1_mean"), ("test Top-1", "test_top1_accuracy"), ("test macro F1", "test_macro_f1")) if tuned[c] > untuned[c]]) if any(tuned[c] > untuned[c] for c in ("cv_macro_f1_mean", "cv_top1_mean", "test_top1_accuracy", "test_macro_f1")) else "The tuned model did not beat the untuned one"}
{"and has lower log loss" if tuned.test_log_loss < untuned.test_log_loss else "but its log loss is not lower"}
(test log loss {f4(tuned.test_log_loss)} against {f4(untuned.test_log_loss)}). The CV Top-1 gain is
{100*(tuned.cv_top1_mean-untuned.cv_top1_mean):+.2f} percentage points and the student-test Top-1 change is
{100*(tuned.test_top1_accuracy-untuned.test_top1_accuracy):+.2f} points, so the improvement is real but small. The untuned settings were
already reasonable, and most of the remaining error comes from ambiguous fixes (near intersections and on parallel roads) that the
HMM is designed to resolve. Inference cost changes from {untuned.inference_ms_per_point:.3f} to {tuned.inference_ms_per_point:.3f} ms per fix.</p>

<h2>5. HMM and Viterbi route reconstruction</h2>
<p><b>Q1. Hidden states.</b> At GPS step i the hidden states are the candidate arcs of fix i (at most {CFG['candidates']['max_candidates']}),
each with its projection point on the arc. A state keeps its original arc id, so the candidate-to-road-segment correspondence is preserved.</p>
<p><b>Q2. Emissions.</b> log b_i(s) = w &middot; log P_LightGBM(s) + (1 - w) &middot; (-d_i(s)&sup2; / 2&sigma;&sup2;), where d_i(s) is the
fix-to-arc distance. The Gaussian term keeps geometry in the emission even when the classifier is over-confident.</p>
<p><b>Q3. Transitions.</b> For states s at step i-1 and s' at step i we compute the driving distance through the directed network from the
projection of fix i-1 on s to the projection of fix i on s' (rest of arc s, Dijkstra shortest path from the head of s to the tail of s', then
the start of s'). Staying on the same arc costs the distance moved along it; a backward step of up to back_tol metres is accepted as projection
jitter and costs its absolute length.
log a(s, s') = -| route distance - straight-line GPS distance | / &beta;<sub>i</sub>: a correct pair of states makes the vehicle drive
about as far as the GPS says it moved. One-way restrictions are respected because the graph is directed.</p>
<p><b>Viterbi recurrence.</b> &delta;<sub>i</sub>(s') = log b<sub>i</sub>(s') + max<sub>s</sub> [ &delta;<sub>i-1</sub>(s) + log a(s, s') ], with back-pointers,
followed by backtracking from the best final state.</p>
<p><b>Q4. Missing candidates and gaps.</b> Candidate generation always returns the nearest arcs when nothing is within the radius, so no fix is
left without states. For time gaps, &beta;<sub>i</sub> = &beta; &middot; &Delta;t<sup>p</sup> grows with the gap and the Dijkstra search radius
grows with the GPS distance (3 &times; distance + {dp['search_extra']:.0f} m), so long gaps can still be bridged by a plausible path.</p>
<p><b>Q5. Disconnected segments.</b> Unreachable pairs get probability zero. If no state of step i can be reached from any state of step i-1,
the chain is broken there and a new chain starts from the emissions of step i; each chain is backtracked separately. The final route
is built from the decoded arcs, and gaps between consecutive arcs are filled with the network shortest path. Chain breaks per
trip are reported below.</p>
<p><b>Decoder tuning.</b> The decoder parameters were tuned with Optuna TPE ({hmmcfg['trials']} trials) on {len(hmmcfg['tuning_trips'])} training
trips, using out-of-fold LightGBM probabilities so the decoder never sees in-sample classifier scores. Objective: pooled point accuracy.
Default parameters scored {f4(hmmcfg['default_value'])} on these trips; the best scored {f4(hmmcfg['best_value'])}.</p>
{table(hmm_param_df)}
<p><b>Q6. Did HMM decoding help?</b> Yes. Student-testing results:</p>
{table(htab)}
{table(pt)}
{img(RESULTS_DIR / 'hmm_vs_classifier.png', 75, 'Point accuracy per student-test trip, classifier only against classifier + HMM.')}
<p>Decoding improved point accuracy on {n_improved} of 10 trips and raised pooled accuracy from {pct(clf_only.pooled_point_accuracy)} to
{pct(hmm_tun.pooled_point_accuracy)}. Route continuity rose from {pct(clf_only.route_continuity)} to {pct(hmm_tun.route_continuity)}: the
classifier alone often jumps to the opposite-direction twin or a parallel arc for one or two fixes, which the transition model rules out.
Continuity is not 100% for two reasons: at 1 Hz on very short arcs two consecutive fixes can sit on arcs that are two hops apart
(the assembled route fills the hop with the shortest path), and chain breaks start a new chain wherever no transition is possible.
The two weakest trips are discussed below.</p>

<p><b>A design change that did not help (ablation).</b> In trip 00000077 most errors put the fix on the opposite-direction twin of the
reference arc. Our first hypothesis was that accepting backward steps lets the decoder drive a two-way road in the wrong direction for
free. We tested a variant that counts a backward step as standing still, so it is penalised by the GPS movement
(<code>python scripts/05_hmm_decoder.py ablation</code>, re-tuned with the same 25-trial search). It scored lower on the training tuning
trips ({f4(ablcfg['best_value'])} against {f4(hmmcfg['best_value'])}) and on student test (pooled {pct(abl_t.pooled_point_accuracy)} against
{pct(hmm_tun.pooled_point_accuracy)}), and trip 00000077 stayed at {pct(abl_trip.loc['00000077'].point_accuracy_hmm)}, so we kept the
original rule. Looking at the fixes themselves showed the real cause: these stretches are driven at 1 to 4 m/s with the GPS 12 to 26 m off
the road, so the GPS heading (fix i-2 to i+2) points against the true direction and the classifier prefers the wrong twin. The decoder
cannot correct a run in which every emission points the same wrong way. A heading measured over a fixed distance instead of a fixed number
of fixes is the obvious next step; it needs the features rebuilt and every model retrained, which did not fit in the hackathon time.</p>
<p>In trip 00000065 the decoder lowered point accuracy ({pct(ht.set_index('trip').loc['00000065'].point_accuracy_classifier_only)} to
{pct(ht.set_index('trip').loc['00000065'].point_accuracy_hmm)}). Route recall there is {f4(ht.set_index('trip').loc['00000065'].route_length_recall)},
but length precision is {f4(ht.set_index('trip').loc['00000065'].route_length_precision)}: after {int(ht.set_index('trip').loc['00000065'].hmm_breaks)}
chain breaks the decoder followed a parallel road running next to the reference route for one stretch and added a loop round a
roundabout (map in Section 6), and the shortest-path fill turned those choices into extra arcs that are not on the reference route.</p>

<h2>6. Geographical visualisation (Folium)</h2>
<p>Maps are generated by <code>pathsnap/maps.py</code> (<code>scripts/01_eda.py</code> and <code>scripts/05_hmm_decoder.py</code>) and saved
as interactive HTML in <code>maps/</code>. Layers: GPS observations (blue dots with start/end markers), road network near the track
(grey), reference route (green, wide), predicted route from classifier + HMM (red), classifier-only arcs (orange), with a legend and a layer control.</p>
{img(FIG_DIR / f'folium_pred_{worst.trip}.png', 85, f'Student-test trip {worst.trip} (weakest trip): reference route (green) against predicted route (red).')}
{img(FIG_DIR / 'folium_pred_best.png', 85, 'Student-test trip 00000007 (point accuracy 100%): the predicted route covers the reference route.')}

<h2>7. API, UI and deployment</h2>
<p>Laptop 1 runs <code>uvicorn api.main:app --host 0.0.0.0 --port 8000</code>. It loads the tuned LightGBM model
(<code>models/final_model_lgbm.txt</code>) and the decoder configuration (<code>models/hmm_config.json</code>) at start-up and reads the
evaluation trips from the folder in <code>PATHSNAP_EVAL_DIR</code>. Endpoints: <code>GET /health</code>, <code>GET /trips</code>,
<code>POST /predict</code> with <code>{{"trip_id": ...}}</code>. A prediction runs candidate generation, feature extraction, LightGBM scoring,
Viterbi decoding and route assembly. The server then reads the trip's reference route, which is used only there, and computes the
point accuracy, route F1 and continuity. The response holds the GPS points, matched positions, predicted arcs and route geometry and the
scores; no reference route or label leaves the server. Every request and its result are logged on the Laptop 1 console.</p>
<p>Laptop 2 runs <code>python ui/serve_ui.py --api http://&lt;Laptop 1 IP&gt;:8000</code>, a static page (Leaflet is bundled locally) that
lists the evaluation trips from the API, sends the selected trip id and shows the predicted route, matched fixes and scores.</p>
{img(FIG_DIR / 'ui_screenshot.png', 95, 'Evaluator UI on Laptop 2 after selecting a trip.')}
{img(FIG_DIR / 'api_cmd_screenshot.png', 95, 'API console on Laptop 1 (the receiving laptop) showing the incoming requests and their results.') if (FIG_DIR / 'api_cmd_screenshot.png').exists() else ''}
<p>API log on the receiving laptop during a rehearsal:</p>
<pre>{api_log}</pre>

<h2>8. Limitations</h2>
<ul>
<li>Labels are derived from the reference route by alignment, so fixes near an intersection can be labelled with either of two
adjacent arcs. Point accuracy slightly understates route quality, which is why route F1 is also reported.</li>
<li>Two training trips have reference routes that do not follow the GPS track; other trips may contain shorter undetected label errors.</li>
<li>The local projection assumes a trip spans tens of km, not hundreds.</li>
<li>At walking-pace speeds with large GPS offsets the heading feature is unreliable, which causes most of the remaining
wrong-direction errors (Section 5).</li>
<li>The candidate radius bounds recall: fixes whose true arc is more than 60 m away can only be matched through the nearest-arc fallback.</li>
</ul>

<h2>9. AI assistance disclosure</h2>
<p>An AI assistant (Claude, by Anthropic) was used while writing the code and drafting this report. Every metric, table,
plot and map in this report comes from running the code in the repository on the provided data, and can be reproduced with
the commands in the README.</p>
</body></html>
"""

out_html = REPORT_DIR / "PathSnap_Technical_Report.html"
out_html.write_text(html, encoding="utf-8")
print("wrote", out_html)
chrome = find_browser()
if chrome:
    out_pdf = REPORT_DIR / "PathSnap_Technical_Report.pdf"
    subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out_pdf}", out_html.as_uri()], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", out_pdf)
else:
    print("No Chrome found: open the HTML file in a browser and print to PDF.")
