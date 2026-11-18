# PathSnap: GPS map matching with a candidate classifier and HMM/Viterbi

PathSnap takes the noisy GPS track of a trip and the trip's road network and returns the road arcs the vehicle most likely drove on. The system has three parts:

1. Candidate generation and feature engineering: for every GPS fix, up to 8 road arcs within 60 m are candidates. Each (fix, candidate) pair gets 25 features (distance, heading agreement, a ±3 fix context window, kinematics, topology).
2. A LightGBM candidate classifier, chosen after comparing nine models and tuned with Bayesian optimisation (Optuna TPE). It scores how likely each candidate is the true arc.
3. An HMM decoded with Viterbi. Hidden states are the candidate arcs, emissions come from the classifier, and transitions compare network driving distance with GPS distance. The decoder turns the per-fix scores into one connected route.

Predictions are served by a FastAPI backend (Laptop 1) and shown in a browser UI (Laptop 2).

Team: Kallepalli Sai Meghana, D R Rushmitha, Shreyan G, Shraddha Bamandi.

## Repository layout

```
pathsnap/                 core package (used by scripts and the API)
  config.py               paths and constants (reads config.json)
  data_io.py              reading and cleaning raw trip files
  geo.py                  local metric projection, point-to-segment geometry
  network.py              per-trip road network, spatial index, shortest paths
  labels.py               per-fix reference labels from .route (monotone DP alignment)
  features.py             candidate generation and the 25 features
  model_zoo.py            the nine candidate classifiers
  metrics.py              Top-1, macro F1, log loss, route continuity, route F1
  hmm.py                  HMM emissions, transitions and Viterbi decoding
  pipeline.py             end-to-end predict_trip() and server-side score_trip()
  maps.py                 Folium map generation
scripts/
  01_eda.py               dataset exploration (tables, plots, Folium maps)
  02_build_dataset.py     preprocessing + candidates + features -> cache/
  03_train_models.py      train and compare 9 models (GroupKFold CV + student test)
  04_bayes_opt.py         Bayesian optimisation of LightGBM, final model export
  05_hmm_decoder.py       decoder tuning, classifier-only vs HMM comparison, maps
  06_plots.py             model comparison figures
  07_capture_figures.py   screenshots of the Folium maps and the UI for the report
  08_build_report.py      builds the draft report (HTML, PDF) from the result files
  09_evaluator_results.py held-out evaluator test results (prediction and scoring only)
api/main.py               FastAPI server (Laptop 1)
ui/index.html             evaluator UI (Laptop 2)
ui/serve_ui.py            serves the UI with the Python standard library
models/                   final_model_lgbm.txt, final_model.joblib, model_config.json, hmm_config.json
results/                  every CSV, log and plot produced by the scripts
maps/                     Folium HTML maps
report/                   final project report (PT-075-ProjectReport.pdf)
config.json               candidate radius, K, feature window, seed, folds
```

## Setup (both laptops have Python 3.10 or newer)

Windows (PowerShell):

```
cd pathsnap
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```
cd pathsnap
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If PowerShell refuses to run the activate script, run `Set-ExecutionPolicy -Scope Process Bypass` once in that window.

## Data placement

The raw data is not committed. Unzip the provided `Student_folder.zip` and copy its two folders so the layout is:

```
pathsnap/data/train/00000001/00000001.track ...
pathsnap/data/student_test/00000000/00000000.track ...
pathsnap/data/evaluator_test/<id>/<id>.track ...   (only on Laptop 1, at evaluation time)
```

## Reproducing every result

Run from the `pathsnap` folder in this order. The times are from a 2-core machine.

```
python scripts/02_build_dataset.py        # ~1 min, writes cache/ and results/candidates_summary.csv
python scripts/01_eda.py                  # ~1 min, writes results/eda/ and maps/eda_*.html
python scripts/03_train_models.py         # ~40 min, all 9 models
python scripts/03_train_models.py curves  # ~3 min, epoch-wise loss curves (MLP, LightGBM, XGBoost)
python scripts/04_bayes_opt.py 30         # ~25 min, Optuna TPE, exports models/final_model*
python scripts/05_hmm_decoder.py 25       # ~3 min, decoder tuning + HMM comparison + prediction maps
python scripts/05_hmm_decoder.py ablation # ~3 min, backward-step ablation (results/hmm_ablation_*)
python scripts/06_plots.py                # seconds, comparison figures
python scripts/07_capture_figures.py      # needs Chrome + API and UI running; report screenshots
python scripts/08_build_report.py         # report/PathSnap_Technical_Report.html and .pdf
```

Held-out evaluator results (run once, after everything above; nothing is trained or tuned here):

```
python scripts/09_evaluator_results.py    # needs PATHSNAP_EVAL_DIR set to the evaluator_test folder
```

`01_eda.py` uses the label alignment from step 2 for one plot, so step 2 runs first. A single model can be rerun with `python scripts/03_train_models.py LightGBM`. The trained model files are committed, so the API runs without retraining.

## Deployment on two laptops

Laptop 1 runs the model and the API. Laptop 2 runs the UI. Both join the same Wi-Fi or hotspot. College Wi-Fi often blocks traffic between devices, so a phone hotspot is the safest choice.

### Laptop 1 (model + API)

1. Find the IP address: Windows `ipconfig` (IPv4 Address of the Wi-Fi adapter), macOS `ipconfig getifaddr en0`, Linux `hostname -I`.
2. Point the server at the evaluation trips and start it:

   Windows PowerShell:
   ```
   $env:PATHSNAP_EVAL_DIR = "C:\path\to\evaluator_test"
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
   macOS / Linux:
   ```
   export PATHSNAP_EVAL_DIR=/path/to/evaluator_test
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
   Without the variable the server reads `data/evaluator_test`. For a rehearsal, use `data/student_test`.
3. On Windows, allow Python through the firewall on private networks when prompted. If no prompt appears, run once as administrator:
   `netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000`
4. Check from Laptop 1: open `http://127.0.0.1:8000/health`.

The console on Laptop 1 prints a line for every request (`REQUEST predict trip=... from <Laptop 2 IP>`) and its result (`RESULT trip=... point_acc=...`). These are the API logs to show the evaluator.

### Laptop 2 (evaluator UI)

The UI needs only Python, with nothing to install.

```
python ui/serve_ui.py --api http://<LAPTOP1_IP>:8000
```

A browser opens at `http://localhost:8501`. Click Connect, pick any of the ten trips, and the route and metrics appear. First check from Laptop 2 that `http://<LAPTOP1_IP>:8000/health` opens in the browser.

## API

| Method | Path | Body / result |
|---|---|---|
| GET | `/health` | status, model name, evaluation folder, number of trips |
| GET | `/trips` | `{"trips": [ids]}` |
| POST | `/predict` | body `{"trip_id": "00000000"}`; returns GPS points, matched points, predicted arc per point, predicted route arcs and geometry, server-side metrics, timings |
| GET | `/predict/{trip_id}` | same as POST |

Interactive docs: `http://<LAPTOP1_IP>:8000/docs`.

## Leakage safeguards

- `predict_trip()` loads only `.track`, `.nodes` and `.arcs`. The `.route` file is read by `load_route()`, and only `score_trip()` calls it, after prediction.
- The API response contains no reference route geometry or labels, only aggregate scores.
- Scalers sit inside model pipelines and are fitted on training folds only. Cross-validation is grouped by trip.
- The evaluator partition is never used for training, tuning, or model and threshold selection. The decoder is tuned on training trips using out-of-fold probabilities.
- Trips 00000033 and 00000054 have reference routes 120 to 140 m from the GPS track (median), so they are excluded from model fitting. `results/eda/median_gps_to_route_distance_per_trip.csv` has the evidence.

## Reproducibility

The random seed is 911 (`config.json`) for every model, fold split, subsample and Optuna sampler. Every metric in the report is in `results/`.
