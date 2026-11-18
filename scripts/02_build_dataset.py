"""Step 2: preprocessing + candidate generation + feature engineering.

Builds the (point, candidate) feature tables for the train and student_test
partitions and stores them in cache/. Also writes a per-trip summary with the
candidate miss rate (share of points whose reference arc is not among the
candidates) to results/candidates_summary.csv.

Run:  python scripts/02_build_dataset.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from pathsnap.config import TRAIN_DIR, STUDENT_TEST_DIR, CACHE_DIR, RESULTS_DIR, CAND
from pathsnap.data_io import list_trips, load_trip, load_route
from pathsnap.features import build_trip_table


def build_partition(name, part_dir):
    tabs, points, summ = [], [], []
    for tid in list_trips(part_dir):
        t0 = time.time()
        trip = load_trip(part_dir, tid)
        route = load_route(part_dir, tid)
        tab, info, net = build_trip_table(trip, route)
        lab = info["label_arc"]
        # does each point's reference arc appear among its candidates?
        hit = tab.groupby("pt")["label"].max().reindex(range(info["n"]), fill_value=0).to_numpy()
        pts = pd.DataFrame({"trip": tid, "pt": np.arange(info["n"]), "label_arc": lab,
                            "label_dist": info["label_dist"], "in_cands": hit,
                            "t": info["t"], "lon": trip.track[:, 0], "lat": trip.track[:, 1]})
        tabs.append(tab)
        points.append(pts)
        summ.append({"partition": name, "trip": tid, "n_points": info["n"],
                     "dropped_points": trip.n_dropped_points,
                     "arcs_total": net.n_arcs_total, "arcs_in_window": len(net.arc_ids),
                     "rows": len(tab), "mean_candidates": len(tab) / max(info["n"], 1),
                     "candidate_miss_rate": 1 - hit.mean(),
                     "median_label_dist_m": float(np.median(info["label_dist"])),
                     "p95_label_dist_m": float(np.percentile(info["label_dist"], 95)),
                     "seconds": round(time.time() - t0, 2)})
        print(f"[{name}] {tid}: {info['n']} pts, {len(tab)} rows, miss={1-hit.mean():.3f}, "
              f"{time.time()-t0:.1f}s", flush=True)
    tab = pd.concat(tabs, ignore_index=True)
    pts = pd.concat(points, ignore_index=True)
    joblib.dump(tab, CACHE_DIR / f"{name}_candidates.joblib", compress=3)
    joblib.dump(pts, CACHE_DIR / f"{name}_points.joblib", compress=3)
    return pd.DataFrame(summ)


if __name__ == "__main__":
    s1 = build_partition("train", TRAIN_DIR)
    s2 = build_partition("student_test", STUDENT_TEST_DIR)
    s = pd.concat([s1, s2], ignore_index=True)
    s.to_csv(RESULTS_DIR / "candidates_summary.csv", index=False)
    print(s.groupby("partition")[["n_points", "rows", "mean_candidates", "candidate_miss_rate",
                                  "median_label_dist_m"]].mean())
    print("radius", CAND["radius_m"], "K", CAND["max_candidates"])
