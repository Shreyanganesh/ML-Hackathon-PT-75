"""Step 1: dataset exploration (train and student_test partitions only).

Outputs in results/eda/:
  inventory.csv            one row per trip: files, sizes, counts, gaps, extent, connectivity
  missing_values.csv       missing / invalid values per file type and partition
  partition_summary.csv    aggregated statistics per partition
  *.png                    distribution plots and trajectory plots
maps/eda_trip_<id>.html    Folium map: GPS points, road network, reference route

Run:  python scripts/01_eda.py
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap.config import TRAIN_DIR, STUDENT_TEST_DIR, RESULTS_DIR, MAPS_DIR
from pathsnap.data_io import list_trips, load_route
from pathsnap.geo import haversine_m

OUT = RESULTS_DIR / "eda"
OUT.mkdir(parents=True, exist_ok=True)
EXT = ["track", "nodes", "arcs", "route"]


def raw(part, tid, ext, ncols, dtype=float):
    p = Path(part) / tid / f"{tid}.{ext}"
    if not p.exists() or p.stat().st_size == 0:
        return None
    # read as text first so malformed rows can be counted instead of crashing
    df = pd.read_csv(p, sep=r"\s+", header=None, engine="python")
    return df


def inspect_trip(name, part, tid):
    row = {"partition": name, "trip": tid}
    for e in EXT:
        p = Path(part) / tid / f"{tid}.{e}"
        row[f"has_{e}"] = p.exists()
        row[f"size_kb_{e}"] = round(p.stat().st_size / 1024, 1) if p.exists() else 0
    tr = raw(part, tid, "track", 3)
    nd = raw(part, tid, "nodes", 2)
    ar = raw(part, tid, "arcs", 2)
    rt = raw(part, tid, "route", 1)
    miss = []
    for e, df, nc in (("track", tr, 3), ("nodes", nd, 2), ("arcs", ar, 2), ("route", rt, 1)):
        if df is None:
            miss.append({"partition": name, "trip": tid, "file": e, "rows": 0, "columns": 0,
                         "missing_cells": 0, "non_numeric_cells": 0})
            continue
        num = df.apply(pd.to_numeric, errors="coerce")
        miss.append({"partition": name, "trip": tid, "file": e, "rows": len(df), "columns": df.shape[1],
                     "missing_cells": int(df.isna().sum().sum()),
                     "non_numeric_cells": int(num.isna().sum().sum() - df.isna().sum().sum())})
    trv = tr.to_numpy(float)
    lon, lat, t = trv[:, 0], trv[:, 1], trv[:, 2]
    dt = np.diff(t)
    step = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    speed = step / np.where(dt > 0, dt, np.nan)
    row.update({
        "n_points": len(trv),
        "duration_s": float(t[-1] - t[0]),
        "track_length_km": float(np.nansum(step) / 1000),
        "share_dt_1s": float(np.mean(dt == 1)),
        "max_gap_s": float(dt.max()) if len(dt) else 0,
        "n_gaps_over_5s": int(np.sum(dt > 5)),
        "non_increasing_timestamps": int(np.sum(dt <= 0)),
        "invalid_coords": int(np.sum((np.abs(lon) > 180) | (np.abs(lat) > 90) | ((lon == 0) & (lat == 0)))),
        "max_speed_mps": float(np.nanmax(speed)) if len(speed) else 0,
        "n_speed_over_60mps": int(np.nansum(speed > 60)),
        "lon_min": lon.min(), "lon_max": lon.max(), "lat_min": lat.min(), "lat_max": lat.max(),
        "extent_ew_km": float(haversine_m(lon.min(), lat.mean(), lon.max(), lat.mean()) / 1000),
        "extent_ns_km": float(haversine_m(lon.mean(), lat.min(), lon.mean(), lat.max()) / 1000),
    })
    V = len(nd)
    a = ar.to_numpy(int)
    row["n_nodes"] = V
    row["n_arcs"] = len(a)
    row["n_route_arcs"] = 0 if rt is None else len(rt)
    bad = (a < 0) | (a >= V)
    row["arcs_bad_node_ref"] = int(bad.any(axis=1).sum())
    used = np.zeros(V, bool)
    used[a[~bad.any(axis=1)].ravel()] = True
    row["isolated_nodes"] = int((~used).sum())
    g = csr_matrix((np.ones(len(a)), (a[:, 0], a[:, 1])), shape=(V, V))
    row["weak_components"], wl = connected_components(g, directed=True, connection="weak")
    row["strong_components"], sl = connected_components(g, directed=True, connection="strong")
    row["largest_scc_share"] = float(np.bincount(sl).max() / V)
    sym = set(zip(a[:, 0].tolist(), a[:, 1].tolist()))
    row["two_way_arc_share"] = float(np.mean([(v, u) in sym for u, v in a.tolist()]))
    alen = haversine_m(nd.iloc[a[:, 0], 0].to_numpy(), nd.iloc[a[:, 0], 1].to_numpy(),
                       nd.iloc[a[:, 1], 0].to_numpy(), nd.iloc[a[:, 1], 1].to_numpy())
    row["median_arc_len_m"] = float(np.median(alen))
    if rt is not None:
        r = rt.to_numpy(int).ravel()
        row["route_bad_ids"] = int(np.sum((r < 0) | (r >= len(a))))
        rr = r[(r >= 0) & (r < len(a))]
        row["route_connected_share"] = float(np.mean(a[rr[:-1], 1] == a[rr[1:], 0])) if len(rr) > 1 else 1.0
        row["route_repeated_arcs"] = int(len(rr) - len(set(rr.tolist())))
        row["route_length_km"] = float(alen[rr].sum() / 1000)
    return row, miss, (lon, lat, t, nd.to_numpy(float), a, None if rt is None else rt.to_numpy(int).ravel())


def main():
    rows, miss, keep = [], [], {}
    for name, part in (("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR)):
        for tid in list_trips(part):
            r, m, data = inspect_trip(name, part, tid)
            rows.append(r)
            miss.extend(m)
            keep[(name, tid)] = data
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "inventory.csv", index=False)
    mv = pd.DataFrame(miss)
    mv.groupby(["partition", "file"])[["rows", "missing_cells", "non_numeric_cells"]].sum().to_csv(OUT / "missing_values.csv")
    summ = inv.groupby("partition").agg(
        trips=("trip", "count"), points_total=("n_points", "sum"), points_median=("n_points", "median"),
        points_min=("n_points", "min"), points_max=("n_points", "max"),
        share_dt_1s=("share_dt_1s", "mean"), max_gap_s=("max_gap_s", "max"),
        trips_with_gap_over_5s=("n_gaps_over_5s", lambda s: int((s > 0).sum())),
        nodes_median=("n_nodes", "median"), arcs_median=("n_arcs", "median"), arcs_max=("n_arcs", "max"),
        isolated_nodes_total=("isolated_nodes", "sum"), weak_components_median=("weak_components", "median"),
        largest_scc_share_median=("largest_scc_share", "median"),
        route_connected_share_mean=("route_connected_share", "mean"),
        route_bad_ids_total=("route_bad_ids", "sum"), track_km_median=("track_length_km", "median"))
    summ.T.to_csv(OUT / "partition_summary.csv")
    print(summ.T)
    print(mv.groupby(["partition", "file"])[["rows", "missing_cells", "non_numeric_cells"]].sum())

    # ---------------- plots ----------------
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    for name in ("train", "student_test"):
        s = inv[inv.partition == name]
        ax[0].hist(s.n_points, bins=30, alpha=0.6, label=name)
        ax[1].hist(s.track_length_km, bins=30, alpha=0.6, label=name)
        ax[2].hist(np.log10(s.n_arcs), bins=30, alpha=0.6, label=name)
    ax[0].set_xlabel("GPS points per trip"); ax[1].set_xlabel("track length (km)"); ax[2].set_xlabel("log10(arcs in trip network)")
    for a_ in ax:
        a_.set_ylabel("trips"); a_.legend()
    fig.suptitle("Trip size distributions (train vs student_test)")
    fig.tight_layout(); fig.savefig(OUT / "trip_size_distributions.png", dpi=130); plt.close(fig)

    dts, spd = [], []
    for (name, tid), (lon, lat, t, nd, a, rt) in keep.items():
        if name != "train":
            continue
        d = np.diff(t)
        dts.append(d)
        stp = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
        spd.append(stp[d > 0] / d[d > 0])
    dts = np.concatenate(dts); spd = np.concatenate(spd)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    vals, cnts = np.unique(np.clip(dts, 0, 10), return_counts=True)
    ax[0].bar(vals, cnts); ax[0].set_yscale("log"); ax[0].set_xlabel("time between fixes (s, clipped at 10)")
    ax[0].set_ylabel("count (log)"); ax[0].set_title("Sampling interval, train")
    ax[1].hist(np.clip(spd * 3.6, 0, 200), bins=80); ax[1].set_xlabel("speed between fixes (km/h)")
    ax[1].set_title("Speed distribution, train")
    fig.tight_layout(); fig.savefig(OUT / "sampling_and_speed.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, c in (("train", "tab:blue"), ("student_test", "tab:orange")):
        s = inv[inv.partition == name]
        ax.scatter((s.lon_min + s.lon_max) / 2, (s.lat_min + s.lat_max) / 2, s=18, c=c, label=name)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_title("Trip locations (centroids)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "trip_locations.png", dpi=130); plt.close(fig)

    # one trip in detail: network + GPS + route (matplotlib)
    tid = "00000048"
    lon, lat, t, nd, a, rt = keep[("train", tid)]
    fig, ax = plt.subplots(figsize=(7, 7))
    pad = 0.003
    m = ((nd[a[:, 0], 0] > lon.min() - pad) & (nd[a[:, 0], 0] < lon.max() + pad) &
         (nd[a[:, 0], 1] > lat.min() - pad) & (nd[a[:, 0], 1] < lat.max() + pad))
    for u, v in a[m]:
        ax.plot([nd[u, 0], nd[v, 0]], [nd[u, 1], nd[v, 1]], c="lightgray", lw=0.7, zorder=1)
    for r in rt:
        u, v = a[r]
        ax.plot([nd[u, 0], nd[v, 0]], [nd[u, 1], nd[v, 1]], c="green", lw=2.5, zorder=2)
    ax.scatter(lon, lat, s=3, c="tab:blue", zorder=3)
    ax.plot([], [], c="lightgray", label="road network (arcs)")
    ax.plot([], [], c="green", lw=2.5, label="reference route")
    ax.scatter([], [], s=6, c="tab:blue", label="GPS fixes")
    ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
    ax.set_title(f"Train trip {tid}: GPS, network and reference route"); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(OUT / f"trip_{tid}_overview.png", dpi=130); plt.close(fig)

    # connectivity overview
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].hist(inv.largest_scc_share, bins=30); ax[0].set_xlabel("share of nodes in largest strongly connected component")
    ax[1].hist(np.log10(inv.weak_components), bins=30); ax[1].set_xlabel("log10(weakly connected components)")
    for a_ in ax:
        a_.set_ylabel("trips")
    fig.suptitle("Road-network connectivity (train + student_test)")
    fig.tight_layout(); fig.savefig(OUT / "network_connectivity.png", dpi=130); plt.close(fig)

    # GPS-to-reference distance (from the label alignment in step 2, if available)
    cache = Path(__file__).resolve().parent.parent / "cache" / "train_points.joblib"
    if cache.exists():
        import joblib
        pts = joblib.load(cache)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(np.clip(pts.label_dist, 0, 100), bins=100)
        ax.set_xlabel("distance GPS fix -> aligned reference arc (m, clipped at 100)")
        ax.set_ylabel("GPS fixes"); ax.set_title("GPS noise relative to the reference route (train)")
        fig.tight_layout(); fig.savefig(OUT / "gps_to_route_distance.png", dpi=130); plt.close(fig)
        med = pts.groupby("trip").label_dist.median().sort_values(ascending=False)
        med.to_csv(OUT / "median_gps_to_route_distance_per_trip.csv")

    # folium map of one train trip
    from pathsnap.maps import eda_map
    eda_map(TRAIN_DIR, tid, MAPS_DIR / f"eda_trip_{tid}.html")
    eda_map(TRAIN_DIR, "00000054", MAPS_DIR / "eda_trip_00000054_label_defect.html")
    print("saved EDA to", OUT)


if __name__ == "__main__":
    main()
