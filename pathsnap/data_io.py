"""Reading raw PathSnap trip folders.

Each trip folder <id>/ holds:
  <id>.track  lon lat timestamp      (GPS observations)
  <id>.nodes  lon lat                (row index = node id)
  <id>.arcs   from_node to_node      (row index = arc id)
  <id>.route  arc_id                 (reference route, server side only)

The prediction pipeline only ever calls load_trip(); the reference route is
read separately by load_route() so labels cannot leak into prediction.
"""
from dataclasses import dataclass
from pathlib import Path
import os
import numpy as np


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray   # (N, 3) lon, lat, t
    nodes: np.ndarray   # (V, 2) lon, lat
    arcs: np.ndarray    # (E, 2) from, to
    n_raw_points: int = 0
    n_dropped_points: int = 0


def list_trips(part_dir):
    """Trip ids in a partition folder. Accepts the nested layout <id>/<id>.track
    (as supplied) and a flat layout <id>.track directly in the folder."""
    part_dir = Path(part_dir)
    if not part_dir.exists():
        return []
    nested = {d.name for d in part_dir.iterdir() if d.is_dir() and (d / f"{d.name}.track").exists()}
    flat = {f.stem for f in part_dir.glob("*.track")}
    return sorted(nested | flat)


def _base(part_dir, trip_id):
    nested = Path(part_dir) / trip_id / trip_id
    if Path(str(nested) + ".track").exists():
        return nested
    return Path(part_dir) / trip_id


def _load(path, ncols, dtype=float):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, ncols), dtype=dtype)
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    return arr


def clean_track(track):
    """Drop invalid GPS rows and enforce strictly increasing timestamps.

    Rules (decided from EDA): non-finite values, coordinates outside valid
    lon/lat ranges, exact (0,0) fixes, and rows whose timestamp is not larger
    than the previous kept row are removed. Order of the file is otherwise kept.
    """
    if len(track) == 0:
        return track, 0
    ok = np.all(np.isfinite(track), axis=1)
    ok &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    ok &= ~((track[:, 0] == 0) & (track[:, 1] == 0))
    tr = track[ok]
    keep = np.ones(len(tr), dtype=bool)
    last = -np.inf
    for i in range(len(tr)):
        if tr[i, 2] <= last:
            keep[i] = False
        else:
            last = tr[i, 2]
    tr = tr[keep]
    return tr, int(len(track) - len(tr))


def load_trip(part_dir, trip_id):
    base = _base(part_dir, trip_id)
    raw = _load(str(base) + ".track", 3)
    track, dropped = clean_track(raw)
    nodes = _load(str(base) + ".nodes", 2)
    arcs = _load(str(base) + ".arcs", 2, dtype=np.int64)
    return Trip(trip_id, track, nodes, arcs, len(raw), dropped)


def load_route(part_dir, trip_id):
    """Reference arc sequence. Only used for labels and server-side scoring."""
    p = Path(str(_base(part_dir, trip_id)) + ".route")
    if not p.exists() or p.stat().st_size == 0:
        return None
    return np.loadtxt(p, dtype=np.int64, ndmin=1)
