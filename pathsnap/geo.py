"""Geometry helpers.

Longitude/latitude are angles, so distances are computed after projecting each
trip to a local tangent plane (equirectangular projection centred on the trip
centroid). Within the extent of one trip (a few tens of km) the scale error of
this projection is far below GPS noise, and it keeps every later computation
(point-to-segment distance, bearings) in metres.
"""
import numpy as np

R_EARTH = 6371008.8  # mean Earth radius in metres


class LocalProjection:
    def __init__(self, lon0, lat0):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self.kx = np.cos(np.radians(self.lat0)) * np.pi / 180.0 * R_EARTH
        self.ky = np.pi / 180.0 * R_EARTH

    @classmethod
    def for_track(cls, track):
        return cls(np.mean(track[:, 0]), np.mean(track[:, 1]))

    def to_xy(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points to segments (all arrays broadcast together).

    Returns (distance, t) where t in [0, 1] is the clamped projection
    parameter along a->b.
    """
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * dx + (py - ay) * dy) / l2
    t = np.where(l2 > 0, t, 0.0)
    t = np.clip(t, 0.0, 1.0)
    qx = ax + t * dx
    qy = ay + t * dy
    return np.hypot(px - qx, py - qy), t


def bearing_xy(dx, dy):
    """Bearing in radians measured from +x axis (east), counter clockwise."""
    return np.arctan2(dy, dx)
