"""Folium map generation (EDA maps and prediction maps)."""
import folium
import numpy as np

from .data_io import load_trip, load_route
from .network import RoadNetwork

LEGEND = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
            padding: 10px 14px; border: 1px solid #999; border-radius: 6px; font-size: 13px;">
<b>{title}</b><br>{rows}
</div>"""


def _legend(title, items):
    rows = "".join(f'<span style="display:inline-block;width:22px;height:4px;background:{c};'
                   f'margin-right:6px;vertical-align:middle"></span>{lbl}<br>' for lbl, c in items)
    return folium.Element(LEGEND.format(title=title, rows=rows))


def _arc_lines(net, locs):
    pr = net.proj
    out = []
    for a in locs:
        out.append([[net.ay[a] / pr.ky + pr.lat0, net.ax[a] / pr.kx + pr.lon0],
                    [net.by[a] / pr.ky + pr.lat0, net.bx[a] / pr.kx + pr.lon0]])
    return out


def _base(trip):
    lat, lon = trip.track[:, 1], trip.track[:, 0]
    m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=14, tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    return m


def _gps_layer(trip, every=1, name="GPS observations"):
    fg = folium.FeatureGroup(name=name)
    tr = trip.track
    for i in range(0, len(tr), every):
        folium.CircleMarker([tr[i, 1], tr[i, 0]], radius=2, color="#1f5fbf", fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"fix {i}, t={tr[i, 2]:.0f}s").add_to(fg)
    folium.Marker([tr[0, 1], tr[0, 0]], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(fg)
    folium.Marker([tr[-1, 1], tr[-1, 0]], tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(fg)
    return fg


def _network_layer(net, max_arcs=40000):
    fg = folium.FeatureGroup(name="Road network (arcs near track)")
    locs = np.arange(len(net.arc_ids))
    if len(locs) > max_arcs:
        locs = locs[:max_arcs]
    folium.PolyLine(_arc_lines(net, locs), color="#9a9a9a", weight=1, opacity=0.6).add_to(fg)
    return fg


def _route_layer(net, route_ids, name, color, weight=5):
    fg = folium.FeatureGroup(name=name)
    locs = [net.local_of[int(a)] for a in route_ids if int(a) in net.local_of]
    if locs:
        folium.PolyLine(_arc_lines(net, locs), color=color, weight=weight, opacity=0.85).add_to(fg)
    return fg


def eda_map(part_dir, trip_id, out_path):
    trip = load_trip(part_dir, trip_id)
    route = load_route(part_dir, trip_id)
    net = RoadNetwork(trip, margin_m=300)
    m = _base(trip)
    _network_layer(net).add_to(m)
    if route is not None:
        _route_layer(net, route, "Reference route", "#2ca02c").add_to(m)
    _gps_layer(trip).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(_legend(f"Trip {trip_id} (EDA)", [
        ("GPS observations", "#1f5fbf"), ("Road network", "#9a9a9a"), ("Reference route", "#2ca02c")]))
    m.save(str(out_path))
    return out_path


def prediction_map(trip, pred, out_path, route=None, title=None):
    """GPS, network, reference route (if given) and predicted route, with layer control."""
    net = pred["net"]
    m = _base(trip)
    _network_layer(net).add_to(m)
    if route is not None:
        _route_layer(net, route, "Reference route", "#2ca02c", weight=8).add_to(m)
    _route_layer(net, pred["route_arcs"], "Predicted route (classifier + HMM)", "#d62728", weight=4).add_to(m)
    clf_route = []
    for a in pred["clf_arcs"]:
        if not clf_route or clf_route[-1] != a:
            clf_route.append(int(a))
    _route_layer(net, clf_route, "Classifier-only arcs", "#ff9f1c", weight=3).add_to(m)
    _gps_layer(trip).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    items = [("GPS observations", "#1f5fbf"), ("Road network", "#9a9a9a")]
    if route is not None:
        items.append(("Reference route", "#2ca02c"))
    items += [("Predicted route (HMM)", "#d62728"), ("Classifier-only arcs", "#ff9f1c")]
    m.get_root().html.add_child(_legend(title or f"Trip {trip.trip_id}", items))
    m.save(str(out_path))
    return out_path
