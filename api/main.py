"""PathSnap FastAPI server (runs on Laptop 1).

Start:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Evaluation trips are read from the folder in the PATHSNAP_EVAL_DIR environment
variable (default: data/evaluator_test). Each trip folder has the same layout
as the training data. The .route file in that folder is used only inside this
server to compute accuracy after prediction; it is never sent to the UI and is
never passed to the prediction pipeline.

Endpoints
    GET  /health                 server, model and data status
    GET  /trips                  ids of the evaluation trips (no labels)
    POST /predict  {"trip_id"}   run the full pipeline and score it server side
    GET  /predict/{trip_id}      same as POST, handy for curl / browser checks
"""
import sys
import json
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap.config import EVAL_DIR, MODELS_DIR
from pathsnap.data_io import list_trips, load_trip, load_route
from pathsnap.pipeline import predict_trip, score_trip, route_geometry, load_final_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap.api")

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app):
    _load()
    yield


app = FastAPI(title="PathSnap map-matching API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATE = {}


def _load():
    STATE["model"] = load_final_model(MODELS_DIR)
    with open(MODELS_DIR / "hmm_config.json") as f:
        STATE["hmm_params"] = json.load(f)["params"]
    with open(MODELS_DIR / "model_config.json") as f:
        STATE["model_config"] = json.load(f)
    STATE["eval_dir"] = Path(EVAL_DIR)
    STATE["trips"] = list_trips(EVAL_DIR)
    log.info("model loaded: %s", STATE["model_config"]["model"])
    log.info("HMM params: %s", STATE["hmm_params"])
    log.info("evaluation folder: %s (%d trips: %s)", EVAL_DIR, len(STATE["trips"]), ", ".join(STATE["trips"]))
    if len(STATE["trips"]) == 0:
        log.warning("No trips found. Set PATHSNAP_EVAL_DIR to the folder with the evaluation trips.")


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model": STATE["model_config"]["model"], "eval_dir": str(STATE["eval_dir"]),
            "n_trips": len(STATE["trips"])}


@app.get("/trips")
def trips(request: Request):
    STATE["trips"] = list_trips(STATE["eval_dir"])  # rescan so a folder dropped in later is picked up
    log.info("GET /trips from %s -> %d trips", request.client.host if request.client else "?", len(STATE["trips"]))
    return {"trips": STATE["trips"]}


def _run(trip_id, client):
    if trip_id not in list_trips(STATE["eval_dir"]):
        raise HTTPException(status_code=404, detail=f"trip {trip_id} not found in evaluation folder")
    t0 = time.time()
    log.info("REQUEST predict trip=%s from %s", trip_id, client)
    trip = load_trip(STATE["eval_dir"], trip_id)           # GPS track + road network only
    pred = predict_trip(trip, STATE["model"], STATE["hmm_params"])
    net = pred["net"]

    # ---- server-side scoring with the internal reference route ----
    route = load_route(STATE["eval_dir"], trip_id)
    metrics = score_trip(pred, trip, route) if route is not None else None

    # matched position of every GPS fix on its decoded arc (for display)
    pr = net.proj
    px, py = pr.to_xy(trip.track[:, 0], trip.track[:, 1])
    loc = np.array([net.local_of[int(a)] for a in pred["hmm_arcs"]])
    dx, dy = net.bx[loc] - net.ax[loc], net.by[loc] - net.ay[loc]
    l2 = np.maximum(dx * dx + dy * dy, 1e-9)
    tt = np.clip(((px - net.ax[loc]) * dx + (py - net.ay[loc]) * dy) / l2, 0, 1)
    mx, my = net.ax[loc] + tt * dx, net.ay[loc] + tt * dy
    matched = np.c_[my / pr.ky + pr.lat0, mx / pr.kx + pr.lon0]

    resp = {
        "trip_id": trip_id,
        "n_gps_points": int(pred["n_points"]),
        "n_candidate_rows": int(pred["n_candidate_rows"]),
        "gps_points": trip.track[:, [1, 0]].round(7).tolist(),
        "matched_points": matched.round(7).tolist(),
        "predicted_arc_per_point": [int(a) for a in pred["hmm_arcs"]],
        "predicted_route_arcs": [int(a) for a in pred["route_arcs"]],
        "predicted_route_geometry": route_geometry(net, pred["route_loc"]),
        "hmm_chain_breaks": int(pred["hmm_breaks"]),
        "metrics": metrics,
        "timing_s": {k: round(v, 3) for k, v in pred["timing_s"].items()},
        "server_time_s": round(time.time() - t0, 3),
        "model": STATE["model_config"]["model"],
    }
    if metrics:
        log.info("RESULT trip=%s points=%d point_acc=%.4f (classifier-only %.4f) route_F1=%.4f "
                 "length_F1=%.4f continuity=%.4f time=%.2fs", trip_id, pred["n_points"],
                 metrics["point_accuracy_hmm"], metrics["point_accuracy_classifier_only"],
                 metrics["route_arc_f1"], metrics["route_length_f1"], metrics["route_continuity_hmm"],
                 resp["server_time_s"])
    else:
        log.info("RESULT trip=%s points=%d (no reference route available) time=%.2fs",
                 trip_id, pred["n_points"], resp["server_time_s"])
    return resp


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    return _run(req.trip_id.strip(), request.client.host if request.client else "?")


@app.get("/predict/{trip_id}")
def predict_get(trip_id: str, request: Request):
    return _run(trip_id.strip(), request.client.host if request.client else "?")
