"""Central configuration. Every script reads paths and constants from here."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "config.json"

with open(CONFIG_FILE) as f:
    CFG = json.load(f)

DATA_DIR = ROOT / CFG["paths"]["data_dir"]
TRAIN_DIR = DATA_DIR / CFG["paths"]["train"]
STUDENT_TEST_DIR = DATA_DIR / CFG["paths"]["student_test"]
# The evaluator partition can be pointed anywhere with an environment variable,
# so the API laptop can use whatever folder the evaluators hand over.
EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", str(DATA_DIR / CFG["paths"]["evaluator_test"])))

RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
MAPS_DIR = ROOT / "maps"
CACHE_DIR = ROOT / "cache"

SEED = CFG["seed"]
CAND = CFG["candidates"]
FEAT = CFG["features"]

for d in (RESULTS_DIR, MODELS_DIR, MAPS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)
