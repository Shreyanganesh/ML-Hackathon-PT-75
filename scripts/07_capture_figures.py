"""Step 7: screenshots of the Folium maps and the running UI for the report.

Needs a local Chrome/Chromium. For the UI screenshot the API (port 8000) and the
UI server (port 8501) must be running, as in the deployment section of the README.

Run:  python scripts/07_capture_figures.py [ui_trip_id]
"""
import sys
import shutil
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from pathsnap.config import RESULTS_DIR, MAPS_DIR, ROOT

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


FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
chrome = find_browser()
if chrome is None:
    sys.exit("Chrome/Chromium not found; take the screenshots manually.")


def shot(url, out, w=1400, h=900, wait=8000):
    subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                    f"--window-size={w},{h}", f"--virtual-time-budget={wait}", f"--screenshot={out}", url],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    print("saved", out)


ht = pd.read_csv(RESULTS_DIR / "hmm_vs_classifier_per_trip.csv", dtype={"trip": str})
ht = ht[ht.decoder == "hmm_tuned"].sort_values("point_accuracy_hmm")
worst, best = ht.iloc[0].trip, ht.iloc[-1].trip
shot((MAPS_DIR / "eda_trip_00000048.html").as_uri(), FIG / "folium_eda_00000048.png")
shot((MAPS_DIR / f"pred_student_test_{worst}.html").as_uri(), FIG / f"folium_pred_{worst}.png")
shot((MAPS_DIR / f"pred_student_test_{best}.html").as_uri(), FIG / "folium_pred_best.png")
trip = sys.argv[1] if len(sys.argv) > 1 else best
shot(f"http://localhost:8501/index.html?api=http://127.0.0.1:8000&trip={trip}", FIG / "ui_screenshot.png", 1500, 900, 20000)
