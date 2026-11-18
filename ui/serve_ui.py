"""Serve the evaluator UI (runs on Laptop 2).

    python ui/serve_ui.py --api http://<LAPTOP1_IP>:8000 [--port 8501]

Then open http://localhost:8501 in a browser on Laptop 2. The UI talks to the
API over the network; it holds no model, no data and no reference routes.
"""
import argparse
import functools
import http.server
import socketserver
import webbrowser
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--api", default="http://127.0.0.1:8000", help="address of the PathSnap API on Laptop 1")
ap.add_argument("--port", type=int, default=8501)
ap.add_argument("--no-browser", action="store_true")
a = ap.parse_args()

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(UI_DIR))
url = f"http://localhost:{a.port}/index.html?api={a.api}"
socketserver.TCPServer.allow_reuse_address = True
with socketserver.ThreadingTCPServer(("0.0.0.0", a.port), handler) as httpd:
    print(f"PathSnap UI on {url}")
    print(f"API expected at {a.api}")
    if not a.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    httpd.serve_forever()
