"""Launch backend (FastAPI) and frontend (Streamlit) together:  python run.py  [--backend-only | --frontend-only]"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description="Run LiteraMind")
    ap.add_argument("--backend-only", action="store_true"); ap.add_argument("--frontend-only", action="store_true")
    ap.add_argument("--api-port", default="8000"); ap.add_argument("--ui-port", default="8501")
    a = ap.parse_args()
    env = {**os.environ, "BACKEND_URL": f"http://127.0.0.1:{a.api_port}", "PYTHONPATH": str(ROOT)}
    procs = []
    try:
        if not a.frontend_only:
            procs.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", a.api_port], cwd=ROOT, env=env))
            time.sleep(2)
        if not a.backend_only:
            procs.append(subprocess.Popen([sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.port", a.ui_port], cwd=ROOT, env=env))
        print(f"\nLiteraMind: API http://127.0.0.1:{a.api_port}/docs  |  UI http://localhost:{a.ui_port}\nPress Ctrl+C to stop.")
        for p in procs: p.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None: p.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
