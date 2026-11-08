# Deployment

> The deployment artefacts (Dockerfile, render.yaml) were written but **not executed** in the development sandbox (no Docker/Render access). Local execution (`python run.py`) *was* tested.

## Local
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # optional: add keys
python run.py                                         # API :8000, UI :8501
```

## Docker
`docker build -t literamind . && docker run -p 8000:8000 -p 8501:8501 --env-file .env -v $(pwd)/data:/app/data literamind`

## Render (two services, `render.yaml`)
1. Push the repo to GitHub. 2. Render → *New → Blueprint* → select the repo. 3. Set secret env vars (`ANTHROPIC_API_KEY`, `S2_API_KEY`, `CONTACT_EMAIL`). 4. The UI service gets `BACKEND_URL` from the API service automatically.
SQLite + Chroma need the persistent disk declared in the blueprint (free instances have ephemeral disks: data is lost on restart). For production use PostgreSQL (`DATABASE_URL`).

## Streamlit Community Cloud (single app: UI + API in one process)
Set `EMBEDDED_BACKEND=1` and the Streamlit app starts the FastAPI backend in a background thread (`frontend/common.py::ensure_backend`, covered by an automated test). Streamlit Cloud then needs only one app. **Not deploy-tested on the real service** (the embedded mode itself is tested locally).
1. Push the project to GitHub (repo root = the inner `LiteraMind` folder, so `frontend/app.py` is at `frontend/app.py`; never commit `.env`).
2. share.streamlit.io → *Create app* → pick the repo/branch → **Main file path: `frontend/app.py`**.
3. *Advanced settings*: Python **3.11**; *Secrets* (TOML):
```toml
EMBEDDED_BACKEND = "1"
EMBEDDING_BACKEND = "hashing"
LLM_PROVIDER = "heuristic"
CONTACT_EMAIL = "you@example.com"
```
4. Deploy. Streamlit Cloud installs `frontend/requirements.txt` (a light set without PyTorch/sentence-transformers) because it sits next to the entry-point file.
Caveats: ~1 GB RAM on the free tier; the disk is **ephemeral** (data resets when the app restarts/sleeps); no login (anyone with the URL can use it; keep `LLM_PROVIDER=heuristic` unless the app is private, otherwise visitors can spend your API credit); if the build fails on a `sqlite3` version error from ChromaDB, add `pysqlite3-binary` to `frontend/requirements.txt` and say so in an issue/message so the import shim can be added.

## Memory note
`sentence-transformers` pulls in PyTorch (~1–2 GB installed; model ~90 MB downloaded on first use). On small instances set `EMBEDDING_BACKEND=hashing` (lexical, no download) and remove `sentence-transformers` from requirements.
