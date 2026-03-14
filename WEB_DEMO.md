# CV Evolution Web Demo

**One link — open in any browser. No app, no download, no signup.**

You can run the demo in two ways:

| Option | How it works | Best for |
|--------|----------------|----------|
| **Streamlit** | Capture a frame from your camera → see 4-panel result. Pure Python, one file. | Easiest to run and deploy; great for sharing one link. |
| **FastAPI** | Live stream: camera sends frames continuously, you see updating 4-panel view. | Smoother “live” feel; needs a bit more setup. |

---

## Option 1: Streamlit (recommended)

### Run locally

```bash
cd /path/to/V2
pip install -r requirements.txt
streamlit run streamlit_app.py --server.port 8501
```

Open http://localhost:8501. Allow camera, capture a frame, and the 4-panel result appears.

### Deploy (e.g. Streamlit Community Cloud or Render)

- **Streamlit Community Cloud**: Connect your GitHub repo, set **Main file path** to `streamlit_app.py`, **Root directory** to `V2` (if repo root is above V2). You get a URL like `https://your-app.streamlit.app`.
- **Render**: New Web Service → connect repo → **Start command:** `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0` (set **Root directory** to `V2` if needed).

---

## Option 2: FastAPI (live stream)

### Run locally

```bash
cd /path/to/V2
pip install -r requirements.txt
uvicorn web_app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. Tap “Tap to start”, allow camera; the 4-panel view updates live.

### Deploy (e.g. Render)

1. Push the repo to GitHub.
2. On [Render](https://render.com): New → Web Service → connect the repo.
3. **Root directory:** `V2` if the service is at repo root.
4. **Start command:** `uvicorn web_app:app --host 0.0.0.0 --port $PORT`
5. Use the generated HTTPS URL and share it.

## Docker (FastAPI app only)

```bash
docker build -t cv-web-demo ./V2
docker run -p 8000:8000 cv-web-demo
```

## Testing on phones

- Use the **HTTPS** public URL (not localhost).
- Allow camera when the browser prompts.
- If the server is on a free tier and sleeps when idle, the first load may take 30–60 s; show “Server starting…” until it responds.
