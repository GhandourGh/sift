# CV Evolution Web Demo

**One link — open in any browser. No app, no download, no signup.** Share your public URL; anyone can open it (e.g. on their phone), tap “Tap to start”, allow the camera when asked, and see the live 4-panel view (Original, Edge/Canny, SIFT, YOLO).

## Run locally

```bash
cd /path/to/V2
pip install -r requirements.txt
uvicorn web_app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in a browser. Use HTTPS and a public URL for phone webcam access (getUserMedia often requires secure context).

## Deploy to a public URL (e.g. Render)

1. Push this repo (or the V2 folder) to GitHub.
2. On [Render](https://render.com): New → Web Service → connect the repo.
3. Configure:
   - **Build command:** (none, or `pip install -r requirements.txt` if not using Docker)
   - **Start command:** `uvicorn web_app:app --host 0.0.0.0 --port $PORT`
   - **Root directory:** `V2` if the service is at repo root.
4. Or use the included **Dockerfile** (Render supports Docker): set start to use the image and expose port 8000; Render sets `PORT` automatically.
5. After deploy, use the generated HTTPS URL (e.g. `https://your-service.onrender.com`). Share that link; classmates open it on their phones and tap “Start webcam demo”.

## Docker (optional)

```bash
docker build -t cv-web-demo ./V2
docker run -p 8000:8000 cv-web-demo
```

## Testing on phones

- Use the **HTTPS** public URL (not localhost).
- Allow camera when the browser prompts.
- If the server is on a free tier and sleeps when idle, the first load may take 30–60 s; show “Server starting…” until it responds.
