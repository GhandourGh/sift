# Deploy live video demo on Render

Use this to get a **public HTTPS link** for the **live** 4-panel demo (FastAPI + web UI). Anyone who opens the link can tap “Tap to start”, allow camera, and see the feed update in real time.

## Steps

1. **Go to [Render](https://render.com)** and sign in (or create a free account).

2. **New → Web Service** and connect your GitHub account. Choose the repo **`GhandourGh/sift`**.

3. **Configure the service:**
   - **Name:** e.g. `cv-demo` (or leave default).
   - **Region:** pick one close to you or your classmates.
   - **Branch:** `main`.
   - **Root Directory:** leave **empty** (your repo root already has all the app files).
   - **Runtime:** **Docker** (Render will use the `Dockerfile` in the repo).
   - **Instance type:** Free.

4. Click **Create Web Service**. Render will build the image (install Python deps, copy app) and then start the app. The first build can take a few minutes.

5. When it’s live, Render shows a URL like **`https://cv-demo-xxxx.onrender.com`**. That’s your public link. Open it on your phone or share it; tap “Tap to start”, allow the camera, and you’ll see the live 4-panel view.

## If you don’t use Docker

- Set **Runtime** to **Python** instead of Docker.
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn web_app:app --host 0.0.0.0 --port $PORT`
- **Root Directory:** leave empty.

## Notes

- **Free tier:** The app may “spin down” after ~15 minutes of no traffic. The first open after that can take 30–60 seconds (Render will show “Server starting…” or similar).
- **HTTPS:** Render gives you HTTPS automatically, so phone cameras will work.
- The same repo also has the **Streamlit** app (`streamlit_app.py`) for the “capture one frame” demo; this Render setup runs the **live** FastAPI app.
