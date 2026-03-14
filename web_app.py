"""
FastAPI web app for the CV Evolution demo.
Serves the 4-panel live demo to browsers (including phones) via a public URL.
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from algorithms.sift_detector import SIFTDetector
from algorithms.yolo_detector import YOLODetector
from pipeline import run_cv_pipeline

# Smaller panel size for web: faster inference and smaller response
WEB_PANEL_W = 320
WEB_PANEL_H = 240
WEB_PROCESS_W = 320
WEB_PROCESS_H = 240

app = FastAPI(title="CV Evolution Web Demo", version="1.0")

# Shared detectors (initialized on startup)
sift_detector: SIFTDetector | None = None
yolo_detector: YOLODetector | None = None


@app.on_event("startup")
def startup():
    global sift_detector, yolo_detector
    sift_detector = SIFTDetector()
    yolo_detector = YOLODetector()


@app.get("/health")
def health():
    return {"status": "ok", "sift": sift_detector is not None, "yolo": yolo_detector is not None}


@app.post("/process-frame")
async def process_frame(file: UploadFile = File(...)):
    """Accept a single image (JPEG/PNG), run the CV pipeline, return the 4-panel composite as JPEG."""
    if sift_detector is None or yolo_detector is None:
        raise HTTPException(status_code=503, detail="Detectors not initialized")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image file (JPEG or PNG)")

    try:
        body = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}") from e

    buf = np.frombuffer(body, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    try:
        combined, _ = run_cv_pipeline(
            frame,
            sift_detector,
            yolo_detector,
            panel_w=WEB_PANEL_W,
            panel_h=WEB_PANEL_H,
            process_w=WEB_PROCESS_W,
            process_h=WEB_PROCESS_H,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}") from e

    _, jpeg = cv2.imencode(".jpg", combined)
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


# Mount static files (web UI) after API routes so / doesn't catch everything
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
if os.path.isdir(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")
