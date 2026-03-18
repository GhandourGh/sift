"""
FastAPI web app for the CV Evolution Demo V3.

Returns structured JSON (metrics + base64 panel images) instead of
raw JPEG composites, enabling rich frontend visualizations.

Endpoints:
  GET  /health         — Check detector status
  POST /process-frame  — Process a single frame, return panels + metrics
  POST /sweep          — Run single perturbation sweep
  POST /sweep-all      — Run all perturbation sweeps
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress blake2 hashlib warnings (macOS pyenv)
class _SuppressHashlibErrors(logging.Filter):
    def filter(self, record):
        return "blake2" not in (record.getMessage() or "")

logging.getLogger().addFilter(_SuppressHashlibErrors())
logger = logging.getLogger(__name__)

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from algorithms.sift_detector import SIFTDetector
from algorithms.yolo_detector import YOLODetector
from algorithms.cnn_classifier import CNNClassifier
from evaluation.perturbations import PerturbationManager
from pipeline import run_cv_pipeline

app = FastAPI(title="CV Evolution Web Demo V3", version="3.0")

# Shared detectors
sift_detector = None
yolo_detector = None
cnn_classifier = None


@app.on_event("startup")
def startup():
    global sift_detector, yolo_detector, cnn_classifier
    print("[V3] Loading SIFT detector...")
    sift_detector = SIFTDetector()
    print("[V3] Loading YOLO detector...")
    yolo_detector = YOLODetector()
    print("[V3] Loading CNN classifier...")
    cnn_classifier = CNNClassifier()
    print("[V3] All detectors loaded successfully!")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "sift": sift_detector is not None,
        "yolo": yolo_detector is not None,
        "cnn": cnn_classifier is not None,
    }


@app.get("/api-test")
def api_test():
    """Quick test to verify API routes are reachable."""
    return {"api": "reachable", "detectors_loaded": cnn_classifier is not None}


def _decode_image(body: bytes):
    """Decode uploaded image bytes to BGR numpy array."""
    buf = np.frombuffer(body, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return frame


def _serialize_metrics(metrics):
    """Make metrics dict JSON-serializable."""
    result = {}
    for algo, m in metrics.items():
        result[algo] = {}
        for k, v in m.items():
            if k == "top5":
                result[algo][k] = [
                    {"label": label, "confidence": round(conf, 2)}
                    for label, conf in v
                ]
            elif isinstance(v, (int, float)):
                result[algo][k] = round(v, 2) if isinstance(v, float) else v
            elif isinstance(v, str):
                result[algo][k] = v
    return result


@app.post("/process-frame")
async def process_frame(
    file: UploadFile = File(...),
    perturbations: str = Form(default="{}"),
):
    """Process a single frame with optional perturbations.

    Args:
        file: Image file (JPEG/PNG).
        perturbations: JSON string of perturbation settings, e.g.:
            {"noise": {"active": true, "value": 30}, ...}

    Returns:
        JSON with base64 panel images and structured metrics.
    """
    print(f"[V3] /process-frame called, file={file.filename}, content_type={file.content_type}")

    if sift_detector is None or yolo_detector is None or cnn_classifier is None:
        raise HTTPException(status_code=503, detail="Detectors not initialized")

    body = await file.read()
    print(f"[V3] Read {len(body)} bytes")
    frame = _decode_image(body)
    print(f"[V3] Decoded frame: {frame.shape}")

    # Apply perturbations if specified
    try:
        perturb_params = json.loads(perturbations)
    except json.JSONDecodeError:
        perturb_params = {}

    if perturb_params:
        perturb = PerturbationManager()
        perturb.set_from_dict(perturb_params)
        if perturb.any_active:
            frame = perturb.apply(frame)

    try:
        panels, metrics = run_cv_pipeline(
            frame, sift_detector, yolo_detector, cnn_classifier,
            encode_panels=True,
        )
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}") from e

    return JSONResponse({
        "panels": panels,
        "metrics": _serialize_metrics(metrics),
    })


# Serve static files — use explicit index route + sub-path mount
# to avoid the catch-all "/" mount intercepting API routes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")


@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


if os.path.isdir(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR), name="static")
