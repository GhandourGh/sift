"""
FastAPI web app for the CV Evolution Demo.

Endpoints:
  GET  /            — Serve the dashboard
  GET  /health      — Check detector status
  POST /process-frame — Process a single frame, return panels + metrics
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

app = FastAPI(title="CV Evolution Demo", version="1.0")

sift_detector = None
yolo_detector = None
cnn_classifier = None


@app.on_event("startup")
def startup():
    global sift_detector, yolo_detector, cnn_classifier
    print("Loading SIFT detector...")
    sift_detector = SIFTDetector()
    print("Loading YOLO detector...")
    yolo_detector = YOLODetector()
    print("Loading CNN classifier...")
    cnn_classifier = CNNClassifier()
    print("All models loaded.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "sift": sift_detector is not None,
        "yolo": yolo_detector is not None,
        "cnn": cnn_classifier is not None,
    }


def _decode_image(body: bytes):
    buf = np.frombuffer(body, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return frame


def _serialize_metrics(metrics):
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
    if sift_detector is None or yolo_detector is None or cnn_classifier is None:
        raise HTTPException(status_code=503, detail="Detectors not initialized")

    body = await file.read()
    frame = _decode_image(body)

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
        raise HTTPException(status_code=500, detail=str(e)) from e

    return JSONResponse({
        "panels": panels,
        "metrics": _serialize_metrics(metrics),
    })


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")


@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


if os.path.isdir(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
