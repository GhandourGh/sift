"""
Streamlit web demo: one link, no app. Uses device camera and runs the 4-panel CV pipeline.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import streamlit as st

from algorithms.sift_detector import SIFTDetector
from algorithms.yolo_detector import YOLODetector
from pipeline import run_cv_pipeline

# Web-friendly panel size
WEB_PANEL_W, WEB_PANEL_H = 320, 240
WEB_PROCESS_W, WEB_PROCESS_H = 320, 240


@st.cache_resource
def get_detectors():
    """Load SIFT and YOLO once per session."""
    return SIFTDetector(), YOLODetector()


st.set_page_config(page_title="CV Demo", page_icon="📷", layout="centered")
st.title("CV Demo")
st.caption("No app — just use your camera. Original · Edge (Canny) · SIFT · YOLO")

img = st.camera_input("Capture a frame")

if img is not None:
    bytes_data = img.getvalue()
    buf = np.frombuffer(bytes_data, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is not None:
        sift_detector, yolo_detector = get_detectors()
        combined, _ = run_cv_pipeline(
            frame,
            sift_detector,
            yolo_detector,
            panel_w=WEB_PANEL_W,
            panel_h=WEB_PANEL_H,
            process_w=WEB_PROCESS_W,
            process_h=WEB_PROCESS_H,
        )
        # OpenCV is BGR; Streamlit expects RGB
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        st.image(combined_rgb, use_container_width=True, caption="Original | Edge | SIFT | YOLO")
    else:
        st.error("Could not decode the image. Try again.")
else:
    st.info("Allow camera access and capture a frame to see the 4-panel result.")
