"""
Reusable per-frame CV pipeline for the 4-panel demo.
Used by both the desktop app (main.py) and the web app (web_app.py).
"""

import cv2
import config as cfg
from algorithms.edge_detector import process as edge_process
from visualization.panels import render_panel, build_grid


def run_cv_pipeline(frame_bgr, sift_detector, yolo_detector,
                    panel_w=None, panel_h=None, process_w=None, process_h=None,
                    original_metrics_lines=None):
    """Run Canny, SIFT, and YOLO on a single BGR frame and return the 4-panel composite.

    Args:
        frame_bgr: Input BGR image (any size; will be resized).
        sift_detector: SIFTDetector instance (shared across calls).
        yolo_detector: YOLODetector instance (shared across calls).
        panel_w, panel_h: Panel size for display. Defaults to cfg.PANEL_W, cfg.PANEL_H.
        process_w, process_h: Size used for SIFT/YOLO inference. Defaults to cfg.PROCESS_W, cfg.PROCESS_H.
        original_metrics_lines: Optional list of strings for the Original panel (e.g. perturbation status).

    Returns:
        combined: BGR image of the 2x2 grid (original, edge, SIFT, YOLO).
        metrics: Dict with edge, sift, yolo sub-dicts for optional overlay/API.
    """
    pw = panel_w if panel_w is not None else cfg.PANEL_W
    ph = panel_h if panel_h is not None else cfg.PANEL_H
    pwc = process_w if process_w is not None else cfg.PROCESS_W
    phc = process_h if process_h is not None else cfg.PROCESS_H
    content_h = ph - cfg.HEADER_H
    display_size = (pw, content_h)

    display_frame = cv2.resize(frame_bgr, (pw, content_h))
    process_frame = cv2.resize(frame_bgr, (pwc, phc))

    gray_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)

    edge_img, edge_metrics = edge_process(gray_display)
    sift_img, sift_metrics = sift_detector.process(
        process_frame, gray_small, display_size
    )
    yolo_img, yolo_metrics = yolo_detector.process(process_frame, display_size)

    orig_lines = original_metrics_lines if original_metrics_lines is not None else []
    panel_orig = render_panel(
        display_frame, "Original", orig_lines, cfg.COLOR_ORIGINAL, panel_size=(pw, ph)
    )
    panel_edge = render_panel(
        edge_img, "Edge Detection (Canny)",
        [f"{edge_metrics['edge_density']:.1f}% density"],
        cfg.COLOR_EDGE, panel_size=(pw, ph)
    )
    panel_sift = render_panel(
        sift_img, "SIFT",
        [f"{int(sift_metrics['keypoints_filtered'])} kp | {int(sift_metrics['matches_good'])} tracked"],
        cfg.COLOR_SIFT, panel_size=(pw, ph)
    )
    person_conf = yolo_metrics.get("person_confidence", 0)
    yolo_line = f"Person: {person_conf:.0f}%" if person_conf > 0 else "No person"
    panel_yolo = render_panel(
        yolo_img, "YOLOv8 Detection", [yolo_line], cfg.COLOR_YOLO, panel_size=(pw, ph)
    )

    combined = build_grid(panel_orig, panel_edge, panel_sift, panel_yolo)
    metrics = {
        "edge": edge_metrics,
        "sift": sift_metrics,
        "yolo": yolo_metrics,
    }
    return combined, metrics
