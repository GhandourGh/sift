"""
Per-frame CV pipeline for the 4-algorithm demo.

Runs Edge Detection, SIFT, CNN Classification, and YOLO Detection
on a single frame. Returns individual panel images and structured metrics.
"""

import base64
import cv2
import config as cfg
from algorithms.edge_detector import process as edge_process
from visualization.panels import render_panel


def _encode_panel_jpeg(panel_img, quality=85):
    """Encode a BGR panel image to base64 JPEG string."""
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buf = cv2.imencode(".jpg", panel_img, params)
    return base64.b64encode(buf).decode("ascii")


def run_cv_pipeline(frame_bgr, sift_detector, yolo_detector, cnn_classifier,
                    panel_w=None, panel_h=None, process_w=None, process_h=None,
                    encode_panels=True):
    """Run all 4 algorithms on a single BGR frame.

    Args:
        frame_bgr: Input BGR image (any size; will be resized).
        sift_detector: SIFTDetector instance.
        yolo_detector: YOLODetector instance.
        cnn_classifier: CNNClassifier instance.
        panel_w, panel_h: Panel size for display.
        process_w, process_h: Size used for inference.
        encode_panels: If True, return base64-encoded JPEG panels.
                       If False, return raw BGR numpy arrays.

    Returns:
        panels: Dict of panel images (base64 strings or BGR arrays).
        metrics: Dict with edge, sift, cnn, yolo sub-dicts.
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

    # Run all 4 algorithms
    edge_img, edge_metrics = edge_process(gray_display)
    sift_img, sift_metrics = sift_detector.process(process_frame, gray_small, display_size)
    cnn_img, cnn_metrics = cnn_classifier.process(process_frame, display_size)
    yolo_img, yolo_metrics = yolo_detector.process(process_frame, display_size)

    # Render panels with headers and metrics overlays
    panel_edge = render_panel(
        edge_img, "Edge Detection (Canny)",
        [f"{edge_metrics['edge_density']:.1f}% density"],
        cfg.COLOR_EDGE, panel_size=(pw, ph)
    )
    panel_sift = render_panel(
        sift_img, "SIFT Features",
        [f"{int(sift_metrics['keypoints_filtered'])} kp | {int(sift_metrics['matches_good'])} tracked"],
        cfg.COLOR_SIFT, panel_size=(pw, ph)
    )

    top1 = cnn_metrics.get("top1_label", "?")
    top1_conf = cnn_metrics.get("top1_confidence", 0)
    panel_cnn = render_panel(
        cnn_img, "CNN (MobileNetV2)",
        [f"{top1}: {top1_conf:.0f}%"],
        cfg.COLOR_CNN, panel_size=(pw, ph)
    )

    person_conf = yolo_metrics.get("person_confidence", 0)
    yolo_line = f"Person: {person_conf:.0f}%" if person_conf > 0 else "No person"
    panel_yolo = render_panel(
        yolo_img, "YOLOv8 Detection",
        [yolo_line],
        cfg.COLOR_YOLO, panel_size=(pw, ph)
    )

    if encode_panels:
        panels = {
            "edge": _encode_panel_jpeg(panel_edge),
            "sift": _encode_panel_jpeg(panel_sift),
            "cnn": _encode_panel_jpeg(panel_cnn),
            "yolo": _encode_panel_jpeg(panel_yolo),
        }
    else:
        panels = {
            "edge": panel_edge,
            "sift": panel_sift,
            "cnn": panel_cnn,
            "yolo": panel_yolo,
        }

    metrics = {
        "edge": edge_metrics,
        "sift": sift_metrics,
        "cnn": cnn_metrics,
        "yolo": yolo_metrics,
    }

    return panels, metrics
