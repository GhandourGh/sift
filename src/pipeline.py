"""
CV pipeline: runs all 4 algorithms on a frame, renders panels, returns results.
Also contains the perturbation system and panel rendering.
"""

import base64
import cv2
import numpy as np
import config as cfg
from algorithms import edge_process


# ── Perturbations ────────────────────────────────────────────

def _apply_noise(frame, sigma):
    noise = np.random.normal(0, sigma, frame.shape).astype(np.float32)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def _apply_blur(frame, kernel_size):
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(frame, (k, k), 0)

def _apply_rotation(frame, angle):
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def _apply_brightness(frame, delta):
    return np.clip(frame.astype(np.int16) + delta, 0, 255).astype(np.uint8)


class PerturbationManager:
    """Manages active perturbations and their strengths."""

    TYPES = ["noise", "blur", "rotation", "brightness"]

    def __init__(self):
        self.active = {t: False for t in self.TYPES}
        self.params = {
            "noise": cfg.NOISE_SIGMA_DEFAULT,
            "blur": cfg.BLUR_KERNEL_DEFAULT,
            "rotation": cfg.ROTATION_ANGLE_DEFAULT,
            "brightness": cfg.BRIGHTNESS_DELTA_DEFAULT,
        }

    @property
    def any_active(self):
        return any(self.active.values())

    def set_from_dict(self, params):
        for ptype in self.TYPES:
            if ptype in params:
                p = params[ptype]
                self.active[ptype] = p.get("active", False)
                if "value" in p:
                    self.params[ptype] = p["value"]

    def apply(self, frame):
        out = frame.copy()
        if self.active["noise"]:
            out = _apply_noise(out, self.params["noise"])
        if self.active["blur"]:
            out = _apply_blur(out, self.params["blur"])
        if self.active["rotation"]:
            out = _apply_rotation(out, self.params["rotation"])
        if self.active["brightness"]:
            out = _apply_brightness(out, self.params["brightness"])
        return out


# ── Panel Rendering ──────────────────────────────────────────

def _draw_text(img, text, pos, scale=cfg.FONT_SCALE_METRIC, color=cfg.COLOR_TEXT,
               thickness=cfg.FONT_THICKNESS):
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), cfg.FONT_FACE, scale, (0, 0, 0), thickness + 1)
    cv2.putText(img, text, (x, y), cfg.FONT_FACE, scale, color, thickness)


def render_panel(image, title, metrics_lines, accent_color, panel_size=None):
    """Render a single algorithm panel with header and metrics overlay."""
    pw = panel_size[0] if panel_size else cfg.PANEL_W
    ph = panel_size[1] if panel_size else cfg.PANEL_H
    content_h = ph - cfg.HEADER_H

    panel = np.full((ph, pw, 3), cfg.COLOR_BG, dtype=np.uint8)

    # Header
    cv2.rectangle(panel, (0, 0), (pw, cfg.HEADER_H), cfg.COLOR_HEADER_BG, -1)
    cv2.rectangle(panel, (0, 0), (4, cfg.HEADER_H), accent_color, -1)
    text_size = cv2.getTextSize(title, cfg.FONT_FACE, cfg.FONT_SCALE_TITLE, cfg.FONT_THICKNESS)[0]
    text_x = (pw - text_size[0]) // 2
    _draw_text(panel, title, (text_x, cfg.HEADER_H - 10),
               scale=cfg.FONT_SCALE_TITLE, color=accent_color)

    # Content image
    content = cv2.resize(image, (pw, content_h))
    panel[cfg.HEADER_H:cfg.HEADER_H + content_h, 0:pw] = content

    # Metrics overlay
    if metrics_lines:
        overlay_h = 18 * len(metrics_lines) + 8
        y_start = ph - overlay_h
        overlay = panel[y_start:ph, 0:pw].copy()
        dark = np.full_like(overlay, 0)
        cv2.addWeighted(overlay, 0.4, dark, 0.6, 0, overlay)
        panel[y_start:ph, 0:pw] = overlay

        for i, line in enumerate(metrics_lines):
            _draw_text(panel, line, (8, y_start + 15 + i * 18),
                       scale=cfg.FONT_SCALE_METRIC, color=cfg.COLOR_TEXT)

    cv2.rectangle(panel, (0, 0), (pw - 1, ph - 1), cfg.COLOR_BORDER, 1)
    return panel


# ── Pipeline ─────────────────────────────────────────────────

def _encode_panel_jpeg(panel_img, quality=85):
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buf = cv2.imencode(".jpg", panel_img, params)
    return base64.b64encode(buf).decode("ascii")


def run_cv_pipeline(frame_bgr, sift_detector, yolo_detector, cnn_classifier,
                    encode_panels=True):
    """Run all 4 algorithms on a single BGR frame.

    Returns:
        panels: dict of base64 JPEG strings (or raw BGR arrays if encode_panels=False)
        metrics: dict with edge, sift, cnn, yolo sub-dicts
    """
    pw, ph = cfg.PANEL_W, cfg.PANEL_H
    content_h = ph - cfg.HEADER_H
    display_size = (pw, content_h)

    display_frame = cv2.resize(frame_bgr, (pw, content_h))
    process_frame = cv2.resize(frame_bgr, (cfg.PROCESS_W, cfg.PROCESS_H))

    gray_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)

    # Run all 4 algorithms
    edge_img, edge_metrics = edge_process(gray_display)
    sift_img, sift_metrics = sift_detector.process(process_frame, gray_small, display_size)
    cnn_img, cnn_metrics = cnn_classifier.process(process_frame, display_size)
    yolo_img, yolo_metrics = yolo_detector.process(process_frame, display_size)

    # Render panels
    panel_edge = render_panel(
        edge_img, "Edge Detection (Canny)",
        [f"{edge_metrics['edge_density']:.1f}% density"],
        cfg.COLOR_EDGE, panel_size=(pw, ph))

    panel_sift = render_panel(
        sift_img, "SIFT Features",
        [f"{int(sift_metrics['keypoints_filtered'])} kp | {int(sift_metrics['matches_good'])} tracked"],
        cfg.COLOR_SIFT, panel_size=(pw, ph))

    top1 = cnn_metrics.get("top1_label", "?")
    top1_conf = cnn_metrics.get("top1_confidence", 0)
    panel_cnn = render_panel(
        cnn_img, "CNN (MobileNetV2)",
        [f"{top1}: {top1_conf:.0f}%"],
        cfg.COLOR_CNN, panel_size=(pw, ph))

    person_conf = yolo_metrics.get("person_confidence", 0)
    yolo_line = f"Person: {person_conf:.0f}%" if person_conf > 0 else "No person"
    panel_yolo = render_panel(
        yolo_img, "YOLOv8 Detection",
        [yolo_line],
        cfg.COLOR_YOLO, panel_size=(pw, ph))

    if encode_panels:
        panels = {
            "edge": _encode_panel_jpeg(panel_edge),
            "sift": _encode_panel_jpeg(panel_sift),
            "cnn": _encode_panel_jpeg(panel_cnn),
            "yolo": _encode_panel_jpeg(panel_yolo),
        }
    else:
        panels = {
            "edge": panel_edge, "sift": panel_sift,
            "cnn": panel_cnn, "yolo": panel_yolo,
        }

    metrics = {
        "edge": edge_metrics, "sift": sift_metrics,
        "cnn": cnn_metrics, "yolo": yolo_metrics,
    }
    return panels, metrics
