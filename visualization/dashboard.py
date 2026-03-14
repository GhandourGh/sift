"""
Dashboard rendering — bottom bar with comparative metrics.

Shows:
  - Horizontal timing bar chart comparing all three algorithms
  - Key metrics summary for each algorithm
  - Active perturbation status
  - Keyboard controls hint
"""

import cv2
import numpy as np
import config as cfg


def _draw_text(img, text, pos, scale=cfg.FONT_SCALE_DASH, color=cfg.COLOR_TEXT,
               thickness=cfg.FONT_THICKNESS):
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), cfg.FONT_FACE, scale, (0, 0, 0), thickness + 1)
    cv2.putText(img, text, (x, y), cfg.FONT_FACE, scale, color, thickness)


def _draw_bar(img, x, y, width, height, value, max_value, color, label, value_text):
    """Draw a single horizontal bar with label."""
    bar_w = int((value / max_value) * width) if max_value > 0 else 0
    bar_w = max(bar_w, 1)

    # Background bar
    cv2.rectangle(img, (x, y), (x + width, y + height), cfg.COLOR_BAR_BG, -1)
    # Value bar
    cv2.rectangle(img, (x, y), (x + bar_w, y + height), color, -1)
    # Border
    cv2.rectangle(img, (x, y), (x + width, y + height), cfg.COLOR_BORDER, 1)
    # Label on the left
    _draw_text(img, label, (x - 50, y + height - 3), scale=cfg.FONT_SCALE_DASH, color=color)
    # Value on the right
    _draw_text(img, value_text, (x + width + 8, y + height - 3),
               scale=cfg.FONT_SCALE_DASH, color=cfg.COLOR_TEXT_DIM)


def render_dashboard(metrics_edge, metrics_sift, metrics_yolo,
                     fps, perturb_status, width):
    """Render the bottom dashboard bar.

    Args:
        metrics_edge: Edge detection metrics dict.
        metrics_sift: SIFT metrics dict.
        metrics_yolo: YOLO metrics dict.
        fps: Current FPS.
        perturb_status: String describing active perturbations.
        width: Total width in pixels.

    Returns:
        BGR image of the dashboard.
    """
    h = cfg.DASHBOARD_H
    dash = np.full((h, width, 3), cfg.COLOR_BG, dtype=np.uint8)

    # Top border line
    cv2.line(dash, (0, 0), (width, 0), cfg.COLOR_BORDER, 1)

    # --- Section 1: Timing bar chart (left third) ---
    section_w = width // 3
    _draw_text(dash, "Processing Time (ms)", (60, 18), scale=cfg.FONT_SCALE_DASH,
               color=cfg.COLOR_TEXT_DIM)

    t_edge = metrics_edge.get("time_ms", 0)
    t_sift = metrics_sift.get("time_ms", 0)
    t_yolo = metrics_yolo.get("time_ms", 0)
    max_t = max(t_edge, t_sift, t_yolo, 1)

    bar_x, bar_w = 110, section_w - 170
    _draw_bar(dash, bar_x, 28, bar_w, 14, t_edge, max_t,
              cfg.COLOR_EDGE, "Edge", f"{t_edge:.1f}ms")
    _draw_bar(dash, bar_x, 48, bar_w, 14, t_sift, max_t,
              cfg.COLOR_SIFT, "SIFT", f"{t_sift:.1f}ms")
    _draw_bar(dash, bar_x, 68, bar_w, 14, t_yolo, max_t,
              cfg.COLOR_YOLO, "YOLO", f"{t_yolo:.1f}ms")

    # FPS
    _draw_text(dash, f"FPS: {fps:.1f}", (60, 98), scale=cfg.FONT_SCALE_METRIC,
               color=cfg.COLOR_TEXT)

    # --- Section 2: Key metrics summary (middle third) ---
    sx = section_w + 20
    _draw_text(dash, "Algorithm Metrics", (sx, 18), scale=cfg.FONT_SCALE_DASH,
               color=cfg.COLOR_TEXT_DIM)

    edge_density = metrics_edge.get("edge_density", 0)
    kp_count = metrics_sift.get("keypoints_filtered", 0)
    matches = metrics_sift.get("matches_good", 0)
    match_ratio = metrics_sift.get("match_ratio", 0)
    det_count = metrics_yolo.get("detections", 0)
    avg_conf = metrics_yolo.get("avg_confidence", 0)
    classes = metrics_yolo.get("classes", [])

    _draw_text(dash, f"Edge density: {edge_density:.1f}%", (sx, 38), color=cfg.COLOR_EDGE)
    _draw_text(dash, f"SIFT keypoints: {kp_count}  |  Tracked: {matches} ({match_ratio:.0f}%)",
               (sx, 58), color=cfg.COLOR_SIFT)
    _draw_text(dash, f"YOLO detections: {det_count}  |  Avg conf: {avg_conf:.0f}%",
               (sx, 78), color=cfg.COLOR_YOLO)
    if classes:
        cls_str = ", ".join(classes[:5])
        if len(classes) > 5:
            cls_str += f" +{len(classes)-5}"
        _draw_text(dash, f"Classes: {cls_str}", (sx, 98), color=cfg.COLOR_TEXT_DIM)

    # --- Section 3: Perturbation status + controls (right third) ---
    rx = 2 * section_w + 20
    _draw_text(dash, "Perturbation", (rx, 18), scale=cfg.FONT_SCALE_DASH,
               color=cfg.COLOR_TEXT_DIM)

    perturb_color = cfg.COLOR_PERTURB if perturb_status != "None" else cfg.COLOR_TEXT_DIM
    _draw_text(dash, perturb_status, (rx, 38), color=perturb_color)

    # Controls help
    _draw_text(dash, "Controls:", (rx, 62), color=cfg.COLOR_TEXT_DIM)
    _draw_text(dash, "N:Noise  B:Blur  R:Rotate  D:Dark", (rx, 78),
               scale=cfg.FONT_SCALE_HELP, color=cfg.COLOR_TEXT_DIM)
    _draw_text(dash, "A:All  +/-:Strength  0:Reset  S:Save", (rx, 93),
               scale=cfg.FONT_SCALE_HELP, color=cfg.COLOR_TEXT_DIM)
    _draw_text(dash, "P:Pause  Q:Quit", (rx, 108),
               scale=cfg.FONT_SCALE_HELP, color=cfg.COLOR_TEXT_DIM)

    return dash
