"""
Panel rendering for the 2x2 algorithm comparison grid.

Each panel has:
  - A colored header bar with the algorithm name
  - The algorithm output image
  - A semi-transparent metrics overlay at the bottom
"""

import cv2
import numpy as np
import config as cfg


def _draw_text(img, text, pos, scale=cfg.FONT_SCALE_METRIC, color=cfg.COLOR_TEXT,
               thickness=cfg.FONT_THICKNESS):
    """Draw text with a subtle shadow for readability."""
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), cfg.FONT_FACE, scale, (0, 0, 0), thickness + 1)
    cv2.putText(img, text, (x, y), cfg.FONT_FACE, scale, color, thickness)


def render_panel(image, title, metrics_lines, accent_color, panel_size=None):
    """Render a single algorithm panel with header and metrics.

    Args:
        image: BGR image (will be resized to panel content area).
        title: Panel title string.
        metrics_lines: List of strings to display as metrics.
        accent_color: BGR tuple for the header accent.
        panel_size: (w, h) total panel size. Defaults to config values.

    Returns:
        BGR image of the complete panel.
    """
    pw = panel_size[0] if panel_size else cfg.PANEL_W
    ph = panel_size[1] if panel_size else cfg.PANEL_H
    content_h = ph - cfg.HEADER_H

    # Create panel canvas
    panel = np.full((ph, pw, 3), cfg.COLOR_BG, dtype=np.uint8)

    # Header bar
    cv2.rectangle(panel, (0, 0), (pw, cfg.HEADER_H), cfg.COLOR_HEADER_BG, -1)
    # Accent stripe on left
    cv2.rectangle(panel, (0, 0), (4, cfg.HEADER_H), accent_color, -1)
    # Title text — centered
    text_size = cv2.getTextSize(title, cfg.FONT_FACE, cfg.FONT_SCALE_TITLE, cfg.FONT_THICKNESS)[0]
    text_x = (pw - text_size[0]) // 2
    _draw_text(panel, title, (text_x, cfg.HEADER_H - 10),
               scale=cfg.FONT_SCALE_TITLE, color=accent_color)

    # Resize and place content image
    content = cv2.resize(image, (pw, content_h))
    panel[cfg.HEADER_H:cfg.HEADER_H + content_h, 0:pw] = content

    # Semi-transparent metrics overlay at bottom
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

    # Border
    cv2.rectangle(panel, (0, 0), (pw - 1, ph - 1), cfg.COLOR_BORDER, 1)

    return panel


def build_grid(panel_orig, panel_edge, panel_sift, panel_yolo):
    """Combine four panels into a 2x2 grid.

    Returns:
        BGR image of the combined grid.
    """
    top = np.hstack((panel_orig, panel_edge))
    bottom = np.hstack((panel_sift, panel_yolo))
    return np.vstack((top, bottom))
