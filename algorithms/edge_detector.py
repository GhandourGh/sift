"""
Edge Detection module using Canny algorithm.

Metrics produced:
  - time_ms: processing time in milliseconds
  - edge_density: percentage of pixels classified as edges
  - edge_pixels: absolute count of edge pixels
"""

import cv2
import time
import config as cfg


def process(gray):
    """Run Canny edge detection on a grayscale frame.

    Args:
        gray: Grayscale image at display resolution.

    Returns:
        (edge_bgr, metrics): BGR edge image and metrics dict.
    """
    t0 = time.perf_counter()
    edges = cv2.Canny(gray, cfg.CANNY_LOW, cfg.CANNY_HIGH)
    dt = (time.perf_counter() - t0) * 1000

    total_pixels = edges.shape[0] * edges.shape[1]
    edge_pixels = int(cv2.countNonZero(edges))
    edge_density = (edge_pixels / total_pixels) * 100

    edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    metrics = {
        "time_ms": dt,
        "edge_density": edge_density,
        "edge_pixels": edge_pixels,
    }
    return edge_bgr, metrics
