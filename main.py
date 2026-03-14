#!/usr/bin/env python3
"""
Computer Vision Evolution Demo — V2 (Research Grade)

Controls:
  N/B/R/D — Toggle noise/blur/rotation/brightness
  A — Toggle all perturbations     0 — Reset all
  +/- — Adjust perturbation strength
  W — Toggle video / webcam
  V — Start/stop recording
  F — Save screenshot
  P — Pause/resume                 S — Save metrics CSV
  Q — Quit
"""

import logging
import os
import sys
import csv
import time

# Suppress hashlib blake2 errors (macOS pyenv)
class _SuppressHashlibErrors(logging.Filter):
    def filter(self, record):
        return "blake2" not in (record.getMessage() or "")

logging.getLogger().addFilter(_SuppressHashlibErrors())

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import config as cfg
from algorithms.sift_detector import SIFTDetector
from algorithms.yolo_detector import YOLODetector
from evaluation.perturbations import PerturbationManager
from pipeline import run_cv_pipeline


# ======================================================================
# EMA Smoother
# ======================================================================

class EMASmoother:
    def __init__(self, alpha=cfg.EMA_ALPHA):
        self.alpha = alpha
        self._state = {}

    def update(self, metrics):
        smoothed = {}
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                if key in self._state:
                    self._state[key] = self.alpha * val + (1 - self.alpha) * self._state[key]
                else:
                    self._state[key] = val
                smoothed[key] = self._state[key]
            else:
                smoothed[key] = val
        return smoothed


# ======================================================================
# Metrics Logger
# ======================================================================

class MetricsLogger:
    def __init__(self):
        self.rows = []

    def log(self, frame_num, perturb_status, edge_m, sift_m, yolo_m, fps):
        row = {
            "frame": frame_num,
            "perturbation": perturb_status,
            "fps": round(fps, 2),
            "edge_time_ms": round(edge_m["time_ms"], 2),
            "edge_density": round(edge_m["edge_density"], 2),
            "sift_time_ms": round(sift_m["time_ms"], 2),
            "sift_keypoints": sift_m["keypoints_filtered"],
            "sift_matches": sift_m["matches_good"],
            "sift_match_ratio": round(sift_m["match_ratio"], 1),
            "yolo_time_ms": round(yolo_m["time_ms"], 2),
            "yolo_detections": yolo_m["detections"],
            "yolo_person_conf": round(yolo_m["person_confidence"], 1),
        }
        self.rows.append(row)

    def save(self, path):
        if not self.rows:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"[INFO] Metrics saved to {path}")


# ======================================================================
# Title Card
# ======================================================================

def show_title_card(window_name):
    """Display a 3-second title card before the demo starts."""
    w = cfg.PANEL_W * 2
    h = cfg.PANEL_H * 2
    card = np.full((h, w, 3), (25, 25, 25), dtype=np.uint8)

    lines = [
        ("Computer Vision Evolution", 0.9, (220, 220, 220)),
        ("Edge Detection  |  SIFT Features  |  YOLOv8", 0.55, (140, 140, 140)),
        ("", 0, (0, 0, 0)),
        ("Robotics AI/ML Project", 0.5, (0, 180, 255)),
    ]

    y = h // 2 - 60
    for text, scale, color in lines:
        if not text:
            y += 30
            continue
        sz = cv2.getTextSize(text, cfg.FONT_FACE, scale, 2)[0]
        x = (w - sz[0]) // 2
        cv2.putText(card, text, (x + 2, y + 2), cfg.FONT_FACE, scale, (0, 0, 0), 3)
        cv2.putText(card, text, (x, y), cfg.FONT_FACE, scale, color, 2)
        y += sz[1] + 25

    # "Press any key to start" at bottom
    hint = "Press any key to start"
    sz = cv2.getTextSize(hint, cfg.FONT_FACE, 0.45, 1)[0]
    cv2.putText(card, hint, ((w - sz[0]) // 2, h - 40),
                cfg.FONT_FACE, 0.45, (100, 100, 100), 1)

    cv2.imshow(window_name, card)
    cv2.waitKey(0)


# ======================================================================
# Main
# ======================================================================

def main():
    # Initialize algorithms
    sift_detector = SIFTDetector()
    yolo_detector = YOLODetector()

    # Initialize systems
    perturb = PerturbationManager()
    ema_edge = EMASmoother()
    ema_sift = EMASmoother()
    ema_yolo = EMASmoother()
    logger = MetricsLogger()

    window_name = "CV Evolution Demo"

    # Title card
    show_title_card(window_name)

    # Open video source
    using_webcam = False
    source = cfg.VIDEO_SOURCE
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    print(f"[INFO] Source: {'webcam' if using_webcam else 'video file'}")

    frame_num = 0
    paused = False
    frozen = False
    frozen_frame = None
    recording = False
    video_writer = None
    combined = None

    while True:
        if not paused:
            ret, raw_frame = cap.read()
            if not ret:
                if not using_webcam:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            frame_num += 1
            t_start = time.perf_counter()

            display_frame = cv2.resize(raw_frame, (cfg.PANEL_W, cfg.PANEL_H - cfg.HEADER_H))

            if perturb.any_active:
                display_frame = perturb.apply(display_frame)

            perturb_status = perturb.status_text()
            combined, pipeline_metrics = run_cv_pipeline(
                display_frame, sift_detector, yolo_detector,
                original_metrics_lines=[perturb_status] if perturb.any_active else [],
            )
            edge_metrics_raw = pipeline_metrics["edge"]
            sift_metrics_raw = pipeline_metrics["sift"]
            yolo_metrics_raw = pipeline_metrics["yolo"]

            edge_metrics = ema_edge.update(edge_metrics_raw)
            sift_metrics = ema_sift.update(sift_metrics_raw)
            yolo_metrics = ema_yolo.update(yolo_metrics_raw)

            total_time = time.perf_counter() - t_start
            fps = 1.0 / total_time if total_time > 0 else 0

            logger.log(frame_num, perturb_status,
                       edge_metrics_raw, sift_metrics_raw, yolo_metrics_raw, fps)

            # Recording indicator
            if recording:
                cv2.circle(combined, (combined.shape[1] - 20, 16), 8, (0, 0, 255), -1)
                video_writer.write(combined)

        if frozen and frozen_frame is not None:
            cv2.imshow(window_name, frozen_frame)
        elif combined is not None:
            cv2.imshow(window_name, combined)

        # Keyboard handling
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
            print(f"[INFO] {'Paused' if paused else 'Resumed'}")
        elif key == ord("n"):
            perturb.toggle("noise")
            print(f"[INFO] Noise: {'ON' if perturb.active['noise'] else 'OFF'}")
        elif key == ord("b"):
            perturb.toggle("blur")
            print(f"[INFO] Blur: {'ON' if perturb.active['blur'] else 'OFF'}")
        elif key == ord("r"):
            perturb.toggle("rotation")
            print(f"[INFO] Rotation: {'ON' if perturb.active['rotation'] else 'OFF'}")
        elif key == ord("d"):
            perturb.toggle("brightness")
            print(f"[INFO] Brightness: {'ON' if perturb.active['brightness'] else 'OFF'}")
        elif key == ord("a"):
            perturb.toggle_all()
            print(f"[INFO] All perturbations: {'ON' if perturb.any_active else 'OFF'}")
        elif key == ord("0"):
            perturb.reset()
            print("[INFO] Perturbations reset")
        elif key in (ord("+"), ord("=")):
            perturb.increase_strength()
            print(f"[INFO] Strength increased: {perturb.status_text()}")
        elif key == ord("-"):
            perturb.decrease_strength()
            print(f"[INFO] Strength decreased: {perturb.status_text()}")
        elif key == ord("s"):
            save_path = os.path.join(cfg.BASE_DIR, "results",
                                     f"metrics_{int(time.time())}.csv")
            logger.save(save_path)

        # F — Screenshot
        elif key == ord("f"):
            if combined is not None:
                os.makedirs(os.path.join(cfg.BASE_DIR, "results"), exist_ok=True)
                path = os.path.join(cfg.BASE_DIR, "results",
                                    f"screenshot_{int(time.time())}.png")
                cv2.imwrite(path, combined)
                print(f"[INFO] Screenshot saved: {path}")

        # V — Toggle recording
        elif key == ord("v"):
            if not recording:
                os.makedirs(os.path.join(cfg.BASE_DIR, "results"), exist_ok=True)
                path = os.path.join(cfg.BASE_DIR, "results",
                                    f"recording_{int(time.time())}.mp4")
                h, w = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(path, fourcc, 20.0, (w, h))
                recording = True
                print(f"[INFO] Recording started: {path}")
            else:
                recording = False
                video_writer.release()
                video_writer = None
                print("[INFO] Recording stopped")

        # W — Toggle video / webcam
        elif key == ord("w"):
            cap.release()
            if using_webcam:
                cap = cv2.VideoCapture(cfg.VIDEO_SOURCE)
                using_webcam = False
                print("[INFO] Switched to video file")
            else:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    using_webcam = True
                    print("[INFO] Switched to webcam")
                else:
                    print("[WARN] Webcam not available, staying on video")
                    cap = cv2.VideoCapture(cfg.VIDEO_SOURCE)

        # Space — Freeze frame
        elif key == 32:
            if not frozen and combined is not None:
                frozen_frame = combined.copy()
                # Draw "FROZEN" indicator
                cv2.putText(frozen_frame, "FROZEN", (20, 40),
                            cfg.FONT_FACE, 0.8, (0, 0, 255), 2)
                frozen = True
                # Auto-save screenshot
                os.makedirs(os.path.join(cfg.BASE_DIR, "results"), exist_ok=True)
                path = os.path.join(cfg.BASE_DIR, "results",
                                    f"freeze_{int(time.time())}.png")
                cv2.imwrite(path, combined)
                print(f"[INFO] Frame frozen & saved: {path}")
            else:
                frozen = False
                frozen_frame = None
                print("[INFO] Unfrozen")

    # Cleanup
    if recording and video_writer:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

    if logger.rows:
        save_path = os.path.join(cfg.BASE_DIR, "results", "metrics_session.csv")
        logger.save(save_path)

    print(f"\n[INFO] Processed {frame_num} frames. Done.")


if __name__ == "__main__":
    main()
