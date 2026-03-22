"""
YOLO v8 Object Detection module.

Metrics produced:
  - time_ms: processing time in milliseconds
  - detections: total number of detections
  - avg_confidence: mean confidence across detections
  - classes: list of detected class names
"""

import cv2
import time
from ultralytics import YOLO
import config as cfg


class YOLODetector:
    def __init__(self):
        self.model = YOLO(cfg.MODEL_PATH)

    def process(self, small_bgr, display_size):
        """Run YOLO inference on a small BGR frame.

        Args:
            small_bgr: BGR image at processing resolution.
            display_size: (w, h) to resize output for display.

        Returns:
            (yolo_img, metrics): Annotated BGR image and metrics dict.
        """
        t0 = time.perf_counter()
        results = self.model(small_bgr, verbose=False, conf=cfg.YOLO_CONF_THRESH)
        dt = (time.perf_counter() - t0) * 1000

        result = results[0]
        boxes = result.boxes

        det_count = len(boxes)
        person_conf = 0.0
        if det_count > 0:
            confs = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            # Find person detections (class 0 in COCO)
            person_mask = class_ids == 0
            if person_mask.any():
                person_conf = float(confs[person_mask].max()) * 100

        yolo_img = result.plot()
        yolo_img = cv2.resize(yolo_img, display_size)

        metrics = {
            "time_ms": dt,
            "detections": det_count,
            "person_confidence": person_conf,
        }
        return yolo_img, metrics
