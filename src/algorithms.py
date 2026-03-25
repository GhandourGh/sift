"""
All four CV algorithms in one module:
  - Canny Edge Detection (1986)
  - SIFT Feature Detection with FLANN matching (2004)
  - MobileNetV2 CNN Classification (2012)
  - YOLOv8 Object Detection (2015)
"""

import cv2
import numpy as np
import time
import config as cfg

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO


# ── Edge Detection ───────────────────────────────────────────

def edge_process(gray):
    """Run Canny edge detection on a grayscale frame."""
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


# ── SIFT Detector ────────────────────────────────────────────

class SIFTDetector:
    """SIFT keypoint detection with FLANN-based temporal tracking."""

    def __init__(self):
        self.sift = cv2.SIFT_create(
            nfeatures=cfg.SIFT_N_FEATURES,
            contrastThreshold=cfg.SIFT_CONTRAST_THRESH,
        )
        index_params = dict(algorithm=cfg.FLANN_INDEX_KDTREE, trees=cfg.FLANN_TREES)
        search_params = dict(checks=cfg.FLANN_CHECKS)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self._prev_des = None

    def process(self, small_bgr, gray_small, display_size):
        t0 = time.perf_counter()

        keypoints, descriptors = self.sift.detectAndCompute(gray_small, None)
        kp_count = len(keypoints)

        good_matches = []
        if (self._prev_des is not None and descriptors is not None
                and len(descriptors) >= 2 and len(self._prev_des) >= 2):
            matches = self.flann.knnMatch(descriptors, self._prev_des, k=2)
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < cfg.LOWE_RATIO * n.distance:
                        good_matches.append(m)

        vis = small_bgr.copy()
        matched_indices = set(m.queryIdx for m in good_matches)
        matched_kp = [kp for i, kp in enumerate(keypoints) if i in matched_indices]
        new_kp = [kp for i, kp in enumerate(keypoints) if i not in matched_indices]

        cv2.drawKeypoints(vis, matched_kp, vis, (0, 220, 0),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.drawKeypoints(vis, new_kp, vis, (0, 0, 220),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        dt = (time.perf_counter() - t0) * 1000
        self._prev_des = descriptors
        sift_img = cv2.resize(vis, display_size)

        metrics = {
            "time_ms": dt,
            "keypoints_raw": kp_count,
            "keypoints_filtered": kp_count,
            "matches_good": len(good_matches),
            "match_ratio": (len(good_matches) / kp_count * 100) if kp_count > 0 else 0.0,
        }
        return sift_img, metrics


# ── CNN Classifier ───────────────────────────────────────────

_IMAGENET_LABELS = None

def _load_imagenet_labels():
    global _IMAGENET_LABELS
    if _IMAGENET_LABELS is not None:
        return _IMAGENET_LABELS
    try:
        from torchvision.models import MobileNet_V2_Weights
        _IMAGENET_LABELS = MobileNet_V2_Weights.IMAGENET1K_V1.meta["categories"]
    except Exception:
        _IMAGENET_LABELS = [f"class_{i}" for i in range(1000)]
    return _IMAGENET_LABELS


class CNNClassifier:
    """MobileNetV2 image classification (ImageNet, 1000 classes)."""

    def __init__(self):
        from torchvision.models import MobileNet_V2_Weights
        self.weights = MobileNet_V2_Weights.IMAGENET1K_V1
        self.model = models.mobilenet_v2(weights=self.weights)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(cfg.CNN_INPUT_SIZE),
            transforms.CenterCrop(cfg.CNN_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.labels = _load_imagenet_labels()

    @torch.no_grad()
    def process(self, small_bgr, display_size):
        t0 = time.perf_counter()

        rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb).unsqueeze(0)

        output = self.model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)

        top_probs, top_indices = torch.topk(probs, cfg.CNN_TOP_K)
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        top5 = []
        for i in range(cfg.CNN_TOP_K):
            idx = int(top_indices[i])
            label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
            conf = float(top_probs[i]) * 100
            top5.append((label, conf))

        dt = (time.perf_counter() - t0) * 1000

        vis = small_bgr.copy()
        vis = self._draw_label(vis, top5)
        cnn_img = cv2.resize(vis, display_size)

        top1_label, top1_conf = top5[0] if top5 else ("unknown", 0.0)
        metrics = {
            "time_ms": dt,
            "top1_label": top1_label,
            "top1_confidence": top1_conf,
            "top5": top5,
        }
        return cnn_img, metrics

    def _draw_label(self, img, top5):
        if not top5:
            return img
        h, w = img.shape[:2]
        top1_label, top1_conf = top5[0]

        badge_h = 28
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - badge_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)

        short_label = top1_label[:24] if len(top1_label) > 24 else top1_label
        text = f"{short_label}  {top1_conf:.0f}%"
        cv2.putText(img, text, (8, h - 8),
                    cfg.FONT_FACE, 0.44, (200, 80, 255), 1, cv2.LINE_AA)
        return img


# ── YOLO Detector ────────────────────────────────────────────

class YOLODetector:
    """YOLOv8 real-time object detection."""

    def __init__(self):
        self.model = YOLO(cfg.MODEL_PATH)

    def process(self, small_bgr, display_size):
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
