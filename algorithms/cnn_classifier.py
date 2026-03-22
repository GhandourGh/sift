"""
CNN Image Classification module using MobileNetV2.

Fills the evolution gap between SIFT (handcrafted features) and YOLO (detection):
  SIFT knows WHERE features are → CNN knows WHAT is in the image → YOLO knows WHAT + WHERE

Metrics produced:
  - time_ms: processing time in milliseconds
  - top1_label: predicted class name
  - top1_confidence: confidence percentage (0-100)
  - top5: list of (label, confidence) tuples
"""

import cv2
import numpy as np
import time
import config as cfg

import torch
import torchvision.models as models
import torchvision.transforms as transforms


# ImageNet class labels (loaded once)
_IMAGENET_LABELS = None


def _load_imagenet_labels():
    """Load ImageNet class labels from torchvision metadata."""
    global _IMAGENET_LABELS
    if _IMAGENET_LABELS is not None:
        return _IMAGENET_LABELS
    try:
        from torchvision.models import MobileNet_V2_Weights
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        _IMAGENET_LABELS = weights.meta["categories"]
    except Exception:
        # Fallback: generate numeric labels
        _IMAGENET_LABELS = [f"class_{i}" for i in range(1000)]
    return _IMAGENET_LABELS


class CNNClassifier:
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
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.labels = _load_imagenet_labels()

    @torch.no_grad()
    def process(self, small_bgr, display_size):
        """Run CNN classification on a small BGR frame.

        Args:
            small_bgr: BGR image at processing resolution.
            display_size: (w, h) to resize output for display.

        Returns:
            (cnn_img, metrics): Annotated BGR image and metrics dict.
        """
        t0 = time.perf_counter()

        # Convert BGR to RGB for torchvision
        rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb).unsqueeze(0)

        # Inference
        output = self.model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)

        # Top-K predictions
        top_k = cfg.CNN_TOP_K
        top_probs, top_indices = torch.topk(probs, top_k)
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        top5 = []
        for i in range(top_k):
            idx = int(top_indices[i])
            label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
            conf = float(top_probs[i]) * 100
            top5.append((label, conf))

        dt = (time.perf_counter() - t0) * 1000

        # Draw annotated image with top-5 label bars
        vis = small_bgr.copy()
        vis = self._draw_labels(vis, top5)
        cnn_img = cv2.resize(vis, display_size)

        top1_label, top1_conf = top5[0] if top5 else ("unknown", 0.0)

        metrics = {
            "time_ms": dt,
            "top1_label": top1_label,
            "top1_confidence": top1_conf,
            "top5": top5,
        }
        return cnn_img, metrics

    def _draw_labels(self, img, top5):
        """Draw top-1 classification as a clean badge at the bottom."""
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
