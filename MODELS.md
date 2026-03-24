# Models Used

This project uses four computer vision algorithms that represent the evolution of the field.

---

## 1. Canny Edge Detection (1986)

Detects edges by finding sharp intensity gradients. No learning involved — purely mathematical.

```python
edges = cv2.Canny(gray, cfg.CANNY_LOW, cfg.CANNY_HIGH)

total_pixels = edges.shape[0] * edges.shape[1]
edge_pixels = int(cv2.countNonZero(edges))
edge_density = (edge_pixels / total_pixels) * 100
```

- **Input:** Grayscale image
- **Output:** Binary edge map + edge density percentage
- **Speed:** ~0.5 ms per frame (fastest algorithm)
- **Limitation:** Finds boundaries only — no understanding of what the objects are

---

## 2. SIFT — Scale-Invariant Feature Transform (2004)

Detects distinctive keypoints that are invariant to scale, rotation, and illumination. Uses FLANN-based matching to track features across consecutive frames.

```python
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

# FLANN matcher for temporal tracking
matcher = cv2.FlannBasedMatcher(
    dict(algorithm=1, trees=5),
    dict(checks=50)
)
matches = matcher.knnMatch(prev_descriptors, descriptors, k=2)

# Lowe's ratio test — filters unreliable matches
good = [m for m, n in matches if m.distance < 0.7 * n.distance]
```

- **Input:** Grayscale image
- **Output:** Keypoint locations + number of tracked matches
- **Speed:** ~15 ms per frame
- **Limitation:** Knows *where* features are, but not *what* they represent

---

## 3. MobileNetV2 — CNN Classification (2012 era)

A lightweight convolutional neural network pretrained on ImageNet (1000 classes). Classifies the entire image into a single category with a confidence score.

```python
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights

model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

with torch.no_grad():
    output = model(transform(rgb).unsqueeze(0))
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top_probs, top_indices = torch.topk(probs, 5)
```

- **Input:** 224x224 RGB image (ImageNet standard)
- **Output:** Top-5 predicted class labels + confidence scores
- **Speed:** ~50 ms per frame
- **Limitation:** Knows *what* is in the image, but not *where* — classifies the whole scene as one label

---

## 4. YOLOv8 — Real-Time Object Detection (2015+)

You Only Look Once. Detects multiple objects in a single forward pass, providing both class labels and bounding box coordinates.

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # nano variant for speed

results = model(frame, verbose=False, conf=0.25)
boxes = results[0].boxes

det_count = len(boxes)
confs = boxes.conf.cpu().numpy()
class_ids = boxes.cls.cpu().numpy().astype(int)

# Find person detections (class 0 in COCO)
person_mask = class_ids == 0
if person_mask.any():
    person_conf = float(confs[person_mask].max()) * 100
```

- **Input:** BGR image (any size)
- **Output:** Bounding boxes + class labels + confidence scores for every detected object
- **Speed:** ~55 ms per frame
- **Key advantage:** Knows *what* AND *where* — the culmination of the CV evolution

---

## Evolution Summary

| Algorithm | Year | Knows What? | Knows Where? | Learning? |
|-----------|------|-------------|--------------|-----------|
| Canny Edge | 1986 | No | Boundaries only | No |
| SIFT | 2004 | No | Feature points | No |
| CNN (MobileNetV2) | 2012 | Yes | No | Yes (ImageNet) |
| YOLOv8 | 2015+ | Yes | Yes (bounding boxes) | Yes (COCO) |
