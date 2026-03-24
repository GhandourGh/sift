# CV Evolution Demo

A real-time web dashboard that runs four computer vision algorithms side-by-side, visualizing the evolution of the field from 1986 to today.

**Edge Detection (1986) → SIFT Features (2004) → CNN Classification (2012) → YOLOv8 Detection (2015)**

Each algorithm runs on the same live camera frame simultaneously. You can apply perturbations (noise, blur, rotation, brightness) and watch how each algorithm responds differently.

---

## Project Structure

```
├── algorithms/
│   ├── edge_detector.py       Canny edge detection
│   ├── sift_detector.py       SIFT keypoint detection + FLANN tracking
│   ├── cnn_classifier.py      MobileNetV2 ImageNet classification
│   └── yolo_detector.py       YOLOv8 object detection
├── evaluation/
│   ├── perturbations.py       Noise / blur / rotation / brightness
│   └── metrics.py             Normalization utilities
├── visualization/
│   └── panels.py              Panel rendering (header, overlay, footer)
├── web/
│   ├── index.html             Dashboard layout
│   ├── style.css              Dark theme, CSS Grid
│   ├── app.js                 Camera capture loop + API calls
│   └── public/sift.mp4        Sample video for demo mode
├── config.py                  All tunable parameters
├── pipeline.py                Runs all 4 algorithms per frame
└── web_app.py                 FastAPI server
```

---

## The Four Algorithms

### 1. Canny Edge Detection — 1986
The oldest algorithm. Finds boundaries in an image by detecting sharp changes in pixel intensity. Fast but produces no semantic understanding — it sees *shapes*, not *objects*.

```python
# edge_detector.py
edges = cv2.Canny(gray, cfg.CANNY_LOW, cfg.CANNY_HIGH)
edge_density = (cv2.countNonZero(edges) / total_pixels) * 100
```

**Metrics:** edge density (%), processing time

---

### 2. SIFT Features — 2004
Scale-Invariant Feature Transform. Detects distinctive keypoints and tracks them across frames using FLANN descriptor matching. Understands *structure* but not meaning.

```python
# sift_detector.py — keypoint detection + temporal tracking
keypoints, descriptors = self.sift.detectAndCompute(gray_small, None)

# FLANN matcher for tracking keypoints across frames
matcher = cv2.FlannBasedMatcher(index_params, search_params)
matches = matcher.knnMatch(self.prev_descriptors, descriptors, k=2)

# Lowe's ratio test to filter good matches
good = [m for m, n in matches if m.distance < cfg.LOWE_RATIO * n.distance]
```

**Metrics:** keypoints detected, matches tracked, processing time

---

### 3. CNN Classification — 2012
MobileNetV2 pretrained on ImageNet (1000 classes). The first algorithm that understands *what* is in the image. No spatial awareness — it classifies the whole scene as one label.

```python
# cnn_classifier.py
self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
self.model.eval()

# ImageNet preprocessing
self.transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Inference
output = self.model(input_tensor)
probs = torch.nn.functional.softmax(output[0], dim=0)
top_probs, top_indices = torch.topk(probs, cfg.CNN_TOP_K)
```

**Metrics:** top-1 label, confidence score, processing time

---

### 4. YOLOv8 Detection — 2015
You Only Look Once. Real-time object detection — knows *what* AND *where* in a single forward pass. The culmination of the evolution: semantic understanding with precise localization.

```python
# yolo_detector.py
self.model = YOLO(cfg.MODEL_PATH)  # yolov8n.pt

results = self.model(small_bgr, verbose=False, conf=cfg.YOLO_CONF_THRESH)
boxes = results[0].boxes

# Extract person confidence (class 0 in COCO dataset)
person_mask = class_ids == 0
person_conf = float(confs[person_mask].max()) * 100
```

**Metrics:** objects detected, person confidence, processing time

---

## Pipeline Architecture

Each browser frame goes through this path:

```
Browser (webcam/video)
    │
    │  JPEG blob via FormData
    ▼
FastAPI  POST /process-frame
    │
    ├── Apply perturbations (if active)
    │
    ├── Edge Detection ──────────┐
    ├── SIFT Detection ──────────┤  run_cv_pipeline()
    ├── CNN Classification ──────┤
    └── YOLOv8 Detection ────────┘
                                 │
                         base64 panel images
                         + metrics JSON
                                 │
                                 ▼
                        Browser updates panels
                        and metrics in real-time
```

The pipeline returns structured JSON so the frontend can render panels independently and display live metrics without page reloads:

```python
# pipeline.py
return JSONResponse({
    "panels": {
        "edge": "<base64 JPEG>",
        "sift": "<base64 JPEG>",
        "cnn":  "<base64 JPEG>",
        "yolo": "<base64 JPEG>",
    },
    "metrics": {
        "edge": {"edge_density": 4.1, "time_ms": 0.6},
        "sift": {"keypoints_filtered": 148, "matches_good": 88, "time_ms": 14.9},
        "cnn":  {"top1_label": "space shuttle", "top1_confidence": 41.0, "time_ms": 50.1},
        "yolo": {"detections": 2, "person_confidence": 85.7, "time_ms": 54.9},
    }
})
```

---

## Perturbation System

Four perturbations can be toggled and adjusted with sliders at the bottom of the dashboard:

| Perturbation | Range | Effect |
|---|---|---|
| Noise | 0 – 120 σ | Gaussian noise added to pixels |
| Blur | 3 – 51 kernel | Gaussian blur |
| Rotation | 0° – 90° | Frame rotation around center |
| Brightness | -180 – +100 | Pixel intensity shift |

```python
# evaluation/perturbations.py
def apply(self, frame):
    if self.active["noise"]:
        noise = np.random.normal(0, self.params["noise"], frame.shape)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if self.active["blur"]:
        k = int(self.params["blur"]) | 1  # ensure odd kernel
        frame = cv2.GaussianBlur(frame, (k, k), 0)
    # ... rotation, brightness
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | Web server |
| `opencv-python-headless` | Image processing, Canny, SIFT |
| `torch` + `torchvision` | MobileNetV2 CNN |
| `ultralytics` | YOLOv8 |
| `numpy` | Array operations |
| `python-multipart` | File upload support |

See [HOWTORUN.md](HOWTORUN.md) to get started.
