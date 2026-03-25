# The Evolution of Computer Vision

A real-time visual simulation that demonstrates how computer vision evolved over three decades — from simple edge detection to deep learning — by running four algorithms on the same camera feed simultaneously.

> **Goal:** Show how machines went from seeing *edges* to understanding *objects*, and let the viewer compare each method's strengths and weaknesses in real time.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [The Evolution](#the-evolution-of-computer-vision)
3. [Stage 1 — Edge Detection (1986)](#stage-1--edge-detection-canny-1986)
4. [Stage 2 — Feature Detection (SIFT, 2004)](#stage-2--feature-detection-sift-2004)
5. [Stage 3 — Deep Learning](#stage-3--deep-learning)
   - [CNN Classification (2012)](#cnn-classification-mobilenetv2-2012)
   - [YOLO Object Detection (2015)](#yolo-object-detection-yolov8-2015)
6. [Comparison Between Methods](#comparison-between-methods)
7. [The Simulation](#the-simulation)
8. [Robustness Testing](#robustness-testing)
9. [Project Structure](#project-structure)
10. [Technologies Used](#technologies-used)

---

## Project Overview

This project is a **visual perception simulation** built with Python and OpenCV.

A camera (or a pre-recorded video) provides frames to the system, and the system processes each frame using **four different computer vision methods** at the same time:

- **Edge Detection** — sees outlines and boundaries
- **SIFT** — finds and tracks distinctive features
- **CNN** — classifies what is in the image
- **YOLO** — detects and locates every object

The results are displayed side-by-side in a web dashboard so you can directly compare what each method "sees" in the same frame. This makes it easy to understand what type of information each method extracts, how robust each one is, and why the field evolved the way it did.

---

## The Evolution of Computer Vision

Computer vision has gone through three major stages:

```
1986                    2004                    2012              2015
 │                       │                       │                 │
 ▼                       ▼                       ▼                 ▼
Edge Detection  →   SIFT Features   →   CNN Classification  →  YOLO Detection
(sees shapes)       (sees structure)     (knows WHAT)          (knows WHAT + WHERE)
```

Each stage solved a problem that the previous one could not:

| Stage | Method | What it understands | What it cannot do |
|-------|--------|--------------------|--------------------|
| 1 | Edge Detection | Where brightness changes (outlines) | Cannot tell what the object is |
| 2 | SIFT | Distinctive points that can be matched | Cannot classify or name objects |
| 3a | CNN | What the object is (classification) | Cannot tell where in the image |
| 3b | YOLO | What the object is AND where it is | Requires large datasets and GPU |

---

## Stage 1 — Edge Detection (Canny, 1986)

### What it is
Edge detection is the oldest and simplest computer vision technique. It finds **boundaries** in an image by looking for sharp changes in pixel brightness — the places where one region ends and another begins.

### How it works (step by step)
1. Convert the image to grayscale
2. Apply the **Canny algorithm**, which scans every pixel and measures the gradient (rate of brightness change)
3. If the gradient is above a threshold, that pixel is marked as an **edge**
4. The result is a black-and-white image where white pixels are edges

### Implementation in this project
```python
# src/algorithms.py — edge_process()
edges = cv2.Canny(gray, cfg.CANNY_LOW, cfg.CANNY_HIGH)

total_pixels = edges.shape[0] * edges.shape[1]
edge_pixels = int(cv2.countNonZero(edges))
edge_density = (edge_pixels / total_pixels) * 100
```
- Uses **OpenCV's Canny** function with two thresholds (low=30, high=80)
- Measures **edge density** — the percentage of pixels classified as edges

### Real-world applications
- Industrial quality inspection (detecting cracks, defects)
- Medical imaging (organ boundary detection)
- Preprocessing step for more advanced algorithms

### Limitations
- Produces only outlines — no understanding of what the object is
- Very sensitive to noise (a noisy image creates false edges everywhere)
- Cannot distinguish between a person and a chair — both are just edges

> This is why the field needed something smarter: a way to identify **distinctive points** that could be recognized and tracked.

---

## Stage 2 — Feature Detection (SIFT, 2004)

### What it is
SIFT (Scale-Invariant Feature Transform) detects **keypoints** — small, distinctive regions in an image — and assigns each one a mathematical **descriptor**. These descriptors can be used to recognize the same point across different images, even if the image has been rotated, scaled, or the lighting has changed.

### How it works (step by step)
1. Scan the image at multiple scales to find **keypoints** (corners, blobs, distinctive regions)
2. For each keypoint, compute a **128-dimensional descriptor** that captures the local pattern
3. Compare descriptors between the current frame and the previous frame using **FLANN matching**
4. Apply **Lowe's ratio test** to keep only reliable matches and reject false ones

### Implementation in this project
```python
# src/algorithms.py — SIFTDetector

# Step 1-2: Detect keypoints and compute descriptors
keypoints, descriptors = self.sift.detectAndCompute(gray_small, None)

# Step 3: Match against previous frame using FLANN
matches = self.flann.knnMatch(descriptors, self._prev_des, k=2)

# Step 4: Lowe's ratio test — keep only good matches
for pair in matches:
    m, n = pair
    if m.distance < cfg.LOWE_RATIO * n.distance:
        good_matches.append(m)
```
- Uses **OpenCV's SIFT** implementation with FLANN-based matching
- Tracks keypoints across consecutive frames (temporal matching)
- Green circles = matched keypoints (stable), Red circles = new keypoints

### Why SIFT is important
SIFT was a breakthrough because it is **invariant** to:
- **Rotation** — if the image is rotated, the same keypoints are still found
- **Scale** — if the object is closer or farther, features still match
- **Lighting changes** — moderate brightness shifts do not break matching

This makes SIFT essential for robotics applications like **SLAM** (Simultaneous Localization and Mapping) and **visual odometry**, where a robot needs to recognize the same landmarks from different angles.

### Real-world applications
- Panorama stitching (matching overlapping photos)
- Object recognition in robotics
- Augmented reality (tracking markers)
- Image search engines

### Limitations
- Knows **where** features are, but not **what** the object is
- Cannot say "this is a person" — only "these keypoints match"
- Slower than edge detection, and not fast enough for real-time on large images
- Replaced by learned features in most modern applications

> This limitation — the inability to understand semantics — is exactly what deep learning solved.

---

## Stage 3 — Deep Learning

### Why deep learning changed everything

In classical computer vision (edge detection, SIFT), features are **manually designed** by researchers. The algorithm looks for specific mathematical patterns like gradients or keypoint descriptors.

In deep learning, the model **learns its own features** automatically from large amounts of training data. A CNN trained on millions of images develops internal representations far more powerful than anything a human could design by hand.

---

### CNN Classification (MobileNetV2, 2012)

#### What it is
A Convolutional Neural Network (CNN) processes an image through layers of learned filters to classify it. MobileNetV2 is a lightweight CNN trained on **ImageNet** (1.2 million images, 1000 classes). It answers the question: **"What is in this image?"**

#### How it works (step by step)
1. Resize the image to 224x224 pixels (ImageNet standard)
2. Normalize pixel values using ImageNet statistics
3. Pass through the **MobileNetV2** network (53 convolutional layers)
4. The output is a probability distribution over 1000 classes
5. Pick the class with the highest probability as the prediction

#### Implementation in this project
```python
# src/algorithms.py — CNNClassifier

# Load pre-trained model (no training needed — transfer learning)
self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
self.model.eval()

# Preprocessing pipeline (ImageNet standard)
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
top_probs, top_indices = torch.topk(probs, 5)
```
- Uses **PyTorch** with a pre-trained MobileNetV2
- Returns the top-5 predictions with confidence scores
- No training required — we use **transfer learning** (a model already trained on ImageNet)

#### Real-world applications
- Medical diagnosis (classifying X-rays)
- Content moderation (detecting inappropriate images)
- Wildlife monitoring (identifying species from camera traps)

#### Limitations
- Classifies the **entire image** as one label — no spatial awareness
- Cannot tell you **where** in the image the object is
- If there are multiple objects, it picks the most dominant one

> CNN tells you WHAT is in the image. But for robotics, you also need to know WHERE. That is what YOLO solves.

---

### YOLO Object Detection (YOLOv8, 2015)

#### What it is
YOLO (You Only Look Once) performs **real-time object detection** — it finds every object in an image and draws a bounding box around each one, with a class label and confidence score. Unlike CNN classification, YOLO knows **what** AND **where**.

#### How it works (step by step)
1. Divide the image into a grid
2. For each grid cell, predict bounding boxes and class probabilities
3. Apply **non-maximum suppression** to remove overlapping boxes
4. Output: a list of detected objects, each with a bounding box, class name, and confidence

The key innovation is that all of this happens in a **single forward pass** through the network — no sliding window, no region proposals. That is why it is fast enough for real-time use.

#### Implementation in this project
```python
# src/algorithms.py — YOLODetector

# Load pre-trained YOLOv8 nano model
self.model = YOLO(cfg.MODEL_PATH)  # yolov8n.pt

# Run detection on a frame
results = self.model(small_bgr, verbose=False, conf=cfg.YOLO_CONF_THRESH)
boxes = results[0].boxes

# Extract detections
det_count = len(boxes)
confs = boxes.conf.cpu().numpy()
class_ids = boxes.cls.cpu().numpy().astype(int)

# Find person detections (class 0 in COCO dataset)
person_mask = class_ids == 0
if person_mask.any():
    person_conf = float(confs[person_mask].max()) * 100
```
- Uses **Ultralytics YOLOv8n** (nano variant — fast, lightweight)
- Pre-trained on **COCO dataset** (80 object classes)
- Draws bounding boxes with class labels and confidence scores directly on the frame

#### Real-world applications
- Self-driving cars (detecting pedestrians, signs, vehicles)
- Security cameras (person and intrusion detection)
- Robotics (obstacle avoidance, pick-and-place)
- Retail (shelf monitoring, customer counting)

#### Why YOLO is the end point of this evolution
YOLO combines everything:
- It understands **what** objects are (like CNN)
- It knows **where** they are (bounding boxes)
- It runs in **real time** (single forward pass)

---

## Comparison Between Methods

### What each method extracts from the same image

| | Edge Detection | SIFT | CNN | YOLO |
|---|---|---|---|---|
| **Output** | Binary edge map | Keypoints + descriptors | Class label + confidence | Bounding boxes + labels |
| **Understands what?** | Brightness changes | Local patterns | Object identity | Object identity + location |
| **Knows where?** | Edge locations only | Keypoint locations | No (whole image) | Yes (bounding boxes) |
| **Uses learning?** | No | No | Yes (trained on ImageNet) | Yes (trained on COCO) |
| **Speed** | < 1 ms | ~15 ms | ~50 ms | ~55 ms |
| **Robustness to noise** | Very low | Medium | High | High |
| **Robustness to rotation** | Low | High (invariant) | Medium | Medium |

### Key observations

- **Edge detection** is extremely fast but breaks easily under noise — because noise creates false brightness changes everywhere
- **SIFT** is the most robust to rotation and scale changes — that is its design purpose — but it cannot name objects
- **CNN** and **YOLO** are slower because they run deep neural networks, but they understand semantics
- **YOLO** is the most complete: it combines classification (what) with localization (where) in a single pass

### The tradeoff

There is a clear tradeoff between **speed** and **understanding**:

```
Fast ◄──────────────────────────────────────────► Slow
Edge Detection      SIFT         CNN           YOLO

Simple ◄────────────────────────────────────────► Complex
  edges only      keypoints    class label    boxes + labels
```

Each step in the evolution adds more understanding but requires more computation.

---

## The Simulation

### What it does

The simulation is a **web-based dashboard** that captures frames from a camera (or video file) and processes each frame through all four algorithms simultaneously. The results are displayed in a 2x2 grid so you can compare them directly.

### How it demonstrates the differences

1. **Start the camera** (or load a video) — the same frame is sent to all four algorithms
2. **Watch the panels** — each panel shows what that algorithm "sees":
   - Edge panel: white outlines on black background
   - SIFT panel: colored circles showing keypoints (green = tracked, red = new)
   - CNN panel: the original image with a classification label at the bottom
   - YOLO panel: the original image with bounding boxes around detected objects
3. **Read the metrics** — each panel shows processing time and algorithm-specific measurements
4. **Open the charts** — real-time graphs show how processing time and metrics change over time

### Architecture

```
Camera / Video
      │
      ▼
  Browser captures frame (JPEG)
      │
      ▼
  FastAPI server receives frame
      │
      ├── Apply perturbations (if sliders are active)
      │
      ├── Run Edge Detection ────┐
      ├── Run SIFT Detection ────┤
      ├── Run CNN Classification ┤    pipeline.py
      └── Run YOLO Detection ────┘
                                 │
                   JSON response with:
                   - 4 panel images (base64)
                   - 4 metric dictionaries
                                 │
                                 ▼
              Browser updates panels and metrics
```

All four algorithms process the **exact same frame** under the **exact same conditions**, making the comparison fair and meaningful.

---

## Robustness Testing

The dashboard includes a **perturbation system** that lets you degrade the input image in controlled ways and observe how each algorithm reacts:

| Perturbation | What it does | Range |
|---|---|---|
| **Noise** | Adds random Gaussian noise to every pixel | 0 – 120 sigma |
| **Blur** | Applies Gaussian blur (smoothing) | kernel 3 – 51 |
| **Rotation** | Rotates the frame around its center | 0° – 90° |
| **Brightness** | Makes the image darker or brighter | -180 to +100 |

### Why this matters for robotics

A robot operating in the real world faces all of these challenges:
- **Noise** from cheap sensors or low-light conditions
- **Blur** from motion or out-of-focus cameras
- **Rotation** when the robot or camera tilts
- **Brightness changes** between indoor/outdoor or day/night

By applying these perturbations, you can see which algorithms are reliable enough for real-world robotics and which ones break under pressure.

### What you will observe

- **Edge detection** breaks first — noise creates false edges everywhere
- **SIFT** handles rotation well (it is invariant by design) but struggles with heavy blur
- **CNN and YOLO** are the most resilient — deep learning models generalize better because they learned from millions of diverse images

---

## Project Structure

```
├── app.py                     Entry point — starts the FastAPI server
├── requirements.txt           Python dependencies
├── src/
│   ├── config.py              All tunable parameters (thresholds, colors, sizes)
│   ├── algorithms.py          All 4 CV algorithms (Edge, SIFT, CNN, YOLO)
│   └── pipeline.py            Pipeline + perturbation system + panel rendering
├── web/
│   ├── index.html             Dashboard layout (2x2 grid + controls)
│   ├── style.css              Dark theme styling
│   ├── app.js                 Camera capture loop and API communication
│   ├── charts.js              Real-time performance charts (Chart.js)
│   └── public/sift.mp4        Sample video for demo mode
└── models/
    └── yolov8n.pt             Pre-trained YOLO model weights
```

---

## Technologies Used

| Technology | Role |
|---|---|
| **Python** | Backend language |
| **OpenCV** | Edge detection (Canny), SIFT, image processing |
| **PyTorch + torchvision** | MobileNetV2 CNN classification |
| **Ultralytics** | YOLOv8 object detection |
| **FastAPI + Uvicorn** | Web server serving the dashboard |
| **Chart.js** | Real-time performance graphs in the browser |
| **NumPy** | Array operations for perturbations |

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser. Click **Camera** or **Video** to start the demo.
