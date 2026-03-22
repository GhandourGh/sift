"""
Configuration constants for the Computer Vision Evolution Demo V3.
All tunable parameters in one place for reproducibility.
"""

import os

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
VIDEO_SOURCE = os.path.join(BASE_DIR, "web", "public", "sift.mp4")

# ----------------------------------------------------------------------
# Resolution
# ----------------------------------------------------------------------
PANEL_W, PANEL_H = 320, 240             # Each panel in the 2x2 grid (web-optimized)
PROCESS_W, PROCESS_H = 320, 240         # Downscaled for inference
HEADER_H = 32                           # Panel header bar height

# ----------------------------------------------------------------------
# Algorithm Parameters
# ----------------------------------------------------------------------
# Canny
CANNY_LOW = 30
CANNY_HIGH = 80

# SIFT
SIFT_N_FEATURES = 0
SIFT_CONTRAST_THRESH = 0.03
SIFT_MIN_SIZE = 0
SIFT_MIN_RESPONSE = 0.0

# FLANN matcher
FLANN_INDEX_KDTREE = 1
FLANN_TREES = 5
FLANN_CHECKS = 50
LOWE_RATIO = 0.7

# YOLO
YOLO_CONF_THRESH = 0.25

# CNN Classifier (MobileNetV2)
CNN_MODEL = "mobilenet_v2"
CNN_TOP_K = 5                            # Number of top predictions to show
CNN_INPUT_SIZE = 224                     # ImageNet standard input size

# ----------------------------------------------------------------------
# Perturbation Defaults
# ----------------------------------------------------------------------
NOISE_SIGMA_DEFAULT = 30
NOISE_SIGMA_STEP = 10
NOISE_SIGMA_MAX = 120

BLUR_KERNEL_DEFAULT = 11
BLUR_KERNEL_STEP = 4
BLUR_KERNEL_MAX = 51

ROTATION_ANGLE_DEFAULT = 15
ROTATION_ANGLE_STEP = 5
ROTATION_ANGLE_MAX = 90

BRIGHTNESS_DELTA_DEFAULT = -60
BRIGHTNESS_DELTA_STEP = 20
BRIGHTNESS_DELTA_MIN = -180
BRIGHTNESS_DELTA_MAX = 100

# ----------------------------------------------------------------------
# Sweep Settings
# ----------------------------------------------------------------------
SWEEP_STEPS = 10
SWEEP_PERTURBATIONS = ["noise", "blur", "rotation", "brightness"]

# Per-perturbation sweep ranges (min, max)
SWEEP_RANGES = {
    "noise": (0, NOISE_SIGMA_MAX),
    "blur": (3, BLUR_KERNEL_MAX),
    "rotation": (0, ROTATION_ANGLE_MAX),
    "brightness": (BRIGHTNESS_DELTA_MIN, 0),
}

# ----------------------------------------------------------------------
# Metrics Smoothing (Exponential Moving Average)
# ----------------------------------------------------------------------
EMA_ALPHA = 0.3

# ----------------------------------------------------------------------
# Colors (BGR for OpenCV)
# ----------------------------------------------------------------------
COLOR_BG = (30, 30, 30)
COLOR_HEADER_BG = (40, 40, 40)
COLOR_TEXT = (220, 220, 220)
COLOR_TEXT_DIM = (140, 140, 140)
COLOR_BORDER = (60, 60, 60)

# Per-algorithm accent colors
COLOR_ORIGINAL = (180, 180, 180)
COLOR_EDGE = (128, 255, 0)              # Green
COLOR_SIFT = (0, 180, 255)              # Orange
COLOR_CNN = (255, 0, 180)               # Purple/Magenta
COLOR_YOLO = (255, 128, 0)              # Blue

# Perturbation indicator
COLOR_PERTURB = (0, 0, 255)             # Red

COLOR_BAR_BG = (50, 50, 50)

# ----------------------------------------------------------------------
# Font
# ----------------------------------------------------------------------
FONT_FACE = 0                           # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_TITLE = 0.55
FONT_SCALE_METRIC = 0.42
FONT_SCALE_DASH = 0.40
FONT_SCALE_HELP = 0.35
FONT_THICKNESS = 1
