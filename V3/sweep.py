"""
Automated perturbation sweep engine.

Systematically increases each perturbation type from 0 to max,
records all algorithm metrics at each level, and produces data
for degradation curve charts.

Can be used:
  - From the web app via /sweep and /sweep-all endpoints
  - Standalone via CLI: python sweep.py --input image.jpg --output results.json
"""

import numpy as np
import config as cfg
from evaluation.perturbations import PerturbationManager
from evaluation.metrics import normalize_metrics, extract_primary_metrics
from pipeline import run_cv_pipeline


def run_sweep(frame_bgr, perturbation_type, sift_detector, yolo_detector,
              cnn_classifier, steps=None):
    """Run a single perturbation sweep on a frame.

    Args:
        frame_bgr: Input BGR image.
        perturbation_type: One of 'noise', 'blur', 'rotation', 'brightness'.
        sift_detector: SIFTDetector instance.
        yolo_detector: YOLODetector instance.
        cnn_classifier: CNNClassifier instance.
        steps: Number of sweep steps (default: cfg.SWEEP_STEPS).

    Returns:
        Dict with 'perturbation_type', 'steps' list of results.
        Each step has 'strength', 'strength_normalized', 'metrics', 'primary', 'normalized'.
    """
    if steps is None:
        steps = cfg.SWEEP_STEPS

    lo, hi = cfg.SWEEP_RANGES[perturbation_type]
    perturb = PerturbationManager()

    # Generate strength values from lo to hi
    if perturbation_type == "blur":
        # Blur kernel must be odd
        values = np.linspace(lo, hi, steps).astype(int)
        values = [v if v % 2 == 1 else v + 1 for v in values]
    else:
        values = np.linspace(lo, hi, steps)

    # First run: baseline (no perturbation) to get reference metrics
    _, baseline_metrics = run_cv_pipeline(
        frame_bgr, sift_detector, yolo_detector, cnn_classifier,
        encode_panels=False,
    )

    results = []
    for i, strength in enumerate(values):
        # Apply single perturbation at this strength
        perturbed = perturb.apply_single(frame_bgr, perturbation_type, float(strength))

        # Run pipeline
        _, step_metrics = run_cv_pipeline(
            perturbed, sift_detector, yolo_detector, cnn_classifier,
            encode_panels=False,
        )

        # Extract and normalize
        primary = extract_primary_metrics(step_metrics)
        normalized = normalize_metrics(step_metrics, baseline_metrics)

        # Normalize strength to 0-1
        strength_norm = (float(strength) - lo) / (hi - lo) if hi != lo else 0.0

        # Make metrics JSON-serializable (remove top5 tuples)
        serializable_metrics = {}
        for algo, m in step_metrics.items():
            serializable_metrics[algo] = {}
            for k, v in m.items():
                if k == "top5":
                    serializable_metrics[algo][k] = [
                        {"label": label, "confidence": conf}
                        for label, conf in v
                    ]
                elif isinstance(v, (int, float, str, bool)):
                    serializable_metrics[algo][k] = v

        results.append({
            "step": i,
            "strength": float(strength),
            "strength_normalized": round(strength_norm, 3),
            "metrics": serializable_metrics,
            "primary": {k: round(v, 2) for k, v in primary.items()},
            "normalized": {k: round(v, 2) for k, v in normalized.items()},
        })

    return {
        "perturbation_type": perturbation_type,
        "steps": results,
        "baseline": {
            k: round(v, 2) for k, v in extract_primary_metrics(baseline_metrics).items()
        },
    }


def run_sweep_all(frame_bgr, sift_detector, yolo_detector, cnn_classifier,
                  steps=None):
    """Run sweeps for all perturbation types.

    Returns:
        Dict of {perturbation_type: sweep_result}.
    """
    results = {}
    for ptype in cfg.SWEEP_PERTURBATIONS:
        results[ptype] = run_sweep(
            frame_bgr, ptype, sift_detector, yolo_detector,
            cnn_classifier, steps=steps,
        )
    return results


if __name__ == "__main__":
    import argparse
    import json
    import sys
    import os
    import cv2

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Run perturbation sweep analysis")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", default="sweep_results.json", help="Output JSON path")
    parser.add_argument("--steps", type=int, default=cfg.SWEEP_STEPS)
    parser.add_argument("--type", choices=cfg.SWEEP_PERTURBATIONS + ["all"], default="all")
    args = parser.parse_args()

    frame = cv2.imread(args.input)
    if frame is None:
        print(f"[ERROR] Cannot read image: {args.input}")
        sys.exit(1)

    from algorithms.sift_detector import SIFTDetector
    from algorithms.yolo_detector import YOLODetector
    from algorithms.cnn_classifier import CNNClassifier

    print("[INFO] Loading detectors...")
    sift = SIFTDetector()
    yolo = YOLODetector()
    cnn = CNNClassifier()

    if args.type == "all":
        print("[INFO] Running full sweep...")
        results = run_sweep_all(frame, sift, yolo, cnn, steps=args.steps)
    else:
        print(f"[INFO] Running {args.type} sweep...")
        results = run_sweep(frame, args.type, sift, yolo, cnn, steps=args.steps)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Results saved to {args.output}")
