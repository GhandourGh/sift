"""
Perturbation system for robustness testing.

Applies controlled degradations to input frames so that
each algorithm's resilience can be compared under identical conditions.

Supported perturbations:
  - Gaussian noise  (key: N)
  - Gaussian blur    (key: B)
  - Rotation         (key: R)
  - Brightness shift (key: D)
  - All combined     (key: A)

Strength adjustment: +/- keys
Reset all: 0
"""

import cv2
import numpy as np
import config as cfg


def apply_noise(frame, sigma):
    """Add Gaussian noise with standard deviation sigma."""
    noise = np.random.normal(0, sigma, frame.shape).astype(np.float32)
    noisy = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_blur(frame, kernel_size):
    """Apply Gaussian blur with given kernel size (must be odd)."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(frame, (k, k), 0)


def apply_rotation(frame, angle):
    """Rotate frame by angle degrees around center."""
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def apply_brightness(frame, delta):
    """Shift brightness by delta (negative = darker)."""
    return np.clip(frame.astype(np.int16) + delta, 0, 255).astype(np.uint8)


class PerturbationManager:
    """Manages active perturbations and their strengths."""

    TYPES = ["noise", "blur", "rotation", "brightness"]

    def __init__(self):
        self.active = {t: False for t in self.TYPES}
        self.params = {
            "noise": cfg.NOISE_SIGMA_DEFAULT,
            "blur": cfg.BLUR_KERNEL_DEFAULT,
            "rotation": cfg.ROTATION_ANGLE_DEFAULT,
            "brightness": cfg.BRIGHTNESS_DELTA_DEFAULT,
        }

    @property
    def any_active(self):
        return any(self.active.values())

    def toggle(self, name):
        """Toggle a single perturbation on/off."""
        if name in self.active:
            self.active[name] = not self.active[name]

    def toggle_all(self):
        """Toggle all perturbations on/off."""
        if self.any_active:
            for t in self.TYPES:
                self.active[t] = False
        else:
            for t in self.TYPES:
                self.active[t] = True

    def reset(self):
        """Turn off all perturbations and reset to defaults."""
        for t in self.TYPES:
            self.active[t] = False
        self.params["noise"] = cfg.NOISE_SIGMA_DEFAULT
        self.params["blur"] = cfg.BLUR_KERNEL_DEFAULT
        self.params["rotation"] = cfg.ROTATION_ANGLE_DEFAULT
        self.params["brightness"] = cfg.BRIGHTNESS_DELTA_DEFAULT

    def increase_strength(self):
        """Increase strength of all active perturbations."""
        if self.active["noise"]:
            self.params["noise"] = min(self.params["noise"] + cfg.NOISE_SIGMA_STEP,
                                       cfg.NOISE_SIGMA_MAX)
        if self.active["blur"]:
            self.params["blur"] = min(self.params["blur"] + cfg.BLUR_KERNEL_STEP,
                                      cfg.BLUR_KERNEL_MAX)
        if self.active["rotation"]:
            self.params["rotation"] = min(self.params["rotation"] + cfg.ROTATION_ANGLE_STEP,
                                          cfg.ROTATION_ANGLE_MAX)
        if self.active["brightness"]:
            self.params["brightness"] = max(self.params["brightness"] + cfg.BRIGHTNESS_DELTA_STEP,
                                            cfg.BRIGHTNESS_DELTA_MIN)

    def decrease_strength(self):
        """Decrease strength of all active perturbations."""
        if self.active["noise"]:
            self.params["noise"] = max(self.params["noise"] - cfg.NOISE_SIGMA_STEP, 0)
        if self.active["blur"]:
            self.params["blur"] = max(self.params["blur"] - cfg.BLUR_KERNEL_STEP, 3)
        if self.active["rotation"]:
            self.params["rotation"] = max(self.params["rotation"] - cfg.ROTATION_ANGLE_STEP, 0)
        if self.active["brightness"]:
            self.params["brightness"] = min(self.params["brightness"] - cfg.BRIGHTNESS_DELTA_STEP,
                                            cfg.BRIGHTNESS_DELTA_MAX)

    def apply(self, frame):
        """Apply all active perturbations to a frame."""
        out = frame.copy()
        if self.active["noise"]:
            out = apply_noise(out, self.params["noise"])
        if self.active["blur"]:
            out = apply_blur(out, self.params["blur"])
        if self.active["rotation"]:
            out = apply_rotation(out, self.params["rotation"])
        if self.active["brightness"]:
            out = apply_brightness(out, self.params["brightness"])
        return out

    def status_text(self):
        """Return a short string describing active perturbations."""
        parts = []
        if self.active["noise"]:
            parts.append(f"Noise(s={self.params['noise']})")
        if self.active["blur"]:
            parts.append(f"Blur(k={self.params['blur']})")
        if self.active["rotation"]:
            parts.append(f"Rot({self.params['rotation']}deg)")
        if self.active["brightness"]:
            parts.append(f"Bright({self.params['brightness']:+d})")
        return " + ".join(parts) if parts else "None"
