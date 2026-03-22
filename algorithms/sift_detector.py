"""
SIFT Feature Detection with temporal matching.

Uses FLANN-based matcher to track keypoints across consecutive frames,
demonstrating feature stability — a key requirement for SLAM and
visual odometry in robotics.
"""

import cv2
import numpy as np
import time
import config as cfg


class SIFTDetector:
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
        """Detect SIFT features and match against previous frame."""
        t0 = time.perf_counter()

        keypoints, descriptors = self.sift.detectAndCompute(gray_small, None)
        kp_count = len(keypoints)

        # Temporal matching against previous frame
        good_matches = []
        if (self._prev_des is not None and descriptors is not None
                and len(descriptors) >= 2 and len(self._prev_des) >= 2):
            matches = self.flann.knnMatch(descriptors, self._prev_des, k=2)
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < cfg.LOWE_RATIO * n.distance:
                        good_matches.append(m)

        # Draw rich keypoints (circle = scale, line = orientation)
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
        match_ratio = (len(good_matches) / kp_count * 100) if kp_count > 0 else 0.0

        metrics = {
            "time_ms": dt,
            "keypoints_raw": kp_count,
            "keypoints_filtered": kp_count,
            "matches_good": len(good_matches),
            "match_ratio": match_ratio,
        }
        return sift_img, metrics
