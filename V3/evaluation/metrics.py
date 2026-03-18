"""
Metrics normalization for cross-algorithm comparison.

Normalizes each algorithm's primary metric to a 0-100 scale
relative to the unperturbed baseline, enabling fair comparison
on shared charts (degradation curves).
"""


# Primary metric keys for each algorithm
PRIMARY_METRICS = {
    "edge": "edge_density",
    "sift": "keypoints_filtered",
    "cnn": "top1_confidence",
    "yolo": "person_confidence",
}


def normalize_metrics(raw_metrics, baseline_metrics):
    """Normalize raw metrics as a percentage of baseline performance.

    Args:
        raw_metrics: Dict of {algo: {metric_key: value, ...}} for current frame.
        baseline_metrics: Dict of {algo: {metric_key: value, ...}} for unperturbed frame.

    Returns:
        Dict of {algo: float} where each value is 0-100 representing
        percentage of baseline performance retained.
    """
    normalized = {}
    for algo, metric_key in PRIMARY_METRICS.items():
        baseline_val = baseline_metrics.get(algo, {}).get(metric_key, 0)
        current_val = raw_metrics.get(algo, {}).get(metric_key, 0)

        if baseline_val > 0:
            normalized[algo] = min(100.0, (current_val / baseline_val) * 100.0)
        else:
            normalized[algo] = 0.0

    return normalized


def extract_primary_metrics(metrics):
    """Extract the primary metric value for each algorithm.

    Args:
        metrics: Dict of {algo: {metric_key: value, ...}}.

    Returns:
        Dict of {algo: primary_metric_value}.
    """
    result = {}
    for algo, metric_key in PRIMARY_METRICS.items():
        result[algo] = metrics.get(algo, {}).get(metric_key, 0)
    return result
