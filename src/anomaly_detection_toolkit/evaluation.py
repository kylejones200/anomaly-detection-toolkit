"""Evaluation utilities for anomaly detection methods."""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .base import BaseDetector


def calculate_lead_time(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate lead time metrics for anomaly detection.

    Lead time is the time difference between when an anomaly is detected
    and when the actual failure/event occurs.

    Parameters
    ----------
    predictions : ndarray
        Binary predictions (-1 for anomaly, 1 for normal).
    true_labels : ndarray
        True binary labels (1 for anomaly, 0 for normal).
    timestamps : ndarray, optional
        Timestamps for each sample. If None, uses indices.

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'mean_lead_time': Mean lead time for detected anomalies
        - 'median_lead_time': Median lead time
        - 'min_lead_time': Minimum lead time
        - 'max_lead_time': Maximum lead time
        - 'early_detections': Number of anomalies detected before event
        - 'late_detections': Number of anomalies detected after event
    """
    if timestamps is None:
        timestamps = np.arange(len(predictions))

    # Convert predictions to binary (1 for anomaly, 0 for normal)
    pred_binary = (predictions == -1).astype(int)
    true_binary = (true_labels == 1).astype(int) if np.any(true_labels == 1) else true_labels

    # Find anomaly events (where true label changes from 0 to 1)
    event_indices = np.where(np.diff(true_binary) == 1)[0] + 1

    if len(event_indices) == 0:
        return {
            "mean_lead_time": 0.0,
            "median_lead_time": 0.0,
            "min_lead_time": 0.0,
            "max_lead_time": 0.0,
            "early_detections": 0,
            "late_detections": 0,
        }

    lead_times = []
    early_count = 0
    late_count = 0

    for event_idx in event_indices:
        # Find first detection before or at event
        detections_before = np.where(pred_binary[: event_idx + 1] == 1)[0]

        if len(detections_before) > 0:
            first_detection_idx = detections_before[-1]  # Last detection before event
            lead_time = timestamps[event_idx] - timestamps[first_detection_idx]

            if lead_time > 0:
                lead_times.append(lead_time)
                early_count += 1
            elif lead_time < 0:
                late_count += 1
                lead_times.append(lead_time)  # Negative for late detections

    if len(lead_times) == 0:
        return {
            "mean_lead_time": 0.0,
            "median_lead_time": 0.0,
            "min_lead_time": 0.0,
            "max_lead_time": 0.0,
            "early_detections": 0,
            "late_detections": 0,
        }

    lead_times = np.array(lead_times)

    return {
        "mean_lead_time": (
            float(np.mean(lead_times[lead_times > 0])) if np.any(lead_times > 0) else 0.0
        ),
        "median_lead_time": (
            float(np.median(lead_times[lead_times > 0])) if np.any(lead_times > 0) else 0.0
        ),
        "min_lead_time": (
            float(np.min(lead_times[lead_times > 0])) if np.any(lead_times > 0) else 0.0
        ),
        "max_lead_time": (
            float(np.max(lead_times[lead_times > 0])) if np.any(lead_times > 0) else 0.0
        ),
        "early_detections": int(early_count),
        "late_detections": int(late_count),
    }


def evaluate_detector(
    detector: BaseDetector,
    X: Union[np.ndarray, pd.DataFrame],
    y_true: np.ndarray,
    scores: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Evaluate an anomaly detector with comprehensive metrics.

    Parameters
    ----------
    detector : BaseDetector
        Fitted anomaly detector.
    X : array-like
        Test data.
    y_true : ndarray
        True binary labels (1 for anomaly, 0 for normal).
    scores : ndarray, optional
        Pre-computed anomaly scores. If None, computed from detector.
    timestamps : ndarray, optional
        Timestamps for lead time calculation.

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'accuracy': Accuracy score
        - 'precision': Precision score
        - 'recall': Recall score
        - 'f1': F1 score
        - 'roc_auc': ROC AUC score (if scores provided)
        - Lead time metrics (if timestamps provided)
    """
    # Get predictions
    predictions = detector.predict(X)

    # Convert to binary for sklearn metrics
    pred_binary = (predictions == -1).astype(int)
    true_binary = (y_true == 1).astype(int) if np.any(y_true == 1) else y_true.astype(int)

    metrics = {
        "accuracy": float(accuracy_score(true_binary, pred_binary)),
        "precision": float(precision_score(true_binary, pred_binary, zero_division=0)),
        "recall": float(recall_score(true_binary, pred_binary, zero_division=0)),
        "f1": float(f1_score(true_binary, pred_binary, zero_division=0)),
    }

    # Add ROC AUC if scores available
    if scores is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(true_binary, scores))
        except ValueError:
            # If only one class present, set to 0
            metrics["roc_auc"] = 0.0

    # Add lead time metrics if timestamps available
    if timestamps is not None:
        lead_time_metrics = calculate_lead_time(predictions, y_true, timestamps)
        metrics.update(lead_time_metrics)

    return metrics


def compare_detectors(
    detectors: Dict[str, BaseDetector],
    X: Union[np.ndarray, pd.DataFrame],
    y_true: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compare multiple anomaly detectors side-by-side.

    Parameters
    ----------
    detectors : dict
        Dictionary mapping detector names to BaseDetector instances.
    X : array-like
        Test data.
    y_true : ndarray
        True binary labels (1 for anomaly, 0 for normal).
    timestamps : ndarray, optional
        Timestamps for lead time calculation.

    Returns
    -------
    comparison_df : DataFrame
        DataFrame with metrics for each detector.
    """
    results = []

    for name, detector in detectors.items():
        # Get scores
        scores = detector.score_samples(X)

        # Evaluate
        metrics = evaluate_detector(detector, X, y_true, scores=scores, timestamps=timestamps)
        metrics["detector"] = name
        results.append(metrics)

    return pd.DataFrame(results)


def calculate_confusion_matrix_metrics(
    predictions: np.ndarray, y_true: np.ndarray
) -> Dict[str, int]:
    """
    Calculate confusion matrix components.

    Parameters
    ----------
    predictions : ndarray
        Binary predictions (-1 for anomaly, 1 for normal).
    y_true : ndarray
        True binary labels (1 for anomaly, 0 for normal).

    Returns
    -------
    metrics : dict
        Dictionary with TP, TN, FP, FN counts.
    """
    pred_binary = (predictions == -1).astype(int)
    true_binary = (y_true == 1).astype(int) if np.any(y_true == 1) else y_true.astype(int)

    tp = np.sum((pred_binary == 1) & (true_binary == 1))
    tn = np.sum((pred_binary == 0) & (true_binary == 0))
    fp = np.sum((pred_binary == 1) & (true_binary == 0))
    fn = np.sum((pred_binary == 0) & (true_binary == 1))

    return {
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }
