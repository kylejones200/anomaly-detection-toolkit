"""Anomaly Detection Toolkit.

A comprehensive Python library for detecting anomalies in time series and multivariate data.
Supports multiple detection methods including statistical, machine learning, and deep learning
approaches.
"""

__version__ = "0.1.1"
__author__ = "Kyle Jones"
__email__ = "kyletjones@gmail.com"

from .autoencoders import LSTMAutoencoderDetector, PyTorchAutoencoderDetector
from .base import BaseDetector
from .ensemble import EnsembleDetector, VotingEnsemble
from .ml_methods import IsolationForestDetector, LOFDetector, RobustCovarianceDetector
from .pca_detector import PCADetector
from .predictive_maintenance import (
    Alert,
    AlertLevel,
    AlertSystem,
    DashboardVisualizer,
    FailureClassifier,
    FeatureExtractor,
    PredictiveMaintenanceSystem,
    RealTimeIngestion,
    RULEstimator,
    add_degradation_rates,
    add_rolling_statistics,
    calculate_rul,
    create_rul_labels,
    prepare_pm_features,
)
from .statistical import (
    IQROutlierDetector,
    SeasonalBaselineDetector,
    StatisticalDetector,
    ZScoreDetector,
)
from .wavelet import WaveletDenoiser, WaveletDetector

# Evaluation and visualization (optional imports)
try:
    from .evaluation import (
        calculate_confusion_matrix_metrics,
        calculate_lead_time,
        compare_detectors,
        evaluate_detector,
    )
    from .visualization import (
        plot_comparison_metrics,
        plot_pca_boundary,
        plot_reconstruction_error,
        plot_sensor_drift,
    )
except ImportError:
    # Allow package to work without optional dependencies
    pass

__all__ = [
    # Base class
    "BaseDetector",
    # Statistical methods
    "StatisticalDetector",
    "ZScoreDetector",
    "IQROutlierDetector",
    "SeasonalBaselineDetector",
    # ML methods
    "IsolationForestDetector",
    "LOFDetector",
    "RobustCovarianceDetector",
    "PCADetector",
    # Wavelet methods
    "WaveletDetector",
    "WaveletDenoiser",
    # Autoencoder methods
    "LSTMAutoencoderDetector",
    "PyTorchAutoencoderDetector",
    # Ensemble methods
    "EnsembleDetector",
    "VotingEnsemble",
    # Predictive maintenance
    "FeatureExtractor",
    "RULEstimator",
    "FailureClassifier",
    "AlertSystem",
    "Alert",
    "AlertLevel",
    "PredictiveMaintenanceSystem",
    # Predictive maintenance utilities
    "calculate_rul",
    "create_rul_labels",
    "add_rolling_statistics",
    "add_degradation_rates",
    "prepare_pm_features",
    # Real-time and visualization
    "RealTimeIngestion",
    "DashboardVisualizer",
    # Visualization functions
    "plot_pca_boundary",
    "plot_reconstruction_error",
    "plot_comparison_metrics",
    "plot_sensor_drift",
    # Evaluation functions
    "evaluate_detector",
    "calculate_lead_time",
    "calculate_confusion_matrix_metrics",
    "compare_detectors",
]
