# Anomaly Detection Toolkit

A comprehensive Python library for detecting anomalies in time series and multivariate data using multiple detection methods including statistical, machine learning, and deep learning approaches.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://readthedocs.org/projects/anomaly-detection-toolkit/badge/?version=latest)](https://anomaly-detection-toolkit.readthedocs.io/en/latest/?badge=latest)

**📚 [Full Documentation](https://anomaly-detection-toolkit.readthedocs.io/)**

## Features

- **Statistical Methods**: Z-score, IQR, seasonal baseline detection
- **Machine Learning**: Isolation Forest, Local Outlier Factor (LOF), Robust Covariance
- **Wavelet Methods**: Wavelet decomposition and denoising for time series
- **Deep Learning**: LSTM and PyTorch autoencoders for anomaly detection
- **Ensemble Methods**: Voting and score combination ensembles
- **Predictive Maintenance**: RUL estimation, failure prediction, feature extraction, alert systems
- **Visualization**: Publication-ready plots with optional signalplot integration
- **Easy to Use**: Scikit-learn compatible API
- **Well Documented**: Comprehensive docstrings and examples

## Installation

### Basic Installation

```bash
pip install anomaly-detection-toolkit
```

### With Deep Learning Support

For LSTM and PyTorch autoencoders:

```bash
pip install anomaly-detection-toolkit[deep]
```

### With Enhanced Visualizations

For publication-ready plots using [signalplot](https://github.com/kylejones200/signalplot):

```bash
pip install anomaly-detection-toolkit[viz]
```

### Development Installation

```bash
git clone https://github.com/kylejones200/anomaly-detection-toolkit.git
cd anomaly-detection-toolkit
pip install -e ".[deep]"
```

### Building Documentation

```bash
pip install -e ".[docs]"
cd docs
make html
```

## Quick Start

### Statistical Methods

```python
from anomaly_detection_toolkit import ZScoreDetector, IQROutlierDetector
import numpy as np

# Generate sample data
data = np.random.randn(1000)
data[100:105] += 5  # Inject anomalies

# Z-score detector
detector = ZScoreDetector(n_std=3.0)
detector.fit(data)
predictions, scores = detector.fit_predict(data)

print(f"Anomalies detected: {(predictions == -1).sum()}")
```

### Machine Learning Methods

```python
from anomaly_detection_toolkit import IsolationForestDetector, LOFDetector
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')
features = ['feature1', 'feature2', 'feature3']
X = df[features]

# Isolation Forest
iso_detector = IsolationForestDetector(contamination=0.05, n_estimators=200)
iso_detector.fit(X)
predictions, scores = iso_detector.fit_predict(X)

# Local Outlier Factor
lof_detector = LOFDetector(contamination=0.05, n_neighbors=20)
lof_detector.fit(X)
predictions, scores = lof_detector.fit_predict(X)
```

### Time Series Anomaly Detection

#### Wavelet-Based Detection

```python
from anomaly_detection_toolkit import WaveletDetector
import pandas as pd

# Time series data
df = pd.read_csv('time_series.csv', parse_dates=['date'])
ts = df['value'].values

# Wavelet detector
wavelet_detector = WaveletDetector(wavelet='db4', threshold_factor=2.5, level=5)
wavelet_detector.fit(ts)
predictions, scores = wavelet_detector.fit_predict(ts)
```

#### Seasonal Baseline Detection

```python
from anomaly_detection_toolkit import SeasonalBaselineDetector

# DataFrame with date and value columns
df = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=365, freq='D'),
    'value': np.random.randn(365) * 10 + 50
})

# Seasonal baseline detector (weekly seasonality)
seasonal_detector = SeasonalBaselineDetector(seasonality='week', threshold_sigma=2.5)
seasonal_detector.fit(df, date_col='date', value_col='value')
predictions = seasonal_detector.predict(df, date_col='date', value_col='value')
```

### Deep Learning Methods

#### LSTM Autoencoder

```python
from anomaly_detection_toolkit import LSTMAutoencoderDetector
import numpy as np

# Time series data
ts = np.sin(np.linspace(0, 50, 1000)) + np.random.randn(1000) * 0.1
ts[450:460] += 3  # Inject anomalies

# LSTM autoencoder
lstm_detector = LSTMAutoencoderDetector(
    window_size=20,
    lstm_units=[32, 16],
    epochs=50,
    threshold_std=3.0
)
lstm_detector.fit(ts)
predictions, scores = lstm_detector.fit_predict(ts)
```

#### PyTorch Autoencoder

```python
from anomaly_detection_toolkit import PyTorchAutoencoderDetector

# PyTorch autoencoder
pytorch_detector = PyTorchAutoencoderDetector(
    window_size=24,
    hidden_dims=[64, 16, 4],
    epochs=200,
    threshold_std=3.0
)
pytorch_detector.fit(ts)
predictions, scores = pytorch_detector.fit_predict(ts)
```

### Ensemble Methods

```python
from anomaly_detection_toolkit import (
    IsolationForestDetector,
    LOFDetector,
    RobustCovarianceDetector,
    VotingEnsemble
)

# Create multiple detectors
detectors = [
    IsolationForestDetector(contamination=0.05),
    LOFDetector(contamination=0.05),
    RobustCovarianceDetector(contamination=0.05)
]

# Voting ensemble (flags if 2+ detectors agree)
ensemble = VotingEnsemble(detectors, voting_threshold=2)
ensemble.fit(X)
predictions, scores = ensemble.fit_predict(X)
```

### Predictive Maintenance

```python
from anomaly_detection_toolkit.predictive_maintenance import (
    FeatureExtractor,
    RULEstimator,
    FailureClassifier,
    AlertSystem,
    PredictiveMaintenanceSystem,
)
from anomaly_detection_toolkit.statistical import ZScoreDetector
import pandas as pd
import numpy as np

# Extract features from time series
extractor = FeatureExtractor(rolling_windows=[5, 10, 20, 50])
features = extractor.extract(sensor_data)

# Train RUL estimator
rul_estimator = RULEstimator(method="regression", random_state=42)
rul_estimator.fit(train_features, train_rul_values)
rul_predictions = rul_estimator.predict(test_features)

# Train failure classifier
failure_classifier = FailureClassifier(random_state=42)
failure_classifier.fit(train_features, train_failure_labels)
failure_proba = failure_classifier.predict_proba(test_features)

# Set up alert system
thresholds = {
    "temperature_rolling_mean_10": {
        "warning": 80.0,
        "critical": 90.0,
        "failure": 100.0,
    }
}
alert_system = AlertSystem(thresholds=thresholds)

# Create complete predictive maintenance system
pm_system = PredictiveMaintenanceSystem(
    feature_extractor=extractor,
    rul_estimator=rul_estimator,
    failure_classifier=failure_classifier,
    alert_system=alert_system,
    anomaly_detector=ZScoreDetector(n_std=2.5),
)

# Process new data
results = pm_system.process(
    new_sensor_data,
    timestamp=pd.Timestamp.now(),
    asset_id="EQUIPMENT_001",
)

print(f"RUL: {results['rul']} hours")
print(f"Failure probability: {results['failure_probability']:.3f}")
print(f"Alerts: {len(results['alerts'])}")
```

#### Complete PM Workflow with Feature Engineering

```python
from anomaly_detection_toolkit.predictive_maintenance import (
    prepare_pm_features,
    calculate_rul,
    create_rul_labels,
    RULEstimator,
)
import pandas as pd

# Load PM data with asset_id, cycle, and sensor columns
df = pd.read_csv('pm_data.csv')

# Prepare all features in one step
df = prepare_pm_features(
    df=df,
    asset_id_col="asset_id",
    cycle_col="cycle",
    feature_cols=None,  # Auto-detect sensor columns
    calculate_rul_flag=True,
    add_labels=True,
    add_rolling_stats=True,
    add_degradation_rates=False,
    rolling_window=5,
    warning_threshold=30,
    critical_threshold=15,
)

# Split by asset for train/test
asset_ids = df['asset_id'].unique()
train_assets = asset_ids[:int(len(asset_ids) * 0.8)]
test_assets = asset_ids[int(len(asset_ids) * 0.8):]

train_df = df[df['asset_id'].isin(train_assets)]
test_df = df[df['asset_id'].isin(test_assets)]

# Train RUL estimator
feature_cols = [col for col in df.columns
                if col not in ['asset_id', 'cycle', 'RUL', 'health_status']]
rul_estimator = RULEstimator(method="regression", random_state=42)
rul_estimator.fit(train_df[feature_cols], train_df['RUL'])

# Predict RUL
rul_predictions = rul_estimator.predict(test_df[feature_cols])
```

#### Real-time Monitoring and Dashboard

```python
from anomaly_detection_toolkit.predictive_maintenance import (
    RealTimeIngestion,
    DashboardVisualizer,
    PredictiveMaintenanceSystem,
)

# Set up PM system (with trained models)
pm_system = PredictiveMaintenanceSystem(...)

# Create real-time ingestion
ingestion = RealTimeIngestion(pm_system=pm_system, window_size=100)

# Ingest streaming data
for sensor_reading in sensor_stream:
    result = ingestion.ingest(
        data=sensor_reading,
        asset_id="EQUIPMENT_001",
        sensor_name="temperature",
    )

    if result.get("alerts"):
        print(f"Alert: {result['alerts']}")

# Get results history and create dashboard
results_history = {
    asset_id: ingestion.get_latest_results(asset_id, n=1000)
    for asset_id in ingestion.get_all_assets()
}

visualizer = DashboardVisualizer()
fig = visualizer.create_dashboard(results_history, save_path="dashboard.png")
```

## API Reference

### Statistical Methods

- **ZScoreDetector**: Z-score based anomaly detection
- **IQROutlierDetector**: Interquartile Range (IQR) based outlier detection
- **SeasonalBaselineDetector**: Seasonal baseline anomaly detection for time series

### Machine Learning Methods

- **IsolationForestDetector**: Isolation Forest anomaly detection
- **LOFDetector**: Local Outlier Factor (LOF) anomaly detection
- **RobustCovarianceDetector**: Robust Covariance (Elliptic Envelope) anomaly detection
- **PCADetector**: PCA-based anomaly detection with Mahalanobis distance or reconstruction error

### Wavelet Methods

- **WaveletDetector**: Wavelet-based anomaly detection for time series
- **WaveletDenoiser**: Wavelet-based signal denoising

### Deep Learning Methods

- **LSTMAutoencoderDetector**: LSTM autoencoder-based anomaly detection (requires TensorFlow/Keras)
- **PyTorchAutoencoderDetector**: PyTorch autoencoder-based anomaly detection (requires PyTorch)

### Ensemble Methods

- **VotingEnsemble**: Voting ensemble that combines predictions from multiple detectors
- **EnsembleDetector**: General ensemble detector with customizable combination methods

### Predictive Maintenance

- **FeatureExtractor**: Extract rolling statistics, change detection, and frequency domain features
- **RULEstimator**: Estimate Remaining Useful Life (RUL) for assets
- **FailureClassifier**: Classify normal vs. failure states
- **AlertSystem**: Threshold-based alert system with escalation rules
- **PredictiveMaintenanceSystem**: Complete system integrating all components
- **RealTimeIngestion**: Real-time data ingestion system for streaming PM data
- **DashboardVisualizer**: Dashboard visualization utilities for PM monitoring

#### Utility Functions

- **calculate_rul**: Calculate Remaining Useful Life from asset cycle data
- **create_rul_labels**: Create health status labels (healthy, warning, critical, failed) from RUL
- **add_rolling_statistics**: Add rolling window statistics grouped by asset
- **add_degradation_rates**: Add degradation rate features (rate of change)
- **prepare_pm_features**: Complete feature engineering pipeline for PM data

## Examples

See the `examples/` directory for complete examples:

- `examples/basic_example.py`: Basic usage examples
- `examples/time_series_example.py`: Time series anomaly detection
- `examples/predictive_maintenance_example.py`: Predictive maintenance features
- `examples/pm_workflow_example.py`: Complete PM workflow with feature engineering and RUL forecasting
- `examples/realtime_pm_example.py`: Real-time data ingestion and dashboard visualization
- `examples/pca_vs_lstm_comparison.py`: PCA vs LSTM comparison for turbomachinery anomaly detection

## Development

### Setting Up Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality before commits and pushes:

```bash
# Install pre-commit hooks
./setup-pre-commit.sh

# Or manually:
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push
```

The pre-push hooks will automatically:
- Check code formatting (Black)
- Sort imports (isort)
- Lint code (flake8)
- Type check (mypy)
- Security scan (bandit)
- Run tests (pytest)

To run checks manually:
```bash
pre-commit run --all-files
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{anomaly_detection_toolkit,
  title={Anomaly Detection Toolkit},
  author={Kyle Jones},
  year={2025},
  url={https://github.com/kylejones200/anomaly-detection-toolkit}
}
```

## Acknowledgments

- Built with scikit-learn, PyWavelets, and other excellent open-source libraries
- Inspired by various anomaly detection research and implementations

## Support

For issues, questions, or contributions, please open an issue on [GitHub](https://github.com/kylejones200/anomaly-detection-toolkit/issues).
