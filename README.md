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

## API Reference

### Statistical Methods

- **ZScoreDetector**: Z-score based anomaly detection
- **IQROutlierDetector**: Interquartile Range (IQR) based outlier detection
- **SeasonalBaselineDetector**: Seasonal baseline anomaly detection for time series

### Machine Learning Methods

- **IsolationForestDetector**: Isolation Forest anomaly detection
- **LOFDetector**: Local Outlier Factor (LOF) anomaly detection
- **RobustCovarianceDetector**: Robust Covariance (Elliptic Envelope) anomaly detection

### Wavelet Methods

- **WaveletDetector**: Wavelet-based anomaly detection for time series
- **WaveletDenoiser**: Wavelet-based signal denoising

### Deep Learning Methods

- **LSTMAutoencoderDetector**: LSTM autoencoder-based anomaly detection (requires TensorFlow/Keras)
- **PyTorchAutoencoderDetector**: PyTorch autoencoder-based anomaly detection (requires PyTorch)

### Ensemble Methods

- **VotingEnsemble**: Voting ensemble that combines predictions from multiple detectors
- **EnsembleDetector**: General ensemble detector with customizable combination methods

## Examples

See the `examples/` directory for complete examples:

- `examples/statistical_example.py`: Statistical methods
- `examples/ml_example.py`: Machine learning methods
- `examples/time_series_example.py`: Time series anomaly detection
- `examples/ensemble_example.py`: Ensemble methods

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
