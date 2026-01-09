Getting Started
===============

This guide will help you get started with the Anomaly Detection Toolkit.

Installation
------------

Install the package using pip:

.. code-block:: bash

   pip install anomaly-detection-toolkit

For deep learning features (LSTM and PyTorch autoencoders):

.. code-block:: bash

   pip install anomaly-detection-toolkit[deep]

Quick Example
-------------

Here's a simple example using the Isolation Forest detector:

.. code-block:: python

   from anomaly_detection_toolkit import IsolationForestDetector
   import numpy as np

   # Generate sample data with anomalies
   np.random.seed(42)
   X = np.random.randn(1000, 3)
   X[100:105] += 5  # Inject anomalies

   # Create and fit detector
   detector = IsolationForestDetector(contamination=0.05)
   detector.fit(X)

   # Detect anomalies
   predictions, scores = detector.fit_predict(X)

   # Get anomaly indices
   anomaly_indices = np.where(predictions == -1)[0]
   print(f"Detected {len(anomaly_indices)} anomalies")

Basic Concepts
--------------

All detectors in the toolkit follow a consistent API:

* ``fit(X)``: Train the detector on data
* ``predict(X)``: Predict anomalies (-1 for anomalies, 1 for normal)
* ``score_samples(X)``: Get anomaly scores (higher = more anomalous)
* ``fit_predict(X)``: Fit and predict in one call

Choosing a Detector
-------------------

* **Statistical Methods** (`ZScoreDetector`, `IQROutlierDetector`): Fast, simple, good for baseline
* **Machine Learning** (`IsolationForestDetector`, `LOFDetector`): Good for multivariate data
* **Wavelet Methods** (`WaveletDetector`): Excellent for time series with sudden changes
* **Autoencoders** (`LSTMAutoencoderDetector`, `PyTorchAutoencoderDetector`): Powerful for complex patterns
* **Ensembles** (`VotingEnsemble`): Combine multiple methods for robust detection

See the :doc:`user_guide/index` for detailed guidance on choosing the right detector for your use case.

Next Steps
----------

* Read the :doc:`user_guide/index` for detailed usage instructions
* Check out the :doc:`examples/index` for more examples
* Explore the :doc:`api/index` for full API reference
