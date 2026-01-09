Anomaly Detection Toolkit
==========================

.. image:: https://readthedocs.org/projects/anomaly-detection-toolkit/badge/?version=latest
    :target: https://anomaly-detection-toolkit.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

A comprehensive Python library for detecting anomalies in time series and multivariate data.
Supports multiple detection methods including statistical, machine learning, and deep learning approaches.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   user_guide/index
   api/index
   examples/index
   contributing

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install anomaly-detection-toolkit

Use a detector:

.. code-block:: python

   from anomaly_detection_toolkit import IsolationForestDetector
   import numpy as np

   # Your data
   X = np.random.randn(1000, 3)

   # Create and fit detector
   detector = IsolationForestDetector(contamination=0.05)
   detector.fit(X)

   # Detect anomalies
   predictions, scores = detector.fit_predict(X)

Features
--------

* **Statistical Methods**: Z-score, IQR, seasonal baseline detection
* **Machine Learning**: Isolation Forest, Local Outlier Factor (LOF), Robust Covariance
* **Wavelet Methods**: Wavelet decomposition and denoising for time series
* **Deep Learning**: LSTM and PyTorch autoencoders for anomaly detection
* **Ensemble Methods**: Voting and score combination ensembles
* **Scikit-learn Compatible**: Familiar API that works with existing workflows

Installation
------------

Basic installation:

.. code-block:: bash

   pip install anomaly-detection-toolkit

With deep learning support:

.. code-block:: bash

   pip install anomaly-detection-toolkit[deep]

Development installation:

.. code-block:: bash

   git clone https://github.com/kylejones200/anomaly-detection-toolkit.git
   cd anomaly-detection-toolkit
   pip install -e ".[deep]"

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
