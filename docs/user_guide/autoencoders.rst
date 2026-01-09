Autoencoders
============

Deep learning autoencoders learn normal patterns and flag points with high reconstruction error as anomalies.

LSTM Autoencoder
----------------

The LSTM autoencoder uses an LSTM architecture to learn sequential patterns in time series data.

.. code-block:: python

   from anomaly_detection_toolkit import LSTMAutoencoderDetector
   import numpy as np

   # Time series data
   ts = np.sin(np.linspace(0, 50, 1000)) + np.random.randn(1000) * 0.1
   ts[450:460] += 3  # Inject anomalies

   detector = LSTMAutoencoderDetector(
       window_size=20,
       lstm_units=[32, 16],
       epochs=50,
       threshold_std=3.0
   )
   detector.fit(ts)
   predictions, scores = detector.fit_predict(ts)

Parameters
~~~~~~~~~~

* ``window_size`` (int, default=20): Size of sliding window
* ``lstm_units`` (list, default=[32, 16]): Number of units in encoder/decoder layers
* ``contamination`` (float, default=0.05): Expected proportion of outliers
* ``threshold_std`` (float, default=3.0): Number of standard deviations for threshold
* ``epochs`` (int, default=50): Number of training epochs
* ``batch_size`` (int, default=32): Batch size for training
* ``random_state`` (int, optional): Random state for reproducibility

Requirements
~~~~~~~~~~~~

Requires TensorFlow/Keras:
.. code-block:: bash

   pip install tensorflow

PyTorch Autoencoder
-------------------

The PyTorch autoencoder uses a feedforward architecture for time series anomaly detection.

.. code-block:: python

   from anomaly_detection_toolkit import PyTorchAutoencoderDetector

   detector = PyTorchAutoencoderDetector(
       window_size=24,
       hidden_dims=[64, 16, 4],
       epochs=200,
       threshold_std=3.0
   )
   detector.fit(ts)
   predictions, scores = detector.fit_predict(ts)

Parameters
~~~~~~~~~~

* ``window_size`` (int, default=24): Size of sliding window
* ``hidden_dims`` (list, default=[64, 16, 4]): Hidden dimensions for encoder
* ``learning_rate`` (float, default=1e-3): Learning rate
* ``epochs`` (int, default=200): Number of training epochs
* ``batch_size`` (int, default=32): Batch size
* ``threshold_std`` (float, default=3.0): Number of standard deviations for threshold
* ``random_state`` (int, optional): Random state for reproducibility

Requirements
~~~~~~~~~~~~

Requires PyTorch:
.. code-block:: bash

   pip install torch
