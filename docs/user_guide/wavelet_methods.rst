Wavelet Methods
===============

Wavelet-based methods are excellent for time series anomaly detection, especially when dealing with sudden changes or anomalies at different time scales.

Wavelet Detector
----------------

The wavelet detector identifies anomalies by finding large coefficients in wavelet detail levels, which indicate sudden changes.

.. code-block:: python

   from anomaly_detection_toolkit import WaveletDetector
   import numpy as np

   # Time series data
   ts = np.sin(np.linspace(0, 50, 1000)) + np.random.randn(1000) * 0.1
   ts[450:460] += 3  # Inject anomalies

   detector = WaveletDetector(wavelet='db4', threshold_factor=2.5, level=5)
   detector.fit(ts)
   predictions, scores = detector.fit_predict(ts)

Parameters
~~~~~~~~~~

* ``wavelet`` (str, default='db4'): Wavelet type ('db4', 'haar', 'bior2.2', etc.)
* ``threshold_factor`` (float, default=3.0): Threshold factor in terms of MAD
* ``level`` (int, default=5): Decomposition level
* ``random_state`` (int, optional): Random state for reproducibility

Wavelet Denoiser
----------------

The wavelet denoiser removes noise from signals using wavelet thresholding.

.. code-block:: python

   from anomaly_detection_toolkit import WaveletDenoiser

   denoiser = WaveletDenoiser(wavelet='db4', threshold_mode='soft', level=5)
   denoised = denoiser.denoise(ts)

Parameters
~~~~~~~~~~

* ``wavelet`` (str, default='db4'): Wavelet type
* ``threshold_mode`` (str, default='soft'): Thresholding mode ('soft' or 'hard')
* ``level`` (int, default=5): Decomposition level

Available Wavelets
------------------

Common wavelet types include:

* ``'db4'``, ``'db8'``: Daubechies wavelets (good for smooth signals)
* ``'haar'``: Simplest wavelet (good for step functions)
* ``'bior2.2'``: Biorthogonal wavelets (good for images)
