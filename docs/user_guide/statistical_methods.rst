Statistical Methods
===================

Statistical methods provide fast and interpretable anomaly detection using statistical principles.

Z-Score Detector
----------------

The Z-score detector flags points that are more than a specified number of standard deviations from the mean.

.. code-block:: python

   from anomaly_detection_toolkit import ZScoreDetector
   import numpy as np

   # Your data
   data = np.random.randn(1000)
   data[100:105] += 5  # Inject anomalies

   # Create detector
   detector = ZScoreDetector(n_std=3.0)
   detector.fit(data)

   # Detect anomalies
   predictions, scores = detector.fit_predict(data)

Parameters
~~~~~~~~~~

* ``n_std`` (float, default=3.0): Number of standard deviations for threshold
* ``random_state`` (int, optional): Random state for reproducibility

IQR Outlier Detector
--------------------

The IQR (Interquartile Range) detector flags points outside Q1 - 1.5*IQR and Q3 + 1.5*IQR.

.. code-block:: python

   from anomaly_detection_toolkit import IQROutlierDetector

   detector = IQROutlierDetector(factor=1.5)
   detector.fit(data)
   predictions, scores = detector.fit_predict(data)

Parameters
~~~~~~~~~~

* ``factor`` (float, default=1.5): IQR multiplier for outlier threshold
* ``random_state`` (int, optional): Random state for reproducibility

Seasonal Baseline Detector
---------------------------

The seasonal baseline detector calculates seasonal baselines (e.g., weekly, monthly) and flags points that deviate significantly from expected seasonal patterns.

.. code-block:: python

   from anomaly_detection_toolkit import SeasonalBaselineDetector
   import pandas as pd

   # Time series data with dates
   df = pd.DataFrame({
       'date': pd.date_range('2020-01-01', periods=365, freq='D'),
       'value': your_values
   })

   detector = SeasonalBaselineDetector(seasonality='week', threshold_sigma=2.5)
   detector.fit(df, date_col='date', value_col='value')
   predictions = detector.predict(df, date_col='date', value_col='value')

Parameters
~~~~~~~~~~

* ``seasonality`` (str, default='week'): Seasonality to use ('week', 'month', 'day', 'hour')
* ``threshold_sigma`` (float, default=2.5): Number of standard deviations for threshold
* ``random_state`` (int, optional): Random state for reproducibility
