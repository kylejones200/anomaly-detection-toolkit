Time Series Anomaly Detection
==============================

Time series anomaly detection has special considerations compared to general multivariate detection.

Preprocessing
-------------

Before applying anomaly detection to time series data:

1. Handle missing values
2. Remove or interpolate outliers that are clearly data errors
3. Consider seasonal decomposition if applicable
4. Normalize if working with multiple series with different scales

Example with Seasonal Decomposition
------------------------------------

.. code-block:: python

   from anomaly_detection_toolkit import SeasonalBaselineDetector
   from statsmodels.tsa.seasonal import STL
   import pandas as pd
   import numpy as np

   # Your time series
   dates = pd.date_range('2020-01-01', periods=365, freq='D')
   values = your_time_series_values
   df = pd.DataFrame({'date': dates, 'value': values})

   # Option 1: Use seasonal baseline detector directly
   detector = SeasonalBaselineDetector(seasonality='week')
   detector.fit(df, date_col='date', value_col='value')
   predictions = detector.predict(df, date_col='date', value_col='value')

   # Option 2: Use STL decomposition first, then detect on residuals
   stl = STL(df['value'], period=7, robust=True)
   result = stl.fit()
   residuals = result.resid

   from anomaly_detection_toolkit import ZScoreDetector
   detector = ZScoreDetector(n_std=3.0)
   detector.fit(residuals.values)
   predictions, scores = detector.fit_predict(residuals.values)

Handling Multiple Time Series
------------------------------

When working with multiple related time series, you can:

1. Apply detection to each series independently
2. Use multivariate methods with all series as features
3. Create features (e.g., rolling statistics) and use ML methods

.. code-block:: python

   # Multiple series as features
   import pandas as pd
   from anomaly_detection_toolkit import IsolationForestDetector

   df = pd.DataFrame({
       'series1': series1_values,
       'series2': series2_values,
       'series3': series3_values
   })

   detector = IsolationForestDetector(contamination=0.05)
   detector.fit(df)
   predictions, scores = detector.fit_predict(df)

Real-time Detection
-------------------

For real-time or streaming time series:

1. Maintain a sliding window of recent data
2. Periodically retrain or update the detector
3. Use incremental methods when possible
4. Consider ensemble methods for robustness

Example with sliding window:

.. code-block:: python

   from anomaly_detection_toolkit import WaveletDetector
   import numpy as np

   window_size = 100
   detector = WaveletDetector()

   for i in range(window_size, len(ts)):
       window = ts[i-window_size:i]
       detector.fit(window)
       prediction = detector.predict(window[-10:])  # Check last 10 points
       if prediction[-1] == -1:
           print(f"Anomaly detected at index {i}")
