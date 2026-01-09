Machine Learning Methods
=========================

Machine learning methods can handle complex patterns in multivariate data.

Isolation Forest
----------------

Isolation Forest is an ensemble method that isolates anomalies by randomly selecting features and splitting values.

.. code-block:: python

   from anomaly_detection_toolkit import IsolationForestDetector
   import numpy as np

   X = np.random.randn(1000, 3)
   detector = IsolationForestDetector(contamination=0.05, n_estimators=200)
   detector.fit(X)
   predictions, scores = detector.fit_predict(X)

Parameters
~~~~~~~~~~

* ``contamination`` (float, default=0.05): Expected proportion of outliers
* ``n_estimators`` (int, default=200): Number of base estimators
* ``random_state`` (int, optional): Random state for reproducibility
* ``n_jobs`` (int, default=-1): Number of parallel jobs

Local Outlier Factor (LOF)
---------------------------

LOF measures the local deviation of density of a given sample with respect to its neighbors.

.. code-block:: python

   from anomaly_detection_toolkit import LOFDetector

   detector = LOFDetector(contamination=0.05, n_neighbors=20)
   detector.fit(X)
   predictions, scores = detector.fit_predict(X)

Parameters
~~~~~~~~~~

* ``contamination`` (float, default=0.05): Expected proportion of outliers
* ``n_neighbors`` (int, default=20): Number of neighbors to use
* ``random_state`` (int, optional): Random state for reproducibility
* ``n_jobs`` (int, default=-1): Number of parallel jobs

Robust Covariance
-----------------

Robust Covariance (Elliptic Envelope) assumes the data is Gaussian distributed and fits an elliptic envelope.

.. code-block:: python

   from anomaly_detection_toolkit import RobustCovarianceDetector

   detector = RobustCovarianceDetector(contamination=0.05, support_fraction=0.8)
   detector.fit(X)
   predictions, scores = detector.fit_predict(X)

Parameters
~~~~~~~~~~

* ``contamination`` (float, default=0.05): Expected proportion of outliers
* ``support_fraction`` (float, default=0.8): Proportion of points to use as support
* ``random_state`` (int, optional): Random state for reproducibility
