Ensemble Methods
================

Ensemble methods combine predictions from multiple detectors for more robust anomaly detection.

Voting Ensemble
---------------

The voting ensemble flags an anomaly if a specified number of detectors agree.

.. code-block:: python

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

   # Create ensemble (flags if 2+ detectors agree)
   ensemble = VotingEnsemble(detectors, voting_threshold=2)
   ensemble.fit(X)
   predictions, scores = ensemble.fit_predict(X)

Parameters
~~~~~~~~~~

* ``detectors`` (list): List of anomaly detectors to ensemble
* ``voting_threshold`` (int, default=2): Minimum number of detectors that must flag a sample
* ``random_state`` (int, optional): Random state for reproducibility

General Ensemble Detector
--------------------------

The general ensemble detector allows custom combination methods.

.. code-block:: python

   from anomaly_detection_toolkit import EnsembleDetector

   ensemble = EnsembleDetector(
       detectors=detectors,
       combination_method='mean',  # 'mean', 'max', 'min', 'median', or callable
       voting_threshold=2  # Optional
   )
   ensemble.fit(X)
   predictions, scores = ensemble.fit_predict(X)

Combination Methods
-------------------

Available combination methods:

* ``'mean'``: Average scores across detectors
* ``'max'``: Maximum score
* ``'min'``: Minimum score
* ``'median'``: Median score
* ``callable``: Custom function taking array of scores and returning combined score

Best Practices
--------------

* Use diverse detectors (different algorithms) for better ensemble performance
* Voting threshold of 2 or more typically works well for 3+ detectors
* Consider the trade-off between precision and recall when choosing threshold
