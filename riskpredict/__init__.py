"""Disease Risk Prediction Demo.

An educational machine-learning project: synthetic health-check data and its cleaning, feature engineering,
model comparison with cross-validation, probability calibration, a screening threshold chosen for high
sensitivity, honest evaluation on a held-out test set, per-person explanations with "what if" scenarios,
and prediction interfaces (menu, local web form, JSON API, batch CSV).

It is a demonstration of the workflow. It is NOT a medical device and must never be used to diagnose,
treat or rule out any condition.
"""

import logging

__version__ = "1.0.0"

logging.getLogger("riskpredict").addHandler(logging.NullHandler())
