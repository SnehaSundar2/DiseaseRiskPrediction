"""Shared fixtures. Training takes a few seconds, so ONE small model is trained for the whole test run."""

import atexit
import shutil
import tempfile
import unittest
from pathlib import Path

from riskpredict import datagen, persistence, training

_SHARED = {}

# an older, heavy-smoking man with high blood pressure, cholesterol and glucose and a family history: very high risk
HIGH_RISK = {"age": 64, "sex": "Male", "bmi": 32, "systolic_bp": 154, "cholesterol": 262, "hdl": 34, "glucose": 121,
             "exercise_hours": 0.5, "smoker": "yes", "family_history": "yes", "bp_medication": "yes", "alcohol": "Heavy"}
# a young, active, non-smoking woman with healthy numbers: very low risk
LOW_RISK = {"age": 26, "sex": "Female", "bmi": 22, "systolic_bp": 108, "cholesterol": 165, "hdl": 64, "glucose": 82,
            "exercise_hours": 7, "smoker": "no", "family_history": "no", "bp_medication": "no", "alcohol": "Moderate"}
MINIMAL = {"age": 50, "sex": "Female", "bmi": 26, "systolic_bp": 128}


def shared():
    """(TrainingResult, saved model directory, raw training DataFrame) - trained once, in fast mode."""
    if not _SHARED:
        folder = Path(tempfile.mkdtemp(prefix="riskpredict_tests_"))
        atexit.register(shutil.rmtree, folder, ignore_errors=True)
        raw = datagen.generate(2500)
        result = training.train(raw, fast=True, tune=False, cv=3, source="synthetic test data")
        persistence.save(result, folder / "model")
        _SHARED.update(result=result, model_dir=folder / "model", raw=raw, folder=folder)
    return _SHARED["result"], _SHARED["model_dir"], _SHARED["raw"]


class TempDirTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def copy_model(self):
        """A private copy of the trained model that a test may damage (each call makes a fresh copy)."""
        _, model_dir, _ = shared()
        self._copies = getattr(self, "_copies", 0) + 1
        target = self.dir / f"model_copy_{self._copies}"
        shutil.copytree(model_dir, target)
        return target
