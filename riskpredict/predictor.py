"""Prediction service: validate a person, estimate the risk, explain it and show "what if" scenarios.

Explanations are computed by replacing one feature at a time with the typical (training median / most common)
value and measuring how the predicted risk changes. They describe what the MODEL associates with risk in its
training data; they are not causes and not medical advice.
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import persistence
from . import schema as S
from . import validators as v
from .exceptions import DataError, RiskError, ValidationError

log = logging.getLogger("riskpredict")

BAND_MESSAGE = {
    "High": "Above the screening threshold: a person like this would be invited for a check-up conversation.",
    "Moderate": "Above average but below the screening threshold: worth keeping an eye on the listed factors.",
    "Low": "Below the level at which this screening rule would raise a flag.",
}
MIN_EFFECT = 0.005         # ignore factors that move the risk by less than half a percentage point
MAX_BATCH = 5000


@dataclass
class Prediction:
    probability: float
    band: str
    threshold: float
    typical_probability: float
    factors: list = field(default_factory=list)       # [{feature, label, value, effect}] sorted by size
    what_ifs: list = field(default_factory=list)      # [{feature, scenario, probability, change}]
    assumed: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    record: dict = field(default_factory=dict)
    disclaimer: str = S.DISCLAIMER

    @property
    def percent(self):
        return 100 * self.probability

    @property
    def message(self):
        return BAND_MESSAGE[self.band]

    def to_dict(self):
        return {"risk_percent": round(self.percent, 1), "band": self.band, "message": self.message,
                "screening_threshold_percent": round(100 * self.threshold, 1),
                "typical_person_percent": round(100 * self.typical_probability, 1),
                "factors": [{**f, "effect_points": round(100 * f["effect"], 1)} for f in self.factors],
                "what_if": [{**w, "risk_percent": round(100 * w["probability"], 1), "change_points": round(100 * w["change"], 1)}
                            for w in self.what_ifs],
                "assumed_typical_values_for": self.assumed, "notes": self.notes, "warnings": self.warnings,
                "disclaimer": self.disclaimer}


def format_value(feature, value):
    if feature in S.BINARY:
        return "yes" if value == 1 else "no"
    if feature in S.CATEGORICAL:
        return str(value)
    unit = f" {S.UNITS[feature]}" if feature in S.UNITS else ""
    return f"{value:g}{unit}"


def _is_worse_than(feature, value, target):
    """True when the person's value is worse than the healthier reference, so a scenario makes sense."""
    if feature == "smoker":
        return value == 1
    if feature == "alcohol":
        return value == "Heavy"
    if feature == "exercise_hours":
        return value < target
    return value > target


class Predictor:
    def __init__(self, loaded):
        self.pipeline = loaded.pipeline
        self.meta = loaded.metadata
        self.reference = loaded.reference
        self.warnings = list(loaded.warnings)
        self.typical = self.meta["typical"]
        self.threshold = self.meta["thresholds"]["screening"]

    # ------------------------------------------------------------------ single person
    def predict(self, raw):
        record, notes, assumed = v.parse_features(raw)
        frame = pd.DataFrame([record])
        warnings = self._range_warnings(record)

        known = [f for f in S.RAW_FEATURES if f not in assumed]
        variants = [frame]
        for feature in known:
            changed = frame.copy()
            changed[feature] = self.typical[feature]
            variants.append(changed)
        scenarios = []
        for feature, (target, text) in S.MODIFIABLE.items():
            value = record[feature]
            if feature in known and _is_worse_than(feature, value, target):
                changed = frame.copy()
                changed[feature] = target
                variants.append(changed)
                scenarios.append((feature, text))
        probabilities = self.pipeline.predict_proba(pd.concat(variants, ignore_index=True))[:, 1]
        p = float(probabilities[0])
        factors = []
        for feature, p_typical in zip(known, probabilities[1:1 + len(known)]):
            effect = p - float(p_typical)
            if abs(effect) >= MIN_EFFECT:
                factors.append({"feature": feature, "label": S.LABELS[feature],
                                "value": format_value(feature, record[feature]), "effect": effect})
        factors.sort(key=lambda f: -abs(f["effect"]))
        what_ifs = []
        for (feature, text), p_new in zip(scenarios, probabilities[1 + len(known):]):
            if p - float(p_new) >= MIN_EFFECT:
                what_ifs.append({"feature": feature, "scenario": f"Risk {text}",
                                 "probability": float(p_new), "change": float(p_new) - p})
        what_ifs.sort(key=lambda w: w["change"])
        typical_record = pd.DataFrame([{f: self.typical[f] for f in S.RAW_FEATURES}])
        typical_p = float(self.pipeline.predict_proba(typical_record)[0, 1])
        return Prediction(p, S.risk_band(p, self.threshold), self.threshold, typical_p, factors[:8], what_ifs,
                          assumed, notes, warnings, record)

    def what_if(self, raw, feature, values):
        """The same person with different values of one detail: [(value, probability)]. Raises ValidationError on bad input."""
        name = v.normalise_key(feature)
        if name not in S.RAW_FEATURES:
            raise ValidationError(f"Unknown detail '{str(feature)[:30]}'.", {name or "field": "Unknown field."})
        if not values:
            raise ValidationError("Give at least one value to try.")
        if len(values) > 12:
            raise ValidationError("Try at most 12 values at once.")
        records = []
        for value in values:
            records.append(v.parse_features({**raw, name: value})[0])
        probabilities = self.pipeline.predict_proba(pd.DataFrame(records))[:, 1]
        return [(records[i][name], float(p)) for i, p in enumerate(probabilities)]

    def suggestions(self, feature):
        """A few sensible values to compare for a detail."""
        if feature in S.CATEGORICAL:
            return list(S.CATEGORICAL[feature])
        if feature in S.BINARY:
            return ["no", "yes"]
        low, high = self.meta["training_ranges"][feature]
        return [round(low + (high - low) * k / 4, 1) for k in range(5)]

    def _range_warnings(self, record):
        out = []
        for feature, (low, high) in self.meta["training_ranges"].items():
            value = record[feature]
            if isinstance(value, float) and math.isnan(value):
                continue
            span = max(high - low, 1e-9)
            if value < low - 0.1 * span or value > high + 0.1 * span:
                out.append(f"{S.LABELS[feature]} {value:g} is outside the range the model learned from "
                           f"({low:g} to {high:g}), so the estimate is less reliable.")
        return out

    # ------------------------------------------------------------------ many people
    def predict_frame(self, df):
        """Score a DataFrame of raw rows. Returns a copy with risk_percent, risk_band, status; bad rows get a message."""
        if len(df) > MAX_BATCH:
            raise DataError(f"At most {MAX_BATCH} people can be scored at once (this file has {len(df)}).")
        df = df.copy()
        df.columns = [v.normalise_key(c) for c in df.columns]
        out = df.copy()
        out["risk_percent"], out["risk_band"], out["status"] = np.nan, "", "ok"
        records, positions = [], []
        for i, (_, row) in enumerate(df.iterrows()):
            try:
                record, _, _ = v.parse_features({k: row[k] for k in S.RAW_FEATURES if k in df.columns})
                records.append(record)
                positions.append(i)
            except ValidationError as exc:
                out.iloc[i, out.columns.get_loc("status")] = f"row {i + 2}: {exc}"[:200]
        if records:
            p = self.pipeline.predict_proba(pd.DataFrame(records))[:, 1]
            for i, prob in zip(positions, p):
                out.iloc[i, out.columns.get_loc("risk_percent")] = round(100 * float(prob), 1)
                out.iloc[i, out.columns.get_loc("risk_band")] = S.risk_band(float(prob), self.threshold)
        return out

    def predict_csv(self, source, target):
        path = Path(str(source).strip().strip('"'))
        if not path.is_file():
            raise DataError(f"File not found: {path}")
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", dtype=object, keep_default_na=False, na_values=[""])
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as exc:
            raise DataError(f"Could not read {path.name}: {exc}") from None
        result = self.predict_frame(df)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(target, index=False)
        return result

    def health(self):
        m = self.meta
        return {"status": "ok", "model": m["model"], "trained_at": m["trained_at"], "source": m["source"],
                "roc_auc": round(m["metrics"]["test"]["roc_auc"], 3), "screening_threshold_percent": round(100 * self.threshold, 1),
                "sensitivity_percent": round(m["metrics"]["test"]["sensitivity"], 1), "disclaimer": S.DISCLAIMER}


def require_predictor(directory):
    return Predictor(persistence.load(directory))
