"""Input validation and the prediction service."""

import math
import unittest

import numpy as np
import pandas as pd

from riskpredict import schema as S
from riskpredict import validators as v
from riskpredict.exceptions import DataError, ValidationError
from riskpredict.predictor import MAX_BATCH, require_predictor
from tests.helpers import HIGH_RISK, LOW_RISK, MINIMAL, TempDirTest, shared


class TestParsing(unittest.TestCase):
    def test_numbers_with_units_and_formats(self):
        self.assertEqual(v.parse_number("age", "54 years")[0], 54)
        self.assertEqual(v.parse_number("age", " 54 ")[0], 54)
        self.assertEqual(v.parse_number("bmi", "27.5")[0], 27.5)
        self.assertEqual(v.parse_number("cholesterol", "195 mg/dL")[0], 195)
        self.assertEqual(v.parse_number("exercise_hours", "3.5 h")[0], 3.5)

    def test_blood_pressure_readings(self):
        number, note = v.parse_number("systolic_bp", "138/88")
        self.assertEqual(number, 138)
        self.assertIn("upper number", note)
        self.assertEqual(v.parse_number("systolic_bp", "120 mmHg")[0], 120)

    def test_glucose_in_mmol_is_converted_with_a_note(self):
        number, note = v.parse_number("glucose", "5.5 mmol/L")
        self.assertEqual(number, 99)
        self.assertIn("mmol/L", note)
        number, note = v.parse_number("glucose", 6.0)         # a bare 6.0 can only be mmol/L
        self.assertEqual(number, 108)
        self.assertEqual(v.parse_number("glucose", 100)[1], None)

    def test_bad_numbers_have_clear_messages(self):
        cases = [("age", "abc", "must be a number"), ("age", "17", "between 18 and 100"), ("age", "45.5", "whole number"),
                 ("bmi", "5", "between 12 and 60"), ("systolic_bp", "400", "between 80 and 240"), ("age", True, "must be a number"),
                 ("age", "inf", "must be a number"), ("age", float("nan"), "normal number"), ("bmi", "30 kg", "must be a number in")]
        for feature, value, message in cases:
            with self.assertRaisesRegex(ValueError, message, msg=f"{feature}={value!r}"):
                v.parse_number(feature, value)

    def test_categories_suggest_corrections(self):
        self.assertEqual(v.parse_category("sex", "female"), "Female")
        with self.assertRaisesRegex(ValueError, "Did you mean 'Female'"):
            v.parse_category("sex", "Femal")
        with self.assertRaisesRegex(ValueError, "Choose one of: None, Moderate, Heavy"):
            v.parse_category("alcohol", "lots")

    def test_a_minimal_record_is_complete(self):
        record, notes, assumed = v.parse_features(MINIMAL)
        self.assertEqual(set(record), set(S.RAW_FEATURES))
        self.assertEqual(record["sex"], "Female")
        self.assertTrue(math.isnan(record["cholesterol"]))
        self.assertEqual(set(assumed), set(S.RAW_FEATURES) - set(S.REQUIRED_INPUT))
        self.assertEqual(notes, [])

    def test_keys_are_forgiving(self):
        record, _, _ = v.parse_features({"Age": "40", " SEX ": "m", "BMI": "25", "Systolic BP": "120", "Family-History": "yes"})
        self.assertEqual((record["age"], record["sex"], record["family_history"]), (40, "Male", 1))

    def test_every_problem_is_reported_at_once(self):
        with self.assertRaises(ValidationError) as ctx:
            v.parse_features({"age": "x", "sex": "robot", "bmi": 99, "smoker": "perhaps", "colour": "red"})
        errors = ctx.exception.errors
        for field in ("age", "sex", "bmi", "smoker", "systolic_bp", "colour"):
            self.assertIn(field, errors)

    def test_missing_required_fields(self):
        with self.assertRaises(ValidationError) as ctx:
            v.parse_features({"age": 50})
        self.assertEqual(set(ctx.exception.errors), {"sex", "bmi", "systolic_bp"})
        with self.assertRaises(ValidationError):
            v.parse_features({**MINIMAL, "age": ""})

    def test_unknown_fields_get_a_hint(self):
        with self.assertRaisesRegex(ValidationError, "Did you mean 'cholesterol'"):
            v.parse_features({**MINIMAL, "cholestrol": 200})

    def test_hdl_cannot_exceed_total_cholesterol(self):
        with self.assertRaisesRegex(ValidationError, "cannot be higher"):
            v.parse_features({**MINIMAL, "cholesterol": 105, "hdl": 115})
        v.parse_features({**MINIMAL, "cholesterol": 150, "hdl": 60})

    def test_non_dict_is_refused(self):
        with self.assertRaises(ValidationError):
            v.parse_features([1, 2, 3])

    def test_settings_validators(self):
        self.assertEqual(v.validate_port("8040"), 8040)
        for bad in ("80", "99999", "x"):
            with self.assertRaises(ValidationError):
                v.validate_port(bad)
        self.assertEqual(v.validate_int(" 5 ", "N", 1, 10), 5)
        with self.assertRaises(ValidationError):
            v.validate_int("11", "N", 1, 10)
        self.assertEqual(v.validate_number("0.85", "S", 0.5, 0.99), 0.85)
        with self.assertRaises(ValidationError):
            v.validate_number("nan", "S", 0.5, 0.99)


class TestPredictions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = require_predictor(shared()[1])

    def test_high_and_low_risk_people(self):
        high, low = self.predictor.predict(HIGH_RISK), self.predictor.predict(LOW_RISK)
        self.assertEqual(high.band, "High")
        self.assertEqual(low.band, "Low")
        self.assertGreater(high.probability, 0.7)
        self.assertLess(low.probability, 0.1)
        self.assertGreater(high.percent, low.percent)

    def test_risk_rises_with_each_obvious_factor(self):
        base = self.predictor.predict(MINIMAL).probability
        for change in ({"age": 70}, {"smoker": "yes"}, {"systolic_bp": 170}, {"sex": "Male"}, {"bmi": 38}):
            self.assertGreater(self.predictor.predict({**MINIMAL, **change}).probability, base, change)
        self.assertLess(self.predictor.predict({**MINIMAL, "age": 25}).probability, base)

    def test_probabilities_stay_inside_zero_and_one(self):
        for person in (HIGH_RISK, LOW_RISK, MINIMAL, {**HIGH_RISK, "age": 100, "systolic_bp": 240}):
            p = self.predictor.predict(person).probability
            self.assertTrue(0 < p < 1)

    def test_factors_explain_the_estimate(self):
        p = self.predictor.predict(HIGH_RISK)
        labels = [f["label"] for f in p.factors]
        self.assertIn("Age", labels)
        self.assertIn("Smoking", labels)
        self.assertEqual([abs(f["effect"]) for f in p.factors], sorted((abs(f["effect"]) for f in p.factors), reverse=True))
        self.assertTrue(all(f["effect"] > 0 for f in p.factors if f["feature"] in ("age", "smoker", "systolic_bp")))
        young = self.predictor.predict(LOW_RISK)
        age = next(f for f in young.factors if f["feature"] == "age")
        self.assertLess(age["effect"], 0)                                   # being young lowers the estimate

    def test_what_ifs_only_offer_improvements(self):
        p = self.predictor.predict(HIGH_RISK)
        self.assertTrue(p.what_ifs)
        self.assertTrue(all(w["change"] < 0 and w["probability"] < p.probability for w in p.what_ifs))
        self.assertEqual([w["change"] for w in p.what_ifs], sorted(w["change"] for w in p.what_ifs))
        self.assertIn("smoke", " ".join(w["scenario"] for w in p.what_ifs))
        healthy = self.predictor.predict(LOW_RISK)
        self.assertEqual([w["feature"] for w in healthy.what_ifs if w["feature"] in ("smoker", "alcohol")], [])

    def test_assumed_values_are_reported_and_not_explained(self):
        p = self.predictor.predict(MINIMAL)
        self.assertIn("cholesterol", p.assumed)
        self.assertTrue(all(f["feature"] not in p.assumed for f in p.factors))
        self.assertEqual(self.predictor.predict(HIGH_RISK).assumed, [])

    def test_unit_notes_are_passed_on(self):
        p = self.predictor.predict({**MINIMAL, "systolic_bp": "130/85", "glucose": "5.4 mmol/L"})
        self.assertEqual(len(p.notes), 2)
        self.assertEqual(p.record["glucose"], 97)

    def test_values_outside_the_training_range_warn(self):
        p = self.predictor.predict({**MINIMAL, "systolic_bp": 238})
        self.assertTrue(any("outside the range" in w for w in p.warnings))
        self.assertEqual(self.predictor.predict(MINIMAL).warnings, [])

    def test_to_dict_is_json_ready(self):
        import json
        d = self.predictor.predict(HIGH_RISK).to_dict()
        json.dumps(d)
        for key in ("risk_percent", "band", "message", "factors", "what_if", "assumed_typical_values_for", "disclaimer",
                    "screening_threshold_percent", "typical_person_percent"):
            self.assertIn(key, d)
        self.assertIn("not a medical device", d["disclaimer"])

    def test_every_band_has_a_message(self):
        for person in (HIGH_RISK, LOW_RISK, {**MINIMAL, "age": 52, "systolic_bp": 135, "smoker": "no", "bmi": 27}):
            p = self.predictor.predict(person)
            self.assertTrue(p.message)
            self.assertIn(p.band, ("Low", "Moderate", "High"))

    def test_bad_input_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            self.predictor.predict({"age": 10})

    def test_what_if_tool(self):
        results = self.predictor.what_if(MINIMAL, "age", [30, 50, 70])
        probabilities = [p for _, p in results]
        self.assertEqual([x for x, _ in results], [30, 50, 70])
        self.assertEqual(probabilities, sorted(probabilities))
        self.assertEqual([x for x, _ in self.predictor.what_if(MINIMAL, "Smoker", ["no", "yes"])], [0, 1])
        for bad in (("colour", [1]), ("age", []), ("age", [10]), ("age", list(range(20, 40)))):
            with self.assertRaises(ValidationError):
                self.predictor.what_if(MINIMAL, *bad)

    def test_suggestions(self):
        self.assertEqual(self.predictor.suggestions("sex"), S.SEXES)
        self.assertEqual(self.predictor.suggestions("smoker"), ["no", "yes"])
        ages = self.predictor.suggestions("age")
        self.assertEqual(len(ages), 5)
        self.assertEqual(ages, sorted(ages))

    def test_health_summary(self):
        h = self.predictor.health()
        self.assertEqual(h["status"], "ok")
        self.assertTrue(0.5 < h["roc_auc"] < 1)


class TestBatch(TempDirTest):
    @classmethod
    def setUpClass(cls):
        cls.predictor = require_predictor(shared()[1])

    def frame(self):
        return pd.DataFrame([
            {S.ID: "a", **HIGH_RISK},
            {S.ID: "b", **LOW_RISK},
            {S.ID: "c", "age": "40", "sex": "F", "bmi": "24", "systolic_bp": "118/76"},
            {S.ID: "d", "age": "5", "sex": "Male", "bmi": "24", "systolic_bp": "118"},
            {S.ID: "e", "age": "40", "sex": "Other", "bmi": "24", "systolic_bp": "118"},
        ])

    def test_rows_are_scored_independently(self):
        out = self.predictor.predict_frame(self.frame())
        self.assertEqual(list(out["status"][:3]), ["ok", "ok", "ok"])
        self.assertIn("between 18 and 100", out.loc[3, "status"])
        self.assertIn("row 5", out.loc[3, "status"])
        self.assertIn("not supported", out.loc[4, "status"])
        self.assertTrue(np.isnan(out.loc[3, "risk_percent"]))
        self.assertEqual(out.loc[0, "risk_band"], "High")
        self.assertEqual(out.loc[1, "risk_band"], "Low")
        self.assertEqual(out[S.ID].tolist(), list("abcde"))

    def test_batch_matches_single_predictions(self):
        out = self.predictor.predict_frame(self.frame())
        single = self.predictor.predict(HIGH_RISK).percent
        self.assertAlmostEqual(out.loc[0, "risk_percent"], single, delta=0.06)

    def test_csv_round_trip(self):
        source = self.dir / "people.csv"
        self.frame().to_csv(source, index=False)
        target = self.dir / "out" / "scored.csv"
        result = self.predictor.predict_csv(source, target)
        saved = pd.read_csv(target)
        self.assertEqual(len(saved), len(result))
        self.assertIn("risk_percent", saved.columns)

    def test_csv_problems(self):
        with self.assertRaisesRegex(DataError, "not found"):
            self.predictor.predict_csv(self.dir / "nope.csv", self.dir / "o.csv")
        empty = self.dir / "empty.csv"
        empty.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(DataError, "Could not read"):
            self.predictor.predict_csv(empty, self.dir / "o.csv")

    def test_batch_size_limit(self):
        big = pd.DataFrame([MINIMAL] * (MAX_BATCH + 1))
        with self.assertRaisesRegex(DataError, "At most"):
            self.predictor.predict_frame(big)

    def test_file_with_no_valid_rows(self):
        out = self.predictor.predict_frame(pd.DataFrame([{"age": 5}]))
        self.assertEqual(out.loc[0, "status"][:6], "row 2:")
        self.assertEqual(out.loc[0, "risk_band"], "")
