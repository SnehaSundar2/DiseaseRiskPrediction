"""Schema helpers, synthetic data, loading and cleaning."""

import unittest

import numpy as np
import pandas as pd

from riskpredict import datagen, schema as S
from riskpredict import preprocessing as P
from riskpredict.exceptions import DataError
from tests.helpers import TempDirTest


def tiny(**overrides):
    """A 400-row clean dataset with some columns replaced, for targeted cleaning tests."""
    df = datagen.generate(400, seed=5, dirty=False).astype(object)   # object columns, like a freshly read CSV
    for column, value in overrides.items():
        df[column] = value
    return df


class TestSchema(unittest.TestCase):
    def test_categories_accept_messy_spellings(self):
        self.assertEqual(S.canonical_category("sex", "  FEMALE "), "Female")
        self.assertEqual(S.canonical_category("sex", "m"), "Male")
        self.assertEqual(S.canonical_category("alcohol", "light"), "Moderate")
        self.assertEqual(S.canonical_category("alcohol", "never"), "None")
        self.assertIsNone(S.canonical_category("sex", "other"))
        self.assertIsNone(S.canonical_category("sex", None))
        self.assertIsNone(S.canonical_category("sex", float("nan")))

    def test_yes_no(self):
        for yes in ("Yes", "y", "TRUE", 1, "1", 1.0):
            self.assertEqual(S.parse_yes_no(yes), 1, yes)
        for no in ("No", "n", "false", 0, "0", 0.0):
            self.assertEqual(S.parse_yes_no(no), 0, no)
        for bad in ("maybe", "", 2, float("nan")):
            self.assertIsNone(S.parse_yes_no(bad), bad)

    def test_labels(self):
        for yes in ("Yes ", 1, 1.0, "disease", "TRUE", True):
            self.assertEqual(S.canonical_label(yes), 1, yes)
        for no in ("no", 0, "healthy", "False", False):
            self.assertEqual(S.canonical_label(no), 0, no)
        for bad in ("unsure", None, float("nan"), 2, ""):
            self.assertIsNone(S.canonical_label(bad), bad)

    def test_risk_bands(self):
        self.assertEqual(S.risk_band(0.30, 0.20), "High")
        self.assertEqual(S.risk_band(0.20, 0.20), "High")
        self.assertEqual(S.risk_band(0.10, 0.20), "Moderate")
        self.assertEqual(S.risk_band(0.099, 0.20), "Low")

    def test_every_feature_is_described_and_labelled(self):
        for feature in S.RAW_FEATURES:
            self.assertTrue(S.describe(feature))
            self.assertIn(feature, S.LABELS)
        for feature, (target, text) in S.MODIFIABLE.items():
            self.assertIn(feature, S.RAW_FEATURES)
            self.assertTrue(text)


class TestGenerator(unittest.TestCase):
    def test_same_seed_same_data(self):
        a, b = datagen.generate(300, seed=3), datagen.generate(300, seed=3)
        pd.testing.assert_frame_equal(a, b)
        self.assertFalse(a.equals(datagen.generate(300, seed=4)))

    def test_clean_data_is_valid_and_has_a_plausible_disease_rate(self):
        df = datagen.generate(3000, dirty=False)
        self.assertEqual(list(df.columns), [S.ID] + S.RAW_FEATURES + [S.TARGET])
        self.assertTrue(0.15 < (df[S.TARGET] == "Yes").mean() < 0.32)
        for column, (kind, low, high, _) in S.NUMERIC.items():
            self.assertTrue(df[column].between(low, high).all(), column)
        self.assertEqual(df[S.ID].nunique(), len(df))

    def test_dirty_data_has_problems(self):
        df = datagen.generate(1500)
        self.assertGreater(len(df), 1500)
        self.assertTrue(df.isna().any().any())
        self.assertTrue((pd.to_numeric(df["age"], errors="coerce") > 100).any())

    def test_true_risk_rises_with_the_obvious_factors(self):
        def p(**kw):
            base = dict(age=45, sex="Female", bmi=25, systolic_bp=120, cholesterol=190, hdl=55, glucose=90,
                        exercise_hours=3, smoker=0, family_history=0, bp_medication=0, alcohol="Moderate")
            base.update(kw)
            return float(datagen.disease_probability(*[np.array([base[k]]) for k in base])[0])
        self.assertGreater(p(age=65), p(age=45))
        self.assertGreater(p(smoker=1), p(smoker=0))
        self.assertGreater(p(systolic_bp=160), p(systolic_bp=120))
        self.assertGreater(p(sex="Male"), p(sex="Female"))
        self.assertGreater(p(exercise_hours=0), p(exercise_hours=8))
        self.assertGreater(p(alcohol="Heavy"), p(alcohol="Moderate"))
        self.assertTrue(0 < p() < 1)


class TestLoading(TempDirTest):
    def write(self, df, name="data.csv"):
        path = self.dir / name
        df.to_csv(path, index=False)
        return path

    def test_loads_a_valid_file_and_normalises_headers(self):
        df = datagen.generate(400, dirty=False).rename(columns={"systolic_bp": " Systolic BP", "bmi": "BMI"})
        df = df.rename(columns={" Systolic BP": "Systolic-BP"})
        out = P.load_dataset(self.write(df))
        self.assertIn("systolic_bp", out.columns)
        self.assertIn("bmi", out.columns)
        self.assertEqual(len(out), 400)

    def test_person_id_is_optional(self):
        out = P.load_dataset(self.write(datagen.generate(400, dirty=False).drop(columns=[S.ID])))
        self.assertNotIn(S.ID, out.columns)

    def test_clear_errors(self):
        with self.assertRaisesRegex(DataError, "not found"):
            P.load_dataset(self.dir / "nope.csv")
        with self.assertRaisesRegex(DataError, "empty"):
            P.load_dataset(self._empty())
        with self.assertRaisesRegex(DataError, "missing column"):
            P.load_dataset(self.write(datagen.generate(400, dirty=False).drop(columns=["glucose", "age"])))
        with self.assertRaisesRegex(DataError, "only 10 rows"):
            P.load_dataset(self.write(datagen.generate(400, dirty=False).head(10)))

    def _empty(self):
        path = self.dir / "empty.csv"
        path.write_text("", encoding="utf-8")
        return path

    def test_not_utf8(self):
        path = self.dir / "latin.csv"
        path.write_bytes(b"age,sex\n\xe9\xe8,x\n")
        with self.assertRaisesRegex(DataError, "UTF-8"):
            P.load_dataset(path)


class TestCleaning(unittest.TestCase):
    def test_clean_data_passes_through_untouched(self):
        df = datagen.generate(500, dirty=False)
        cleaned, report = P.clean(df)
        self.assertEqual(len(cleaned), 500)
        self.assertEqual(report.steps, [])
        self.assertEqual(report.rows_dropped, 0)

    def test_labels_become_zero_one_and_bad_labels_are_dropped(self):
        df = tiny()
        df.loc[0, S.TARGET] = "unsure"
        df.loc[1, S.TARGET] = np.nan
        df.loc[2, S.TARGET] = "healthy"
        df.loc[3, S.TARGET] = "1"
        cleaned, report = P.clean(df)
        self.assertEqual(len(cleaned), 398)
        self.assertTrue(set(cleaned[S.TARGET]) <= {0, 1})
        self.assertIn("missing or unrecognised outcome label", report.steps[0][0])
        self.assertEqual(report.steps[0][1], 2)

    def test_spellings_are_standardised_and_missing_sex_drops_the_row(self):
        df = tiny()
        df.loc[0, "sex"] = "F"
        df.loc[1, "sex"] = "MALE "
        df.loc[2, "sex"] = np.nan
        df.loc[3, "alcohol"] = "light"
        cleaned, report = P.clean(df)
        self.assertEqual(len(cleaned), 399)
        self.assertEqual(set(cleaned["sex"]), {"Female", "Male"})
        self.assertEqual(cleaned.loc[cleaned[S.ID] == df.loc[3, S.ID], "alcohol"].iloc[0], "Moderate")
        self.assertTrue(any("unknown or missing sex" in s[0] for s in report.steps))

    def test_unreadable_numbers_and_out_of_range_become_missing(self):
        df = tiny()
        df.loc[0, "bmi"] = "n/a"
        df.loc[1, "age"] = 250
        df.loc[2, "systolic_bp"] = 12
        df.loc[3, "cholesterol"] = 9000
        cleaned, report = P.clean(df)
        self.assertEqual(len(cleaned), 400)
        for index, column in ((0, "bmi"), (1, "age"), (2, "systolic_bp"), (3, "cholesterol")):
            self.assertTrue(np.isnan(cleaned.loc[index, column]), column)
        self.assertTrue(any("Out-of-range age" in s[0] for s in report.steps))

    def test_glucose_in_mmol_is_converted(self):
        df = tiny()
        df.loc[0, "glucose"] = 5.5
        cleaned, report = P.clean(df)
        self.assertEqual(cleaned.loc[0, "glucose"], 99)
        self.assertTrue(any("mmol/L" in s[0] for s in report.steps))

    def test_hdl_above_total_cholesterol_is_blanked(self):
        df = tiny()
        df.loc[0, ["cholesterol", "hdl"]] = [105, 115]
        cleaned, _ = P.clean(df)
        self.assertTrue(np.isnan(cleaned.loc[0, "hdl"]) and np.isnan(cleaned.loc[0, "cholesterol"]))

    def test_yes_no_variants(self):
        df = tiny()
        df.loc[0, "smoker"] = "Yes"
        df.loc[1, "smoker"] = "N"
        df.loc[2, "smoker"] = "maybe"
        cleaned, _ = P.clean(df)
        self.assertEqual(cleaned.loc[0, "smoker"], 1)
        self.assertEqual(cleaned.loc[1, "smoker"], 0)
        self.assertTrue(np.isnan(cleaned.loc[2, "smoker"]))

    def test_duplicates_and_repeated_ids(self):
        df = tiny()
        df = pd.concat([df, df.head(5)], ignore_index=True)
        changed = df.head(3).copy()
        changed["age"] = changed["age"] + 1
        df = pd.concat([df, changed], ignore_index=True)
        cleaned, report = P.clean(df)
        self.assertEqual(len(cleaned), 400)
        self.assertEqual(cleaned[S.ID].nunique(), 400)
        labels = [s[0] for s in report.steps]
        self.assertIn("Duplicate rows", labels)
        self.assertTrue(any("more than once" in label for label in labels))

    def test_input_is_not_modified(self):
        df = datagen.generate(500)
        before = df.copy(deep=True)
        P.clean(df)
        pd.testing.assert_frame_equal(df, before)

    def test_too_little_data_or_one_class_is_refused(self):
        with self.assertRaisesRegex(DataError, "usable rows remain"):
            P.clean(datagen.generate(400, dirty=False).head(100))
        df = tiny()
        df[S.TARGET] = "No"
        df.loc[:10, S.TARGET] = "Yes"
        with self.assertRaisesRegex(DataError, "at least"):
            P.clean(df)

    def test_report_summary(self):
        _, report = P.clean(datagen.generate(800))
        self.assertGreater(report.rows_dropped, 0)
        self.assertEqual(report.rows_out, sum(report.class_counts.values()))
        self.assertIn("rows read", report.lines()[0])
        self.assertEqual(report.to_dict()["rows_in"], report.rows_in)


class TestFeatures(unittest.TestCase):
    def test_engineered_features_use_inputs_only(self):
        df = pd.DataFrame({"cholesterol": [200, np.nan], "hdl": [50, 40], "systolic_bp": [150, np.nan], "glucose": [130, 90]})
        out = P.add_features(df)
        self.assertEqual(out.loc[0, "chol_ratio"], 4.0)
        self.assertTrue(np.isnan(out.loc[1, "chol_ratio"]))
        self.assertEqual(out.loc[0, "high_bp"], 1.0)
        self.assertTrue(np.isnan(out.loc[1, "high_bp"]))
        self.assertEqual((out.loc[0, "high_glucose"], out.loc[1, "high_glucose"]), (1.0, 0.0))
        self.assertNotIn(S.TARGET, out.columns)

    def test_preprocessor_handles_missing_and_unseen_values(self):
        cleaned, _ = P.clean(datagen.generate(600))
        X, y = P.split_xy(cleaned)
        from riskpredict import models as M
        pipe = M.build_pipeline(M.candidates()["Logistic Regression"]).fit(X, y)
        odd = X.head(3).copy()
        odd.loc[odd.index[0], "alcohol"] = "Brandy"
        odd.loc[odd.index[1], "age"] = np.nan
        p = pipe.predict_proba(odd)[:, 1]
        self.assertTrue(((p > 0) & (p < 1)).all())
