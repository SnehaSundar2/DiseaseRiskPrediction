"""Models, evaluation metrics, the training workflow and model persistence."""

import json
import unittest

import numpy as np
import pandas as pd

from riskpredict import datagen, persistence, report, schema as S, training
from riskpredict import evaluation as E
from riskpredict import models as M
from riskpredict import preprocessing as P
from riskpredict.exceptions import DataError, ModelError, NotTrainedError
from tests.helpers import TempDirTest, shared


class TestMetrics(unittest.TestCase):
    def test_confusion_counts_and_rates(self):
        actual = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        prob = [0.9, 0.8, 0.4, 0.1, 0.7, 0.3, 0.2, 0.1, 0.05, 0.0]
        m = E.classification_metrics(actual, prob, 0.5)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (2, 1, 2, 5))
        self.assertAlmostEqual(m["sensitivity"], 50.0)
        self.assertAlmostEqual(m["specificity"], 100 * 5 / 6)
        self.assertAlmostEqual(m["precision"], 100 * 2 / 3)
        self.assertAlmostEqual(m["npv"], 100 * 5 / 7)
        self.assertAlmostEqual(m["accuracy"], 70.0)
        self.assertEqual(m["flagged"], 3)

    def test_nothing_flagged_does_not_divide_by_zero(self):
        m = E.classification_metrics([1, 0, 1], [0.1, 0.1, 0.1], 0.9)
        self.assertEqual((m["precision"], m["f1"], m["sensitivity"]), (0.0, 0.0, 0.0))

    def test_ranking_metrics(self):
        r = E.ranking_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(r["roc_auc"], 1.0)
        self.assertAlmostEqual(E.ranking_metrics([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])["roc_auc"], 0.5)
        self.assertEqual(E.ranking_metrics([1, 1], [0.4, 0.6])["roc_auc"], 0.0)       # one class only: no AUC

    def test_screening_threshold_meets_the_target(self):
        rng = np.random.default_rng(0)
        actual = (rng.random(2000) < 0.25).astype(int)
        prob = np.clip(0.2 + 0.4 * actual + rng.normal(0, 0.15, 2000), 0, 1)
        for target in (0.7, 0.85, 0.95):
            info = E.choose_screening_threshold(actual, prob, target)
            sensitivity = E.classification_metrics(actual, prob, info["screening"])["sensitivity"] / 100
            self.assertGreaterEqual(sensitivity, target - 1e-9)
            self.assertLess(sensitivity, target + 0.02)
        higher = E.choose_screening_threshold(actual, prob, 0.95)["screening"]
        lower = E.choose_screening_threshold(actual, prob, 0.70)["screening"]
        self.assertLess(higher, lower)                       # catching more cases needs a lower bar

    def test_screening_threshold_with_no_cases(self):
        self.assertEqual(E.choose_screening_threshold([0, 0, 0], [0.1, 0.2, 0.3])["screening"], 0.5)

    def test_bands_add_up(self):
        rng = np.random.default_rng(1)
        prob = rng.random(1000)
        actual = (rng.random(1000) < prob).astype(int)
        bands = E.band_table(actual, prob, 0.4)
        self.assertEqual([b["band"] for b in bands], ["Low", "Moderate", "High"])
        self.assertEqual(sum(b["people"] for b in bands), 1000)
        self.assertEqual(sum(b["with_disease"] for b in bands), int(actual.sum()))
        self.assertLess(bands[0]["rate"], bands[1]["rate"])
        self.assertLess(bands[1]["rate"], bands[2]["rate"])

    def test_calibration_and_roc_tables(self):
        prob = np.linspace(0.01, 0.99, 400)
        actual = (np.random.default_rng(2).random(400) < prob).astype(int)
        for row in E.calibration_table(actual, prob):
            self.assertLess(abs(row["predicted"] - row["observed"]), 20)
        roc = E.roc_points(actual, prob)
        self.assertEqual(roc[0], (0.0, 0.0))
        self.assertEqual(roc[-1], (1.0, 1.0))
        self.assertTrue(all(b[0] >= a[0] and b[1] >= a[1] for a, b in zip(roc, roc[1:])))

    def test_threshold_table_trades_sensitivity_for_specificity(self):
        rng = np.random.default_rng(3)
        prob = rng.random(800)
        actual = (rng.random(800) < prob).astype(int)
        rows = E.threshold_table(actual, prob)
        sens = [r["sensitivity"] for r in rows]
        spec = [r["specificity"] for r in rows]
        self.assertEqual(sens, sorted(sens, reverse=True))
        self.assertEqual(spec, sorted(spec))

    def test_age_groups(self):
        self.assertEqual([E.age_group(a) for a in (30, 40, 54, 55, 69, 70, 90)],
                         ["under 40", "40-54", "40-54", "55-69", "55-69", "70 or older", "70 or older"])


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cleaned, _ = P.clean(datagen.generate(1200, seed=11))
        cls.X, cls.y = P.split_xy(cleaned)

    def test_five_candidates_with_a_baseline(self):
        names = list(M.candidates(fast=True))
        self.assertEqual(len(names), 5)
        self.assertIn(M.BASELINE, names)

    def test_comparison_is_sorted_and_beats_the_baseline(self):
        results = M.compare(self.X, self.y, cv=3, fast=True)
        aucs = [r["roc_auc"] for r in results]
        self.assertEqual(aucs, sorted(aucs, reverse=True))
        baseline = next(r for r in results if r["model"] == M.BASELINE)
        self.assertAlmostEqual(baseline["roc_auc"], 0.5, places=2)
        best = M.best_candidate(results)
        self.assertNotEqual(best["model"], M.BASELINE)
        self.assertGreater(best["roc_auc"], 0.75)

    def test_tuning_returns_known_parameters(self):
        _, params = M.tune("Logistic Regression", self.X, self.y, cv=3, iterations=3)
        self.assertIn("C", params)
        self.assertEqual(M.tune("Baseline (nobody gets the disease)", self.X, self.y)[1], {})
        pipe = M.pipeline_with_params("Logistic Regression", params).fit(self.X, self.y)
        self.assertEqual(pipe.named_steps["model"].C, params["C"])

    def test_calibration_choice_is_never_worse(self):
        pipe = M.build_pipeline(M.candidates(fast=True)["Random Forest"])
        model, oof, info = M.choose_calibration(pipe, self.X, self.y, cv=3)
        self.assertLessEqual(info["brier_after"], info["brier_before"] + 1e-4)
        self.assertEqual(len(oof), len(self.y))
        self.assertTrue(((oof >= 0) & (oof <= 1)).all())
        model.fit(self.X, self.y)
        self.assertEqual(model.predict_proba(self.X.head(4)).shape, (4, 2))


class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result, _, cls.raw = shared()
        cls.meta = cls.result.metadata

    def test_the_model_is_good_and_close_to_the_ceiling(self):
        test = self.meta["metrics"]["test"]
        self.assertGreater(test["roc_auc"], 0.78)
        self.assertGreater(test["roc_auc"], self.meta["metrics"]["baseline_cv"]["roc_auc"] + 0.2)
        ceiling = self.meta["ceiling"]["roc_auc"]
        self.assertLess(test["roc_auc"], ceiling + 0.04)                       # nothing can beat the truth by much
        self.assertGreater(test["roc_auc"], ceiling - 0.08)

    def test_test_set_is_untouched_and_stratified(self):
        rows = self.meta["rows"]
        self.assertEqual(rows["train"] + rows["test"], rows["clean"])
        self.assertAlmostEqual(rows["test"] / rows["clean"], 0.2, delta=0.01)
        frame = self.result.test_frame
        self.assertEqual(len(frame), rows["test"])
        train_rate = self.result.reference[S.TARGET].mean()
        self.assertAlmostEqual(frame["actual"].mean(), train_rate, delta=0.02)

    def test_screening_threshold_comes_from_training_data(self):
        sc = self.meta["screening"]
        self.assertGreaterEqual(sc["reached"], sc["target"] - 1e-9)
        self.assertEqual(self.meta["thresholds"]["screening"], sc["screening"])
        self.assertAlmostEqual(self.meta["thresholds"]["watch"], sc["screening"] / 2)
        test = self.meta["metrics"]["test"]
        self.assertGreater(test["sensitivity"], 70)                            # near the target on unseen people
        self.assertGreater(test["sensitivity"], test["default_threshold"]["sensitivity"])   # better than a 50% cut-off

    def test_bands_separate_people(self):
        low, moderate, high = self.meta["bands"]
        self.assertLess(low["rate"], moderate["rate"])
        self.assertLess(moderate["rate"], high["rate"])
        self.assertLess(low["rate"], 8)
        self.assertEqual(low["people"] + moderate["people"] + high["people"], self.meta["rows"]["test"])

    def test_importance_puts_age_first(self):
        top = self.meta["importance"][0]
        self.assertEqual(top["feature"], "age")
        self.assertGreater(top["importance"], 0.05)

    def test_metadata_is_complete_and_json_safe(self):
        for key in ("model", "calibration", "metrics", "screening", "bands", "segments", "roc", "comparison", "cleaning",
                    "training_ranges", "typical", "environment", "threshold_table", "calibration_table"):
            self.assertIn(key, self.meta)
        json.dumps(self.meta, default=persistence._json_default)
        self.assertEqual(set(self.meta["typical"]), set(S.RAW_FEATURES))

    def test_segments_cover_sexes_and_ages(self):
        self.assertEqual({s["group"] for s in self.meta["segments"]["sex"]}, {"Female", "Male"})
        self.assertEqual([s["group"] for s in self.meta["segments"]["age"]][0], "under 40")
        self.assertEqual(sum(s["n"] for s in self.meta["segments"]["age"]), self.meta["rows"]["test"])

    def test_same_seed_same_result(self):
        raw = datagen.generate(1200, seed=9)
        a = training.train(raw, fast=True, tune=False, cv=3, source="synthetic")
        b = training.train(raw, fast=True, tune=False, cv=3, source="synthetic")
        self.assertEqual(a.metadata["metrics"]["test"]["roc_auc"], b.metadata["metrics"]["test"]["roc_auc"])
        self.assertEqual(a.metadata["screening"]["screening"], b.metadata["screening"]["screening"])

    def test_bad_settings_and_data_are_refused(self):
        with self.assertRaisesRegex(DataError, "sensitivity"):
            training.train(self.raw, target_sensitivity=1.2)
        with self.assertRaises(DataError):
            training.train(self.raw.head(50))

    def test_progress_is_reported(self):
        steps = []
        training.train(datagen.generate(800, seed=4), fast=True, tune=False, cv=3, progress=steps.append)
        self.assertGreaterEqual(len(steps), 8)
        self.assertTrue(steps[0].startswith("Cleaning"))

    def test_a_higher_target_flags_more_people(self):
        raw = datagen.generate(1500, seed=6)
        low = training.train(raw, fast=True, tune=False, cv=3, target_sensitivity=0.6)
        high = training.train(raw, fast=True, tune=False, cv=3, target_sensitivity=0.95)
        self.assertLess(high.metadata["thresholds"]["screening"], low.metadata["thresholds"]["screening"])
        self.assertGreater(high.metadata["metrics"]["test"]["flagged"], low.metadata["metrics"]["test"]["flagged"])


class TestPersistence(TempDirTest):
    def test_round_trip_gives_identical_predictions(self):
        result, model_dir, raw = shared()
        loaded = persistence.load(model_dir)
        X = result.test_frame[S.RAW_FEATURES].head(30)
        np.testing.assert_allclose(result.pipeline.predict_proba(X), loaded.pipeline.predict_proba(X))
        self.assertTrue(persistence.is_trained(model_dir))
        self.assertEqual(loaded.metadata["model"], result.metadata["model"])
        self.assertEqual(len(loaded.reference), len(result.reference))

    def test_no_model_gives_instructions(self):
        with self.assertRaisesRegex(NotTrainedError, "Train one first"):
            persistence.load(self.dir / "empty")

    def test_tampering_is_detected(self):
        target = self.copy_model()
        with open(target / persistence.MODEL_FILE, "ab") as fh:
            fh.write(b"x")
        with self.assertRaisesRegex(ModelError, "checksum"):
            persistence.load(target)

    def test_missing_files_and_damaged_metadata(self):
        target = self.copy_model()
        (target / persistence.REFERENCE_FILE).unlink()
        with self.assertRaisesRegex(ModelError, "missing"):
            persistence.load(target)
        target = self.copy_model()
        (target / persistence.META_FILE).write_text("{not json", encoding="utf-8")
        with self.assertRaisesRegex(ModelError, "damaged"):
            persistence.load(target)
        target = self.copy_model()
        meta = json.loads((target / persistence.META_FILE).read_text(encoding="utf-8"))
        meta.pop("files")
        (target / persistence.META_FILE).write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaisesRegex(ModelError, "checksums"):
            persistence.load(target)

    def test_version_mismatch_is_a_warning(self):
        target = self.copy_model()
        meta = json.loads((target / persistence.META_FILE).read_text(encoding="utf-8"))
        meta["environment"]["sklearn"] = "0.0.1"
        (target / persistence.META_FILE).write_text(json.dumps(meta), encoding="utf-8")
        self.assertTrue(any("0.0.1" in w for w in persistence.load(target).warnings))


class TestReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = shared()[0].metadata

    def test_text_report_has_the_key_sections(self):
        text = report.model_text(self.meta)
        for part in ("HOW WELL IT RANKS", "SCREENING AT", "RISK BANDS", "WHAT THE MODEL RELIES ON", "WHAT THIS MEANS",
                     "not a medical device", "ROC AUC", "Sensitivity"):
            self.assertIn(part, text)
        self.assertIn("MODEL COMPARISON", report.comparison_text(self.meta))
        self.assertIn("chosen", report.comparison_text(self.meta))

    def test_html_report_is_self_contained_and_accessible(self):
        page = report.render_html(self.meta)
        self.assertTrue(page.startswith("<!doctype html>"))
        self.assertEqual(page.count("<svg"), page.count("</svg>"))
        self.assertGreaterEqual(page.count('role="img"'), 4)
        self.assertGreaterEqual(page.count('<details class="tv">'), 4)
        self.assertIn("prefers-color-scheme:dark", page)
        self.assertNotIn("http://", page.replace("http://www.w3.org", ""))      # nothing is fetched from outside
        self.assertIn("not a medical device", page)

    def test_html_report_escapes_text_from_the_data(self):
        meta = json.loads(json.dumps(self.meta, default=persistence._json_default))
        meta["source"] = "<script>alert(1)</script>"
        page = report.render_html(meta)
        self.assertNotIn("<script>alert(1)", page)
        self.assertIn("&lt;script&gt;alert(1)", page)

    def test_old_models_without_results_are_refused(self):
        with self.assertRaises(ModelError):
            report.render_html({"metrics": {}})

    def test_save_html(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = report.save_html(self.meta, Path(tmp) / "sub" / "r.html")
            self.assertTrue(path.exists() and path.stat().st_size > 5000)


if __name__ == "__main__":
    unittest.main()
