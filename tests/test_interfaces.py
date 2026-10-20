"""The web interface (over real HTTP on a random local port), the interactive menu and main.py."""

import http.client
import io
import json
import threading
import urllib.parse
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import main as entry
from riskpredict import datagen, persistence, web
from riskpredict.cli import RiskCLI
from riskpredict.predictor import require_predictor
from tests.helpers import HIGH_RISK, LOW_RISK, MINIMAL, TempDirTest, shared


class TestWeb(TempDirTest):
    @classmethod
    def setUpClass(cls):
        cls.predictor = require_predictor(shared()[1])
        cls.server = web.make_server(cls.predictor, port=0)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read().decode("utf-8")
        conn.close()
        return resp.status, data, {k.lower(): v for k, v in resp.getheaders()}

    def post_form(self, **values):
        return self.request("POST", "/", urllib.parse.urlencode(values), {"Content-Type": "application/x-www-form-urlencoded"})

    def post_json(self, payload, path="/api/predict"):
        return self.request("POST", path, json.dumps(payload), {"Content-Type": "application/json"})

    def test_form_page_and_security_headers(self):
        status, body, headers = self.request("GET", "/")
        self.assertEqual(status, 200)
        for part in ("Disease Risk Prediction", "Estimate risk", "not a medical device", 'name="age"', 'name="systolic_bp"'):
            self.assertIn(part, body)
        self.assertNotIn("<script", body.lower())
        self.assertIn("default-src 'none'", headers["content-security-policy"])
        self.assertEqual(headers["x-content-type-options"], "nosniff")
        self.assertEqual(self.request("GET", "/nope")[0], 404)
        self.assertEqual(self.request("POST", "/elsewhere", b"")[0], 404)

    def test_form_prediction_for_a_high_risk_person(self):
        form = {k: ("yes" if v == "yes" else v) for k, v in HIGH_RISK.items()}
        status, body, _ = self.post_form(**{k: str(v) for k, v in form.items()})
        self.assertEqual(status, 200)
        for part in ("High risk", "estimated chance", "What if?", "What drives this estimate", "Smoking", "not a medical device"):
            self.assertIn(part, body)
        self.assertIn('value="64"', body)                                   # the form keeps what was typed

    def test_low_risk_and_minimal_forms(self):
        status, body, _ = self.post_form(**{k: str(v) for k, v in LOW_RISK.items()})
        self.assertEqual(status, 200)
        self.assertIn("Low risk", body)
        status, body, _ = self.post_form(**{k: str(v) for k, v in MINIMAL.items()}, cholesterol="", smoker="")
        self.assertEqual(status, 200)
        self.assertIn("typical values were assumed", body.lower())

    def test_mistakes_are_explained_next_to_the_field(self):
        status, body, _ = self.post_form(age="17", sex="Female", bmi="25", systolic_bp="120")
        self.assertEqual(status, 422)
        self.assertIn("between 18 and 100", body)
        self.assertIn('class="invalid"', body)
        status, body, _ = self.post_form(age="40", sex="Female", bmi="", systolic_bp="")
        self.assertEqual(status, 422)
        self.assertIn("BMI is required", body)
        self.assertIn("Systolic blood pressure is required", body)

    def test_text_from_the_user_is_escaped(self):
        evil = '"><script>alert(1)</script><img src=x onerror=alert(2)>'
        status, body, _ = self.post_form(age=evil, sex=evil, bmi="25", systolic_bp="120")
        self.assertEqual(status, 422)
        self.assertNotIn("<script>alert", body)
        self.assertNotIn("<img src=x", body)
        self.assertIn("&lt;script&gt;", body)

    def test_json_api(self):
        status, body, headers = self.post_json(HIGH_RISK)
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertIn("application/json", headers["content-type"])
        self.assertEqual(data["band"], "High")
        self.assertTrue(data["factors"] and data["what_if"])
        self.assertIn("not a medical device", data["disclaimer"])
        data = json.loads(self.post_json(MINIMAL)[1])
        self.assertTrue(data["assumed_typical_values_for"])
        health = json.loads(self.request("GET", "/api/health")[1])
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["required"], ["age", "sex", "bmi", "systolic_bp"])

    def test_json_errors_name_the_fields(self):
        status, body, _ = self.post_json({"age": 5, "sex": "x"})
        data = json.loads(body)
        self.assertEqual(status, 422)
        self.assertIn("age", data["fields"])
        self.assertIn("bmi", data["fields"])

    def test_bad_requests_are_refused(self):
        self.assertEqual(self.request("POST", "/api/predict", "{}", {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/predict", "{nope", {"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.request("POST", "/api/predict", "x" * 30_000, {"Content-Type": "application/json"})[0], 413)
        self.assertEqual(self.request("POST", "/", "x" * 30_000, {"Content-Type": "application/x-www-form-urlencoded"})[0], 413)
        self.assertEqual(self.request("GET", "/", headers={"Host": "evil.example.com"})[0], 403)
        self.assertEqual(self.request("GET", "/", headers={"Host": f"localhost:{self.port}"})[0], 200)
        status, body, _ = self.post_json(["not", "a", "dict"])
        self.assertEqual(status, 422)


def run_cli(cli, inputs):
    out = io.StringIO()
    with patch("builtins.input", side_effect=inputs), redirect_stdout(out):
        cli.run()
    return out.getvalue()


# answers for the 12 questions of ask_person(): the four required ones, then Enter to skip the rest
BASIC = ["58", "Male", "29", "145"] + [""] * 8
FULL = ["64", "male", "32", "154/90", "262", "34", "121", "0.5", "yes", "yes", "yes", "heavy"]


class TestCLI(TempDirTest):
    @classmethod
    def setUpClass(cls):
        cls.model_dir = shared()[1]

    def make(self, model_dir=None):
        return RiskCLI(model_dir or self.model_dir, self.dir / "reports")

    def test_estimate_a_person(self):
        out = run_cli(self.make(), ["1"] + FULL + ["0"])
        for part in ("Estimated risk", "High risk", "What if?", "What drives this estimate", "not a medical device"):
            self.assertIn(part, out)

    def test_skipping_optional_questions(self):
        out = run_cli(self.make(), ["1"] + BASIC + ["0"])
        self.assertIn("Typical values were assumed for", out)

    def test_invalid_answers_are_asked_again(self):
        out = run_cli(self.make(), ["1", "abc", "17", "58", "robot", "Male", "29", "145"] + [""] * 8 + ["0"])
        self.assertIn("must be a number", out)
        self.assertIn("between 18 and 100", out)
        self.assertIn("not supported", out)
        self.assertIn("Estimated risk", out)

    def test_three_bad_answers_cancel(self):
        out = run_cli(self.make(), ["1", "a", "b", "c", "0"])
        self.assertIn("Too many invalid answers", out)
        self.assertIn("Goodbye", out)

    def test_required_answer_cannot_be_skipped(self):
        out = run_cli(self.make(), ["1", "", "", "", "0"])
        self.assertIn("Age is required", out)

    def test_what_if_menu(self):
        out = run_cli(self.make(), ["2"] + BASIC + ["smoker", "no, yes", "0"])
        self.assertIn("different smoking", out)
        self.assertRegex(out, r"\n\s+0\s+\d+\.\d%")
        out = run_cli(self.make(), ["2"] + BASIC + ["1", "", "0"])                # number choice, default suggestions
        self.assertIn("different age", out)
        out = run_cli(self.make(), ["2"] + BASIC + ["colour", "1", "0"])
        self.assertIn("Unknown detail", out)

    def test_score_a_csv(self):
        source = self.dir / "people.csv"
        pd.DataFrame([{"age": 60, "sex": "Male", "bmi": 30, "systolic_bp": 150, "smoker": "yes"},
                      {"age": 22, "sex": "Female", "bmi": 22, "systolic_bp": 110},
                      {"age": 3, "sex": "Female", "bmi": 22, "systolic_bp": 110}]).to_csv(source, index=False)
        target = self.dir / "scored.csv"
        out = run_cli(self.make(), ["3", str(source), str(target), "0"])
        self.assertIn("Scored 2 person(s)", out)
        self.assertIn("1 row(s) could not be scored", out)
        self.assertEqual(len(pd.read_csv(target)), 3)
        out = run_cli(self.make(), ["3", str(self.dir / "missing.csv"), "", "0"])
        self.assertIn("not found", out)

    def test_reports_and_menu_errors(self):
        out = run_cli(self.make(), ["6", "7", "8", "", "99", "abc", "0"])
        self.assertIn("MODEL REPORT", out)
        self.assertIn("MODEL COMPARISON", out)
        self.assertIn("Report saved to", out)
        self.assertTrue((self.dir / "reports" / "model_report.html").exists())
        self.assertEqual(out.count("Invalid option"), 2)

    def test_dataset_tools(self):
        target = self.dir / "mine.csv"
        out = run_cli(self.make(), ["9", "400", "y", str(target), "10", str(target), "9", "400", "n", str(target), "0"])
        self.assertIn("Saved 400 people", out)
        self.assertIn("rows read", out)
        self.assertIn("already exists", out)
        self.assertTrue(target.exists())

    def test_no_model_offers_training(self):
        empty = self.make(self.dir / "nomodel")
        out = run_cli(empty, ["n", "1", "0"])
        self.assertIn("No trained model yet", out)
        self.assertIn("Train one first", out)

    def test_training_through_the_menu(self):
        folder = self.dir / "trainme"
        data = self.dir / "train.csv"
        datagen.generate(700, seed=8).to_csv(data, index=False)
        cli = self.make(folder)
        out = run_cli(cli, ["n", "5", str(data), "", "y", "0"])                 # default sensitivity, quick mode
        self.assertIn("MODEL REPORT", out)
        self.assertIn("Model saved to", out)
        self.assertTrue(persistence.is_trained(folder))


class TestMain(TempDirTest):
    def run_main(self, *args):
        out = io.StringIO()
        with redirect_stdout(out), patch("logging.basicConfig"):
            code = entry.main(list(args))
        return code, out.getvalue()

    def test_predict_from_the_command_line(self):
        model = str(shared()[1])
        code, out = self.run_main("--model-dir", model, "--predict", "age=60", "sex=Male", "bmi=30", "systolic_bp=150", "smoker=yes")
        self.assertEqual(code, 0)
        self.assertIn("Estimated risk", out)
        code, out = self.run_main("--model-dir", model, "--predict", "age=17")
        self.assertEqual(code, 1)
        self.assertIn("Error:", out)
        code, out = self.run_main("--model-dir", model, "--predict", "age")
        self.assertEqual(code, 1)
        self.assertIn("key=value", out)

    def test_score_and_report(self):
        model = str(shared()[1])
        source = self.dir / "p.csv"
        pd.DataFrame([{"age": 50, "sex": "Male", "bmi": 27, "systolic_bp": 130}]).to_csv(source, index=False)
        target = self.dir / "o.csv"
        code, out = self.run_main("--model-dir", model, "--score", str(source), "--output", str(target))
        self.assertEqual(code, 0)
        self.assertIn("Scored 1 person", out)
        html = self.dir / "r.html"
        code, _ = self.run_main("--model-dir", model, "--report", str(html))
        self.assertEqual(code, 0)
        self.assertTrue(html.exists())

    def test_generate_refuses_to_overwrite(self):
        target = self.dir / "gen.csv"
        code, out = self.run_main("--generate", str(target), "--rows", "400")
        self.assertEqual(code, 0)
        self.assertEqual(len(pd.read_csv(target)), len(datagen.generate(400)))
        code, out = self.run_main("--generate", str(target), "--rows", "400")
        self.assertEqual(code, 1)
        self.assertIn("already exists", out)
        code, out = self.run_main("--generate", str(self.dir / "x.csv"), "--rows", "5")
        self.assertEqual(code, 1)

    def test_train_from_the_command_line(self):
        data = self.dir / "t.csv"
        datagen.generate(700, seed=3).to_csv(data, index=False)
        model = self.dir / "m"
        code, out = self.run_main("--train", "--data", str(data), "--fast", "--model-dir", str(model), "--sensitivity", "0.8")
        self.assertEqual(code, 0)
        self.assertIn("MODEL REPORT", out)
        self.assertTrue(persistence.is_trained(model))
        code, out = self.run_main("--train", "--data", str(data), "--fast", "--model-dir", str(model), "--sensitivity", "2")
        self.assertEqual(code, 1)

    def test_prediction_without_a_model(self):
        code, out = self.run_main("--model-dir", str(self.dir / "none"), "--predict", "age=50", "sex=Male", "bmi=25", "systolic_bp=120")
        self.assertEqual(code, 1)
        self.assertIn("Train one first", out)
