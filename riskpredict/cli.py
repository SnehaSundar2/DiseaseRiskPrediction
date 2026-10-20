"""Interactive menu-driven command-line interface.

All RiskError subclasses are caught here and shown as friendly messages; unexpected errors are
logged and the menu keeps running.
"""

import logging
from pathlib import Path

from . import datagen, persistence, report, schema as S, training
from . import preprocessing as P
from . import validators as v
from .exceptions import RiskError, ValidationError
from .predictor import require_predictor
from .web import serve

log = logging.getLogger("riskpredict")

MENU = """
============== DISEASE RISK PREDICTION DEMO ================
 PREDICT                            MODEL
  1. Estimate one person's risk      5. Train a new model
  2. What-if: change one detail      6. Model report (accuracy, screening)
  3. Score people from a CSV         7. Compare models
  4. Start the web interface         8. Save an HTML report
 DATA
  9. Create a sample dataset (CSV)  10. Check a dataset's quality
  0. Exit
  (a learning demo on synthetic data - not medical advice)
============================================================"""

# (feature, prompt, hint) in the order they are asked; the first four are required
QUESTIONS = [
    ("age", "Age in years", "18-100"),
    ("sex", "Sex", ", ".join(S.SEXES)),
    ("bmi", "BMI", "12-60, e.g. 24.5"),
    ("systolic_bp", "Systolic blood pressure (upper number)", "80-240, or 120/80"),
    ("cholesterol", "Total cholesterol", "mg/dL, Enter = skip"),
    ("hdl", "HDL cholesterol", "mg/dL, Enter = skip"),
    ("glucose", "Fasting glucose", "mg/dL or mmol/L, Enter = skip"),
    ("exercise_hours", "Exercise hours per week", "Enter = skip"),
    ("smoker", "Smoker? yes/no", "Enter = skip"),
    ("family_history", "Early heart disease in a parent or sibling? yes/no", "Enter = skip"),
    ("bp_medication", "Taking blood-pressure medication? yes/no", "Enter = skip"),
    ("alcohol", "Alcohol use", ", ".join(S.ALCOHOL) + ", Enter = skip"),
]
SYNTHETIC_ROWS = 3000
PLAUSIBLE = {"age": 45, "sex": "Female", "bmi": 25, "systolic_bp": 120}


def ask(prompt):
    return input(f"  {prompt}: ").strip()


def confirm(prompt, default=False):
    answer = ask(f"{prompt} [{'Y/n' if default else 'y/N'}]").lower()
    return default if not answer else answer in {"y", "yes"}


class RiskCLI:
    def __init__(self, model_dir="models", reports_dir="reports"):
        self.model_dir = Path(model_dir)
        self.reports_dir = Path(reports_dir)
        self._predictor = None
        self.actions = {"1": self.predict, "2": self.what_if, "3": self.predict_csv, "4": self.web,
                        "5": self.train, "6": self.model_report, "7": self.compare, "8": self.save_report,
                        "9": self.make_dataset, "10": self.check_quality}

    # ---------------------------------------------------------- plumbing
    def predictor(self):
        if self._predictor is None:
            self._predictor = require_predictor(self.model_dir)
            for warning in self._predictor.warnings:
                print(f"  ! {warning}")
        return self._predictor

    def run(self):
        print(f"\n  {S.DISCLAIMER}")
        if not persistence.is_trained(self.model_dir):
            print("\n  No trained model yet.")
            if confirm("Train one now on the built-in sample data (about half a minute)?", default=True):
                self._run_action(self.train_default)
        while True:
            print(MENU)
            choice = ask("Choose an option")
            if choice == "0":
                print("  Goodbye!")
                return
            action = self.actions.get(choice)
            if action is None:
                print("  ! Invalid option, please enter a number from the menu.")
                continue
            self._run_action(action)

    def _run_action(self, action):
        try:
            action()
        except RiskError as exc:
            log.warning("%s failed: %s", action.__name__, exc)
            print(f"  ! Error: {exc}")
        except (KeyboardInterrupt, EOFError):
            print("\n  Cancelled.")
        except Exception:  # safety net; the full traceback goes to the log file
            log.exception("Unexpected error in %s", action.__name__)
            print("  ! An unexpected error occurred. See riskpredict.log for details.")

    # ---------------------------------------------------------- prediction
    def ask_person(self):
        """Ask for each detail, re-asking (up to 3 times) when a value is invalid. Enter skips an optional one."""
        print("  Describe the person (Enter skips optional details; typical values are then assumed):")
        raw = {}
        context = dict(PLAUSIBLE)       # each answer is checked together with the earlier ones
        for name, prompt, hint in QUESTIONS:
            for attempt in range(3):
                value = ask(f"{prompt}  ({hint})")
                if value == "" and name not in S.REQUIRED_INPUT:
                    break
                try:
                    if value == "":
                        raise ValidationError(f"{S.LABELS[name]} is required.", {name: f"{S.LABELS[name]} is required."})
                    v.parse_features({**context, **raw, name: value})
                except ValidationError as exc:
                    print(f"    ! {exc.errors.get(name) or exc}")
                    if attempt == 2:
                        raise ValidationError("Too many invalid answers; estimate cancelled.") from None
                else:
                    raw[name] = context[name] = value
                    break
        return raw

    def show_prediction(self, p):
        print(f"\n  Estimated risk:  {p.percent:.1f}%   ({p.band} risk)   typical person: {100 * p.typical_probability:.1f}%")
        print(f"  {p.message}")
        for note in p.notes:
            print(f"  · {note}")
        for warning in p.warnings:
            print(f"  ! {warning}")
        if p.assumed:
            print("  Typical values were assumed for: " + ", ".join(S.LABELS[a].lower() for a in p.assumed))
        if p.what_ifs:
            print("\n  What if? (patterns in the data, not promises):")
            for w in p.what_ifs:
                print(f"   {w['change'] * 100:+4.0f} points  {w['scenario']}  ({p.percent:.0f}% → {w['probability'] * 100:.0f}%)")
        if p.factors:
            print("\n  What drives this estimate (compared with a typical person):")
            top = max(abs(f["effect"]) for f in p.factors) or 1
            for f in p.factors:
                sign = "+" if f["effect"] >= 0 else "-"
                print(f"   {f['label']:<26} {f['value']:<14} {sign}{abs(f['effect']) * 100:4.0f} pts  {report.bar(abs(f['effect']), top, 12)}")
        print(f"\n  {p.disclaimer}")

    def predict(self):
        predictor = self.predictor()
        self.show_prediction(predictor.predict(self.ask_person()))

    def what_if(self):
        predictor = self.predictor()
        raw = self.ask_person()
        for i, name in enumerate(S.RAW_FEATURES, 1):
            print(f"   {i:>2}. {S.LABELS[name]}")
        choice = ask("Which detail to vary? (number or name)")
        name = S.RAW_FEATURES[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(S.RAW_FEATURES) else choice
        key = v.normalise_key(name)
        suggestions = predictor.suggestions(key) if key in S.RAW_FEATURES else []
        typed = ask(f"Values to try, comma-separated (Enter = {', '.join(map(str, suggestions)) or '?'})")
        values = [x.strip() for x in typed.split(",") if x.strip()] or suggestions
        results = predictor.what_if(raw, name, values)
        print(f"\n  Estimated risk of the same person with a different {S.LABELS[key].lower()}:")
        for value, p in results:
            print(f"   {str(value):<22} {p * 100:5.1f}%  {report.bar(p, 1, 28)}")

    def predict_csv(self):
        predictor = self.predictor()
        print(f"  The CSV needs the columns {', '.join(S.REQUIRED_INPUT)}; other details are optional "
              f"({S.ID} is kept if present).")
        source = ask("CSV file with people")
        default = Path("predictions") / f"{Path(source).stem}_scored.csv"
        target = ask(f"Save scores to [{default}]") or default
        result = predictor.predict_csv(source, target)
        failed = result[result["status"] != "ok"]
        bands = result["risk_band"].value_counts()
        print(f"  Scored {len(result) - len(failed)} person(s); saved to {target}.")
        print(f"  High risk {bands.get('High', 0)} · moderate {bands.get('Moderate', 0)} · low {bands.get('Low', 0)}.")
        if len(failed):
            print(f"  {len(failed)} row(s) could not be scored:")
            for line in failed["status"].head(15):
                print(f"    - {line}")

    def web(self):
        port = v.validate_port(ask("Port [8040]") or 8040)
        serve(self.predictor(), port)

    # ---------------------------------------------------------- training and reports
    def train_default(self):
        self._train(datagen.generate(SYNTHETIC_ROWS), "synthetic (generated)", fast=False, tune=True)

    def train(self):
        source = ask("CSV file to train on (Enter = generate the built-in sample data)")
        if source:
            df, label = P.load_dataset(source), Path(source.strip('"')).name
        else:
            rows = v.validate_int(ask(f"People to generate [{SYNTHETIC_ROWS}]") or SYNTHETIC_ROWS, "Rows", 500, 30000)
            df, label = datagen.generate(rows), "synthetic (generated)"
        target = v.validate_number(ask(f"Target sensitivity, 0.5-0.99 [{S.TARGET_SENSITIVITY:g}]") or S.TARGET_SENSITIVITY,
                                   "Target sensitivity", 0.5, 0.99)
        fast = confirm("Quick mode (smaller models, faster, a little less accurate)?")
        tune = False if fast else confirm("Tune the best model? (adds a few seconds)", default=True)
        self._train(df, label, fast, tune, target_sensitivity=target)

    def _train(self, df, label, fast, tune, **settings):
        print()
        result = training.train(df, fast=fast, tune=tune, source=label, progress=lambda m: print(f"   {m}"), **settings)
        persistence.save(result, self.model_dir)
        self._predictor = None
        print("\n  Data cleaning:")
        print("\n".join("  " + line for line in result.cleaning.lines()))
        print()
        print(report.comparison_text(result.metadata))
        print()
        print(report.model_text(result.metadata))
        print(f"\n  Model saved to {self.model_dir}/ - you can now estimate risk.")

    def model_report(self):
        print(report.model_text(self.predictor().meta))

    def compare(self):
        print(report.comparison_text(self.predictor().meta))

    def save_report(self):
        default = self.reports_dir / "model_report.html"
        path = Path(ask(f"File name [{default}]") or default)
        if path.suffix.lower() not in (".html", ".htm"):
            path = path.with_suffix(".html")
        report.save_html(self.predictor().meta, path)
        print(f"  Report saved to {path} - open it in a web browser.")

    # ---------------------------------------------------------- data
    def make_dataset(self):
        rows = v.validate_int(ask(f"Number of people [{SYNTHETIC_ROWS}]") or SYNTHETIC_ROWS, "Rows", 300, 50000)
        dirty = not ask("Include data problems for practising cleaning? [Y/n]").lower().startswith("n")
        default = Path("sample_data") / ("people_dirty.csv" if dirty else "people_clean.csv")
        path = Path(ask(f"File name [{default}]") or default)
        if path.exists():
            raise ValidationError(f"{path} already exists; choose another name.")
        path.parent.mkdir(parents=True, exist_ok=True)
        datagen.generate(rows, dirty=dirty).to_csv(path, index=False)
        print(f"  Saved {rows} people to {path} (synthetic data).")

    def check_quality(self):
        df = P.load_dataset(ask("CSV file to check"))
        _, cleaning = P.clean(df)
        print("\n".join("  " + line for line in cleaning.lines()))
        if cleaning.missing_before:
            print("  Missing values before cleaning: " + ", ".join(f"{c} {n}" for c, n in cleaning.missing_before.items()))
            print("  (remaining gaps are filled during training using values learned from the training rows only)")
