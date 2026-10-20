# Disease Risk Prediction Demo

A machine-learning demo that estimates a person's chance of developing a disease (cardiovascular disease within ten years) from a basic health check. It covers the full workflow: **data preprocessing → model training → evaluation → prediction interface**.

Developed by **Sneha S**, Skyrovix virtual internship (Intern ID SKX-2026-6425).

> **This is a learning demo trained on generated (synthetic) data. It is not a medical device and gives no diagnosis. For any health concern, talk to a qualified clinician.** The notice is repeated in every interface and report.

## What it does

- **Cleans messy health-check data** and logs every fix: blanks, impossible values (age 240, systolic pressure 12), glucose written in mmol/L instead of mg/dL, text such as "n/a", messy spellings (`F`, `light`), yes/no written many ways, HDL above total cholesterol, repeated people.
- **Compares models fairly** with cross-validation on training data: a baseline, logistic regression, decision tree, random forest and gradient boosting, all in the same pipeline (feature engineering → imputing/scaling/encoding → model). The best is lightly tuned, and its probabilities are calibrated only if that improves the Brier score.
- **Chooses a screening threshold for high sensitivity.** Missing a future case is worse than inviting a healthy person for a check-up, so the threshold is the highest one that still catches 85 % of cases on training data (out-of-fold). The test set is used once, at the end.
- **Evaluates honestly:** ROC AUC, PR AUC, Brier score, sensitivity, specificity, precision, confusion matrix, risk bands, calibration, a threshold table, checks by sex and age group, and (because the data is generated) the best possible score.
- **Explains every estimate:** which details push the risk up or down compared with a typical person, and "what if" scenarios (no smoking, blood pressure 120, …). These are patterns in the data, not causes or advice.
- **Interfaces:** interactive menu, command line, local web form, JSON API and batch CSV scoring.

## Quick start

```bash
pip install -r requirements.txt
python main.py --train                  # about 15 seconds
python main.py                          # interactive menu
python main.py --predict age=58 sex=Male bmi=29 systolic_bp=145 smoker=yes
python main.py --score sample_data/people_to_score.csv
python main.py --serve                  # web page at http://127.0.0.1:8040
python main.py --report reports/model_report.html
```

Only `age`, `sex`, `bmi` and `systolic_bp` are required; for anything else left out, typical values are assumed and the result says so. Blood pressure can be written `138/88`; glucose in mmol/L is converted.

Your own data: a CSV with the columns `age, bmi, systolic_bp, cholesterol, hdl, glucose, exercise_hours, smoker, family_history, bp_medication, sex, alcohol, disease` (plus an optional `person_id`) and at least 300 rows, then `python main.py --train --data my_people.csv`. `python main.py --generate sample_data/people.csv` writes a synthetic file; `sample_data/people_dirty.csv` is a ready-made messy one.

JSON API: `POST /api/predict` with `{"age": 58, "sex": "Male", "bmi": 29, "systolic_bp": 145}`; `GET /api/health`.

## Results (generated data, 2,986 people after cleaning, test set 598)

| Measure | Value |
|---|---|
| ROC AUC (best possible on this data: 0.839) | **0.840** |
| PR AUC (guessing: 0.231) | 0.625 |
| Brier score | 0.128 |
| Screening at 16.3 % risk: sensitivity | **83 %** (target 85 % on training data) |
| Specificity / precision | 65 % / 42 % |
| Disease rate in the Low / Moderate / High band | 1.6 % / 15.6 % / 41.8 % |

Logistic regression won the cross-validation (AUC 0.833 against 0.822 for random forest). The generated risk rule is close to logistic, so a simple model suits it; with real data a different model could win. Accuracy (69 %) is not the headline: always answering "no disease" would score 77 % while catching nobody.

**Honest limitations:** many flags are false alarms (precision 42 %), the price of catching most cases; and with one shared threshold sensitivity is lower for women (75 %) than men (88 %) because their average risk is lower. Whether to use separate thresholds is a clinical decision. Both are shown in the report.

## Project layout

```
main.py                 command line and entry point
riskpredict/
  schema.py             features, ranges, labels, risk bands
  datagen.py            synthetic population with a known risk rule + dirty data
  preprocessing.py      loading, cleaning, feature engineering, scikit-learn transformer
  models.py             candidates, cross-validated comparison, tuning, calibration
  evaluation.py         metrics, screening threshold, bands, importance, ceiling
  training.py           the workflow and metadata
  persistence.py        model saving/loading with SHA-256 checks
  validators.py         input parsing with per-field messages
  predictor.py          prediction, explanations, what-if, batch scoring
  report.py             text and HTML report (SVG charts, light/dark)
  web.py / cli.py       local web form + JSON API, interactive menu
tests/                  117 tests
docs/PROJECT_REPORT.md
```

## Testing

```bash
python -m unittest discover -s tests -t .
```

117 tests (data and cleaning 26, models/training/persistence/reports 34, validation and prediction 33, web over real HTTP, menu and command line 24), all passing. Requires Python 3.10+.
