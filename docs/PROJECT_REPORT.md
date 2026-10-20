# Project Report – Disease Risk Prediction Demo

**Intern:** Sneha S · **Intern ID:** SKX-2026-6425 · **Programme:** Skyrovix virtual internship · **Due:** 30 Nov 2026

> A learning demo on synthetic data. Not a medical device; it gives no diagnosis and no medical advice.

## 1. Objective
Build a working disease risk prediction demo using machine-learning concepts, with a documented workflow covering data preprocessing, model training, evaluation and a prediction interface, plus testing and a submission-ready project.

## 2. Workflow

1. **Data.** Real patient data is private, so a generator builds a population (age, sex, BMI, blood pressure, cholesterol, HDL, glucose, exercise, smoking, family history, blood-pressure medication, alcohol) with a *known* risk rule; whether each person develops the disease is a random draw from their true chance. The best possible score is therefore known, and data problems are injected on purpose.
2. **Preprocessing** (`clean`, before the split, row-by-row so nothing leaks from test data): labels standardised (rows with an unknown outcome removed), spellings fixed, unreadable text and out-of-range values blanked, glucose in mmol/L converted, HDL above total cholesterol blanked, repeats removed, rows without a sex removed. Sample run: 3,051 rows read → 2,986 kept, 65 removed, all logged. Learned steps (median imputing, scaling, one-hot encoding) sit inside the model pipeline and are fitted on training rows only. Engineered features: cholesterol ratio, high blood pressure, high glucose.
3. **Split.** 80 % train / 20 % test, stratified so both keep the same disease share (23 %).
4. **Model comparison.** Five candidates in the same pipeline, 5-fold cross-validation on training data, ranked by ROC AUC. The best is tuned (randomised search) and kept only if cross-validation improves; calibration is applied only if it lowers the Brier score.
5. **Screening threshold.** The highest threshold that still catches 85 % of cases among out-of-fold training predictions (16.3 % risk). Risk bands: Low below half of it, Moderate up to it, High from it.
6. **Evaluation, once, on the untouched test set.**
7. **Explanation.** For each person, one detail at a time is replaced by the typical value to see how the estimate moves; "what if" scenarios do the same with healthier values.
8. **Interfaces.** Menu, command line, local web form, JSON API, batch CSV; shared validation with a message per field.

## 3. Results (test set, 598 people)

| Model (cross-validated ROC AUC) | AUC | PR AUC | Brier |
|---|---|---|---|
| **Logistic regression (tuned)** | **0.833** | 0.625 | 0.129 |
| Logistic regression | 0.832 | 0.624 | 0.129 |
| Random forest | 0.822 | 0.597 | 0.133 |
| Gradient boosting | 0.820 | 0.588 | 0.135 |
| Decision tree | 0.788 | 0.519 | 0.148 |
| Baseline (nobody gets the disease) | 0.500 | 0.231 | 0.177 |

| Test-set measure | Value |
|---|---|
| ROC AUC | 0.840 (training 0.840, best possible 0.839: no overfitting) |
| Sensitivity at 16.3 % | 83 % (114 of 138 cases flagged; 24 missed) |
| Specificity | 65 % (159 false alarms among 460 healthy) |
| Precision / accuracy | 42 % / 69 % (always "no disease": 77 %) |
| Risk bands | Low 1.6 % · Moderate 15.6 % · High 41.8 % had the disease |
| Calibration | already honest, so calibration was tried but not applied |

**Findings**
- Age is by far the most important feature (AUC drop 0.222); smoking, blood pressure and sex follow.
- The three risk bands separate people clearly: the High band has a 26 times higher disease rate than the Low band.
- Sensitivity (83 %) falls slightly short of the 85 % training target; the report shows this rather than hiding it.
- Fairness check: one shared threshold gives 75 % sensitivity for women and 88 % for men, because women's average risk is lower. Separate thresholds would even this out with more false alarms; that is a clinical choice.

## 4. Prediction interface
- **Inputs:** four required details, eight optional; units and formats are understood (`138/88`, `5.4 mmol/L`, `54 years`); missing details are assumed typical and reported.
- **Outputs:** estimated risk, band and message, comparison with a typical person, top factors, what-if scenarios, warnings for values outside the training range, and the disclaimer.
- **Safety:** web server on localhost only with Host check, size limits, escaped output, no scripts, strict Content-Security-Policy; model files verified by SHA-256 before loading.

## 5. Testing
117 automated tests: data and cleaning (26), models/training/persistence/reports (34), validation and prediction (33), web over real HTTP, menu sessions and command line (24). They check, among other things, that cleaning removes each injected problem, that the screening threshold meets its sensitivity target, that the test set stays untouched, that risk rises with age/smoking/blood pressure, that tampered models are refused, and that script injection in form fields is escaped. Training with a fixed seed reproduces the same numbers.

## 6. Limitations and next steps
- Synthetic data: scores show how the methods behave, not how they would perform on real patients. Real use would need clinical data, ethics review, validation and regulatory approval.
- Precision is moderate by design (a screening rule trades false alarms for catching cases).
- Possible extensions: separate thresholds per group, time-to-event models, more risk factors, a comparison with an established clinical score.
