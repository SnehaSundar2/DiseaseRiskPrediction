"""Synthetic health-check data for the project.

Real patient records are private, so the project ships a generator that builds a population with a KNOWN
risk rule. Each person's true chance of developing the disease is computed from their age, blood pressure,
cholesterol, lifestyle and family history; whether they actually develop it is then a random draw from that
chance. Two things follow:

* no model can ever be perfect (even the true rule only ranks people so well: the "ceiling"), and
* the evaluation can show how close a model gets to that ceiling.

The data is clearly SYNTHETIC: it teaches the workflow and must never be used to judge a real person.
The generated file is also deliberately DIRTY (blanks, impossible values, unit mix-ups, messy spellings,
repeated people) so the preprocessing stage has real work to do.
"""

import numpy as np
import pandas as pd

from . import schema as S

INTERCEPT = -4.8                        # tuned so that roughly a quarter of people develop the disease
ALCOHOL_EFFECT = {"None": 0.0, "Moderate": -0.05, "Heavy": 0.45}


def disease_probability(age, sex, bmi, systolic_bp, cholesterol, hdl, glucose, exercise_hours, smoker,
                        family_history, bp_medication, alcohol):
    """The 'ground truth' rule: the chance (0-1) that each person develops the disease. Vectorised."""
    age, bmi, sbp = (np.asarray(x, float) for x in (age, bmi, systolic_bp))
    chol, hdl, glucose = (np.asarray(x, float) for x in (cholesterol, hdl, glucose))
    exercise, smoker, family, med = (np.asarray(x, float) for x in (exercise_hours, smoker, family_history, bp_medication))
    male = np.asarray(sex) == "Male"
    z = (INTERCEPT
         + 0.075 * (age - 20)                                # risk rises steadily with age
         + 0.55 * male
         + 0.045 * np.maximum(bmi - 22, 0)
         + 0.020 * (sbp - 110) + 0.30 * med                  # treated blood pressure still signals risk
         + 0.0045 * (chol - 150) - 0.018 * (hdl - 50)
         + 0.012 * np.maximum(glucose - 85, 0)
         - 0.075 * np.minimum(exercise, 10)
         + 0.85 * smoker + 0.55 * family
         + np.array([ALCOHOL_EFFECT[a] for a in alcohol]))
    return 1 / (1 + np.exp(-z))


def generate(rows=3000, seed=2026, dirty=True):
    """Return a DataFrame with person_id, the schema's feature columns and disease ('Yes'/'No')."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(48, 15, rows), 18, 90).astype(int)
    sex = rng.choice(S.SEXES, rows)
    male = sex == "Male"
    bmi = np.clip(rng.normal(25.5 + 0.04 * (age - 48) + 0.8 * male, 4.5, rows), 15, 52).round(1)
    smoker = (rng.random(rows) < np.where(male, 0.27, 0.17)).astype(int)
    family = (rng.random(rows) < 0.18).astype(int)
    exercise = np.clip(rng.gamma(2.0, 1.6, rows) - 0.03 * (bmi - 25), 0, 20).round(1)
    sbp = np.clip(rng.normal(108 + 0.45 * (age - 20) + 0.7 * (bmi - 25), 12, rows), 85, 220).astype(int)
    med = ((sbp > 140) & (rng.random(rows) < 0.55)).astype(int)
    chol = np.clip(rng.normal(168 + 0.5 * (age - 20) + 1.2 * (bmi - 25), 30, rows), 105, 380).astype(int)
    hdl = np.clip(rng.normal(55 - 5 * male - 0.5 * (bmi - 25) + 1.0 * exercise, 11, rows), 20, 110).astype(int)
    glucose = np.clip(rng.normal(86 + 0.25 * (age - 20) + 0.9 * (bmi - 25), 11, rows), 60, 300).astype(int)
    alcohol = rng.choice(S.ALCOHOL, rows, p=[0.30, 0.55, 0.15])
    p = disease_probability(age, sex, bmi, sbp, chol, hdl, glucose, exercise, smoker, family, med, alcohol)
    ill = rng.random(rows) < p
    df = pd.DataFrame({
        S.ID: [f"P{10000 + i}" for i in range(rows)], "age": age, "bmi": bmi, "systolic_bp": sbp,
        "cholesterol": chol, "hdl": hdl, "glucose": glucose, "exercise_hours": exercise, "smoker": smoker,
        "family_history": family, "bp_medication": med, "sex": sex, "alcohol": alcohol,
        S.TARGET: np.where(ill, "Yes", "No")})
    return make_dirty(df, rng) if dirty else df


def make_dirty(df, rng):
    """Inject realistic data-quality problems (about 10% of rows are affected)."""
    df = df.copy().astype(object)
    n = len(df)

    def rows(count):
        return rng.choice(n, count, replace=False)

    for col, rate in (("bmi", 0.03), ("systolic_bp", 0.02), ("cholesterol", 0.04), ("hdl", 0.05), ("glucose", 0.05),
                      ("exercise_hours", 0.04), ("smoker", 0.02), ("family_history", 0.04), ("bp_medication", 0.02),
                      ("alcohol", 0.03)):
        df.loc[rng.random(n) < rate, col] = np.nan                                       # missing values
    for idx in rows(4):
        df.loc[idx, "age"] = int(rng.choice([-5, 0, 150, 240]))                          # impossible age
    for idx in rows(4):
        df.loc[idx, "bmi"] = float(rng.choice([0, 3, 140]))
    for idx in rows(5):
        df.loc[idx, "systolic_bp"] = int(rng.choice([0, 12, 900]))
    for idx in rows(4):
        df.loc[idx, "cholesterol"] = int(rng.choice([-1, 9000]))
    for idx in rows(6):                                                                  # glucose in mmol/L instead of mg/dL
        df.loc[idx, "glucose"] = round(float(rng.uniform(4.0, 9.0)), 1)
    for idx in rows(4):
        df.loc[idx, "exercise_hours"] = 400
    for idx in rows(8):                                                                  # unreadable text
        df.loc[idx, str(rng.choice(["bmi", "cholesterol", "glucose"]))] = str(rng.choice(["n/a", "high", "??", "120/80"]))
    for idx in rows(5):                                                                  # HDL above total cholesterol: impossible
        df.loc[idx, ["cholesterol", "hdl"]] = [105, 115]
    spellings = {"sex": ["F", "female ", "MALE", "m", "Woman"], "alcohol": ["no", "light", "heavy ", "Daily", "social"]}
    for col, options in spellings.items():
        for idx in np.flatnonzero(rng.random(n) < 0.05):
            df.loc[idx, col] = str(rng.choice(options))
    for idx in rows(5):
        df.loc[idx, "sex"] = np.nan                                                      # unusable: sex is essential
    for col in ("smoker", "family_history", "bp_medication"):                            # yes/no written in many ways
        for idx in np.flatnonzero((rng.random(n) < 0.20) & df[col].notna().to_numpy()):
            value = df.loc[idx, col]
            if value in (0, 1):
                df.loc[idx, col] = str(rng.choice(["Yes", "Y", "true", "1"] if value == 1 else ["No", "N", "false", "0"]))
    for idx in np.flatnonzero(rng.random(n) < 0.15):                                     # label spellings
        df.loc[idx, S.TARGET] = str(rng.choice(["yes ", "1", "True", "disease"] if df.loc[idx, S.TARGET] == "Yes"
                                               else ["no ", "0", "healthy", "False"]))
    for idx in rows(5):
        df.loc[idx, S.TARGET] = np.nan
    for idx in rows(4):
        df.loc[idx, S.TARGET] = "unsure"
    dupes = df.sample(int(n * 0.015), random_state=1)                                    # exact repeats
    reused = df.sample(6, random_state=2).copy()                                         # same id, different details
    reused["age"] = reused["age"].map(lambda a: a if pd.isna(a) or a in (-5, 0, 150, 240) else min(int(a) + 4, 100))
    return pd.concat([df, dupes, reused], ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)
