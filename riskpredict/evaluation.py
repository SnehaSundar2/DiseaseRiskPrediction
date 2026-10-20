"""Model evaluation for disease risk: ranking quality, honest probabilities and screening performance.

A screening tool is judged differently from a plain classifier. Missing someone who goes on to have the disease
(a false negative) is worse than inviting a healthy person for a check-up (a false positive), so the screening
threshold is chosen to reach a target SENSITIVITY (share of future cases caught) while flagging as few healthy
people as possible. All thresholds are chosen on training data only (out-of-fold predictions).
"""

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import schema as S


def classification_metrics(actual, probability, threshold):
    """Counts and percentages when everyone scoring at or above `threshold` is flagged."""
    actual = np.asarray(actual, int)
    flagged = np.asarray(probability, float) >= threshold
    sick = actual == 1
    tp, fp = int((flagged & sick).sum()), int((flagged & ~sick).sum())
    fn, tn = int((~flagged & sick).sum()), int((~flagged & ~sick).sum())
    sensitivity = 100 * tp / (tp + fn) if tp + fn else 0.0          # also called recall
    specificity = 100 * tn / (tn + fp) if tn + fp else 0.0
    precision = 100 * tp / (tp + fp) if tp + fp else 0.0             # also called positive predictive value
    npv = 100 * tn / (tn + fn) if tn + fn else 0.0
    return {
        "n": int(len(actual)), "threshold": float(threshold), "flagged": tp + fp,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": 100 * (tp + tn) / len(actual),
        "sensitivity": sensitivity, "specificity": specificity, "precision": precision, "npv": npv,
        "f1": 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0,
    }


def ranking_metrics(actual, probability):
    """Threshold-free scores: ROC AUC, average precision (PR AUC) and the Brier score (lower = more honest)."""
    actual, probability = np.asarray(actual, int), np.asarray(probability, float)
    if len(set(actual.tolist())) < 2:
        return {"roc_auc": 0.0, "average_precision": 0.0, "brier": float(brier_score_loss(actual, probability))}
    return {"roc_auc": float(roc_auc_score(actual, probability)),
            "average_precision": float(average_precision_score(actual, probability)),
            "brier": float(brier_score_loss(actual, probability))}


def choose_screening_threshold(actual, probability, target=S.TARGET_SENSITIVITY):
    """The highest threshold that still catches at least `target` of the people who have the disease.

    A higher threshold flags fewer healthy people, so the highest one meeting the sensitivity target is the one
    with the fewest false alarms. Chosen on out-of-fold training predictions, never on the test set.
    """
    actual, probability = np.asarray(actual, int), np.asarray(probability, float)
    sick = np.sort(probability[actual == 1])[::-1]
    if len(sick) == 0:
        return {"screening": 0.5, "target": float(target), "reached": 0.0}
    needed = max(int(np.ceil(target * len(sick))), 1)
    threshold = float(sick[needed - 1])
    reached = float((probability[actual == 1] >= threshold).mean())
    return {"screening": threshold, "target": float(target), "reached": reached,
            "training_specificity": classification_metrics(actual, probability, threshold)["specificity"]}


def threshold_table(actual, probability, thresholds=(0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)):
    """Sensitivity, specificity and precision at a range of thresholds (how the choice matters)."""
    rows = []
    for t in thresholds:
        m = classification_metrics(actual, probability, t)
        rows.append({"threshold": t, **{k: m[k] for k in ("sensitivity", "specificity", "precision", "flagged")}})
    return rows


def band_table(actual, probability, screening):
    """Low / Moderate / High: how many people fall in each band and how many of them really had the disease."""
    actual, probability = np.asarray(actual, int), np.asarray(probability, float)
    bands = np.array([S.risk_band(p, screening) for p in probability])
    overall = actual.mean() or 1.0
    rows = []
    for name in ("Low", "Moderate", "High"):
        mask = bands == name
        n = int(mask.sum())
        rate = float(actual[mask].mean()) if n else 0.0
        rows.append({"band": name, "people": n, "share": 100 * n / len(actual), "with_disease": int(actual[mask].sum()),
                     "rate": 100 * rate, "relative": float(rate / overall) if n else 0.0})
    return rows


def calibration_table(actual, probability, bins=10):
    """Predicted chance against what really happened, per probability band. Empty bands are skipped."""
    actual, probability = np.asarray(actual, int), np.asarray(probability, float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = which == b
        if mask.any():
            rows.append({"band": f"{edges[b] * 100:.0f}-{edges[b + 1] * 100:.0f}%", "people": int(mask.sum()),
                         "predicted": 100 * float(probability[mask].mean()), "observed": 100 * float(actual[mask].mean())})
    return rows


def roc_points(actual, probability, steps=40):
    """[(false-positive rate, true-positive rate)] for the ROC chart, thinned to about `steps` points."""
    actual, probability = np.asarray(actual, int), np.asarray(probability, float)
    order = np.argsort(-probability, kind="stable")
    sick, healthy = max(actual.sum(), 1), max((1 - actual).sum(), 1)
    tpr = np.concatenate([[0], np.cumsum(actual[order]) / sick])
    fpr = np.concatenate([[0], np.cumsum(1 - actual[order]) / healthy])
    keep = np.unique(np.linspace(0, len(tpr) - 1, steps + 1).round().astype(int))
    return [(round(float(fpr[i]), 4), round(float(tpr[i]), 4)) for i in keep]


def segment_rows(frame, by):
    """Per group: people, real disease rate, mean predicted rate, the gap, and sensitivity at the screening threshold."""
    rows = []
    for key, group in frame.groupby(by, observed=True):
        sick = group["actual"] == 1
        flagged = group["predicted"] == 1
        rows.append({"group": str(key), "n": int(len(group)), "actual": float(100 * group["actual"].mean()),
                     "predicted": float(100 * group["probability"].mean()),
                     "sensitivity": float(100 * (flagged & sick).sum() / sick.sum()) if sick.sum() else None})
    return rows


def age_group(age):
    return "under 40" if age < 40 else "40-54" if age < 55 else "55-69" if age < 70 else "70 or older"


def feature_importance(pipeline, X_test, y_test, seed=42, repeats=8):
    """Permutation importance: how much ROC AUC drops when one feature's values are shuffled (test set)."""
    result = permutation_importance(pipeline, X_test, y_test, scoring="roc_auc", n_repeats=repeats, random_state=seed,
                                    n_jobs=1)
    rows = [{"feature": f, "label": S.LABELS[f], "importance": max(0.0, float(m)), "std": float(s)}
            for f, m, s in zip(X_test.columns, result.importances_mean, result.importances_std)]
    return sorted(rows, key=lambda r: r["importance"], reverse=True)


def true_risk_ceiling(df_clean):
    """For SYNTHETIC data only: the scores of the TRUE disease probabilities against what actually happened.

    No model can systematically beat this: the rest is pure luck in who actually develops the disease.
    Missing values are filled with the median / most common value, as the model's own pipeline does.
    """
    from .datagen import disease_probability
    d = df_clean
    p = disease_probability(
        d["age"].fillna(d["age"].median()), d["sex"], d["bmi"].fillna(d["bmi"].median()),
        d["systolic_bp"].fillna(d["systolic_bp"].median()), d["cholesterol"].fillna(d["cholesterol"].median()),
        d["hdl"].fillna(d["hdl"].median()), d["glucose"].fillna(d["glucose"].median()),
        d["exercise_hours"].fillna(d["exercise_hours"].median()), d["smoker"].fillna(0), d["family_history"].fillna(0),
        d["bp_medication"].fillna(0), d["alcohol"].fillna("Moderate"))
    return ranking_metrics(d[S.TARGET], p)
