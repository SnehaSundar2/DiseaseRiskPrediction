"""The dataset schema: every feature, its type, valid range and allowed values.

Training-data cleaning and prediction-input validation both read from here, so the two can never disagree
about what a valid record looks like.

IMPORTANT: this is a teaching project on SYNTHETIC data. The disease modelled is "cardiovascular disease
within ten years" in a generated population. Nothing here is medical advice.
"""

TARGET = "disease"                # 1 = developed the disease, 0 = did not
ID = "person_id"                  # optional column in a dataset; never used as a feature

DISCLAIMER = ("This is a learning demo trained on generated (synthetic) data. It is not a medical device and "
              "gives no diagnosis. For any health concern, talk to a qualified clinician.")

SEXES = ["Female", "Male"]
ALCOHOL = ["None", "Moderate", "Heavy"]

# name -> (type, low, high, description). Numeric limits are hard limits for valid data.
NUMERIC = {
    "age": ("int", 18, 100, "Age in years"),
    "bmi": ("float", 12, 60, "Body mass index (kg/m²)"),
    "systolic_bp": ("int", 80, 240, "Systolic blood pressure in mmHg (the upper number)"),
    "cholesterol": ("int", 100, 400, "Total cholesterol in mg/dL"),
    "hdl": ("int", 15, 120, "HDL ('good') cholesterol in mg/dL"),
    "glucose": ("int", 50, 400, "Fasting blood glucose in mg/dL"),
    "exercise_hours": ("float", 0, 25, "Hours of exercise per week"),
}
BINARY = {
    "smoker": "Currently smokes (yes/no)",
    "family_history": "A parent or sibling had early heart disease (yes/no)",
    "bp_medication": "Takes blood-pressure medication (yes/no)",
}
CATEGORICAL = {"sex": SEXES, "alcohol": ALCOHOL}

NUMERIC_FEATURES = list(NUMERIC)
BINARY_FEATURES = list(BINARY)
CATEGORICAL_FEATURES = list(CATEGORICAL)
RAW_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = RAW_FEATURES + [TARGET]
REQUIRED_INPUT = ["age", "sex", "bmi", "systolic_bp"]       # the rest may be left out; typical values are then assumed

ENGINEERED = ["chol_ratio", "high_bp", "high_glucose"]      # built from other inputs, never from the label

MIN_ROWS = 300
MIN_PER_CLASS = 40
TARGET_SENSITIVITY = 0.85         # screening aim: catch at least this share of people who go on to have the disease

LABELS = {
    "age": "Age", "bmi": "BMI", "systolic_bp": "Systolic blood pressure", "cholesterol": "Total cholesterol",
    "hdl": "HDL cholesterol", "glucose": "Fasting glucose", "exercise_hours": "Exercise",
    "smoker": "Smoking", "family_history": "Family history", "bp_medication": "Blood-pressure medication",
    "sex": "Sex", "alcohol": "Alcohol use",
}
UNITS = {"age": "years", "bmi": "kg/m²", "systolic_bp": "mmHg", "cholesterol": "mg/dL", "hdl": "mg/dL",
         "glucose": "mg/dL", "exercise_hours": "hours/week"}

# Features a person can influence, and a healthier reference value used for the "what if" scenarios.
MODIFIABLE = {
    "smoker": (0, "if you did not smoke"),
    "exercise_hours": (5.0, "with 5 hours of exercise a week"),
    "bmi": (24.0, "at a BMI of 24"),
    "systolic_bp": (120, "with a blood pressure of 120"),
    "cholesterol": (180, "with total cholesterol of 180"),
    "glucose": (90, "with fasting glucose of 90"),
    "alcohol": ("Moderate", "with moderate alcohol use"),
}

ALIASES = {
    "sex": {"f": "Female", "woman": "Female", "female": "Female", "m": "Male", "man": "Male", "male": "Male"},
    "alcohol": {"no": "None", "never": "None", "nil": "None", "teetotal": "None", "light": "Moderate",
                "social": "Moderate", "moderate": "Moderate", "occasional": "Moderate", "high": "Heavy",
                "heavy": "Heavy", "daily": "Heavy"},
}

DISEASE_YES = {"yes", "y", "1", "true", "disease", "positive", "sick", "case"}
DISEASE_NO = {"no", "n", "0", "false", "healthy", "negative", "none", "control"}


def describe(feature):
    if feature in NUMERIC:
        return NUMERIC[feature][3]
    if feature in BINARY:
        return BINARY[feature]
    return {"sex": "Sex (Female or Male)", "alcohol": "Alcohol use (None, Moderate, Heavy)"}.get(feature, feature)


def risk_band(probability, screening_threshold):
    """'High' at or above the screening threshold, 'Moderate' from half of it, otherwise 'Low'."""
    if probability >= screening_threshold:
        return "High"
    if probability >= screening_threshold / 2:
        return "Moderate"
    return "Low"


def _key(text):
    return " ".join(str(text).replace("_", " ").replace("-", " ").split()).lower()


def _missing(value):
    return value is None or (isinstance(value, float) and value != value)


def canonical_category(feature, value):
    """'  FEMALE ' -> 'Female'. Returns None when the value is missing or not recognised."""
    if _missing(value):
        return None
    lookup = {_key(name): name for name in CATEGORICAL[feature]}
    lookup.update({_key(alias): name for alias, name in ALIASES.get(feature, {}).items()})
    return lookup.get(_key(value))


def parse_yes_no(value):
    """'Yes' / 'y' / true / 1 -> 1;  'no' / 'n' / false / 0 -> 0;  anything else -> None."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value == value:
        return {1: 1, 0: 0}.get(int(value)) if value in (0, 1) else None
    return {"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}.get(str(value).strip().lower())


def canonical_label(value):
    """'Yes ' / 1 / 'disease' -> 1;  'No' / 0 / 'healthy' -> 0;  anything else (or missing) -> None."""
    if _missing(value):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and value == int(value):
        value = int(value)
    key = _key(value)
    if key in DISEASE_YES:
        return 1
    if key in DISEASE_NO:
        return 0
    return None
