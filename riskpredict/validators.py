"""Validation of person records (prediction input) and of command-line settings.

parse_features() turns raw text values (from the menu, a web form, a JSON request or a CSV row) into a clean,
consistent record - or raises ValidationError listing EVERY problem at once, with a message per field.
It uses the same schema as training-data cleaning, so both agree on what is valid.
"""

import difflib
import math
import re

import numpy as np

from . import schema as S
from .exceptions import ValidationError

_NUMBER = re.compile(r"^([-+]?\d+(?:\.\d+)?|[-+]?\.\d+)\s*([a-zA-Z/²%]*)$")
_BP = re.compile(r"^(\d{2,3})\s*/\s*(\d{2,3})(?:\s*mmhg)?$", re.I)
_MMOL = (2.5, 25.0)            # a glucose in this range is taken to be mmol/L (mg/dL values are at least 50)


def normalise_key(key):
    return str(key).strip().lower().replace(" ", "_").replace("-", "_")


def _is_blank(value):
    return value is None or (isinstance(value, float) and value != value) or (isinstance(value, str) and not value.strip())


def parse_number(feature, value):
    """'120/80' -> 120 (blood pressure) ; '5.4 mmol/L' glucose -> 97 ; '95 kg' is refused. Returns (number, note|None)."""
    kind, low, high, _ = S.NUMERIC[feature]
    label = S.LABELS[feature]
    note = None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number.")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().replace(",", "")
        bp = _BP.match(text) if feature == "systolic_bp" else None
        if bp:
            number = float(bp.group(1))
            note = f"Used {bp.group(1)} (the upper number) as the systolic blood pressure."
        else:
            match = _NUMBER.match(text)
            if not match:
                raise ValueError(f"{label} must be a number (you entered '{text[:30]}').")
            number, unit = float(match.group(1)), match.group(2).lower()
            allowed = {"", S.UNITS.get(feature, "").lower(), S.UNITS.get(feature, "").lower().replace("²", "2"),
                       "mmhg", "years", "yrs", "yr", "y", "h", "hrs", "hours", "hr"}
            if feature == "glucose" and unit in ("mmol/l", "mmol"):
                number, note = round(number * 18), f"Glucose {match.group(1)} mmol/L converted to {round(number * 18)} mg/dL."
            elif unit not in allowed:
                raise ValueError(f"{label} must be a number in {S.UNITS.get(feature, 'plain numbers')} "
                                 f"(you entered '{text[:30]}').")
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a normal number.")
    if feature == "glucose" and note is None and _MMOL[0] <= number <= _MMOL[1]:
        number, note = round(number * 18), f"Glucose {value} looks like mmol/L; converted to {round(number * 18)} mg/dL."
    if kind == "int":
        if number != int(number):
            raise ValueError(f"{label} must be a whole number (you entered {number:g}).")
        number = int(number)
    if not low <= number <= high:
        raise ValueError(f"{label} must be between {low} and {high} {S.UNITS.get(feature, '')}".rstrip() + f" (you entered {number:g}).")
    return number, note


def parse_category(feature, value):
    canon = S.canonical_category(feature, value)
    if canon is None:
        allowed = S.CATEGORICAL[feature]
        guess = difflib.get_close_matches(str(value).strip().title(), allowed, n=1, cutoff=0.6)
        hint = f" Did you mean '{guess[0]}'?" if guess else ""
        raise ValueError(f"{S.LABELS[feature]} '{str(value).strip()[:30]}' is not supported.{hint} "
                         f"Choose one of: {', '.join(allowed)}.")
    return canon


def parse_features(raw):
    """Validate one person. Returns (record, notes, assumed).

    record   every feature; optional ones left out are np.nan (the model's pipeline fills in typical values)
    notes    friendly remarks about conversions that were applied
    assumed  names of the optional features that were not given
    Raises ValidationError with .errors = {field: message} for every problem found.
    """
    if not isinstance(raw, dict):
        raise ValidationError("A person's details must be given as field/value pairs.")
    given = {}
    errors = {}
    for key, value in raw.items():
        name = normalise_key(key)
        if name not in S.RAW_FEATURES:
            guess = difflib.get_close_matches(name, S.RAW_FEATURES, n=1, cutoff=0.6)
            errors[name or "field"] = f"Unknown field '{str(key)[:30]}'." + (f" Did you mean '{guess[0]}'?" if guess else "")
            continue
        given[name] = value
    for name in S.REQUIRED_INPUT:
        if _is_blank(given.get(name)):
            errors[name] = f"{S.LABELS[name]} is required."
    record, notes = {}, []
    for name in S.RAW_FEATURES:
        value = given.get(name)
        if _is_blank(value):
            record[name] = np.nan
            continue
        try:
            if name in S.NUMERIC:
                record[name], note = parse_number(name, value)
                if note:
                    notes.append(note)
            elif name in S.BINARY:
                flag = S.parse_yes_no(value)
                if flag is None:
                    raise ValueError(f"{S.LABELS[name]} must be yes or no (you entered '{str(value).strip()[:20]}').")
                record[name] = flag
            else:
                record[name] = parse_category(name, value)
        except ValueError as exc:
            errors[name] = str(exc)
    if not errors and not _is_blank(given.get("hdl")) and not _is_blank(given.get("cholesterol")) \
            and record["hdl"] > record["cholesterol"]:
        errors["hdl"] = "HDL cholesterol cannot be higher than total cholesterol."
    if errors:
        raise ValidationError("; ".join(errors.values()), errors)
    assumed = [n for n in S.RAW_FEATURES if n not in S.REQUIRED_INPUT and _is_blank(record[n])]
    return record, notes, assumed


def validate_port(value):
    try:
        port = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValidationError("Port must be a whole number.") from None
    if not 1024 <= port <= 65535:
        raise ValidationError("Port must be between 1024 and 65535.")
    return port


def validate_int(value, field, low, high):
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValidationError(f"{field} must be a whole number.") from None
    if not low <= number <= high:
        raise ValidationError(f"{field} must be between {low} and {high}.")
    return number


def validate_number(value, field, low, high):
    try:
        number = float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        raise ValidationError(f"{field} must be a number.") from None
    if not math.isfinite(number) or not low <= number <= high:
        raise ValidationError(f"{field} must be between {low:g} and {high:g}.")
    return number
