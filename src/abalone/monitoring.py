"""Reference-profile creation and lightweight data/performance monitoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES


def _probabilities(values: pd.Series, bins: list[float]) -> list[float]:
    categories = pd.cut(values, bins=bins, include_lowest=True)
    counts = categories.value_counts(sort=False, dropna=False)
    return (counts / max(len(values), 1)).astype(float).tolist()


def build_reference_profile(features: pd.DataFrame) -> dict[str, Any]:
    """Capture training distributions for later drift comparisons."""
    profile: dict[str, Any] = {"numeric": {}, "categorical": {}}
    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(features[column], errors="coerce")
        finite = values.dropna()
        # Quantile bins are stable for skewed physical measurements.
        internal_edges = np.unique(finite.quantile(np.linspace(0.1, 0.9, 9)).to_numpy()).tolist()
        bins = [-float("inf"), *[float(edge) for edge in internal_edges], float("inf")]
        profile["numeric"][column] = {
            "bins": bins,
            "proportions": _probabilities(values, bins),
            "missing_rate": float(values.isna().mean()),
            "mean": float(finite.mean()),
            "std": float(finite.std(ddof=1)),
        }
    for column in CATEGORICAL_FEATURES:
        values = features[column].astype("string").fillna("__MISSING__")
        proportions = values.value_counts(normalize=True, dropna=False)
        profile["categorical"][column] = {
            "proportions": {str(key): float(value) for key, value in proportions.items()},
            "missing_rate": float(features[column].isna().mean()),
        }
    return profile


def population_stability_index(expected: list[float], actual: list[float]) -> float:
    """Calculate PSI; values above 0.2 are commonly treated as material drift."""
    expected_array = np.clip(np.asarray(expected, dtype=float), 1e-6, None)
    actual_array = np.clip(np.asarray(actual, dtype=float), 1e-6, None)
    return float(np.sum((actual_array - expected_array) * np.log(actual_array / expected_array)))


def assess_drift(features: pd.DataFrame, reference: dict[str, Any]) -> dict[str, Any]:
    """Compare a scoring batch to the training reference profile."""
    numeric: dict[str, Any] = {}
    categorical: dict[str, Any] = {}
    for column in NUMERIC_FEATURES:
        saved = reference["numeric"][column]
        values = pd.to_numeric(features[column], errors="coerce")
        actual = _probabilities(values, saved["bins"])
        psi = population_stability_index(saved["proportions"], actual)
        numeric[column] = {
            "psi": psi,
            "missing_rate": float(values.isna().mean()),
            "mean_shift_standard_deviations": float(
                (values.mean() - saved["mean"]) / max(saved["std"], 1e-6)
            ),
            "status": "alert" if psi >= 0.2 else "watch" if psi >= 0.1 else "ok",
        }

    for column in CATEGORICAL_FEATURES:
        expected = reference["categorical"][column]["proportions"]
        actual_values = features[column].astype("string").fillna("__MISSING__")
        actual = actual_values.value_counts(normalize=True, dropna=False).to_dict()
        categories = sorted(set(expected) | {str(value) for value in actual})
        total_variation = 0.5 * sum(abs(actual.get(key, 0.0) - expected.get(key, 0.0)) for key in categories)
        categorical[column] = {
            "total_variation_distance": float(total_variation),
            "unseen_categories": sorted(set(actual) - set(expected)),
            "missing_rate": float(features[column].isna().mean()),
            "status": "alert" if total_variation >= 0.2 else "watch" if total_variation >= 0.1 else "ok",
        }

    statuses = [item["status"] for item in numeric.values()] + [item["status"] for item in categorical.values()]
    overall = "alert" if "alert" in statuses else "watch" if "watch" in statuses else "ok"
    return {"overall_status": overall, "numeric": numeric, "categorical": categorical}
