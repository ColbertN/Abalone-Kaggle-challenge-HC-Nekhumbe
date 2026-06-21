"""Regression metrics and model diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return metrics in units that are meaningful for predicted age."""
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return {
        "mae_years": float(mean_absolute_error(y_true, y_pred)),
        "rmse_years": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "bias_years": float(residuals.mean()),
        "residual_std_years": float(residuals.std(ddof=1)),
    }


def residual_diagnostics(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Create a row-level test-set diagnostic table."""
    output = pd.DataFrame(
        {
            "actual_age_years": np.asarray(y_true, dtype=float),
            "predicted_age_years": np.asarray(y_pred, dtype=float),
        },
        index=y_true.index,
    )
    output["residual_years"] = output["actual_age_years"] - output["predicted_age_years"]
    output["absolute_error_years"] = output["residual_years"].abs()
    return output
