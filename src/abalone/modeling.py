"""Model selection, hyperparameter tuning, and model artifact creation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from .config import (
    FIGURES_DIR,
    METADATA_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    REFERENCE_PROFILE_PATH,
    REPORTS_DIR,
    ensure_output_directories,
)
from .evaluation import regression_metrics, residual_diagnostics
from .features import AGE_TARGET, build_preprocessor, load_training_data, make_features
from .monitoring import build_reference_profile


def _pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


def _search_definitions() -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    """Define high-value parameter spaces for complementary tree ensembles."""
    return {
        "extra_trees": (
            _pipeline(ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1)),
            {
                "model__n_estimators": [100, 150, 200],
                "model__max_depth": [None, 12, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": [0.7, 1.0, "sqrt"],
            },
        ),
        "gradient_boosting": (
            _pipeline(GradientBoostingRegressor(random_state=RANDOM_STATE)),
            {
                "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
                "model__n_estimators": [100, 150, 200],
                "model__max_depth": [2, 3, 4],
                "model__min_samples_leaf": [5, 10, 20],
                "model__subsample": [0.7, 0.85, 1.0],
            },
        ),
    }


def _save_search_trials(name: str, search: RandomizedSearchCV) -> None:
    columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_fit_time",
        "params",
    ]
    trials = pd.DataFrame(search.cv_results_).loc[:, columns].copy()
    trials.rename(
        columns={
            "mean_test_score": "mean_cv_neg_rmse",
            "std_test_score": "std_cv_neg_rmse",
        },
        inplace=True,
    )
    trials["mean_cv_rmse_years"] = -trials["mean_cv_neg_rmse"]
    trials.sort_values("rank_test_score").to_csv(REPORTS_DIR / f"hyperopt_trials_{name}.csv", index=False)


def _plot_model_comparison(results: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    chart = results.sort_values("test_rmse_years")
    sns.barplot(data=chart, x="test_rmse_years", y="model", hue="model", legend=False, ax=axis, palette="viridis")
    axis.set(title="Hold-out RMSE by model", xlabel="RMSE (years)", ylabel="")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison_rmse.png", dpi=160)
    plt.close(fig)


def _plot_residuals(diagnostics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.scatterplot(
        data=diagnostics,
        x="predicted_age_years",
        y="residual_years",
        alpha=0.55,
        s=24,
        ax=axes[0],
    )
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(title="Residuals vs predicted age", xlabel="Predicted age (years)", ylabel="Residual (years)")
    sns.histplot(diagnostics["residual_years"], bins=30, kde=True, ax=axes[1], color="#4c78a8")
    axes[1].set(title="Hold-out residual distribution", xlabel="Residual (years)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "best_model_residuals.png", dpi=160)
    plt.close(fig)


def train_and_compare(data_path: str | Path, n_iter: int = 4) -> dict[str, Any]:
    """Tune three non-linear models and select the lowest hold-out RMSE model."""
    ensure_output_directories()
    data = load_training_data(data_path)
    features = make_features(data)
    target = data[AGE_TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE
    )
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    results: list[dict[str, Any]] = []

    dummy = _pipeline(DummyRegressor(strategy="mean"))
    dummy_scores = -cross_val_score(dummy, x_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
    dummy.fit(x_train, y_train)
    dummy_predictions = dummy.predict(x_test)
    results.append(
        {
            "model": "dummy_mean",
            "cv_rmse_years": float(dummy_scores.mean()),
            "cv_rmse_std_years": float(dummy_scores.std(ddof=1)),
            **{f"test_{key}": value for key, value in regression_metrics(y_test, dummy_predictions).items()},
            "best_parameters": "{}",
        }
    )

    trained_models: dict[str, Pipeline] = {}
    for name, (pipeline, parameter_space) in _search_definitions().items():
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=parameter_space,
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            random_state=RANDOM_STATE,
            # This avoids joblib worker-pipe errors on locked-down Windows hosts.
            n_jobs=1,
            refit=True,
            return_train_score=False,
        )
        search.fit(x_train, y_train)
        _save_search_trials(name, search)
        predictions = search.predict(x_test)
        trained_models[name] = search.best_estimator_
        results.append(
            {
                "model": name,
                "cv_rmse_years": float(-search.best_score_),
                "cv_rmse_std_years": float(search.cv_results_["std_test_score"][search.best_index_]),
                **{f"test_{key}": value for key, value in regression_metrics(y_test, predictions).items()},
                "best_parameters": json.dumps(search.best_params_, sort_keys=True),
            }
        )

    # Select using cross-validation only. The untouched hold-out metrics remain
    # an honest final estimate of the chosen model's generalisation error.
    comparison = pd.DataFrame(results).sort_values("cv_rmse_years").reset_index(drop=True)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    _plot_model_comparison(comparison)

    winner_name = str(comparison.loc[0, "model"])
    winner = trained_models[winner_name]
    winner_predictions = winner.predict(x_test)
    diagnostics = residual_diagnostics(y_test, winner_predictions)
    diagnostics.to_csv(REPORTS_DIR / "best_model_test_predictions.csv", index_label="source_row")
    _plot_residuals(diagnostics)

    winner_metrics = regression_metrics(y_test, winner_predictions)
    metadata = {
        "model_name": winner_name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target": "Age in years (Rings + 1.5)",
        "data_path": str(Path(data_path)),
        "train_records": int(len(x_train)),
        "test_records": int(len(x_test)),
        "test_metrics": winner_metrics,
        "approximate_95pct_interval_half_width_years": float(1.96 * winner_metrics["residual_std_years"]),
    }
    joblib.dump(winner, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    REFERENCE_PROFILE_PATH.write_text(json.dumps(build_reference_profile(x_train), indent=2), encoding="utf-8")

    summary = {
        "best_model": winner_name,
        "best_test_metrics": winner_metrics,
        "search_iterations_per_model": n_iter,
        "artifacts": {
            "model": str(MODEL_PATH),
            "comparison": str(REPORTS_DIR / "model_comparison.csv"),
            "reference_profile": str(REFERENCE_PROFILE_PATH),
        },
    }
    (REPORTS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
