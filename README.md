# Abalone age prediction

This project predicts abalone age in years (`Rings + 1.5`) from physical measurements. It replaces the original one-off notebook workflow with leakage-safe pipelines, tuned model selection, SHAP explanations, and data/performance monitoring.

## Project layout

- `abalone.csv` - immutable source data (the scripts also support `data/raw/abalone.csv` when data is moved there).
- `notebooks/abalone_modeling_original.ipynb` - the original exploratory notebook, retained for reference.
- `src/abalone/` - reusable feature, modelling, evaluation, and monitoring code.
- `scripts/` - commands for EDA, training, explanation, scoring, and monitoring.
- `models/` - persisted selected model and its metadata/reference distribution.
- `reports/figures/` - EDA, residual, comparison, and SHAP figures.
- `reports/monitoring/` - monitoring outputs. The two report subfolders keep visuals and operational checks separate.

## Run the workflow

```powershell
python scripts/profile_data.py
python scripts/train_compare.py --iterations 4
python scripts/explain_model.py
python scripts/score_model.py --input abalone.csv --output submissions/abalone_predictions.csv
python scripts/monitor_data.py --input abalone.csv
```

`train_compare.py` uses a fixed hold-out test set and 3-fold cross-validated random search. It compares a dummy baseline with tuned extra-trees and gradient-boosting models, selecting by cross-validation RMSE and reporting the untouched hold-out result. Every feature transformation is fitted inside the pipeline, so validation data cannot influence training.

The scoring script produces age predictions and an approximate 95% prediction interval based on held-out residual variation. Monitoring evaluates missingness, invalid values, numeric PSI, categorical distribution shift, and-when `Rings` is supplied-realised regression accuracy.
# Machine-learning
