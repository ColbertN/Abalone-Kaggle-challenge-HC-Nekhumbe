# Abalone data profile

- Records: 4,177
- Columns in source data: 9
- Prediction target: age in years, calculated as `Rings + 1.5`.
- Duplicate-row rate: 0.00%
- Unexpected sex values: 0

## Target and cohort

- Age range: 2.5 to 30.5 years.
- Median age: 10.5 years.
- Sex counts: M=1528, I=1342, F=1307

## Data-quality findings

- Missing values: Sex=0.00%, Length=0.00%, Diameter=0.00%, Height=0.00%, Whole weight=0.00%, Shucked weight=0.00%, Viscera weight=0.00%, Shell weight=0.00%
- Non-positive measurements: Length=0, Diameter=0, Height=2, Whole weight=0, Shucked weight=0, Viscera weight=0, Shell weight=0

## Strongest linear associations with age

- Shell weight: 0.628
- Diameter: 0.575
- Height: 0.557
- Length: 0.557
- Whole weight: 0.540

## Modelling implications

- Age is right-skewed, so MAE and RMSE in years are reported alongside R².
- Measurements are highly correlated, so the workflow uses tree ensembles and avoids target-informed feature filtering.
- No observations are dropped as outliers before the train/test split; extreme animals are part of the prediction problem.
