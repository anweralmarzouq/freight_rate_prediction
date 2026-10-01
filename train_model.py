from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight"]
CATEGORICAL_COLUMNS = ["pickup", "delivery", "equipment"]
NUMERIC_COLUMNS = ["distance", "weight", "month", "day_of_week", "day_of_month"]


def add_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    dates = pd.to_datetime(result["date"])
    result["month"] = dates.dt.month
    result["day_of_week"] = dates.dt.dayofweek
    result["day_of_month"] = dates.dt.day
    return result


def build_model() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
            ("numeric", "passthrough", NUMERIC_COLUMNS),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=0.06,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    train = pd.read_csv(DATA / "train-test.csv")
    validation = pd.read_csv(DATA / "validation.csv")
    template = pd.read_csv(DATA / "validation-predictions-template.csv")
    december = pd.read_csv(DATA / "december-chart-inputs.csv")

    train = add_date_features(train)
    validation = add_date_features(validation)
    december = add_date_features(december)

    train_dates = pd.to_datetime(train["date"])
    cutoff = train_dates.quantile(0.8) # type: ignore
    train_part = train[train_dates < cutoff]
    holdout_part = train[train_dates >= cutoff]

    validation_model = build_model()
    model_columns = MODEL_COLUMNS + NUMERIC_COLUMNS[2:]
    validation_model.fit(train_part[model_columns], train_part["posted_rate"])
    holdout_predictions = validation_model.predict(holdout_part[model_columns])
    print(
        f"Time-based holdout MAE: ${mean_absolute_error(holdout_part['posted_rate'], holdout_predictions):,.2f}"
    )
    print(
        f"Time-based holdout RMSE: ${np.sqrt(mean_squared_error(holdout_part['posted_rate'], holdout_predictions)):,.2f}"
    )

    model = build_model()
    model.fit(train[model_columns], train["posted_rate"])

    validation_predictions = model.predict(validation[model_columns])
    output = template.copy()
    output["predicted_rate"] = np.maximum(validation_predictions, 0.01)
    output.to_csv(ROOT / "validation_predictions.csv", index=False)

    december_predictions = model.predict(december[model_columns])
    december["predicted_rate"] = np.maximum(december_predictions, 0.01)
    december.drop(columns=["month", "day_of_week", "day_of_month"]).to_csv(
        DATA / "december-chart-inputs.csv", index=False
    )

    print("Created validation_predictions.csv")
    print("Filled data/december-chart-inputs.csv")


if __name__ == "__main__":
    main()