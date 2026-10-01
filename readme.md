# Freight Rate Prediction Challenge

This project predicts freight rates for the assessment validation loads and for a fixed
December 2025 route scenario.

## Project files

- `train_model.py`: trains the model, prints holdout metrics, and creates the prediction files.
- `score.py`: validates the required output format and creates the December chart.
- `data/train-test.csv`: labeled training data with the `posted_rate` target.
- `data/validation.csv`: 12,000 loads requiring predictions.
- `data/validation-predictions-template.csv`: required validation output format.
- `data/december-chart-inputs.csv`: 31 December scenarios; the script fills `predicted_rate`.

## Run the model

From the project root, install the dependencies and run:

```powershell
python -m pip install -r requirements.txt
python train_model.py
```

The script uses a time-based holdout for validation, then retrains on all labeled training
data. It creates:

- `validation_predictions.csv`
- completed `data/december-chart-inputs.csv`

## Validate the outputs

```powershell
python score.py `
	--predictions validation_predictions.csv `
	--december-predictions data/december-chart-inputs.csv
```

The scorer checks 12,000 validation predictions and 31 December predictions, then creates:
`scorer_results/candidate_december.png`.

## Model approach

The pipeline uses `HistGradientBoostingRegressor`. Pickup, delivery, and equipment are
one-hot encoded. The model uses distance, weight, and date-derived month, weekday, and
day-of-month features. A chronological 80/20 holdout produced:

- Holdout MAE: `$132.47`
- Holdout RMSE: `$632.38`

## Submission files

- `validation_predictions.csv`
- `freight_rate_prediction_report.docx`
- `scorer_results/candidate_december.png`
