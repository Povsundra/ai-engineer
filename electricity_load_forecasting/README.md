# Electricity Load Forecasting

Lightweight project to prototype hourly electricity load forecasting with a simple calendar + weather feature set. The default run uses a synthetic dataset, but you can point the pipeline at your own CSV to evaluate a quick baseline.

## What this project does
- Generates a small synthetic hourly dataset with temperature and weekend effects (if no CSV is provided).
- Trains a Random Forest baseline on calendar + temperature features.
- Evaluates on a hold-out window (MAE, RMSE, MAPE, R²).
- Produces a 24-hour (configurable) forecast and saves it to `artifacts/forecast.csv`.
- Saves the recent history window used for evaluation to `artifacts/history_sample.csv`.

## Project structure
```
electricity_load_forecasting/
├── README.MD
├── requirements.txt
├── .gitignore
├── main.py                 # CLI entry point
└── src/
    ├── __init__.py
    └── pipeline.py         # Data generation, feature engineering, training, forecasting
```

## Setup
```bash
cd electricity_load_forecasting
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
Run with the built-in synthetic sample:
```bash
python main.py
```

Point to your own CSV (columns: `timestamp`, `load_mw`, `temperature_c`):
```bash
python main.py --data /path/to/my_loads.csv --test-hours 168 --horizon 24 --output-dir artifacts
```

### Output files
- `artifacts/forecast.csv` — next horizon predictions
- `artifacts/history_sample.csv` — recent history used for evaluation

### Notes
- `--test-hours` controls how many trailing hours are held out for evaluation.
- `--horizon` sets the forecast horizon (hours). Default is 24.
- The baseline model is intentionally simple and intended for quick experiments; swap in your preferred model or feature set as needed.
