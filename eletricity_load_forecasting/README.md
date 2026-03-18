## Eletricity Load Forecasting

This mini project provides a **sequence-to-sequence (encoder–decoder) LSTM** baseline for short‑term electricity load forecasting with **13 input features**. The model is implemented in PyTorch and includes data preparation, scaling, training, validation, and inference utilities.

### Project layout

```
eletricity_load_forecasting/
├── model.py           # Encoder–decoder LSTM, dataset prep, training loop, CLI
├── requirements.txt   # Minimal dependencies (PyTorch + data utilities)
└── README.md          # This guide
```

### Requirements

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows use .venv\Scripts\activate
pip install -r eletricity_load_forecasting/requirements.txt
```

### Input data expectations

- A CSV file with **at least 13 feature columns**.
- A `target` column (default: `load`) that represents the electricity load to forecast.
- The target column should be included among the 13 feature columns so the model can use past load values.

If no dataset is provided, the CLI can generate a small synthetic series to smoke‑test the pipeline.

### Quick start (synthetic data demo)

Runs a tiny demo with synthetic data (fast, CPU‑friendly):

```bash
python eletricity_load_forecasting/model.py --epochs 3 --batch_size 64
```

### Training on your dataset

```bash
python eletricity_load_forecasting/model.py \
  --data_path /path/to/electricity.csv \
  --target_col load \
  --feature_cols f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 load \
  --input_window 48 \
  --output_window 12 \
  --epochs 20 \
  --hidden_size 128 \
  --num_layers 2 \
  --dropout 0.15
```

Key flags:

- `--data_path`: CSV file to train on. If omitted, synthetic data is used.
- `--feature_cols`: Space‑separated list of exactly **13** feature columns (must include the target column).
- `--target_col`: Name of the load column (default `load`).
- `--input_window`: Historical time steps consumed by the encoder.
- `--output_window`: Future time steps predicted by the decoder.

### Outputs

The script prints train/validation losses each epoch and final MAE/MAPE metrics on the hold‑out test split. You can adapt the code to persist checkpoints or export forecasts by extending the `save_path` argument.
