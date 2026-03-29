# NIFTY 50 Next-Day Close Price Predictor

A machine learning pipeline that predicts the next trading day's closing price of the NIFTY 50 index using technical indicators and lagged price features, built with `scikit-learn` and `pandas`.

---

## Features Engineered

| Feature | Description |
|---|---|
| `close_lag1` | Previous day's closing price |
| `close_lag2` | Closing price from 2 days ago |
| `shared_value` | Lagged shares traded (volume) |
| `15_ema` | 15-day rolling mean of Close |
| `25_ema` | 25-day rolling mean of Close |
| `50_ema` | 50-day rolling mean of Close |
| `pct_close` | Daily % change in Close |
| `RSI` | 14-period Relative Strength Index |

---

## Requirements
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## How to Run

1. Place `NIFTY 50.csv` in the project directory
2. Update the file path in `main.py`
3. Run:
```bash
python main.py
```

---

## Pipeline
```
Raw CSV → Feature Engineering → dropna() → Train/Test Split (80/20, no shuffle) → LinearRegression → MAE · MSE · R²
```

---

## Results

| Metric | Meaning |
|---|---|
| R² ≈ 0.999 | Expected for next-day prediction using lagged close |
| MAE | Average error in index points |
| MSE | Penalises large errors more heavily |

> High R² is normal here — `close_lag1` is nearly identical to tomorrow's close. It reflects autocorrelation, not trading alpha.

---

## Plots

- Actual vs Predicted close prices
- Residual distribution (error histogram + KDE)
- Open vs Close scatter coloured by volume
