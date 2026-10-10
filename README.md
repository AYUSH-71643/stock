# 📈 Stock Price Predictor (Python)

A machine learning pipeline that fetches historical stock data, cleans it, engineers technical
indicators, trains **Linear Regression** and **Random Forest** models, and visualises backtests and a
future price-trend forecast with matplotlib.

**Libraries:** pandas · scikit-learn · matplotlib · yfinance (+ numpy)

## Quick start

```powershell
pip install -r requirements.txt
python main.py                          # AAPL, 2015 -> today, last 365 days as test set
python main.py --ticker TSLA --forecast-days 20
python main.py --ticker MSFT --test-start 2024-01-01 --end 2025-01-01
python main.py --ticker RELIANCE.NS --no-show   # save the chart only
python -m unittest -v                   # offline tests (synthetic data)
```

| Option | Default | Meaning |
|---|---|---|
| `--ticker` | `AAPL` | Any Yahoo Finance symbol |
| `--start` / `--end` | `2015-01-01` / today | History window |
| `--test-start` | last 365 days | First date of the held-out test set |
| `--forecast-days` | `30` | Business days to forecast |
| `--refresh` | off | Ignore the CSV cache and re-download |
| `--no-show` | off | Don't open the chart window |

Outputs go to `outputs/`: `<TICKER>_dashboard.png`, `_backtest.csv`, `_forecast.csv`, `_metrics.csv`.

## Architecture

| Phase | Module | What happens |
|---|---|---|
| 1. Data acquisition | `stock_predictor/data.py` | `yfinance` download of OHLCV → CSV cache in `data/` → cleaning (duplicates, invalid/missing prices, High/Low consistency) |
| 2. Feature engineering | `stock_predictor/features.py` | SMA 10/50/200, EMA 20, RSI 14, multi-horizon returns, lags, volatility, volume ratio |
| 3. Models | `stock_predictor/models.py` | `MinMaxScaler → LinearRegression`, `RandomForestRegressor`, and `LSTM` pipelines |
| 4. Evaluation | `stock_predictor/evaluation.py` | RMSE, MAE, MAPE, R², direction accuracy vs a naive baseline; long/flat strategy backtest |
| 5. Sentiment (NLP) | `stock_predictor/nlp.py` | Fetches live news from Yahoo Finance, scores sentiment with VADER (-1 to +1) |
| Forecast | `stock_predictor/forecast.py` | Recursive N-day forecast with a widening ~95 % band |
| Visuals | `stock_predictor/visualize.py` | 5-panel dark dashboard |

### Design decisions
- **Predicts next-day return, not raw price.** Prices drift outside their training range, which
  breaks scaling and tree models. The predicted return is converted back to a price.
- **No data leakage.** Features at day *t* use only data up to *t* (covered by a unit test). The
  scaler is fit on training data only (inside the Pipeline), and the split is by target date.
- **Honest baseline.** Every model is compared with "tomorrow = today". If a model can't beat it,
  that is the Efficient Market Hypothesis at work, not a bug.

## Next steps
- Level 3: an LSTM (TensorFlow/Keras or PyTorch) over 30–60 day sequences.
- Sentiment features from financial news (NLP).

> ⚠️ Educational project, not financial advice.
