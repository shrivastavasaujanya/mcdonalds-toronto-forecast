"""
McDonald's Toronto — Prophet vs LSTM comparison.

Evaluation: 30-day holdout (most recent 30 days of historical data).
Forecast:   Next 30 days from today using both models.
"""

import warnings, json, sys, os
import concurrent.futures
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet

from mcdonalds_forecast import (
    trends_agent, weather_agent, events_agent, build_features
)

FEATURE_COLS = ["temp_max", "is_rainy", "is_weekend", "is_friday", "event_boost"]
LOOKBACK = 14   # LSTM sequence length (days)
HORIZON  = 30


# ── Data pipeline ─────────────────────────────────────────────────────────────

print("Running data agents in parallel...", flush=True)
with concurrent.futures.ThreadPoolExecutor() as pool:
    t = pool.submit(trends_agent)
    w = pool.submit(weather_agent)
    trends_df, weather_df = t.result(), w.result()
# Events agent requires an interactive API key; skip here (same fallback as original)
events_df = pd.DataFrame(columns=["ds", "event", "event_boost"])

features_df = build_features(trends_df, weather_df, events_df)

today = pd.Timestamp(datetime.today().date())
hist  = features_df[features_df["ds"] < today].copy().dropna(subset=["trend"]).reset_index(drop=True)

# Hold out the last 30 days for evaluation
train = hist.iloc[:-HORIZON].copy()
test  = hist.iloc[-HORIZON:].copy()

print(f"Train: {len(train)} days  |  Test (holdout): {len(test)} days\n", flush=True)


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(actual, predicted):
    actual    = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    mae  = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    return mae, rmse, mape


# ── Prophet ───────────────────────────────────────────────────────────────────

def run_prophet(train_df, test_df, future_df):
    print("[Prophet] Training...", flush=True)
    prophet_train = train_df[["ds", "trend"] + FEATURE_COLS].rename(columns={"trend": "y"})

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
    )
    for col in FEATURE_COLS:
        m.add_regressor(col)
    m.fit(prophet_train)

    def predict(df):
        pdf = df[["ds"] + FEATURE_COLS].copy()
        for col in FEATURE_COLS:
            pdf[col] = pdf[col].fillna(train_df[col].mean())
        fc = m.predict(pdf)
        return fc["yhat"].values

    test_pred   = predict(test_df)
    future_pred = predict(future_df)
    print("[Prophet] Done.", flush=True)
    return test_pred, future_pred


# ── LSTM ──────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def make_sequences(values, features, lookback):
    """Returns (X, y) tensors. X shape: (N, lookback, n_features+1)."""
    X, y = [], []
    all_cols = np.column_stack([values] + [features[:, i] for i in range(features.shape[1])])
    for i in range(lookback, len(all_cols)):
        X.append(all_cols[i - lookback:i])
        y.append(values[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def run_lstm(train_df, test_df, future_df):
    print("[LSTM] Preparing data...", flush=True)

    # Scale target
    scaler_y = MinMaxScaler()
    scaler_X = MinMaxScaler()

    # Combine train+test for scaling (no data leak — we fit only on train)
    train_y = train_df["trend"].values.reshape(-1, 1)
    train_X = train_df[FEATURE_COLS].fillna(train_df[FEATURE_COLS].mean()).values

    scaler_y.fit(train_y)
    scaler_X.fit(train_X)

    y_scaled = scaler_y.transform(train_y).flatten()
    X_scaled = scaler_X.transform(train_X)

    X_seq, y_seq = make_sequences(y_scaled, X_scaled, LOOKBACK)
    X_t = torch.tensor(X_seq)
    y_t = torch.tensor(y_seq)

    input_size = X_t.shape[2]
    model = LSTMModel(input_size=input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.MSELoss()

    print("[LSTM] Training (100 epochs)...", flush=True)
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch+1:3d}  loss={loss.item():.6f}", flush=True)

    # ── Rolling 30-day test prediction ───────────────────────────────────────
    model.eval()

    # Seed window = last LOOKBACK days of training
    window_y = y_scaled[-LOOKBACK:].tolist()
    window_X = X_scaled[-LOOKBACK:].tolist()

    test_X_raw  = test_df[FEATURE_COLS].fillna(train_df[FEATURE_COLS].mean()).values
    test_X_sc   = scaler_X.transform(test_X_raw)

    test_preds_sc = []
    for i in range(len(test_df)):
        seq_y = np.array(window_y[-LOOKBACK:], dtype=np.float32)
        seq_X = np.array(window_X[-LOOKBACK:], dtype=np.float32)
        combined = np.column_stack([seq_y.reshape(-1, 1)] +
                                   [seq_X[:, j] for j in range(seq_X.shape[1])])
        inp = torch.tensor(combined[np.newaxis], dtype=torch.float32)
        with torch.no_grad():
            p = model(inp).item()
        test_preds_sc.append(p)
        window_y.append(p)
        window_X.append(test_X_sc[i].tolist())

    test_pred = scaler_y.inverse_transform(
        np.array(test_preds_sc, dtype=np.float32).reshape(-1, 1)
    ).flatten()

    # ── 30-day future prediction ──────────────────────────────────────────────
    # Re-seed from end of all historical data (train+test)
    full_y = np.array(hist["trend"].values.reshape(-1, 1), dtype=np.float32)
    full_X = hist[FEATURE_COLS].fillna(train_df[FEATURE_COLS].mean()).values
    full_y_sc = scaler_y.transform(full_y).flatten()
    full_X_sc = scaler_X.transform(full_X)

    window_y = full_y_sc[-LOOKBACK:].tolist()
    window_X = full_X_sc[-LOOKBACK:].tolist()

    future_X_raw = future_df[FEATURE_COLS].fillna(train_df[FEATURE_COLS].mean()).values
    future_X_sc  = scaler_X.transform(future_X_raw)

    future_preds_sc = []
    for i in range(len(future_df)):
        seq_y = np.array(window_y[-LOOKBACK:], dtype=np.float32)
        seq_X = np.array(window_X[-LOOKBACK:], dtype=np.float32)
        combined = np.column_stack([seq_y.reshape(-1, 1)] +
                                   [seq_X[:, j] for j in range(seq_X.shape[1])])
        inp = torch.tensor(combined[np.newaxis], dtype=torch.float32)
        with torch.no_grad():
            p = model(inp).item()
        future_preds_sc.append(p)
        window_y.append(p)
        window_X.append(future_X_sc[i].tolist())

    future_pred = scaler_y.inverse_transform(
        np.array(future_preds_sc, dtype=np.float32).reshape(-1, 1)
    ).flatten()

    print("[LSTM] Done.", flush=True)
    return test_pred, future_pred


# ── Build future feature rows ─────────────────────────────────────────────────

future_dates = pd.date_range(start=today, periods=HORIZON, freq="D")
future_df = features_df[features_df["ds"].isin(future_dates)].copy()
if len(future_df) < HORIZON:
    extra = pd.DataFrame({"ds": future_dates})
    future_df = extra.merge(future_df, on="ds", how="left")
for col in FEATURE_COLS:
    future_df[col] = future_df[col].fillna(hist[col].mean())
future_df["dow"]        = future_df["ds"].dt.dayofweek
future_df["is_weekend"] = (future_df["dow"] >= 5).astype(int)
future_df["is_friday"]  = (future_df["dow"] == 4).astype(int)
future_df = future_df.reset_index(drop=True)


# ── Run both models ───────────────────────────────────────────────────────────

prophet_test, prophet_future = run_prophet(train, test, future_df)
lstm_test,    lstm_future    = run_lstm(train, test, future_df)

actual = test["trend"].values


# ── Normalize to demand index (100 = avg) ────────────────────────────────────

baseline = np.median(hist["trend"])
def to_index(arr): return np.clip(arr / baseline * 100, 50, 200)

prophet_test_idx  = to_index(prophet_test)
lstm_test_idx     = to_index(lstm_test)
actual_idx        = to_index(actual)

prophet_future_idx = to_index(prophet_future)
lstm_future_idx    = to_index(lstm_future)


# ── Ensemble (simple average) ────────────────────────────────────────────────

ensemble_test_idx   = (prophet_test_idx + lstm_test_idx) / 2
ensemble_future_idx = (prophet_future_idx + lstm_future_idx) / 2


# ── Evaluation ────────────────────────────────────────────────────────────────

p_mae, p_rmse, p_mape = metrics(actual_idx, prophet_test_idx)
l_mae, l_rmse, l_mape = metrics(actual_idx, lstm_test_idx)
e_mae, e_rmse, e_mape = metrics(actual_idx, ensemble_test_idx)

def winner(vals):
    best = min(vals)
    return ["Prophet", "LSTM", "Ensemble"][vals.index(best)] + " ✓"

print("\n" + "=" * 66)
print("MODEL COMPARISON — 30-day holdout (demand index)")
print("=" * 66)
print(f"{'Metric':<10} {'Prophet':>12} {'LSTM':>12} {'Ensemble':>12}  {'Winner':>10}")
print("-" * 66)
print(f"{'MAE':<10} {p_mae:>12.2f} {l_mae:>12.2f} {e_mae:>12.2f}  {winner([p_mae, l_mae, e_mae]):>10}")
print(f"{'RMSE':<10} {p_rmse:>12.2f} {l_rmse:>12.2f} {e_rmse:>12.2f}  {winner([p_rmse, l_rmse, e_rmse]):>10}")
print(f"{'MAPE %':<10} {p_mape:>12.2f} {l_mape:>12.2f} {e_mape:>12.2f}  {winner([p_mape, l_mape, e_mape]):>10}")
print("=" * 66)


# ── Side-by-side 30-day forecast ─────────────────────────────────────────────

print("\nMcDonald's Toronto — 30-Day Forecast Comparison")
print("=" * 84)
print(f"{'Date':<12} {'Day':<4} {'Actual':>8} {'Prophet':>9} {'LSTM':>9} {'Ensemble':>10}  Note")
print("-" * 84)

for i, (_, row) in enumerate(test.iterrows()):
    d   = row["ds"].date()
    dow = row["ds"].strftime("%a")
    a   = int(round(actual_idx[i]))
    p   = int(round(prophet_test_idx[i]))
    l   = int(round(lstm_test_idx[i]))
    en  = int(round(ensemble_test_idx[i]))
    print(f"{str(d):<12} {dow:<4} {a:>8} {p:>9} {l:>9} {en:>10}  ← holdout")

print("-" * 84)

for i, d in enumerate(future_dates):
    dow = d.strftime("%a")
    p   = int(round(prophet_future_idx[i]))
    l   = int(round(lstm_future_idx[i]))
    en  = int(round(ensemble_future_idx[i]))
    print(f"{str(d.date()):<12} {dow:<4} {'—':>8} {p:>9} {l:>9} {en:>10}")

print("=" * 84)
print(f"\n{'FORECAST AVG':<12} {'':4} {'':>8} {np.mean(prophet_future_idx):>9.0f} {np.mean(lstm_future_idx):>9.0f} {np.mean(ensemble_future_idx):>10.0f}")
print(f"{'FORECAST PEAK':<12} {'':4} {'':>8} {np.max(prophet_future_idx):>9.0f} {np.max(lstm_future_idx):>9.0f} {np.max(ensemble_future_idx):>10.0f}")
