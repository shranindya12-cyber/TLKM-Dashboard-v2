import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

# =====================================================
# FOLDER
# =====================================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =====================================================
# DOWNLOAD DATA
# =====================================================

ticker = "TLKM.JK"

df = yf.download(
    ticker,
    period="10y",
    interval="1d",
    auto_adjust=True,
    progress=False
)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()

# =====================================================
# FEATURE ENGINEERING
# =====================================================

df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()

delta = df["Close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

df["RSI"] = 100 - (100 / (1 + rs))

ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()

df["MACD"] = ema12 - ema26

df.dropna(inplace=True)

# =====================================================
# FEATURES
# =====================================================

features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "RSI",
    "MACD",
    "MA20",
    "MA50"
]

dataset = df[features]

# =====================================================
# SCALING
# =====================================================

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

# =====================================================
# LOOKBACK
# =====================================================

LOOKBACK = 60

X = []
y = []

for i in range(LOOKBACK, len(scaled_data)):
    X.append(scaled_data[i-LOOKBACK:i])
    y.append(scaled_data[i, 3])

X = np.array(X)
y = np.array(y)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

# =====================================================
# MODEL LSTM
# =====================================================

model = Sequential([
    LSTM(
        128,
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    ),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse"
)

# =====================================================
# EARLY STOPPING
# =====================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# =====================================================
# TRAINING
# =====================================================

history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# =====================================================
# PREDICTION TEST
# =====================================================

y_pred = model.predict(X_test, verbose=0)

# =====================================================
# EVALUATION
# =====================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

mape = np.mean(
    np.abs((y_test - y_pred.flatten()) / y_test)
) * 100

r2 = r2_score(y_test, y_pred)

accuracy = 100 - mape

print("\n===== HASIL EVALUASI =====")
print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"MAPE     : {mape:.2f}%")
print(f"ACC      : {accuracy:.2f}%")
print(f"R2 Score : {r2:.4f}")

# =====================================================
# SAVE MODEL
# =====================================================

model.save("models/best_model.keras")

# =====================================================
# SAVE METRICS
# =====================================================

metrics = {
    "model_name": "Multivariate LSTM",
    "epochs": 50,
    "batch_size": 32,
    "lookback": LOOKBACK,
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "ticker": ticker,
    "mae": float(mae),
    "rmse": float(rmse),
    "mape": float(mape),
    "accuracy": float(accuracy),
    "r2": float(r2)
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(
    "outputs/metrics.csv",
    index=False
)

# =====================================================
# ACTUAL VS PREDICTION
# =====================================================

actual_pred = pd.DataFrame({
    "Actual": y_test,
    "Prediction": y_pred.flatten()
})

actual_pred.to_csv(
    "outputs/actual_vs_prediction.csv",
    index=False
)

# =====================================================
# LOSS HISTORY
# =====================================================

loss_df = pd.DataFrame({
    "loss": history.history["loss"],
    "val_loss": history.history["val_loss"]
})

loss_df.to_csv(
    "outputs/loss_history.csv",
    index=False
)

# =====================================================
# SAVE HISTORICAL DATA
# =====================================================

historical_data = df[[
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]]

historical_data.to_csv(
    "outputs/historical_data.csv",
    index=False
)

# =====================================================
# FORECAST 30 HARI
# =====================================================

future_days = 30

last_sequence = X[-1].copy()
forecast_scaled = []

for _ in range(future_days):

    pred = model.predict(
        last_sequence.reshape(
            1,
            LOOKBACK,
            len(features)
        ),
        verbose=0
    )

    forecast_scaled.append(pred[0][0])

    next_row = last_sequence[-1].copy()
    next_row[3] = pred[0][0]

    last_sequence = np.vstack([
        last_sequence[1:],
        next_row
    ])

dummy = np.zeros(
    (len(forecast_scaled), len(features))
)

dummy[:, 3] = forecast_scaled

forecast_price = (
    scaler.inverse_transform(dummy)
)[:, 3]

forecast_std = np.std(
    actual_pred["Actual"] - actual_pred["Prediction"]
)

future_dates = pd.date_range(
    start=df["Date"].iloc[-1] + pd.Timedelta(days=1),
    periods=30
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast": forecast_price,
    "Lower_Bound": forecast_price - (1.96 * forecast_std),
    "Upper_Bound": forecast_price + (1.96 * forecast_std)
})

forecast_df.to_csv(
    "outputs/forecast.csv",
    index=False
)

print("\\nForecast 30 Hari berhasil dibuat.")
print("Model berhasil disimpan.")
