
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

DATA_PATH = Path("data/processed_energy.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "energy_lstm.keras"
SCALER_PATH = MODEL_DIR / "scaler.npy"
CONFIG_PATH = MODEL_DIR / "config.json"

LOOKBACK = 96  # Previous 24 hours: 96 x 15 minutes
HORIZON = 4    # Next 60 minutes: four 15-minute forecasts

print("Loading processed energy data...")
df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"], index_col="Datetime")
values = df["Global_active_power"].to_numpy(dtype=np.float32)

# Chronological split: 70% train, 15% validation, 15% test.
n = len(values)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

# Fit scaler on training data only.
scaler = MinMaxScaler()
scaler.fit(values[:train_end].reshape(-1, 1))
scaled = scaler.transform(values.reshape(-1, 1)).flatten()

# Create sequences without shuffling the time series.
def make_sequences(series, start, end, lookback, horizon):
    X, y = [], []

    for i in range(start + lookback, end - horizon + 1):
        X.append(series[i - lookback:i])
        y.append(series[i:i + horizon])

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

X_train, y_train = make_sequences(
    scaled, 0, train_end, LOOKBACK, HORIZON
)

# Include earlier historical context for validation and testing.
X_val, y_val = make_sequences(
    scaled, train_end - LOOKBACK, val_end, LOOKBACK, HORIZON
)
X_test, y_test = make_sequences(
    scaled, val_end - LOOKBACK, n, LOOKBACK, HORIZON
)

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print(f"Training sequences:   {len(X_train):,}")
print(f"Validation sequences: {len(X_val):,}")
print(f"Testing sequences:    {len(X_test):,}")

print("\nBuilding LSTM model...")
model = Sequential([
    Input(shape=(LOOKBACK, 1)),
    LSTM(32, return_sequences=True),
    Dropout(0.2),
    LSTM(16),
    Dense(HORIZON)
])

model.compile(optimizer="adam", loss="mse")

model.summary()

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("\nTraining model. This may take several minutes...")

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=128,
    callbacks=[early_stopping],
    shuffle=False,
    verbose=1
)

print("\nEvaluating on unseen test data...")
pred_scaled = model.predict(X_test, verbose=0)

predictions = scaler.inverse_transform(
    pred_scaled.reshape(-1, 1)
).reshape(-1, HORIZON)

actual = scaler.inverse_transform(
    y_test.reshape(-1, 1)
).reshape(-1, HORIZON)

mae = mean_absolute_error(actual.flatten(), predictions.flatten())
rmse = np.sqrt(mean_squared_error(actual.flatten(), predictions.flatten()))

model.save(MODEL_PATH)
np.save(SCALER_PATH, scaler.scale_)
np.save(MODEL_DIR / "scaler_min.npy", scaler.min_)

config = {
    "lookback": LOOKBACK,
    "horizon": HORIZON,
    "frequency_minutes": 15,
    "train_rows": train_end,
    "validation_end_row": val_end,
    "test_rows": n - val_end,
    "test_mae_kw": float(mae),
    "test_rmse_kw": float(rmse)
}

CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

print("\nTraining completed successfully!")
print(f"Test MAE:  {mae:.4f} kW")
print(f"Test RMSE: {rmse:.4f} kW")
print(f"Model saved: {MODEL_PATH}")
print(f"Scaler parameters saved: {SCALER_PATH}")
print(f"Configuration saved: {CONFIG_PATH}")
