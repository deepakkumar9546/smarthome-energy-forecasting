
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

# ---------------- Configuration ----------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed_energy.csv"
MODEL_PATH = BASE_DIR / "models" / "energy_lstm.keras"
SCALER_PATH = BASE_DIR / "models" / "scaler.npy"
CONFIG_PATH = BASE_DIR / "models" / "config.json"

st.set_page_config(
    page_title="SmartHome AI | Energy Forecasting",
    page_icon="⚡",
    layout="wide"
)

# ---------------- Load saved artifacts ----------------
@st.cache_resource
def load_artifacts():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_energy_data():
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["Datetime"],
        index_col="Datetime"
    )
    return df.sort_index()

# ---------------- Header ----------------
st.title("⚡ SmartHome AI")
st.subheader("Smart Home Energy Forecasting Dashboard")
st.caption(
    "LSTM-powered forecasting using historical household electricity data"
)

# Check project files
required_files = [
    DATA_PATH,
    MODEL_PATH,
    SCALER_PATH,
    SCALER_PATH.with_name("scaler_min.npy"),
    CONFIG_PATH,
]

if not all(path.exists() for path in required_files):
    st.error("Required project files are missing. Please train the model first.")
    st.stop()

try:
    model = load_artifacts()
    energy_df = load_energy_data()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
except Exception as exc:
    st.error(f"Could not load the model or data: {exc}")
    st.stop()

power = energy_df["Global_active_power"].astype(float)
latest_time = power.index[-1]
current_power = float(power.iloc[-1])

# ---------------- Forecast ----------------
lookback = int(config["lookback"])
history = power.tail(lookback).to_numpy().reshape(-1, 1)


scaler = MinMaxScaler()
scaler.scale_ = np.load(SCALER_PATH)
scaler.min_ = np.load(SCALER_PATH.with_name("scaler_min.npy"))
scaler.n_features_in_ = 1

scaled_history = scaler.transform(history).reshape(1, lookback, 1)

forecast_scaled = model.predict(scaled_history, verbose=0)[0]
forecast = scaler.inverse_transform(
    forecast_scaled.reshape(-1, 1)
).flatten()

# Power is non-negative; clip small negative predictions if any.
forecast = np.maximum(forecast, 0)

# ---------------- Summary metrics ----------------
daily = power[power.index.date == latest_time.date()]
daily_energy_kwh = float(daily.sum() * 0.25)
peak_power = float(daily.max())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest Recorded Power", f"{current_power:.2f} kW")
m2.metric("Today's Estimated Energy", f"{daily_energy_kwh:.2f} kWh")
m3.metric("Today's Peak Power", f"{peak_power:.2f} kW")
m4.metric("Next 15-Min Forecast", f"{forecast[0]:.2f} kW")

st.caption(
    f"Latest dataset timestamp: {latest_time:%d %b %Y, %H:%M}. "
    "This is historical data, not a live smart-meter feed."
)

# ---------------- Forecast table ----------------
st.markdown("### 🔮 Upcoming Energy Forecast")

forecast_times = [
    latest_time + pd.Timedelta(minutes=15 * i)
    for i in range(1, 5)
]

forecast_df = pd.DataFrame({
    "Forecast Time": forecast_times,
    "Horizon": ["15 minutes", "30 minutes", "45 minutes", "60 minutes"],
    "Predicted Active Power (kW)": np.round(forecast, 3)
})

st.dataframe(
    forecast_df,
    use_container_width=True,
    hide_index=True
)

# ---------------- Historical chart ----------------
st.markdown("### 📈 Recent Electricity Consumption")

history_hours = st.selectbox(
    "Historical period",
    [6, 12, 24, 48, 72],
    index=2
)

recent = power.tail(history_hours * 4)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=recent.index,
    y=recent.values,
    mode="lines",
    name="Actual historical power"
))

fig.add_trace(go.Scatter(
    x=forecast_times,
    y=forecast,
    mode="lines+markers",
    name="LSTM forecast",
    line=dict(dash="dash")
))

fig.update_layout(
    xaxis_title="Date and time",
    yaxis_title="Active power (kW)",
    hovermode="x unified",
    legend=dict(orientation="h")
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- Usage warning ----------------
st.markdown("### ⚠️ High-Consumption Warning")

threshold = st.slider(
    "High-power threshold (kW)",
    min_value=1.0,
    max_value=10.0,
    value=3.0,
    step=0.5
)

if current_power >= threshold:
    st.warning(
        f"Latest recorded power ({current_power:.2f} kW) "
        f"is at or above your {threshold:.1f} kW threshold."
    )
elif float(np.max(forecast)) >= threshold:
    st.warning(
        "The forecast indicates that power may reach or exceed "
        f"{threshold:.1f} kW in the next hour."
    )
else:
    st.success("No high-consumption threshold breach detected.")

# ---------------- Recommendations ----------------
st.markdown("### 💡 Energy-Saving Recommendations")

if current_power >= threshold:
    st.write(
        "- Check whether high-power appliances are running together."
    )
    st.write(
        "- Turn off appliances that are not needed."
    )
else:
    st.write(
        "- Prefer efficient appliances and switch off unused devices."
    )

if float(np.max(forecast)) >= threshold:
    st.write(
        "- Review scheduled appliance usage during the forecast peak."
    )

st.info(
    "Forecasts are estimates based on historical household readings. "
    "They do not measure current electricity use in real time."
)

with st.expander("Model evaluation"):
    st.write(f"Test MAE: {config['test_mae_kw']:.4f} kW")
    st.write(f"Test RMSE: {config['test_rmse_kw']:.4f} kW")
