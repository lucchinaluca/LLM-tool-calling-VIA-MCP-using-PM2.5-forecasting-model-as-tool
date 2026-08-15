import os
import sys
os.environ["MCP_USE_ANONYMIZED_TELEMETRY"] = "false"
os.environ["DEBUG"] = "0"
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
import torch
import torch.nn as nn
from mcp.server.fastmcp import FastMCP
from retry_requests import retry
from sklearn.preprocessing import MinMaxScaler

mcp = FastMCP("bath-model-mcp")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models_saved" / "4.5transformer_lstm_optimized_cpu.pkl"

# Bath, UK coordinates (same as used for the training data source)
BATH_LAT = 51.3751
BATH_LON = -2.3617

FEATURE_COLS = ["pm2_5 (μg/m³)", "hour_sin", "hour_cos"]
TARGET_COL = "pm2_5 (μg/m³)"
TARGET_SCALER_IDX = FEATURE_COLS.index(TARGET_COL)
N_FEATURES = len(FEATURE_COLS)
INPUT_WINDOW = 24 * 3   # 72 hours = last 3 days
OUTPUT_WINDOW = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Open-Meteo air quality client with caching + retry (mirrors weather_mcp.py pattern)
_cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
_retry_session = retry(_cache_session, retries=5, backoff_factor=0.2)
_openmeteo = openmeteo_requests.Client(session=_retry_session)


class TransformerLSTMModel(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_transformer_layers,
        hidden_dim,
        num_lstm_layers,
        output_dim,
        seq_len,
        dropout=0.1,
    ):
        super(TransformerLSTMModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        x = self.input_projection(src)
        x = x + self.pos_encoder[:, : src.size(1), :]
        x = self.dropout(x)
        transformer_out = self.transformer_encoder(x)
        lstm_out, _ = self.lstm(transformer_out)
        out = lstm_out[:, -1, :]
        predictions = self.fc(self.dropout(out))
        return predictions


def _load_model() -> nn.Module:
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "TransformerLSTMModel"):
        main_module.TransformerLSTMModel = TransformerLSTMModel

    try:
        with MODEL_PATH.open("rb") as f:
            model = torch.load(f, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(
            "The saved PM2.5 model was created on a CUDA-enabled environment and cannot be loaded on this CPU-only machine. "
            "Please re-save it from the training notebook with final_model = final_model.cpu() before pickling, "
            "or retrain/export a CPU-safe model artifact."
        ) from exc

    if not isinstance(model, nn.Module):
        raise TypeError("The saved object is not a torch.nn.Module instance.")

    return model.to("cpu").eval()

def _fetch_live_pm25() -> pd.DataFrame:
    """Fetch the last 3 days of hourly PM2.5 data for Bath from Open-Meteo Air Quality API.

    Returns a DataFrame indexed by UTC timestamp with a single column 'pm2_5 (μg/m³)'.
    The index is a UTC-aware DatetimeIndex at hourly frequency.
    Excludes the current (incomplete) hour so all rows are finalised values.
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": BATH_LAT,
        "longitude": BATH_LON,
        "hourly": ["pm2_5"],
        "past_days": 3,   # returns yesterday-2, yesterday-1, yesterday (72 h)
        "forecast_days": 0,  # do not include future data
        "timezone": "UTC",
    }

    responses = _openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()

    times = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )
    pm25_values = hourly.Variables(0).ValuesAsNumpy()

    df = pd.DataFrame({"pm2_5 (μg/m³)": pm25_values}, index=times)

    # Exclude the current (in-progress) hour — keep only completed hours
    now_utc = pd.Timestamp.utcnow().floor("h")
    df = df[df.index < now_utc]

    # Ensure hourly frequency, forward-fill small gaps (≤2 h) to handle API gaps
    df = df.asfreq("h")
    df["pm2_5 (μg/m³)"] = df["pm2_5 (μg/m³)"].interpolate(method="time", limit=2)

    # Drop any remaining NaNs
    df = df.dropna()

    return df


def _build_feature_frame(pm25_df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical hour encodings and return only the FEATURE_COLS."""
    df = pm25_df.copy()
    df["hour_sin"] = df.index.map(lambda t: np.sin(2 * np.pi * t.hour / 24.0))
    df["hour_cos"] = df.index.map(lambda t: np.cos(2 * np.pi * t.hour / 24.0))
    return df[FEATURE_COLS]


def _inverse_transform_target(
    scaled_data: np.ndarray, scaler: MinMaxScaler
) -> np.ndarray:
    flat_data = scaled_data.flatten()
    dummy = np.zeros((len(flat_data), N_FEATURES))
    dummy[:, TARGET_SCALER_IDX] = flat_data
    unscaled_dummy = scaler.inverse_transform(dummy)
    return unscaled_dummy[:, TARGET_SCALER_IDX].reshape(scaled_data.shape)


@mcp.tool()
def predict_bath_pm25_forecast(hours_to_forecast: int = 24) -> dict:
    """Predict the next N hours of Bath PM2.5 using the saved Transformer-LSTM model.

    The model input is the last 3 complete days (72 hours) of real-time PM2.5 data
    fetched live from the Open-Meteo Air Quality API for Bath, UK. The scaler is fit
    on this same 72-hour window (consistent with univariate-only features: pm2_5,
    hour_sin, hour_cos). Predictions cover the next N hours starting from the first
    hour after the last observed value.
    """
    try:
        hours_to_forecast = int(hours_to_forecast)
    except (TypeError, ValueError):
        hours_to_forecast = 24

    if hours_to_forecast <= 0:
        raise ValueError("hours_to_forecast must be a positive integer.")

    model = _load_model()

    # --- fetch live data ---
    pm25_df = _fetch_live_pm25()

    # --- build feature matrix ---
    feature_df = _build_feature_frame(pm25_df)

    if len(feature_df) < INPUT_WINDOW:
        raise ValueError(
            f"Not enough live PM2.5 history for inference. "
            f"Need at least {INPUT_WINDOW} hourly rows, found {len(feature_df)}. "
            "The Open-Meteo API may be temporarily unavailable or has data gaps."
        )

    # Use only the most-recent INPUT_WINDOW rows
    recent_window = feature_df[FEATURE_COLS].iloc[-INPUT_WINDOW:].values

    # Fit the scaler on this window — identical to training preprocessing but
    # scoped to the live context window (univariate model only sees these 3 cols)
    scaler = MinMaxScaler()
    scaler.fit(recent_window)
    scaled_window = scaler.transform(recent_window)

    # --- run model inference ---
    with torch.no_grad():
        model_input = (
            torch.tensor(scaled_window[np.newaxis, :, :], dtype=torch.float32)
            .to(DEVICE)
        )
        scaled_predictions = model(model_input).squeeze(0).cpu().numpy()

    raw_predictions = _inverse_transform_target(scaled_predictions, scaler)

    # Build forecast timestamps starting from the next hour after observed data
    forecast_start = feature_df.index[-1] + pd.Timedelta(hours=1)
    output_window = min(int(hours_to_forecast), OUTPUT_WINDOW)
    forecast_index = pd.date_range(start=forecast_start, periods=output_window, freq="h")

    forecast = [
        {
            "timestamp": ts.isoformat(),
            "pm2_5_ug_per_m3": round(float(value), 3),
        }
        for ts, value in zip(forecast_index, raw_predictions[:output_window])
    ]

    return {
        "tool_name": "predict_bath_pm25_forecast",
        "source": "Transformer-LSTM model | live PM2.5 via Open-Meteo Air Quality API",
        "location": "Bath, UK",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "input_window_hours": INPUT_WINDOW,
        "forecast_horizon_hours": output_window,
        "context_start": feature_df.index[-INPUT_WINDOW].isoformat(),
        "context_end": feature_df.index[-1].isoformat(),
        "forecast": forecast,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
