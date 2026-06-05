
# weather_mcp.py
from datetime import datetime, timezone
from mcp_use.server import FastMCP
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

mcp = FastMCP("bath-weather-mcp")

# Setup Open-Meteo client
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

@mcp.tool()
def get_weather_bath() -> dict:
    """Return real-time + forecast weather for Bath, UK using Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 51.3751,
        "longitude": -2.3617,
        "hourly": [
            "temperature_2m", "wind_speed_10m", "wind_direction_10m",
            "wind_gusts_10m", "rain", "precipitation",
            "apparent_temperature", "dew_point_2m"
        ],
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()

    df = pd.DataFrame({
        "time": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "wind_speed_10m": hourly.Variables(1).ValuesAsNumpy(),
        "wind_direction_10m": hourly.Variables(2).ValuesAsNumpy(),
        "wind_gusts_10m": hourly.Variables(3).ValuesAsNumpy(),
        "rain": hourly.Variables(4).ValuesAsNumpy(),
        "precipitation": hourly.Variables(5).ValuesAsNumpy(),
        "apparent_temperature": hourly.Variables(6).ValuesAsNumpy(),
        "dew_point_2m": hourly.Variables(7).ValuesAsNumpy(),
    })

    # Return only the next 12 hours for compactness
    df = df.head(12)

    return {
        "tool_name": "get_weather_bath",
        "source": "Open-Meteo hourly forecast",
        "location": "Bath, UK",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "summary": "Next 12 hours of weather data for Bath, including temperature, wind, precipitation, and humidity.",
        "forecast": df.to_dict(orient="list")
    }

if __name__ == "__main__":
    mcp.run()
