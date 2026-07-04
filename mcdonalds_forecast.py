"""
McDonald's Toronto — 30-day daily demand forecast.

Multi-agent system:
  Trends Agent  — 2 years of Google Trends for "McDonald's Toronto"
  Weather Agent — historical + 14-day forecast from Open-Meteo (free, no key)
  Events Agent  — upcoming Toronto events via web search
  Forecast Agent — Prophet model trained on all signals
  Report Agent  — formats the 30-day demand index
"""

import json
import warnings
import concurrent.futures
from datetime import datetime, timedelta

import anthropic
import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry
from pytrends.request import TrendReq
from prophet import Prophet

warnings.filterwarnings("ignore")

client = anthropic.Anthropic()
SEARCH_TOOLS = [{"type": "web_search_20250305", "name": "web_search"}]


# ── Agent 1: Google Trends ────────────────────────────────────────────────────

def trends_agent():
    """Pulls 2 years of daily Google Trends for McDonald's Toronto."""
    print("  [Trends Agent] Fetching Google Trends...", flush=True)
    try:
        pytrends = TrendReq(hl="en-CA", tz=-300)
        pytrends.build_payload(
            ["McDonald's Toronto"],
            timeframe="today 5-y",
            geo="CA-ON",
        )
        df = pytrends.interest_over_time()
        if df.empty:
            raise ValueError("Empty trends response")
        df = df.reset_index()[["date", "McDonald's Toronto"]]
        df.columns = ["ds", "trend"]
        df["ds"] = pd.to_datetime(df["ds"])
        # weekly → daily via interpolation
        df = df.set_index("ds").resample("D").interpolate().reset_index()
        print(f"  [Trends Agent] Got {len(df)} days of trend data", flush=True)
        return df
    except Exception as e:
        print(f"  [Trends Agent] Failed: {e} — using synthetic baseline", flush=True)
        dates = pd.date_range(end=datetime.today(), periods=365 * 2, freq="D")
        trend = 60 + 20 * np.sin(2 * np.pi * dates.dayofyear / 365) + np.random.normal(0, 5, len(dates))
        return pd.DataFrame({"ds": dates, "trend": trend.clip(0, 100)})


# ── Agent 2: Weather ──────────────────────────────────────────────────────────

def weather_agent():
    """Pulls historical + forecast weather for Toronto from Open-Meteo."""
    print("  [Weather Agent] Fetching weather data...", flush=True)
    try:
        cache_session = requests_cache.CachedSession(".weather_cache", expire_after=3600)
        retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
        om = openmeteo_requests.Client(session=retry_session)

        today = datetime.today().date()
        start = (today - timedelta(days=365 * 2)).isoformat()
        end_hist = (today - timedelta(days=1)).isoformat()
        end_fcast = (today + timedelta(days=14)).isoformat()

        def fetch(url, start_date, end_date):
            responses = om.weather_api(url, params={
                "latitude": 43.7,
                "longitude": -79.42,
                "daily": ["temperature_2m_max", "precipitation_sum"],
                "start_date": start_date,
                "end_date": end_date,
                "timezone": "America/Toronto",
            })
            r = responses[0].Daily()
            dates = pd.date_range(
                start=pd.Timestamp(r.Time(), unit="s", tz="UTC").tz_localize(None),
                end=pd.Timestamp(r.TimeEnd(), unit="s", tz="UTC").tz_localize(None),
                freq=pd.Timedelta(seconds=r.Interval()),
                inclusive="left",
            )
            return pd.DataFrame({
                "ds": dates,
                "temp_max": r.Variables(0).ValuesAsNumpy(),
                "precip": r.Variables(1).ValuesAsNumpy(),
            })

        hist = fetch("https://archive-api.open-meteo.com/v1/archive", start, end_hist)
        fcast = fetch("https://api.open-meteo.com/v1/forecast", today.isoformat(), end_fcast)
        df = pd.concat([hist, fcast]).drop_duplicates("ds").sort_values("ds").reset_index(drop=True)
        print(f"  [Weather Agent] Got {len(df)} days of weather data", flush=True)
        return df
    except Exception as e:
        print(f"  [Weather Agent] Failed: {e} — using synthetic weather", flush=True)
        dates = pd.date_range(end=datetime.today() + timedelta(days=30), periods=365 * 2 + 30, freq="D")
        temp = 10 + 15 * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365) + np.random.normal(0, 3, len(dates))
        precip = np.random.exponential(2, len(dates)) * (np.random.rand(len(dates)) < 0.3)
        return pd.DataFrame({"ds": dates, "temp_max": temp, "precip": precip})


# ── Agent 3: Events ───────────────────────────────────────────────────────────

def events_agent():
    """Searches for major Toronto events in the next 30 days."""
    print("  [Events Agent] Searching for Toronto events...", flush=True)
    today = datetime.today().date()
    end = today + timedelta(days=30)

    messages = [{
        "role": "user",
        "content": (
            f"Search for major events in Toronto between {today} and {end}. "
            "Include: Raptors or Leafs games, concerts at Scotiabank Arena, "
            "festivals, Canada Day, long weekends, and any large public events. "
            "Return a JSON array like: "
            '[{"date": "2026-07-01", "event": "Canada Day", "impact": "high"}]. '
            "Impact: high = city-wide, medium = neighborhood, low = minor. JSON only."
        ),
    }]

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            system="You are a Toronto events researcher. Return only a valid JSON array with no markdown, no code fences, no other text.",
            messages=messages,
            tools=SEARCH_TOOLS,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break

    try:
        import re
        text = next((b.text for b in response.content if hasattr(b, "text")), "[]")
        # strip markdown code fences if present
        text = re.sub(r"```(?:json)?", "", text).strip()
        match = re.search(r'\[.*\]', text, re.DOTALL)
        events = json.loads(match.group()) if match else []
        df = pd.DataFrame(events)
        if df.empty or "date" not in df.columns:
            raise ValueError("No valid events returned")
        df["ds"] = pd.to_datetime(df["date"])
        df["event_boost"] = df.get("impact", pd.Series(["medium"] * len(df))).map(
            {"high": 30, "medium": 15, "low": 5}
        ).fillna(10)
        print(f"  [Events Agent] Found {len(df)} events", flush=True)
        return df[["ds", "event", "event_boost"]]
    except Exception as e:
        print(f"  [Events Agent] Parse failed: {e} — no events", flush=True)
        return pd.DataFrame(columns=["ds", "event", "event_boost"])


# ── Feature Engineering ───────────────────────────────────────────────────────

def build_features(trends_df, weather_df, events_df):
    """Merges all signals into a single daily feature table."""
    df = trends_df.copy()
    df = df.merge(weather_df, on="ds", how="left")

    # Events
    events_agg = events_df.groupby("ds")["event_boost"].sum().reset_index()
    df = df.merge(events_agg, on="ds", how="left")
    df["event_boost"] = df["event_boost"].fillna(0)

    # Day-of-week and holiday features
    df["dow"] = df["ds"].dt.dayofweek          # 0=Mon, 6=Sun
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_friday"] = (df["dow"] == 4).astype(int)

    # Weather features — forward/backward fill for any gaps then fallback to seasonal median
    df["temp_max"] = df["temp_max"].ffill().bfill().fillna(10.0)
    df["precip"] = df["precip"].ffill().bfill().fillna(0.0)
    df["is_rainy"] = (df["precip"] > 5).astype(int)
    df["is_cold"] = (df["temp_max"] < 5).astype(int)
    df["is_warm"] = (df["temp_max"] > 20).astype(int)

    # Trend
    df["trend"] = df["trend"].fillna(df["trend"].median())

    return df


# ── Forecast Agent ────────────────────────────────────────────────────────────

def forecast_agent(features_df, horizon=30):
    """Trains Prophet on historical features and forecasts the next 30 days."""
    print("  [Forecast Agent] Training Prophet model...", flush=True)

    today = pd.Timestamp(datetime.today().date())
    hist = features_df[features_df["ds"] < today].copy()
    future_dates = pd.date_range(start=today, periods=horizon, freq="D")

    # Target: trend index as proxy for demand
    prophet_df = hist[["ds", "trend"]].rename(columns={"trend": "y"})

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
    )
    m.add_regressor("temp_max")
    m.add_regressor("is_rainy")
    m.add_regressor("is_weekend")
    m.add_regressor("is_friday")
    m.add_regressor("event_boost")

    # Add regressors to training data
    for col in ["temp_max", "is_rainy", "is_weekend", "is_friday", "event_boost"]:
        prophet_df[col] = hist[col].values

    m.fit(prophet_df)

    # Build future dataframe with known/forecasted features
    future_features = features_df[features_df["ds"].isin(future_dates)].copy()
    future_df = pd.DataFrame({"ds": future_dates})
    future_df = future_df.merge(future_features[["ds", "temp_max", "is_rainy", "is_weekend", "is_friday", "event_boost"]], on="ds", how="left")

    # Fill any missing future weather with seasonal average
    for col in ["temp_max", "is_rainy", "is_weekend", "is_friday", "event_boost"]:
        future_df[col] = future_df[col].fillna(hist[col].mean())

    forecast = m.predict(future_df)

    # Normalize to demand index (100 = average day)
    baseline = np.median(hist["trend"])
    def norm(col): return (forecast[col].values / baseline * 100)

    demand_index = norm("yhat").clip(50, 200)

    # Component contributions (scaled to index points relative to 100)
    components = pd.DataFrame({
        "date": future_dates.date,
        "demand_index": demand_index.round(0).astype(int),
        "lower": norm("yhat_lower").clip(40, 200).round(0).astype(int),
        "upper": norm("yhat_upper").clip(60, 220).round(0).astype(int),
        "c_trend":    norm("trend").round(1),
        "c_weekly":   norm("weekly").round(1),
        "c_yearly":   norm("yearly").round(1),
        "c_temp":     (forecast["temp_max"].values / baseline * 100).round(1) if "temp_max" in forecast else 0,
        "c_rain":     (forecast["is_rainy"].values / baseline * 100).round(1) if "is_rainy" in forecast else 0,
        "c_weekend":  (forecast["is_weekend"].values / baseline * 100).round(1) if "is_weekend" in forecast else 0,
        "c_friday":   (forecast["is_friday"].values / baseline * 100).round(1) if "is_friday" in forecast else 0,
        "c_event":    (forecast["event_boost"].values / baseline * 100).round(1) if "event_boost" in forecast else 0,
    })

    print("  [Forecast Agent] Forecast complete.", flush=True)
    return components


# ── Report Agent ──────────────────────────────────────────────────────────────

def report_agent(forecast_df, events_df):
    """Formats the 30-day forecast into a readable report."""
    events_dict = {}
    if not events_df.empty:
        for _, row in events_df.iterrows():
            key = row["ds"].date()
            events_dict[key] = row.get("event", "")

    bars = {range(0, 80): "▁", range(80, 90): "▂", range(90, 100): "▃",
            range(100, 110): "▄", range(110, 120): "▅", range(120, 135): "▆",
            range(135, 155): "▇", range(155, 300): "█"}

    def bar(val):
        for r, b in bars.items():
            if val in r:
                return b
        return "█"

    lines = ["McDonald's Toronto — 30-Day Demand Forecast",
             "=" * 52,
             "Index: 100 = average day  |  >120 = high  |  <80 = low",
             ""]

    for _, row in forecast_df.iterrows():
        d = row["date"]
        idx = row["demand_index"]
        dow = datetime.strptime(str(d), "%Y-%m-%d").strftime("%a")
        event = f"  ← {events_dict[d]}" if d in events_dict else ""
        flag = " 🔴" if idx >= 130 else (" 🟡" if idx >= 110 else "")
        lines.append(f"{d} {dow}  {bar(idx)} {idx:3d}{flag}{event}")

    lines += ["", f"Peak day:  {forecast_df.loc[forecast_df['demand_index'].idxmax(), 'date']}  ({forecast_df['demand_index'].max()} index)",
              f"Low day:   {forecast_df.loc[forecast_df['demand_index'].idxmin(), 'date']}  ({forecast_df['demand_index'].min()} index)",
              f"Avg index: {forecast_df['demand_index'].mean():.0f}"]

    return "\n".join(lines)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_forecast():
    print("Starting McDonald's Toronto demand forecast...\n", flush=True)

    # Run data agents in parallel
    print("Running data agents in parallel...", flush=True)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        trends_future = pool.submit(trends_agent)
        weather_future = pool.submit(weather_agent)
        events_future = pool.submit(events_agent)
        trends_df = trends_future.result()
        weather_df = weather_future.result()
        events_df = events_future.result()

    print("\nEngineering features...", flush=True)
    features_df = build_features(trends_df, weather_df, events_df)

    print("Running forecast model...", flush=True)
    forecast_df = forecast_agent(features_df)

    print("Generating report...\n", flush=True)
    report = report_agent(forecast_df, events_df)
    return report


if __name__ == "__main__":
    print(run_forecast())
