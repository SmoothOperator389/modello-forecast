"""Feature PDV + meteo per TESTING.

Default: meteo simulato per città.
Con FORECAST_WEATHER_API=1 usa Open-Meteo (stesso connettore di production).
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from testing.config import DATA_DIR

MQ_BANDS = 4
WEATHER_REGIMES = 4


def load_pdv_master(path: Path | None = None) -> pd.DataFrame:
    path = path or (DATA_DIR / "pdv_master.csv")
    df = pd.read_csv(path, parse_dates=["data_apertura"])
    df["data_apertura"] = df["data_apertura"].dt.date
    if "transit_time_days" not in df.columns and "transit_time" in df.columns:
        df["transit_time_days"] = df["transit_time"]
    return df


def mq_to_band(mq: float) -> int:
    if mq < 80:
        return 0
    if mq < 100:
        return 1
    if mq < 120:
        return 2
    return 3


def weather_to_regime(temp_media: float, giorni_pioggia: int) -> int:
    if giorni_pioggia >= 2:
        return 3
    if temp_media >= 26:
        return 2
    if temp_media <= 10:
        return 0
    return 1


MQ_BANDS = 4
WEATHER_REGIMES = 4

# Temperatura media mensile per paese (°C) — ciclo reale inverno/estate
_COUNTRY_MONTHLY_TEMP: dict[str, list[float]] = {
    "IT": [5.5, 7.0, 11.0, 15.0, 19.5, 24.0, 27.0, 26.5, 22.0, 16.0, 10.0, 6.5],
    "FR": [4.5, 5.5, 9.0, 12.0, 16.0, 20.0, 23.0, 22.5, 18.5, 13.5, 8.0, 5.0],
    "DE": [0.5, 1.5, 5.5, 10.0, 14.5, 18.0, 20.0, 19.5, 15.5, 10.0, 5.0, 1.5],
    "AT": [0.0, 1.5, 6.0, 10.5, 15.0, 18.5, 20.5, 20.0, 15.5, 10.0, 4.5, 1.0],
    "ES": [7.0, 8.5, 11.5, 14.5, 19.0, 24.5, 28.5, 28.0, 23.5, 17.0, 11.0, 7.5],
    "UK": [5.0, 5.5, 7.5, 10.0, 13.5, 16.5, 19.0, 18.5, 15.5, 12.0, 8.0, 5.5],
}

# Scostamento città vs media paese
CITY_TEMP_ADJ: dict[str, float] = {
    "Milan": 0.5, "Rome": 3.0, "Naples": 2.5,
    "Paris": 0.5, "Lyon": 0.0, "Marseille": 3.0,
    "Berlin": 0.0, "Munich": -0.5, "Hamburg": -1.0,
    "Vienna": 0.5, "Salzburg": -1.5, "Graz": 0.0,
    "Madrid": 1.5, "Barcelona": 2.5, "Valencia": 3.0,
    "London": 0.5, "Manchester": -1.0, "Birmingham": -0.5,
}


def daily_weather(country: str, city: str, d: date) -> dict[str, float | int]:
    """Meteo giornaliero per città (testing fake / stesso schema produzione API)."""
    monthly = _COUNTRY_MONTHLY_TEMP.get(country, _COUNTRY_MONTHLY_TEMP["IT"])
    temp = monthly[d.month - 1] + CITY_TEMP_ADJ.get(city, 0.0)
    # oscillazione deterministica giorno-giorno (±2.5°C)
    jitter = ((hash((city, d.isoformat())) % 21) - 10) / 4.0
    temp += jitter
    is_rain = hash((city, d.toordinal())) % 11 == 0
    if country in {"UK", "DE", "AT"}:
        is_rain = hash((city, d.toordinal())) % 8 == 0
    if is_rain:
        temp -= 2.5
    return {
        "temp_media": float(round(temp, 1)),
        "pioggia_mm": 10.0 if is_rain else 0.0,
        "is_pioggia": int(is_rain),
    }


def _use_weather_api() -> bool:
    return os.environ.get("FORECAST_WEATHER_API", "").strip().lower() in {"1", "true", "yes", "on"}


def _store_hour_temps_from_daily(daily: pd.DataFrame) -> dict[str, float]:
    """Proxy ore di vendita: negozio 10–20, picco pomeriggio 14–17."""
    tm = float(daily["temp_media"].mean()) if not daily.empty else 18.0
    return {
        "temp_orario_negozio": tm + 1.2,
        "temp_pomeriggio": tm + 3.5,
    }


def _store_hour_temps_from_api(country: str, city: str, start: date, end: date, daily: pd.DataFrame) -> dict[str, float]:
    from production.connectors.weather_api import fetch_hourly_weather

    h = fetch_hourly_weather(city, start, end, country)
    if h.empty:
        return _store_hour_temps_from_daily(daily)
    shop = h[(h["ora"] >= 10) & (h["ora"] < 20)]
    pm = h[(h["ora"] >= 14) & (h["ora"] < 18)]
    base = _store_hour_temps_from_daily(daily)
    if not shop.empty:
        base["temp_orario_negozio"] = float(shop["temp_ora"].mean())
    if not pm.empty:
        base["temp_pomeriggio"] = float(pm["temp_ora"].mean())
    return base


def ora_picco_vendita(df: pd.DataFrame) -> pd.Series:
    """Ora media di vendita dai scontrini (``ora_media_14d`` / ``ora``).

    Fallback: orario tipico per tipo SKU se il fact non ha data+ora.
    """
    if "ora_media_14d" in df.columns and df["ora_media_14d"].notna().any():
        return df["ora_media_14d"].astype(float).fillna(15.0)
    if "ora" in df.columns and df["ora"].notna().any():
        return df["ora"].astype(float).fillna(15.0)
    peak = {
        "COLLANA": 16, "BRACCIALE": 16, "ORECCHINI": 16, "ANELLO": 16, "CHARM": 16, "CLIP": 16,
        "AVIATOR": 15, "BIKINI": 14, "PUFFER": 12, "COAT": 12, "PARKA": 12,
        "KNIT": 13, "CARDIGAN": 13, "BEANIE": 12, "SCARF": 12, "ANKLE": 13,
        "CROP": 15, "TSHIRT": 15, "TANK": 15, "SHORT": 15, "SKIRT": 15, "MINI": 15,
        "FLAT": 15, "LOW": 15, "HOODIE": 14, "JEANS": 16, "PANT": 16,
    }
    if "ssclasse" in df.columns:
        return df["ssclasse"].astype(str).map(peak).fillna(15.0).astype(float)
    return pd.Series(15.0, index=df.index)


def temp_at_sale_hour(temp_media: pd.Series, temp_pomeriggio: pd.Series, ora: pd.Series) -> pd.Series:
    """Temperatura stimata all'ora di vendita: 15:00 ≈ pomeriggio, 10:00 ≈ media giorno."""
    t0 = temp_media.astype(float)
    t1 = temp_pomeriggio.astype(float)
    w = ((ora.astype(float) - 10.0) / 5.0).clip(lower=0.0, upper=1.2)
    return t0 + w * (t1 - t0)


def fetch_weather_window(
    country: str,
    city: str,
    start: date,
    end: date,
    rng_seed: int | None = None,
) -> pd.DataFrame:
    """Meteo per città: Open-Meteo se FORECAST_WEATHER_API=1, altrimenti simulato."""
    if _use_weather_api():
        from production.connectors.weather_api import fetch_weather_window as api_fetch

        return api_fetch(country, city, start, end)
    rows = []
    d = start
    while d <= end:
        w = daily_weather(country, city, d)
        rows.append({"data": d, "country": country, "city": city, **w})
        d += timedelta(days=1)
    return pd.DataFrame(rows)


def attach_store_weather(
    df: pd.DataFrame,
    as_of: date,
    run_rate_window: int = 14,
    pdv_master: pd.DataFrame | None = None,
) -> pd.DataFrame:
    master = pdv_master if pdv_master is not None else load_pdv_master()
    out = df.merge(master, on=["pdv_id", "country"], how="left", suffixes=("", "_store"))

    if "data_apertura_store" in out.columns:
        out["data_apertura"] = out["data_apertura"].fillna(out["data_apertura_store"])
        out = out.drop(columns=["data_apertura_store"])

    if "mq_totali" not in out.columns or out["mq_totali"].isna().any():
        out["mq_totali"] = out["mq_espositivi"] + out["mq_stock"]

    w_start = as_of - timedelta(days=run_rate_window - 1)
    city_frames: dict[tuple[str, str], pd.DataFrame] = {}
    weather_parts = []
    for _, s in master.iterrows():
        key = (str(s["country"]), str(s["city"]))
        if key not in city_frames:
            city_frames[key] = fetch_weather_window(s["country"], s["city"], w_start, as_of)
        w = city_frames[key].copy()
        w["pdv_id"] = s["pdv_id"]
        weather_parts.append(w)
    weather = pd.concat(weather_parts, ignore_index=True)
    w_agg = weather.groupby("pdv_id", as_index=False).agg(
        temp_media=("temp_media", "mean"),
        giorni_pioggia=("is_pioggia", "sum"),
        pioggia_mm_tot=("pioggia_mm", "sum"),
    )
    hour_rows = []
    for _, s in master.iterrows():
        key = (str(s["country"]), str(s["city"]))
        daily = city_frames[key]
        if _use_weather_api():
            ht = _store_hour_temps_from_api(s["country"], s["city"], w_start, as_of, daily)
        else:
            ht = _store_hour_temps_from_daily(daily)
        hour_rows.append({"pdv_id": s["pdv_id"], **ht})
    w_hours = pd.DataFrame(hour_rows)
    out = out.merge(w_agg, on="pdv_id", how="left")
    out = out.merge(w_hours, on="pdv_id", how="left")
    out = out.merge(master[["pdv_id", "city"]], on="pdv_id", how="left", suffixes=("", "_pdv"))
    if "city_pdv" in out.columns:
        out["city"] = out["city"].fillna(out["city_pdv"])
        out = out.drop(columns=["city_pdv"])

    out["mq_espositivi_band"] = out["mq_espositivi"].apply(lambda x: mq_to_band(float(x))).astype(int)
    out["mq_stock_band"] = out["mq_stock"].apply(lambda x: mq_to_band(float(x))).astype(int)
    out["weather_regime"] = out.apply(
        lambda r: weather_to_regime(float(r["temp_media"]), int(r["giorni_pioggia"])),
        axis=1,
    ).astype(int)
    return out
