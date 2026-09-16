"""
Feature PDV (mq) + meteo per il modello.
- mq espositivi / stock: numerici + bande per embedding
- meteo: numerici + regime per embedding
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


# Master PDV allineato a data/manual/pdv_master.csv (riunione 11 ago)
STORE_MASTER = pd.DataFrame(
    [
        {
            "pdv_id": "MIL_01",
            "country": "IT",
            "city": "Milan",
            "commerciale_riferimento": "Rossi",
            "mq_espositivi": 120,
            "mq_stock": 40,
            "mq_totali": 160,
            "numero_fatturazioni": 320,
            "transit_time_days": 5,
            "data_apertura": date(2024, 3, 1),
            "pdv_strategico": 1,
        },
        {
            "pdv_id": "ROM_NEW",
            "country": "IT",
            "city": "Rome",
            "commerciale_riferimento": "Bianchi",
            "mq_espositivi": 90,
            "mq_stock": 30,
            "mq_totali": 120,
            "numero_fatturazioni": 40,
            "transit_time_days": 5,
            "data_apertura": date(2026, 7, 20),
            "pdv_strategico": 0,
        },
        {
            "pdv_id": "NAP_02",
            "country": "IT",
            "city": "Naples",
            "commerciale_riferimento": "Verdi",
            "mq_espositivi": 100,
            "mq_stock": 35,
            "mq_totali": 135,
            "numero_fatturazioni": 210,
            "transit_time_days": 6,
            "data_apertura": date(2023, 6, 1),
            "pdv_strategico": 0,
        },
    ]
)

# Bande mq per embedding (S/M/L/XL negozio o area stock)
MQ_BANDS = 4
WEATHER_REGIMES = 4  # 0=fresco 1=normale 2=caldo 3=piovoso


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
    if temp_media >= 30:
        return 2
    if temp_media <= 22:
        return 0
    return 1


def fetch_weather_window(country: str, city: str, start: date, end: date) -> pd.DataFrame:
    """Simula API meteo sulla finestra feature."""
    rows = []
    d = start
    while d <= end:
        is_rain = 1 if (city == "Milan" and d == date(2026, 8, 9)) else 0
        temp = 31.0 if city == "Rome" else (29.0 if city == "Naples" else 28.0)
        rows.append(
            {
                "data": d,
                "country": country,
                "city": city,
                "temp_media": temp - (2 if is_rain else 0),
                "pioggia_mm": 12.0 if is_rain else 0.0,
                "is_pioggia": is_rain,
            }
        )
        d += timedelta(days=1)
    return pd.DataFrame(rows)


def attach_store_weather(
    df: pd.DataFrame,
    as_of: date,
    run_rate_window: int = 14,
) -> pd.DataFrame:
    """Join master PDV + aggregato meteo finestra + bande per embedding."""
    out = df.merge(STORE_MASTER, on=["pdv_id", "country"], how="left", suffixes=("", "_store"))

    if "data_apertura_store" in out.columns:
        out["data_apertura"] = out["data_apertura"].fillna(out["data_apertura_store"])
        out = out.drop(columns=["data_apertura_store"])

    if "mq_totali" not in out.columns or out["mq_totali"].isna().any():
        out["mq_totali"] = out["mq_espositivi"] + out["mq_stock"]

    w_start = as_of - timedelta(days=run_rate_window - 1)
    weather_parts = []
    for _, s in STORE_MASTER.iterrows():
        w = fetch_weather_window(s["country"], s["city"], w_start, as_of)
        w["pdv_id"] = s["pdv_id"]
        weather_parts.append(w)
    weather = pd.concat(weather_parts, ignore_index=True)
    w_agg = weather.groupby("pdv_id", as_index=False).agg(
        temp_media=("temp_media", "mean"),
        giorni_pioggia=("is_pioggia", "sum"),
        pioggia_mm_tot=("pioggia_mm", "sum"),
    )
    out = out.merge(w_agg, on="pdv_id", how="left")

    out["mq_espositivi_band"] = out["mq_espositivi"].apply(lambda x: mq_to_band(float(x))).astype(int)
    out["mq_stock_band"] = out["mq_stock"].apply(lambda x: mq_to_band(float(x))).astype(int)
    out["weather_regime"] = out.apply(
        lambda r: weather_to_regime(float(r["temp_media"]), int(r["giorni_pioggia"])),
        axis=1,
    ).astype(int)

    return out
