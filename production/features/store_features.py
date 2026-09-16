"""Bande mq + regime meteo (production)."""

from __future__ import annotations

import pandas as pd

MQ_BANDS = 4
WEATHER_REGIMES = 4

_ORA_PICCO = {
    "COLLANA": 16, "BRACCIALE": 16, "ORECCHINI": 16, "ANELLO": 16, "CHARM": 16, "CLIP": 16,
    "AVIATOR": 15, "BIKINI": 14, "PUFFER": 12, "COAT": 12, "PARKA": 12,
    "KNIT": 13, "CARDIGAN": 13, "BEANIE": 12, "SCARF": 12, "ANKLE": 13,
    "CROP": 15, "TSHIRT": 15, "TANK": 15, "SHORT": 15, "SKIRT": 15, "MINI": 15,
    "FLAT": 15, "LOW": 15, "HOODIE": 14, "JEANS": 16, "PANT": 16,
}


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


def ora_picco_vendita(df: pd.DataFrame) -> pd.Series:
    """Ora di vendita: media scontrini se c'è data_ora, altrimenti tipica per tipo SKU."""
    if "ora_media_14d" in df.columns and df["ora_media_14d"].notna().any():
        return df["ora_media_14d"].astype(float).fillna(15.0)
    if "ora" in df.columns and df["ora"].notna().any():
        return df["ora"].astype(float).fillna(15.0)
    if "ssclasse" in df.columns:
        return df["ssclasse"].astype(str).map(_ORA_PICCO).fillna(15.0).astype(float)
    return pd.Series(15.0, index=df.index)


def temp_at_sale_hour(temp_media: pd.Series, temp_pomeriggio: pd.Series, ora: pd.Series) -> pd.Series:
    t0 = temp_media.astype(float)
    t1 = temp_pomeriggio.astype(float)
    w = ((ora.astype(float) - 10.0) / 5.0).clip(lower=0.0, upper=1.2)
    return t0 + w * (t1 - t0)
