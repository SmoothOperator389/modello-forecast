"""Profilo meteo per SKU imparato dalle vendite — niente tag ESTATE/INVERNO.

Per ogni SKU, prima di as_of: temperatura media ponderata delle vendite,
quota pezzi venduti al caldo/freddo, e quanto il meteo attuale «fitta»
quel profilo. Causale: usa solo scontrini con data < as_of.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from testing.features.store_features import fetch_weather_window

HOT_C = 22.0
COLD_C = 12.0
NEUTRAL_C = 18.0
MIN_SALE_DAYS = 8
MIN_UNITS = 8.0


def _daily_store_weather(master: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    city_wx: dict[tuple[str, str], pd.DataFrame] = {}
    parts: list[pd.DataFrame] = []
    for _, s in master.iterrows():
        key = (str(s["country"]), str(s["city"]))
        if key not in city_wx:
            city_wx[key] = fetch_weather_window(s["country"], s["city"], start, end)
        w = city_wx[key][["data", "temp_media"]].copy()
        w["pdv_id"] = s["pdv_id"]
        parts.append(w)
    if not parts:
        return pd.DataFrame(columns=["data", "pdv_id", "temp_media"])
    return pd.concat(parts, ignore_index=True)


def build_sku_sale_weather_cumsum(fact: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Cumulati giornalieri SKU×data (qty, qty×temp, qty caldo/freddo)."""
    cols = ["sku_id", "data", "cum_qty", "cum_qty_x_temp", "cum_qty_hot", "cum_qty_cold", "n_sale_days"]
    if fact.empty:
        return pd.DataFrame(columns=cols)
    start = min(fact["data"])
    end = max(fact["data"])
    wx = _daily_store_weather(master, start, end)
    sales = fact.loc[fact["vendite_qty"].astype(float) > 0, ["data", "pdv_id", "sku_id", "vendite_qty"]].copy()
    if sales.empty:
        return pd.DataFrame(columns=cols)
    sales = sales.merge(wx, on=["data", "pdv_id"], how="left")
    sales["temp_media"] = sales["temp_media"].fillna(NEUTRAL_C).astype(float)
    sales["qty"] = sales["vendite_qty"].astype(float)
    sales["qty_x_temp"] = sales["qty"] * sales["temp_media"]
    sales["qty_hot"] = np.where(sales["temp_media"] >= HOT_C, sales["qty"], 0.0)
    sales["qty_cold"] = np.where(sales["temp_media"] <= COLD_C, sales["qty"], 0.0)
    daily = sales.groupby(["sku_id", "data"], as_index=False).agg(
        qty=("qty", "sum"),
        qty_x_temp=("qty_x_temp", "sum"),
        qty_hot=("qty_hot", "sum"),
        qty_cold=("qty_cold", "sum"),
    )
    daily = daily.sort_values(["sku_id", "data"])
    g = daily.groupby("sku_id", sort=False)
    daily["cum_qty"] = g["qty"].cumsum()
    daily["cum_qty_x_temp"] = g["qty_x_temp"].cumsum()
    daily["cum_qty_hot"] = g["qty_hot"].cumsum()
    daily["cum_qty_cold"] = g["qty_cold"].cumsum()
    daily["n_sale_days"] = g.cumcount() + 1
    return daily[cols]


def attach_sku_weather_response(df: pd.DataFrame, cumsum: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Aggiunge il profilo meteo dello SKU visto fino a ieri (as_of escluso)."""
    out = df.copy()
    defaults = {
        "sku_temp_vendita": NEUTRAL_C,
        "sku_share_hot": 0.0,
        "sku_share_cold": 0.0,
        "sku_temp_gap": 0.0,
        "sku_temp_fit": 0.0,
        "run_rate_x_sku_temp": 0.0,
        "sku_meteo_giorni": 0.0,
    }
    if cumsum.empty or "sku_id" not in out.columns:
        for col, val in defaults.items():
            out[col] = val
        return out

    hist = cumsum.loc[cumsum["data"] < as_of]
    if hist.empty:
        for col, val in defaults.items():
            out[col] = val
        return out

    last = hist.sort_values("data").groupby("sku_id", as_index=False).tail(1)
    last = last.rename(
        columns={
            "cum_qty": "_wx_qty",
            "cum_qty_x_temp": "_wx_qty_t",
            "cum_qty_hot": "_wx_hot",
            "cum_qty_cold": "_wx_cold",
            "n_sale_days": "_wx_days",
        }
    )
    out = out.merge(
        last[["sku_id", "_wx_qty", "_wx_qty_t", "_wx_hot", "_wx_cold", "_wx_days"]],
        on="sku_id",
        how="left",
    )
    qty = out["_wx_qty"].fillna(0.0).astype(float)
    days = out["_wx_days"].fillna(0.0).astype(float)
    weak = (days < MIN_SALE_DAYS) | (qty < MIN_UNITS)
    safe_qty = qty.clip(lower=1e-6)
    temp_v = np.where(qty > 0, out["_wx_qty_t"].fillna(0.0) / safe_qty, NEUTRAL_C)
    share_hot = np.where(qty > 0, out["_wx_hot"].fillna(0.0) / safe_qty, 0.0)
    share_cold = np.where(qty > 0, out["_wx_cold"].fillna(0.0) / safe_qty, 0.0)
    temp_v = np.where(weak, NEUTRAL_C, temp_v)
    share_hot = np.where(weak, 0.0, share_hot)
    share_cold = np.where(weak, 0.0, share_cold)
    store_t = out["temp_media"].astype(float) if "temp_media" in out.columns else NEUTRAL_C
    affinity = share_hot - share_cold
    temp_dev = (store_t - NEUTRAL_C) / 10.0
    sku_temp_fit = np.where(weak, 0.0, affinity * temp_dev)
    out["sku_temp_vendita"] = temp_v
    out["sku_share_hot"] = share_hot
    out["sku_share_cold"] = share_cold
    out["sku_temp_gap"] = temp_v - store_t
    out["sku_temp_fit"] = sku_temp_fit
    rr = out["run_rate_14d"].astype(float) if "run_rate_14d" in out.columns else 0.0
    out["run_rate_x_sku_temp"] = rr * sku_temp_fit
    out["sku_meteo_giorni"] = days
    return out.drop(columns=["_wx_qty", "_wx_qty_t", "_wx_hot", "_wx_cold", "_wx_days"], errors="ignore")


SEASON_CALDO = "tende al caldo"
SEASON_FREDDO = "tende al freddo"
SEASON_FLAT = "tutto l'anno"
SEASON_THIN = "poco storico"
AFFINITY_TH = 0.12


def season_labels_from_profile(
    share_hot: pd.Series,
    share_cold: pd.Series,
    n_days: pd.Series | None = None,
) -> pd.Series:
    """Etichetta leggibile: profilo imparato dalle vendite, non tag catalogo."""
    h = pd.to_numeric(share_hot, errors="coerce").fillna(0.0)
    c = pd.to_numeric(share_cold, errors="coerce").fillna(0.0)
    aff = h - c
    out = pd.Series(SEASON_FLAT, index=h.index, dtype="object")
    out = out.mask(aff >= AFFINITY_TH, SEASON_CALDO)
    out = out.mask(aff <= -AFFINITY_TH, SEASON_FREDDO)
    if n_days is not None:
        d = pd.to_numeric(n_days, errors="coerce").fillna(0.0)
        out = out.mask(d < MIN_SALE_DAYS, SEASON_THIN)
    return out


def add_season_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    days = out["sku_meteo_giorni"] if "sku_meteo_giorni" in out.columns else None
    if "sku_share_hot" not in out.columns or "sku_share_cold" not in out.columns:
        out["stagione_stimata"] = SEASON_THIN
        return out
    out["stagione_stimata"] = season_labels_from_profile(
        out["sku_share_hot"],
        out["sku_share_cold"],
        days,
    )
    return out


def write_sku_season_lookup(
    as_of: date,
    data_dir=None,
    fact: pd.DataFrame | None = None,
    master: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Una riga per SKU: profilo meteo e stagione stimata (causale, data < as_of)."""
    from pathlib import Path

    from testing.config import DATA_DIR
    from testing.features.store_features import load_pdv_master

    root = Path(data_dir) if data_dir is not None else DATA_DIR
    if master is None:
        master = load_pdv_master(root / "pdv_master.csv")
    if fact is None:
        fact_path = root / "fact_daily.csv"
        fact = pd.read_csv(
            fact_path,
            usecols=["data", "pdv_id", "sku_id", "vendite_qty"],
            parse_dates=["data"],
        )
        fact["data"] = pd.to_datetime(fact["data"]).dt.date
    cumsum = build_sku_sale_weather_cumsum(fact, master)
    skus = pd.DataFrame({"sku_id": fact["sku_id"].drop_duplicates().astype(str)})
    skus["temp_media"] = NEUTRAL_C
    skus["run_rate_14d"] = 0.0
    prof = attach_sku_weather_response(skus, cumsum, as_of)
    prof = add_season_labels(prof)
    cols = [
        "sku_id",
        "sku_share_hot",
        "sku_share_cold",
        "sku_temp_vendita",
        "sku_meteo_giorni",
        "stagione_stimata",
    ]
    out = prof[cols].copy()
    out["sku_share_hot"] = out["sku_share_hot"].astype(float).round(3)
    out["sku_share_cold"] = out["sku_share_cold"].astype(float).round(3)
    out["sku_temp_vendita"] = out["sku_temp_vendita"].astype(float).round(2)
    out["sku_meteo_giorni"] = out["sku_meteo_giorni"].astype(float).round(0)
    path = root / "sku_meteo_profilo.csv"
    out.to_csv(path, index=False)
    return out
