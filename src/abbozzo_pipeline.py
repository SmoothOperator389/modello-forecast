"""
abbozzo_pipeline.py
Fonti: DB + manuale + API (meteo) + calendario → feature → forecast → operativo
Demo eseguibile con dati finti (MIL_01 maturo, ROM_NEW nuovo, SKU_A caldo, SKU_Z nuovo).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from store_features import MQ_BANDS, STORE_MASTER, WEATHER_REGIMES, attach_store_weather


# =============================================================================
# 0) CONFIG
# =============================================================================

@dataclass
class RunConfig:
    as_of: date
    horizon_sku_days: int = 15
    horizon_cat_days: int = 30
    run_rate_window: int = 14


# =============================================================================
# 1) CONTRATTI FONTI
# =============================================================================

class SalesStockSource(Protocol):
    def fetch_fact(self, start: date, end: date) -> pd.DataFrame: ...


class StoreMasterSource(Protocol):
    def fetch_stores(self) -> pd.DataFrame: ...


class ProductMasterSource(Protocol):
    def fetch_products(self) -> pd.DataFrame: ...


class WeatherSource(Protocol):
    def fetch_weather(self, country: str, city: str, start: date, end: date) -> pd.DataFrame: ...


class CalendarSource(Protocol):
    """Saldi / festivi / stagione — può arrivare da DB, file manuale o API."""
    def fetch_calendar(self, country: str, start: date, end: date) -> pd.DataFrame: ...


# =============================================================================
# 2) FONTI DEMO (simulano DB / manuale / API)
# =============================================================================

class FakeDemoSources:
    """
    In produzione spezzeresti così:
      - DatabaseSource     → Valido (vendite, stock, prezzi, anagrafica SKU)
      - ManualStoreFile    → CSV team (mq, apertura, transit, strategico)
      - WeatherAPI         → provider meteo
      - CalendarSource     → DB corporate OPPURE file saldi OPPURE API festività
    Qui tutto è fake per far girare la pipeline.
    """

    def fetch_fact(self, start: date, end: date) -> pd.DataFrame:
        # Per run_rate realistico: inventiamo anche giorni precedenti nella finestra 14gg
        rows = []
        # MIL_01 + SKU_A: ritmo ~2/giorno
        d = start
        while d <= end:
            # weekend un po' più alti
            base = 3 if d.weekday() >= 5 else 2
            rows.append((d, "IT", "MIL_01", "SKU_A", "MAG", base, 20 - (end - d).days, 29.99))
            d += timedelta(days=1)

        # ROM_NEW + SKU_A: PDV nuovo, ritmo basso ~0.5/giorno
        d = start
        stock = 28
        while d <= end:
            qty = 1 if d.day % 2 == 0 else 0
            stock = max(0, stock - qty)
            rows.append((d, "IT", "ROM_NEW", "SKU_A", "MAG", qty, stock, 29.99))
            d += timedelta(days=1)

        # MIL_01 + SKU_Z: brand new, 0 vendite, presente solo da data lancio 8 ago
        launch_z = date(2026, 8, 8)
        d = max(start, launch_z)
        while d <= end:
            rows.append((d, "IT", "MIL_01", "SKU_Z", "MAG", 0, 15, 23.99))
            d += timedelta(days=1)

        df = pd.DataFrame(
            rows,
            columns=[
                "data", "country", "pdv_id", "sku_id", "categoria_id",
                "vendite_qty", "giacenza_qty", "prezzo_pagato",
            ],
        )
        return df.copy()

    def fetch_stores(self) -> pd.DataFrame:
        # Fonte tipicamente MANUALE / file interno (allineato a data/manual/pdv_master.csv)
        return STORE_MASTER.copy()

    def fetch_products(self) -> pd.DataFrame:
        # Fonte tipicamente DB (anagrafica) + qualche campo manuale
        return pd.DataFrame(
            [
                {
                    "sku_id": "SKU_A",
                    "categoria_id": "MAG",
                    "colore": "Nero",
                    "modello": "Mod_A",
                    "data_lancio_sku": date(2026, 2, 1),
                },
                {
                    "sku_id": "SKU_Z",
                    "categoria_id": "MAG",
                    "colore": "Blu",
                    "modello": "Mod_Z",
                    "data_lancio_sku": date(2026, 8, 8),
                },
            ]
        )

    def fetch_weather(self, country: str, city: str, start: date, end: date) -> pd.DataFrame:
        # Fonte API meteo
        days = pd.date_range(start, end, freq="D")
        # Roma un po' più calda; un giorno di pioggia inventato a Milano
        rows = []
        for d in days:
            dd = d.date()
            is_rain = 1 if (city == "Milan" and dd == date(2026, 8, 9)) else 0
            temp = 31.0 if city == "Rome" else 28.0
            rows.append(
                {
                    "data": dd,
                    "country": country,
                    "city": city,
                    "temp_media": temp - (2 if is_rain else 0),
                    "pioggia_mm": 12.0 if is_rain else 0.0,
                    "is_pioggia": is_rain,
                }
            )
        return pd.DataFrame(rows)

    def fetch_calendar(self, country: str, start: date, end: date) -> pd.DataFrame:
        """
        Calendario commerciale per country.
        In reale:
          - DB HR/corporate per festivi
          - file manuale per finestre saldi (diversi per country)
          - oppure API festività pubbliche + override manuale saldi
        """
        # IT demo: ferragosto festivo; ipotetico mid-season flag off
        festivi = {date(2026, 8, 15)}  # Ferragosto (fuori finestra run ma ok)
        # nella nostra finestra 28 lug–10 ago: nessun festivo nazionale tipico
        # segniamo comunque struttura colonne
        days = pd.date_range(start, end, freq="D")
        rows = []
        for d in days:
            dd = d.date()
            week = int(dd.strftime("%W"))
            month = dd.month
            # ciclicità annuale
            angle = 2 * np.pi * week / 52.0
            rows.append(
                {
                    "data": dd,
                    "country": country,
                    "mese": month,
                    "settimana_anno": week,
                    "sin_anno": float(np.sin(angle)),
                    "cos_anno": float(np.cos(angle)),
                    "is_festivo": int(dd in festivi),
                    "is_saldi": 0,  # estate: post-saldi in questo esempio
                    "is_mid_season": 0,
                    "giorni_da_inizio_collezione": (dd - date(2026, 7, 1)).days,
                }
            )
        return pd.DataFrame(rows)


# =============================================================================
# 3) FEATURE BUILDER
# =============================================================================

def build_feature_table(cfg: RunConfig, sources: FakeDemoSources) -> pd.DataFrame:
    start = cfg.as_of - timedelta(days=cfg.run_rate_window - 1)
    fact = sources.fetch_fact(start, cfg.as_of)
    products = sources.fetch_products()

    cal_parts = []
    for country in STORE_MASTER["country"].unique():
        cal_parts.append(sources.fetch_calendar(country, start, cfg.as_of))
    calendar = pd.concat(cal_parts, ignore_index=True)

    g = (
        fact.groupby(["country", "pdv_id", "sku_id", "categoria_id"], as_index=False)
        .agg(
            vendite_window=("vendite_qty", "sum"),
            giorni_con_vendita=("vendite_qty", lambda s: int((s > 0).sum())),
            giacenza=("giacenza_qty", "last"),
            prezzo_pagato=("prezzo_pagato", "last"),
        )
    )
    g["run_rate_14d"] = g["vendite_window"] / float(cfg.run_rate_window)
    g["has_history"] = (g["giorni_con_vendita"] > 0).astype(int)

    # PDV (mq espositivi, mq stock) + meteo API + bande per embedding
    feat = attach_store_weather(g, cfg.as_of, cfg.run_rate_window)
    feat = feat.merge(products, on=["sku_id", "categoria_id"], how="left")

    # calendario: usiamo lo stato al giorno as_of (+ medie utili)
    cal_asof = calendar[calendar["data"] == cfg.as_of][
        [
            "country",
            "mese",
            "settimana_anno",
            "sin_anno",
            "cos_anno",
            "is_festivo",
            "is_saldi",
            "is_mid_season",
            "giorni_da_inizio_collezione",
        ]
    ]
    feat = feat.merge(cal_asof, on="country", how="left")

    cal_agg = calendar.groupby("country", as_index=False).agg(
        festivi_in_finestra=("is_festivo", "sum"),
        giorni_saldi_in_finestra=("is_saldi", "sum"),
    )
    feat = feat.merge(cal_agg, on="country", how="left")

    # maturità continua
    feat["giorni_da_lancio"] = feat["data_lancio_sku"].apply(lambda d: (cfg.as_of - d).days)
    feat["pdv_age_days"] = feat["data_apertura"].apply(lambda d: (cfg.as_of - d).days)
    feat["is_new_pdv"] = (feat["pdv_age_days"] < 60).astype(int)
    feat["as_of"] = cfg.as_of

    feat["fonte_fact"] = "DB(sim)"
    feat["fonte_pdv"] = "MANUALE(sim)"
    feat["fonte_meteo"] = "API(sim)"
    feat["fonte_calendario"] = "CALENDARIO(sim: DB/file/API)"

    return feat


# =============================================================================
# 4) MODELLO (abbozzo PyTorch — non addestrato in questa demo)
# =============================================================================

class DualHeadForecast(nn.Module):
    def __init__(self, n_numeric: int, n_country: int, n_pdv: int, n_sku: int, n_cat: int):
        super().__init__()
        self.emb_country = nn.Embedding(n_country, 4)
        self.emb_pdv = nn.Embedding(n_pdv, 4)
        self.emb_sku = nn.Embedding(n_sku, 6)
        self.emb_cat = nn.Embedding(n_cat, 4)
        self.emb_mq_espo = nn.Embedding(MQ_BANDS, 3)
        self.emb_mq_stock = nn.Embedding(MQ_BANDS, 3)
        self.emb_weather = nn.Embedding(WEATHER_REGIMES, 3)
        in_dim = 4 + 4 + 6 + 4 + 3 + 3 + 3 + n_numeric
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.head_sku = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus())
        self.head_cat = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus())

    def forward(self, country, pdv, sku, cat, mq_espo_b, mq_stock_b, weather_reg, numeric):
        x = torch.cat(
            [
                self.emb_country(country),
                self.emb_pdv(pdv),
                self.emb_sku(sku),
                self.emb_cat(cat),
                self.emb_mq_espo(mq_espo_b),
                self.emb_mq_stock(mq_stock_b),
                self.emb_weather(weather_reg),
                numeric,
            ],
            dim=1,
        )
        h = self.encoder(x)
        return self.head_sku(h).squeeze(1), self.head_cat(h).squeeze(1)


def build_untrained_model_summary() -> str:
    """Mostra che il pezzo NN esiste; la demo usa ancora placeholder leggibile."""
    m = DualHeadForecast(n_numeric=10, n_country=5, n_pdv=20, n_sku=50, n_cat=10)
    n = sum(p.numel() for p in m.parameters())
    return f"DualHeadForecast creato (non trainato in demo) — parametri: {n}"


# =============================================================================
# 5) LAYER OPERATIVO
# =============================================================================

def suggest_replenishment(
    forecast_sku: float,
    giacenza: float,
    transit_days: float,
    horizon_days: int,
    is_immature: bool,
) -> dict:
    daily = forecast_sku / max(horizon_days, 1)
    need = forecast_sku + daily * transit_days
    raw = max(0.0, need - giacenza)
    if is_immature:
        raw *= 0.5
    qty = float(np.rint(raw))
    return {
        "vendite_previste_15gg": round(forecast_sku, 1),
        "giacenza": giacenza,
        "qty_suggerita": qty,
        "azione": "monitora" if qty == 0 else "valuta_riassortimento",
    }


def placeholder_forecast(row: pd.Series, horizon_days: int) -> float:
    """
    Sostituto temporaneo della rete (solo per vedere i 3 casi didattici).
    Poi: DualHeadForecast.forward(...)
    """
    base = float(row["run_rate_14d"]) * horizon_days
    if int(row["has_history"]) == 0:
        base = 12.0  # cold SKU: stima cauta da simili/categoria
    if int(row["is_new_pdv"]) == 1:
        base *= 0.5  # PDV nuovo
    # piccolo aggiustamento meteo didattico
    if float(row.get("giorni_pioggia", 0)) >= 1:
        base *= 0.95
    return float(base)


# =============================================================================
# 6) RUN
# =============================================================================

def demo_run() -> None:
    cfg = RunConfig(as_of=date(2026, 8, 10))
    sources = FakeDemoSources()
    feat = build_feature_table(cfg, sources)

    print(build_untrained_model_summary())
    print()
    print("=== FONTI (concetto) ===")
    print("fact/vendite     → DB (sim)")
    print("anagrafica PDV   → MANUALE/file (sim)")
    print("meteo            → API (sim)")
    print("calendario       → DB o file saldi o API festivi (sim)")
    print()

    print("=== FEATURE TABLE ===")
    cols = [
        "pdv_id",
        "sku_id",
        "mq_espositivi",
        "mq_stock",
        "mq_totali",
        "mq_espositivi_band",
        "mq_stock_band",
        "weather_regime",
        "temp_media",
        "giorni_pioggia",
        "run_rate_14d",
        "has_history",
        "giorni_da_lancio",
        "pdv_age_days",
        "is_new_pdv",
        "giacenza",
        "mese",
        "settimana_anno",
        "sin_anno",
        "cos_anno",
        "is_saldi",
        "is_festivo",
        "giorni_da_inizio_collezione",
        "transit_time_days",
    ]
    out = feat[cols].copy()
    out["run_rate_14d"] = out["run_rate_14d"].round(3)
    out["temp_media"] = out["temp_media"].round(2)
    out["sin_anno"] = out["sin_anno"].round(3)
    out["cos_anno"] = out["cos_anno"].round(3)
    print(out.to_string(index=False))

    outputs = []
    for _, r in feat.iterrows():
        pred = placeholder_forecast(r, cfg.horizon_sku_days)
        immature = (int(r["has_history"]) == 0) or (int(r["is_new_pdv"]) == 1)
        op = suggest_replenishment(
            forecast_sku=pred,
            giacenza=float(r["giacenza"]),
            transit_days=float(r["transit_time_days"]),
            horizon_days=cfg.horizon_sku_days,
            is_immature=bool(immature),
        )
        outputs.append(
            {
                "pdv_id": r["pdv_id"],
                "sku_id": r["sku_id"],
                "caso": (
                    "SKU_nuovo"
                    if int(r["has_history"]) == 0
                    else ("PDV_nuovo" if int(r["is_new_pdv"]) == 1 else "caldo")
                ),
                **op,
            }
        )

    print()
    print("=== OUTPUT (forecast + operativo, stessa importanza) ===")
    print(pd.DataFrame(outputs).to_string(index=False))


if __name__ == "__main__":
    demo_run()
