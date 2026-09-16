"""Feature engineering PRODUCTION."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from production.connectors.pdv_master import load_pdv_master
from production.connectors.weather_api import fetch_weather_for_pdv_master
from production.features.store_features import (
    MQ_BANDS,
    WEATHER_REGIMES,
    mq_to_band,
    ora_picco_vendita,
    temp_at_sale_hour,
    weather_to_regime,
)

SCONTO_SOGLIA_PCT = 5.0

NUMERIC_COLS = [
    "run_rate_14d",
    "has_history",
    "giorni_da_lancio",
    "pdv_age_days",
    "is_new_pdv",
    "giacenza",
    "prezzo_listino",
    "prezzo_pagato",
    "rapporto_prezzo_pagato",
    "is_in_sconto",
    "sconto_pct",
    "sconto_pct_medio_14d",
    "giorni_in_sconto_14d",
    "sconto_impatto_stimato",
    "mq_espositivi",
    "mq_stock",
    "mq_totali",
    "numero_fatturazioni",
    "temp_media",
    "giorni_pioggia",
    "temp_orario_negozio",
    "temp_pomeriggio",
    "ora_picco",
    "temp_ora_vendita",
    "mese",
    "sin_anno",
    "cos_anno",
    "is_festivo",
    "is_saldi",
    "giorni_da_inizio_collezione",
]


def _ensure_discount_columns(fact: pd.DataFrame) -> pd.DataFrame:
    out = fact.copy()
    if "prezzo_listino" not in out.columns:
        if "prezzo_pubblico" in out.columns:
            out["prezzo_listino"] = out["prezzo_pubblico"].astype(float)
        elif "prezzo_pagato" in out.columns:
            out["prezzo_listino"] = out["prezzo_pagato"].astype(float)
        else:
            out["prezzo_listino"] = 0.0
    if "prezzo_pagato" not in out.columns:
        if "pagato_effettivo" in out.columns:
            out["prezzo_pagato"] = out["pagato_effettivo"].astype(float)
        else:
            out["prezzo_pagato"] = out["prezzo_listino"]
    if "sconto_pct" not in out.columns:
        listino = out["prezzo_listino"].astype(float).clip(lower=0.01)
        pagato = out["prezzo_pagato"].astype(float)
        out["sconto_pct"] = ((1.0 - pagato / listino) * 100.0).clip(0.0, 90.0)
    if "is_in_sconto" not in out.columns:
        out["is_in_sconto"] = (out["sconto_pct"] >= SCONTO_SOGLIA_PCT).astype(int)
    return out


def build_feature_table(
    fact: pd.DataFrame,
    calendar: pd.DataFrame,
    as_of: date,
    run_rate_window: int = 14,
    pdv_master: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Aggrega vendite reali fino a as_of → una riga per country×pdv×sku.
    Rolling: usa SOLO dati reali, mai forecast passati.
    """
    master = pdv_master if pdv_master is not None else load_pdv_master()
    w_start = as_of - timedelta(days=run_rate_window - 1)
    hist = _ensure_discount_columns(fact[(fact["data"] >= w_start) & (fact["data"] <= as_of)])
    if "data_ora" in hist.columns:
        hist = hist.copy()
        ts = pd.to_datetime(hist["data_ora"])
        hist["ora"] = ts.dt.hour + ts.dt.minute / 60.0

    agg_kw = dict(
        vendite_window=("vendite_qty", "sum"),
        giorni_con_vendita=("vendite_qty", lambda s: int((s > 0).sum())),
        giacenza=("giacenza_qty", "last"),
        prezzo_listino=("prezzo_listino", "last"),
        prezzo_pagato=("prezzo_pagato", "last"),
        sconto_pct=("sconto_pct", "last"),
        sconto_pct_medio_14d=("sconto_pct", "mean"),
        giorni_in_sconto_14d=("is_in_sconto", "mean"),
        data_lancio_sku=("data_lancio_sku", "last"),
    )
    if "ora" in hist.columns:
        agg_kw["ora_media_14d"] = ("ora", "mean")
    if "ssclasse" in hist.columns:
        agg_kw["ssclasse"] = ("ssclasse", "last")

    g = hist.groupby(["country", "pdv_id", "sku_id", "categoria_id"], as_index=False).agg(**agg_kw)
    g["run_rate_14d"] = g["vendite_window"] / float(run_rate_window)
    g["has_history"] = (g["giorni_con_vendita"] > 0).astype(float)
    g["giorni_da_lancio"] = g["data_lancio_sku"].apply(lambda d: (as_of - d).days)

    g["sconto_pct"] = g["sconto_pct"].fillna(0.0)
    g["sconto_pct_medio_14d"] = g["sconto_pct_medio_14d"].fillna(0.0)
    g["giorni_in_sconto_14d"] = g["giorni_in_sconto_14d"].fillna(0.0)
    g["rapporto_prezzo_pagato"] = (g["prezzo_pagato"] / g["prezzo_listino"].clip(lower=0.01)).clip(0.1, 1.0)
    g["is_in_sconto"] = (g["sconto_pct"] >= SCONTO_SOGLIA_PCT).astype(float)
    g["sconto_impatto_stimato"] = g["run_rate_14d"] * (g["sconto_pct_medio_14d"] / 100.0)

    feat = g.merge(master, on=["pdv_id", "country"], how="left")
    feat["pdv_age_days"] = feat["data_apertura"].apply(lambda d: (as_of - d).days)
    feat["is_new_pdv"] = (feat["pdv_age_days"] < 60).astype(float)

    cal = calendar[calendar["data"] == as_of]
    feat = feat.merge(
        cal[["country", "mese", "sin_anno", "cos_anno", "is_festivo", "is_saldi", "giorni_da_inizio_collezione"]],
        on="country",
        how="left",
    )

    w_agg = fetch_weather_for_pdv_master(as_of, run_rate_window, master)
    feat = feat.merge(w_agg, on="pdv_id", how="left")
    feat["mq_espositivi_band"] = feat["mq_espositivi"].apply(mq_to_band).astype(int)
    feat["mq_stock_band"] = feat["mq_stock"].apply(mq_to_band).astype(int)
    feat["weather_regime"] = feat.apply(
        lambda r: weather_to_regime(float(r["temp_media"]), int(r["giorni_pioggia"])),
        axis=1,
    ).astype(int)
    if "temp_pomeriggio" not in feat.columns:
        feat["temp_pomeriggio"] = feat["temp_media"].astype(float) + 3.5
    if "temp_orario_negozio" not in feat.columns:
        feat["temp_orario_negozio"] = feat["temp_media"].astype(float) + 1.2
    feat["ora_picco"] = ora_picco_vendita(feat)
    feat["temp_ora_vendita"] = temp_at_sale_hour(
        feat["temp_media"], feat["temp_pomeriggio"], feat["ora_picco"]
    )
    feat["as_of"] = as_of
    return feat
