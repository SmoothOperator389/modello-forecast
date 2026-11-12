"""Feature YoY / seasonality per LGBM h30 (allineate allo spirito tip_budget).

Aggiunge colonne numeriche causali (niente leakage):
- yoy_rr14: run_rate_14d allo stesso giorno dell'anno precedente (se disponibile)
- yoy_ratio: rr14 / max(yoy_rr14, eps)
- season_sin / season_cos: già tipicamente in samples come sin_anno/cos_anno;
  qui si garantiscono alias stabili
- month_oh: mese 1–12 (già spesso presente come ``mese``)

Usare in rebuild samples / train h30::

    from production.ops.h30_yoy_features import attach_yoy_features, YOY_NUMERIC_COLS
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

YOY_NUMERIC_COLS = [
    "yoy_rr14",
    "yoy_ratio",
    "yoy_has_hist",
    "season_sin",
    "season_cos",
    "month_num",
]


def _ensure_as_of_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.date


def attach_yoy_features(
    samples: pd.DataFrame,
    fact: pd.DataFrame | None = None,
    *,
    window: int = 14,
) -> pd.DataFrame:
    """Attacca YoY run-rate 14d per pdv×sku rispetto a as_of − 365g.

    Se ``fact`` è None, prova a riusare colonne già presenti o mette 0.
    """
    out = samples.copy()
    out["as_of"] = _ensure_as_of_date(out["as_of"])
    out["pdv_id"] = out["pdv_id"].astype(str)
    out["sku_id"] = out["sku_id"].astype(str)

    if "sin_anno" in out.columns:
        out["season_sin"] = out["sin_anno"].astype(float)
    else:
        # Fourier da mese se manca
        m = out["mese"].astype(float) if "mese" in out.columns else out["as_of"].map(lambda d: float(d.month))
        out["season_sin"] = np.sin(2 * np.pi * (m - 1) / 12.0)
    if "cos_anno" in out.columns:
        out["season_cos"] = out["cos_anno"].astype(float)
    else:
        m = out["mese"].astype(float) if "mese" in out.columns else out["as_of"].map(lambda d: float(d.month))
        out["season_cos"] = np.cos(2 * np.pi * (m - 1) / 12.0)

    if "mese" in out.columns:
        out["month_num"] = out["mese"].astype(float)
    else:
        out["month_num"] = out["as_of"].map(lambda d: float(d.month))

    if fact is None or fact.empty:
        out["yoy_rr14"] = 0.0
        out["yoy_ratio"] = 1.0
        out["yoy_has_hist"] = 0.0
        return out

    fact = fact[["data", "pdv_id", "sku_id", "vendite_qty"]].copy()
    fact["data"] = pd.to_datetime(fact["data"]).dt.date
    fact["pdv_id"] = fact["pdv_id"].astype(str)
    fact["sku_id"] = fact["sku_id"].astype(str)

    # Chiavi uniche as_of → finestra YoY
    as_ofs = sorted(out["as_of"].unique())
    parts = []
    for a in as_ofs:
        a_yoy = a - timedelta(days=365)
        c0, c1 = a_yoy - timedelta(days=window - 1), a_yoy
        if c1 < min(fact["data"]):
            continue
        sub = fact[(fact["data"] >= c0) & (fact["data"] <= c1)]
        if sub.empty:
            continue
        g = sub.groupby(["pdv_id", "sku_id"], as_index=False, observed=True).agg(
            yoy_vendite=("vendite_qty", "sum")
        )
        g["yoy_rr14"] = g["yoy_vendite"] / float(window)
        g["as_of"] = a
        parts.append(g[["pdv_id", "sku_id", "as_of", "yoy_rr14"]])

    if not parts:
        out["yoy_rr14"] = 0.0
        out["yoy_ratio"] = 1.0
        out["yoy_has_hist"] = 0.0
        return out

    yoy = pd.concat(parts, ignore_index=True)
    out = out.merge(yoy, on=["pdv_id", "sku_id", "as_of"], how="left")
    out["yoy_rr14"] = out["yoy_rr14"].fillna(0.0).astype(float)
    out["yoy_has_hist"] = (out["yoy_rr14"] > 0).astype(float)
    rr = out["run_rate_14d"].astype(float) if "run_rate_14d" in out.columns else 0.0
    out["yoy_ratio"] = (rr / np.maximum(out["yoy_rr14"], 1e-3)).clip(0.05, 20.0)
    out.loc[out["yoy_has_hist"] < 0.5, "yoy_ratio"] = 1.0
    return out
