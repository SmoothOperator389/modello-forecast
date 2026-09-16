"""Modello originale CDM / Subdued (regole) — baseline da battere sempre.

Ricostruzione del documento CDM:
- coeff stagionali mese su mese (mediana + cap)
- 3 cluster performance fissi 0.70 / 1.00 / 1.30
- proxy country/PDV (nuovo negozio = mediana country×categoria)
- maturità a stati discreti (nuovo / provvisorio / stabile)
- base = run rate 14 giorni × orizzonte
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERF_LOW = 0.70
PERF_MID = 1.00
PERF_HIGH = 1.30
PERF_P40 = 0.40
PERF_P70 = 0.70
SEASONAL_CAP = (0.50, 2.00)

# Proxy country del documento (es. IE→UK). Estendere quando Valido ha altri remap.
COUNTRY_PROXY: dict[str, str] = {
    "IE": "UK",
}


def _proxy_country(country: pd.Series) -> pd.Series:
    return country.astype(str).map(lambda c: COUNTRY_PROXY.get(c, c))


def fit_cdm_baseline(train_df: pd.DataFrame, horizon: int) -> dict:
    """Stima coeff stagionali (mediana y / naive, cap) e mediane categoria sul train."""
    df = train_df.copy()
    naive = df["run_rate_14d"].astype(float) * float(horizon)
    ratio = (df["y_sku"].astype(float) / naive.clip(lower=0.1)).clip(*SEASONAL_CAP)

    stag_keys = [c for c in ("mese",) if c in df.columns]
    seasonal = (
        df.assign(_r=ratio).groupby(stag_keys, dropna=False)["_r"].median()
        if stag_keys
        else pd.Series(dtype=float)
    )

    df["country_p"] = _proxy_country(df["country"])
    cat_rr = (
        df.groupby(["country_p", "categoria_id"], dropna=False)["run_rate_14d"]
        .median()
        .astype(float)
    )
    return {
        "horizon": int(horizon),
        "seasonal": seasonal,
        "seasonal_keys": stag_keys,
        "cat_run_rate": cat_rr,
    }


def _lookup(df: pd.DataFrame, series: pd.Series, keys: list[str]) -> np.ndarray:
    if series is None or len(series) == 0 or not keys:
        return np.full(len(df), np.nan, dtype=float)
    cols = [c for c in keys if c in df.columns]
    if not cols:
        return np.full(len(df), np.nan, dtype=float)
    idx = pd.MultiIndex.from_frame(df[cols]) if len(cols) > 1 else df[cols[0]]
    try:
        mapped = series.reindex(idx)
    except Exception:
        mapped = idx.map(series)
    return pd.to_numeric(mapped, errors="coerce").to_numpy(dtype=float)


def _performance_cluster(df: pd.DataFrame) -> np.ndarray:
    """Percentile run-rate nella categoria (stesso as_of) → 0.70 / 1.00 / 1.30."""
    rr = df["run_rate_14d"].astype(float)
    if "as_of" in df.columns and "categoria_id" in df.columns:
        pct = rr.groupby([df["as_of"], df["categoria_id"]]).rank(pct=True)
    elif "categoria_id" in df.columns:
        pct = rr.groupby(df["categoria_id"]).rank(pct=True)
    else:
        pct = rr.rank(pct=True)
    p = pct.to_numpy(dtype=float)
    return np.where(p < PERF_P40, PERF_LOW, np.where(p < PERF_P70, PERF_MID, PERF_HIGH))


def predict_cdm_baseline(
    df: pd.DataFrame,
    fitted: dict,
    horizon: int | None = None,
) -> np.ndarray:
    """Previsione del modello CDM originale, allineata riga per riga al fact di test."""
    horizon = int(horizon or fitted.get("horizon") or 15)
    out = df.copy()
    out["country_p"] = _proxy_country(out["country"])
    rr = out["run_rate_14d"].astype(float).to_numpy()

    cat_rr = _lookup(out, fitted.get("cat_run_rate"), ["country_p", "categoria_id"])
    cat_rr = np.where(np.isnan(cat_rr), rr, cat_rr)

    is_new_pdv = np.zeros(len(out), dtype=bool)
    if "is_new_pdv" in out.columns:
        is_new_pdv |= out["is_new_pdv"].astype(float).to_numpy() >= 0.5
    if "pdv_age_days" in out.columns:
        is_new_pdv |= out["pdv_age_days"].astype(float).to_numpy() < 60
    rr = np.where(is_new_pdv, cat_rr, rr)

    seasonal = _lookup(out, fitted.get("seasonal"), fitted.get("seasonal_keys") or [])
    seasonal = np.where(np.isnan(seasonal), 1.0, seasonal)
    seasonal = np.clip(seasonal, *SEASONAL_CAP)

    perf = _performance_cluster(out)
    pred = rr * float(horizon) * seasonal * perf
    cat_fc = cat_rr * float(horizon)

    days = out["giorni_da_lancio"].astype(float).to_numpy() if "giorni_da_lancio" in out.columns else np.full(len(out), 90.0)
    nuovo = days < 14
    provvisorio = (days >= 14) & (days < 60)
    pred = np.where(nuovo, cat_fc * PERF_LOW, pred)
    pred = np.where(provvisorio, 0.5 * pred + 0.5 * cat_fc, pred)
    return np.maximum(pred.astype(float), 0.0)
