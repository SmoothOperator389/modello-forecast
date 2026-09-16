"""
Confidence score e blend cold start (post-modello).

Allineato al PDF CDM: SKU giovane → peso alto su analoghi (attributi prodotto + categoria);
SKU maturo → peso alto su storico (modello + run rate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# SKU «solido» per calibrare lookup analoghi
MATURE_DAYS = 45
MATURE_MIN_HISTORY = 0.5

# Analoghi condizionati al mese (niente tag stagione prodotto):
# un piumino ad agosto non deve ereditare il run-rate di gennaio.
ANALOGUE_LEVELS: list[tuple[str, list[str]]] = [
    ("attr_mese", ["country", "pdv_id", "categoria_id", "modello", "tessuto", "colore", "mese"]),
    ("attr", ["country", "pdv_id", "categoria_id", "modello", "tessuto", "colore"]),
    ("pdv_cat_mese", ["country", "pdv_id", "categoria_id", "mese"]),
    ("pdv_cat", ["country", "pdv_id", "categoria_id"]),
    ("country_cat_mese", ["country", "categoria_id", "mese"]),
    ("country_cat", ["country", "categoria_id"]),
]


def _forecast_from_run_rate(run_rate: pd.Series, horizon: int) -> np.ndarray:
    return run_rate.astype(float).to_numpy() * horizon


def build_analogue_lookup(train_df: pd.DataFrame, horizon: int) -> dict[str, pd.Series]:
    """Mediana run rate × orizzonte su SKU maturi, per livelli di fallback."""
    mature = train_df[
        (train_df["giorni_da_lancio"].astype(float) >= MATURE_DAYS)
        & (train_df["has_history"].astype(float) >= MATURE_MIN_HISTORY)
    ].copy()
    if mature.empty:
        mature = train_df.copy()

    mature = mature.assign(_fc=_forecast_from_run_rate(mature["run_rate_14d"], horizon))

    lookups: dict[str, pd.Series] = {}
    for name, keys in ANALOGUE_LEVELS:
        cols = [c for c in keys if c in mature.columns]
        if not cols:
            continue
        lookups[name] = mature.groupby(cols, dropna=False)["_fc"].median()

    global_median = float(mature["_fc"].median()) if len(mature) else 1.0
    lookups["global"] = global_median
    return lookups


def _lookup_series(df: pd.DataFrame, series: pd.Series, keys: list[str]) -> np.ndarray:
    cols = [c for c in keys if c in df.columns]
    idx = df[cols].apply(tuple, axis=1)
    mapped = idx.map(series)
    return mapped.to_numpy(dtype=float)


def analogue_forecast(df: pd.DataFrame, lookups: dict[str, pd.Series], horizon: int) -> np.ndarray:
    """Mediana 15gg di SKU solidi simili, nello stesso mese."""
    out = np.full(len(df), np.nan, dtype=float)
    for name, keys in ANALOGUE_LEVELS:
        if name not in lookups:
            continue
        fallback = _lookup_series(df, lookups[name], keys)
        out = np.where(np.isnan(out), fallback, out)

    global_med = lookups.get("global", 1.0)
    if isinstance(global_med, pd.Series):
        global_med = float(global_med.median())
    out = np.where(np.isnan(out), float(global_med), out)

    # Ancoraggio categoria da run-rate (niente vendite future: serve anche in inferenza).
    n_in_cat = df.groupby(["pdv_id", "categoria_id"], dropna=False)["sku_id"].transform("count")
    cat_rr = df.groupby(["pdv_id", "categoria_id"], dropna=False)["run_rate_14d"].transform("sum")
    cat_anchor = cat_rr.astype(float) * float(horizon) / n_in_cat.clip(lower=1)
    immature = df["giorni_da_lancio"].astype(float) < MATURE_DAYS
    out = np.where(immature.to_numpy(), 0.7 * out + 0.3 * cat_anchor.to_numpy(), out)

    return np.maximum(out, 0.0)


def maturity_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Quattro segnali PDF CDM normalizzati 0–1."""
    days = df["giorni_da_lancio"].astype(float)
    weeks = days / 7.0
    units = df.get("units_14_net", df["run_rate_14d"].astype(float) * 14.0).astype(float)
    pdv_act = df.get("pdv_active_14", df["has_history"]).astype(float)
    rr = df["run_rate_14d"].astype(float)
    cv = df["vendite_volatilita_14d"].astype(float) / (rr + 0.1)

    sig_weeks = np.clip(weeks / (MATURE_DAYS / 7.0), 0.0, 1.0)
    sig_units = np.clip(np.log1p(units) / np.log1p(80.0), 0.0, 1.0)
    sig_pdv = np.clip(pdv_act / 15.0, 0.0, 1.0)
    sig_stability = 1.0 - np.clip(cv / 2.0, 0.0, 1.0)

    return pd.DataFrame(
        {
            "sig_weeks": sig_weeks,
            "sig_units": sig_units,
            "sig_pdv_net": sig_pdv,
            "sig_stability": sig_stability,
        },
        index=df.index,
    )


def history_weight(df: pd.DataFrame) -> np.ndarray:
    """Peso rete+storico vs analoghi. Maturi → quasi tutta rete; giovani → aiuto analoghi."""
    days = df["giorni_da_lancio"].astype(float).to_numpy()
    curve = np.interp(days, [0, 7, 21, 45, 90], [0.35, 0.65, 0.82, 0.92, 0.96])
    sigs = maturity_signals(df)
    composite = sigs.mean(axis=1).to_numpy()
    w = 0.65 * curve + 0.35 * composite
    return np.clip(w, 0.35, 0.96)


def blend_forecast(
    df: pd.DataFrame,
    pred_model: np.ndarray,
    lookups: dict[str, pd.Series],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SKU maturo: pred_history (≈ rete). SKU giovane: mix con analoghi se aiutano.
    Gli analoghi spostano la previsione solo se lo SKU ha pochi giorni di vita.
    """
    days = df["giorni_da_lancio"].astype(float).to_numpy()
    w = history_weight(df)
    pred_analog = analogue_forecast(df, lookups, horizon)
    naive = _forecast_from_run_rate(df["run_rate_14d"], horizon)

    w_model = np.interp(days, [0, 14, 45, 90], [0.50, 0.75, 0.90, 1.0])
    pred_history = w_model * pred_model + (1.0 - w_model) * naive
    has_hist = (
        df["has_history"].astype(float).to_numpy() > 0
        if "has_history" in df.columns
        else np.ones(len(df), dtype=bool)
    )
    hist_ref = np.maximum(np.maximum(pred_model, naive), 0.0)
    analog_cap = np.maximum(hist_ref * 1.8, 1.5)
    pred_analog = np.where(has_hist, np.minimum(pred_analog, analog_cap), pred_analog)
    pred_final = (1.0 - w) * pred_analog + w * pred_history
    # Maturi: non lasciare che analoghi peggiorino una rete già buona
    mature = (days >= MATURE_DAYS) & has_hist
    pred_final = np.where(mature, pred_history, pred_final)
    w = np.where(mature, np.maximum(w, 0.92), w)
    return pred_final.astype(float), w.astype(float), pred_analog.astype(float)


def apply_confidence_to_frame(
    df: pd.DataFrame,
    pred_model: np.ndarray,
    lookups: dict[str, pd.Series],
    horizon: int,
) -> pd.DataFrame:
    out = df.copy()
    pred_final, w_hist, pred_analog = blend_forecast(df, pred_model, lookups, horizon)
    out["pred_model"] = pred_model
    out["pred_analogue"] = pred_analog
    out["confidence_history"] = w_hist
    out["pred_final"] = pred_final
    sigs = maturity_signals(df)
    for col in sigs.columns:
        out[col] = sigs[col].values
    out["confidence_score"] = sigs.mean(axis=1).values
    return out
