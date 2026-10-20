"""Inferenza isolata per orizzonti nativi 30/60/90gg."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from production.config import MODEL_DIR
from testing.features.build_features import NUMERIC_COLS
from testing.features.schema import BAND_INPUT_COLS, CATEGORICAL_COLS

CAT_COLS = list(CATEGORICAL_COLS) + list(BAND_INPUT_COLS.keys())


def art_dir(horizon: int) -> Path:
    return MODEL_DIR / "real" / f"h{int(horizon)}"


def load_horizon_bundle(horizon: int, pilot: bool = False) -> tuple[dict, dict]:
    sfx = "_pilot" if pilot else ""
    root = art_dir(horizon)
    bundle = joblib.load(root / f"lgbm_bundle{sfx}.joblib")
    maps = {
        k: {str(a): int(b) for a, b in v.items()}
        for k, v in json.loads((root / f"lgbm_maps{sfx}.json").read_text()).items()
    }
    return bundle, maps


def predict_horizon(
    feat: pd.DataFrame,
    horizon: int,
    bundle: dict | None = None,
    maps: dict | None = None,
) -> np.ndarray:
    horizon = int(horizon)
    if bundle is None or maps is None:
        bundle, maps = load_horizon_bundle(horizon, pilot=False)
    out = feat.copy()
    for col in CAT_COLS:
        if col not in out.columns:
            out[col] = "NA"
        out[col] = out[col].astype(str).fillna("NA").map(maps.get(col, {})).fillna(-1).astype(int)
    for col in NUMERIC_COLS:
        if col not in out.columns:
            out[col] = 0.0
    out["sku_prior"] = (out["run_rate_14d"].astype(float) * float(horizon)).clip(lower=0.05)
    feats = bundle["features"]
    X = out[feats].copy()
    for col in CAT_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    pred_tw = np.maximum(bundle["tweedie"].predict(X[feats]), 0.0)
    pred_hi = np.maximum(np.expm1(bundle["high"].predict(X[feats])), 0.0)
    prior = feat["run_rate_14d"].astype(float).to_numpy() * float(horizon)
    thr = float(bundle.get("high_prior_thr", 15.0 * horizon / 90.0))
    mix = float(bundle.get("high_mix", 0.45))
    floor = float(bundle.get("naive_floor_ratio", 0.35))
    high = prior >= thr
    pred = pred_tw.copy()
    pred[high] = (1.0 - mix) * pred_tw[high] + mix * pred_hi[high]
    pred[high] = np.maximum(pred[high], floor * prior[high])
    return pred


def has_horizon_bundle(horizon: int) -> bool:
    return (art_dir(horizon) / "lgbm_bundle.joblib").exists()
