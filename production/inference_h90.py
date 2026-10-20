"""Inferenza isolata orizzonte 90gg (non tocca il percorso 15gg)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from production.config import MODEL_DIR
from testing.features.build_features import NUMERIC_COLS
from testing.features.schema import BAND_INPUT_COLS, CATEGORICAL_COLS

ART_H90 = MODEL_DIR / "real" / "h90"
CAT_COLS = list(CATEGORICAL_COLS) + list(BAND_INPUT_COLS.keys())
HORIZON = 90


def load_h90_bundle(pilot: bool = False) -> tuple[dict, dict]:
    sfx = "_pilot" if pilot else ""
    bundle = joblib.load(ART_H90 / f"lgbm_bundle{sfx}.joblib")
    maps = {
        k: {str(a): int(b) for a, b in v.items()}
        for k, v in json.loads((ART_H90 / f"lgbm_maps{sfx}.json").read_text()).items()
    }
    return bundle, maps


def prepare_xy_h90(df: pd.DataFrame, maps: dict) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    for col in CAT_COLS:
        if col not in out.columns:
            out[col] = "NA"
        out[col] = out[col].astype(str).fillna("NA").map(maps.get(col, {})).fillna(-1).astype(int)
    for col in NUMERIC_COLS:
        if col not in out.columns:
            out[col] = 0.0
    out["sku_prior"] = (out["run_rate_14d"].astype(float) * float(HORIZON)).clip(lower=0.05)
    feats = NUMERIC_COLS + ["sku_prior"] + CAT_COLS
    X = out[feats].copy()
    for col in CAT_COLS:
        X[col] = X[col].astype("category")
    return X, feats


def predict_h90(feat: pd.DataFrame, bundle: dict | None = None, maps: dict | None = None) -> np.ndarray:
    """Predizione nativa a 90 giorni su un frame feature (stesso schema del 15gg)."""
    if bundle is None or maps is None:
        bundle, maps = load_h90_bundle(pilot=False)
    X, _ = prepare_xy_h90(feat, maps)
    feats = bundle["features"]
    pred_tw = np.maximum(bundle["tweedie"].predict(X[feats]), 0.0)
    pred_hi = np.maximum(np.expm1(bundle["high"].predict(X[feats])), 0.0)
    prior = feat["run_rate_14d"].astype(float).to_numpy() * HORIZON
    thr = float(bundle.get("high_prior_thr", 15.0))
    mix = float(bundle.get("high_mix", 0.45))
    floor = float(bundle.get("naive_floor_ratio", 0.35))
    high = prior >= thr
    pred = pred_tw.copy()
    pred[high] = (1.0 - mix) * pred_tw[high] + mix * pred_hi[high]
    pred[high] = np.maximum(pred[high], floor * prior[high])
    return pred
