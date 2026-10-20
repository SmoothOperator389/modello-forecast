#!/usr/bin/env python3
"""Addestra un modello NATIVO a N giorni in isolamento (come h90).

NON modifica artefatti del forecast 15gg.
Scrive in ``production/artifacts/real/h{N}/``.

  FORECAST_NATIVE_HORIZON=30 OMP_NUM_THREADS=1 .venv2/bin/python -m production.run_train_horizon_native_real
  FORECAST_NATIVE_HORIZON=60 OMP_NUM_THREADS=1 .venv2/bin/python -m production.run_train_horizon_native_real
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from production.config import MODEL_DIR
from production.run_train_real import load_real_tables
from testing.blas_threads import limit_native_threads

limit_native_threads()

from testing.config import TrainConfig
from testing.features.build_features import NUMERIC_COLS, apply_train_test_split
from testing.features.schema import BAND_INPUT_COLS, CATEGORICAL_COLS

ART15 = MODEL_DIR / "real"
HORIZON = int(os.environ.get("FORECAST_NATIVE_HORIZON", "30"))
if HORIZON not in {30, 60, 90}:
    raise SystemExit("FORECAST_NATIVE_HORIZON deve essere 30, 60 o 90")
OUT = ART15 / f"h{HORIZON}"
OUT.mkdir(parents=True, exist_ok=True)
YCOL = f"y_{HORIZON}"

CAT_COLS = list(CATEGORICAL_COLS) + list(BAND_INPUT_COLS.keys())

_PROTECTED = [
    "samples.pkl",
    "samples_pilot.pkl",
    "dual_head.keras",
    "dual_head_pilot.keras",
    "encoders.npz",
    "encoders_pilot.npz",
    "lgbm_quantile_bundle_full.joblib",
    "lgbm_quantile_bundle.joblib",
    "lgbm_high_volume_full.joblib",
    "lgbm_high_volume.joblib",
    "ensemble_improved_params_full.json",
    "ensemble_improved_params.json",
    "lgbm_quantile_maps_full.json",
    "lgbm_quantile_maps.json",
]


def _pilot() -> bool:
    return os.environ.get("FORECAST_PILOT", "0").strip().lower() not in {"0", "false", "no", "off"}


def _fingerprint_protected() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in _PROTECTED:
        p = ART15 / name
        if not p.exists():
            continue
        st = p.stat()
        out[name] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    return out


def _assert_untouched(before: dict[str, dict]) -> None:
    after = _fingerprint_protected()
    for name, b in before.items():
        a = after.get(name)
        if a != b:
            raise RuntimeError(f"Artefatto 15gg modificato durante train h{HORIZON}: {name}")


def _prepare_xy(df: pd.DataFrame, cat_maps: dict | None = None, fit: bool = False):
    out = df.copy()
    maps = cat_maps or {}
    if fit:
        maps = {}
        for col in CAT_COLS:
            if col not in out.columns:
                out[col] = "NA"
            vals = sorted({str(v) for v in out[col].astype(str).fillna("NA").unique()})
            maps[col] = {v: i for i, v in enumerate(vals)}
    for col in CAT_COLS:
        if col not in out.columns:
            out[col] = "NA"
        mapping = maps.get(col, {})
        out[col] = out[col].astype(str).fillna("NA").map(mapping).fillna(-1).astype(int)
    for col in NUMERIC_COLS:
        if col not in out.columns:
            out[col] = 0.0
    out["sku_prior"] = (out["run_rate_14d"].astype(float) * float(HORIZON)).clip(lower=0.05)
    feature_cols = NUMERIC_COLS + ["sku_prior"] + CAT_COLS
    X = out[feature_cols].copy()
    for col in CAT_COLS:
        X[col] = X[col].astype("category")
    y = out[YCOL].astype(float).to_numpy()
    return X, y, maps, feature_cols


def _val_mask(train_df: pd.DataFrame, n_val_as_of: int) -> np.ndarray:
    as_ofs = sorted(train_df["as_of"].unique())
    n_val = min(max(1, n_val_as_of), max(1, len(as_ofs) // 4))
    val_set = set(as_ofs[-n_val:])
    return train_df["as_of"].isin(val_set).to_numpy()


def _sample_weights(tr: pd.DataFrame) -> np.ndarray:
    w = np.ones(len(tr), dtype=np.float32)
    prior = tr["run_rate_14d"].astype(float).to_numpy() * HORIZON
    w *= (1.0 + np.log1p(np.clip(prior, 0, None))).astype(np.float32)
    if "is_new_sku" in tr.columns:
        w[tr["is_new_sku"].astype(float).to_numpy() >= 0.5] *= 1.4
    return w


def _metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    mae = float(np.abs(pred - y).mean())
    wmape = float(np.abs(pred - y).sum() / max(float(y.sum()), 1.0) * 100)
    bias = float((pred - y).sum())
    return {
        "n": int(len(y)),
        "sum_y": round(float(y.sum()), 1),
        "sum_pred": round(float(pred.sum()), 1),
        "mae": round(mae, 4),
        "wmape": round(wmape, 2),
        "bias": round(bias, 1),
    }


def _attach_y(samples: pd.DataFrame, fact: pd.DataFrame) -> pd.DataFrame:
    s = samples.copy()
    s["as_of"] = pd.to_datetime(s["as_of"]).dt.date
    s["pdv_id"] = s["pdv_id"].astype(str)
    s["sku_id"] = s["sku_id"].astype(str)

    fact = fact[["data", "pdv_id", "sku_id", "vendite_qty"]].copy()
    fact["data"] = pd.to_datetime(fact["data"]).dt.date
    fact["pdv_id"] = fact["pdv_id"].astype(str)
    fact["sku_id"] = fact["sku_id"].astype(str)
    max_d = max(fact["data"])
    s = s[s["as_of"].map(lambda d: d + timedelta(days=HORIZON) <= max_d)].copy()
    print(f"  samples con {YCOL} ammissibile: {len(s):,}  as_of {s['as_of'].min()} → {s['as_of'].max()}")

    skus = s["sku_id"].unique()
    fact = fact[fact["sku_id"].isin(skus)]
    print(f"  fact filtrato sku: {len(fact):,}")

    parts = []
    as_ofs = sorted(s["as_of"].unique())
    for i, a in enumerate(as_ofs):
        c0, c1 = a + timedelta(days=1), a + timedelta(days=HORIZON)
        sub = fact[(fact["data"] >= c0) & (fact["data"] <= c1)]
        g = sub.groupby(["pdv_id", "sku_id"], as_index=False, observed=True).agg(**{YCOL: ("vendite_qty", "sum")})
        g["as_of"] = a
        parts.append(g)
        if (i + 1) % 10 == 0 or i == len(as_ofs) - 1:
            print(f"  {YCOL} as_of {i+1}/{len(as_ofs)} ({a})", flush=True)
    ydf = pd.concat(parts, ignore_index=True)
    out = s.merge(ydf, on=["pdv_id", "sku_id", "as_of"], how="left")
    out[YCOL] = out[YCOL].fillna(0.0)
    return out


def main() -> None:
    before = _fingerprint_protected()
    (OUT / "protected_fingerprint_before.json").write_text(json.dumps(before, indent=2))

    cfg = TrainConfig()
    cfg.horizon_sku_days = HORIZON
    cfg.train_window_days = 270 if _pilot() else 365
    np.random.seed(cfg.seed)

    samples_path = ART15 / ("samples_pilot.pkl" if _pilot() else "samples.pkl")
    if not samples_path.exists():
        raise SystemExit(f"Manca {samples_path}")

    print(f"=== Train nativo h{HORIZON} (isolato) ===")
    print("=== 1) Load samples 15gg (feature only) ===")
    samples = pd.read_pickle(samples_path)
    print(f"  rows={len(samples):,}")

    print(f"=== 2) Load fact + attach {YCOL} ===")
    fact, _calendar, _pdv = load_real_tables()
    samples_h = _attach_y(samples, fact)
    del samples, fact
    out_samples = OUT / (f"samples_pilot_{YCOL}.pkl" if _pilot() else f"samples_{YCOL}.pkl")
    samples_h.to_pickle(out_samples)
    print(f"  saved {out_samples} rows={len(samples_h):,} mean_{YCOL}={samples_h[YCOL].mean():.2f}")

    print("=== 3) Split train/test ===")
    train_df, test_df, cut, train_start = apply_train_test_split(samples_h, cfg)
    print(f"  train={len(train_df):,} test={len(test_df):,} cut>={cut} start={train_start}")

    mask_val = _val_mask(train_df, cfg.n_val_as_of)
    tr = train_df.loc[~mask_val]
    va = train_df.loc[mask_val]
    X_tr, y_tr, maps, feats = _prepare_xy(tr, fit=True)
    X_va, y_va, _, _ = _prepare_xy(va, cat_maps=maps, fit=False)
    X_te, y_te, _, _ = _prepare_xy(test_df, cat_maps=maps, fit=False)
    w = _sample_weights(tr)

    print(f"=== 4) Train LGBM Tweedie {HORIZON}d ===")
    params = dict(
        objective="tweedie",
        tweedie_variance_power=1.3,
        n_estimators=2500,
        learning_rate=0.04,
        num_leaves=95,
        min_child_samples=50,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=1.2,
        random_state=cfg.seed,
        n_jobs=2,
        verbosity=-1,
    )
    m_tw = lgb.LGBMRegressor(**params)
    m_tw.fit(
        X_tr[feats],
        y_tr,
        sample_weight=w,
        eval_set=[(X_va[feats], y_va)],
        eval_metric="l1",
        categorical_feature=CAT_COLS,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    print(f"  Tweedie best_iter={m_tw.best_iteration_}")

    print(f"=== 5) Train high-volume specialist {HORIZON}d ===")
    prior_tr = tr["run_rate_14d"].astype(float).to_numpy() * HORIZON
    thr = 15.0 * (HORIZON / 90.0)
    hi_mask = prior_tr >= thr
    if hi_mask.sum() < 500:
        hi_mask = prior_tr >= np.quantile(prior_tr, 0.85)
    X_hi, y_hi = X_tr.loc[hi_mask, feats], np.log1p(y_tr[hi_mask])
    m_hi = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=cfg.seed,
        n_jobs=2,
        verbosity=-1,
    )
    prior_va = va["run_rate_14d"].astype(float).to_numpy() * HORIZON
    va_hi = prior_va >= thr
    if va_hi.sum() < 50:
        va_hi = prior_va >= np.quantile(prior_va, 0.8) if len(prior_va) else va_hi
    eval_set = [(X_va.loc[va_hi, feats], np.log1p(y_va[va_hi]))] if va_hi.any() else None
    m_hi.fit(
        X_hi,
        y_hi,
        eval_set=eval_set,
        eval_metric="l1",
        categorical_feature=CAT_COLS,
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)] if eval_set else [lgb.log_evaluation(0)],
    )
    print(f"  High best_iter={getattr(m_hi, 'best_iteration_', None)}")

    print("=== 6) Eval test ===")
    pred_tw = np.maximum(m_tw.predict(X_te[feats]), 0.0)
    pred_hi = np.maximum(np.expm1(m_hi.predict(X_te[feats])), 0.0)
    prior_te = test_df["run_rate_14d"].astype(float).to_numpy() * HORIZON
    naive = prior_te.copy()
    high = prior_te >= thr
    pred = pred_tw.copy()
    pred[high] = 0.55 * pred_tw[high] + 0.45 * pred_hi[high]
    pred[high] = np.maximum(pred[high], 0.35 * naive[high])

    metrics = {
        "horizon_days": HORIZON,
        "pilot": _pilot(),
        "naive": _metrics(naive, y_te),
        "tweedie": _metrics(pred_tw, y_te),
        "deploy_blend": _metrics(pred, y_te),
        "note": f"Modello isolato h{HORIZON}; artifact 15gg non modificati.",
    }
    print(json.dumps(metrics, indent=2))

    bundle = {
        "tweedie": m_tw,
        "high": m_hi,
        "features": feats,
        "horizon_days": HORIZON,
        "high_prior_thr": float(thr),
        "high_mix": 0.45,
        "naive_floor_ratio": 0.35,
        "log_target_high": True,
    }
    sfx = "_pilot" if _pilot() else ""
    joblib.dump(bundle, OUT / f"lgbm_bundle{sfx}.joblib")
    (OUT / f"lgbm_maps{sfx}.json").write_text(json.dumps(maps, ensure_ascii=False, indent=2))
    (OUT / f"metrics{sfx}.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved {OUT / f'lgbm_bundle{sfx}.joblib'}")

    _assert_untouched(before)
    after = _fingerprint_protected()
    (OUT / "protected_fingerprint_after.json").write_text(json.dumps(after, indent=2))
    digest = hashlib.md5(json.dumps(before, sort_keys=True).encode()).hexdigest()
    print(f"OK: artifact 15gg invariati (fingerprint {digest[:12]})")


if __name__ == "__main__":
    main()
