#!/usr/bin/env python3
"""P0: rebuild samples (no poisoned Jul as_of) + YoY features + train LGBM h30.

Usa lo stesso artefatto isolato di ``run_train_horizon_native_real`` ma:
- esclude as_of Jul/Aug (e POISONED_AS_OF_EXTRA)
- attacca feature YoY/seasonality
- acquisisce ``bi_training_lock`` per tutta la durata

  OMP_NUM_THREADS=1 python -m production.run_train_h30_p0
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

# Force h30
os.environ["FORECAST_NATIVE_HORIZON"] = "30"

from production.config import MODEL_DIR
from production.ops.bi_training_lock import bi_training_lock, heartbeat
from production.ops.h30_yoy_features import YOY_NUMERIC_COLS, attach_yoy_features
from production.ops.poisoned_as_of import filter_samples_exclude_poisoned
from production.run_train_horizon_native_real import (
    CAT_COLS,
    HORIZON,
    OUT,
    YCOL,
    _assert_untouched,
    _attach_y,
    _fingerprint_protected,
    _metrics,
    _pilot,
    _sample_weights,
    _val_mask,
)
from production.run_train_real import load_real_tables
from testing.blas_threads import limit_native_threads

limit_native_threads()

from testing.config import TrainConfig
from testing.features.build_features import NUMERIC_COLS, apply_train_test_split

ART15 = MODEL_DIR / "real"


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

    extra_num = [c for c in YOY_NUMERIC_COLS if c in out.columns]
    for col in list(NUMERIC_COLS) + extra_num:
        if col not in out.columns:
            out[col] = 0.0
    out["sku_prior"] = (out["run_rate_14d"].astype(float) * float(HORIZON)).clip(lower=0.05)
    feature_cols = list(NUMERIC_COLS) + extra_num + ["sku_prior"] + CAT_COLS
    # dedupe preserve order
    seen = set()
    feats = []
    for c in feature_cols:
        if c not in seen:
            seen.add(c)
            feats.append(c)
    X = out[feats].copy()
    for col in CAT_COLS:
        X[col] = X[col].astype("category")
    y = out[YCOL].astype(float).to_numpy()
    return X, y, maps, feats


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with bi_training_lock(reason="h30_p0_yoy_retrain"):
        before = _fingerprint_protected()
        (OUT / "protected_fingerprint_before.json").write_text(json.dumps(before, indent=2))

        cfg = TrainConfig()
        cfg.horizon_sku_days = HORIZON
        cfg.train_window_days = 270 if _pilot() else 365
        np.random.seed(cfg.seed)

        samples_path = ART15 / ("samples_pilot.pkl" if _pilot() else "samples.pkl")
        if not samples_path.exists():
            raise SystemExit(f"Manca {samples_path} — eseguire prima su VM con dati reali")

        print("=== P0 h30: load samples + exclude poisoned as_of ===")
        samples = pd.read_pickle(samples_path)
        n0 = len(samples)
        samples = filter_samples_exclude_poisoned(samples)
        print(
            f"  rows {n0:,} → {len(samples):,} "
            f"(dropped_rows={samples.attrs.get('poisoned_dropped_rows')} "
            f"as_ofs={samples.attrs.get('poisoned_dropped_as_ofs')})"
        )
        heartbeat(phase="samples_filtered", rows=len(samples))

        print("=== Load fact + attach y_30 + YoY ===")
        fact, _calendar, _pdv = load_real_tables()
        samples_h = _attach_y(samples, fact)
        samples_h = attach_yoy_features(samples_h, fact)
        del samples, fact
        out_samples = OUT / (
            f"samples_pilot_{YCOL}_p0.pkl" if _pilot() else f"samples_{YCOL}_p0.pkl"
        )
        samples_h.to_pickle(out_samples)
        print(
            f"  saved {out_samples} rows={len(samples_h):,} "
            f"mean_{YCOL}={samples_h[YCOL].mean():.2f} "
            f"yoy_has={samples_h['yoy_has_hist'].mean():.3f}"
        )
        heartbeat(phase="y_attached", rows=len(samples_h))

        print("=== Split + train ===")
        train_df, test_df, cut, train_start = apply_train_test_split(samples_h, cfg)
        print(f"  train={len(train_df):,} test={len(test_df):,} cut>={cut} start={train_start}")

        mask_val = _val_mask(train_df, cfg.n_val_as_of)
        tr = train_df.loc[~mask_val]
        va = train_df.loc[mask_val]
        X_tr, y_tr, maps, feats = _prepare_xy(tr, fit=True)
        X_va, y_va, _, _ = _prepare_xy(va, cat_maps=maps, fit=False)
        X_te, y_te, _, _ = _prepare_xy(test_df, cat_maps=maps, fit=False)
        w = _sample_weights(tr)

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
        heartbeat(phase="tweedie_done", best_iter=int(m_tw.best_iteration_ or 0))

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
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)]
            if eval_set
            else [lgb.log_evaluation(0)],
        )

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
            "p0": True,
            "poisoned_excluded": True,
            "yoy_features": YOY_NUMERIC_COLS,
            "feature_count": len(feats),
            "naive": _metrics(naive, y_te),
            "tweedie": _metrics(pred_tw, y_te),
            "deploy_blend": _metrics(pred, y_te),
            "cut": str(cut),
            "train_start": str(train_start),
            "note": "P0 h30: poisoned Jul/Aug as_of esclusi; YoY/seasonality on.",
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
            "p0_yoy": True,
            "yoy_cols": YOY_NUMERIC_COLS,
        }
        sfx = "_pilot" if _pilot() else ""
        # Scrivi sia bundle p0 dedicato sia bundle standard (bi_engine tipicamente legge lgbm_bundle)
        joblib.dump(bundle, OUT / f"lgbm_bundle_p0{sfx}.joblib")
        joblib.dump(bundle, OUT / f"lgbm_bundle{sfx}.joblib")
        (OUT / f"lgbm_maps{sfx}.json").write_text(json.dumps(maps, ensure_ascii=False, indent=2))
        (OUT / f"metrics_p0{sfx}.json").write_text(json.dumps(metrics, indent=2))
        (OUT / f"metrics{sfx}.json").write_text(json.dumps(metrics, indent=2))
        print(f"Saved {OUT / f'lgbm_bundle{sfx}.joblib'}")

        _assert_untouched(before)
        after = _fingerprint_protected()
        (OUT / "protected_fingerprint_after.json").write_text(json.dumps(after, indent=2))
        print("OK: artifact 15gg invariati; lock verrà rilasciato dal context manager")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise
