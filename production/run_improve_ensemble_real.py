#!/usr/bin/env python3
"""Migliora ensemble: LGBM high-volume + calibrazione routing.

Pilota (default):
  samples_pilot.pkl + dual_head_pilot.keras
Rete intera:
  FORECAST_PILOT=0 → samples.pkl + dual_head.keras → artefatti *_full

Uso:
  OMP_NUM_THREADS=1 .venv2/bin/python -m production.run_improve_ensemble_real
  FORECAST_PILOT=0 OMP_NUM_THREADS=1 .venv2/bin/python -m production.run_improve_ensemble_real
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from production.config import MODEL_DIR
from production.operational.routing import (
    apply_calibrated_route,
    calibrate_route_params,
    route_forecast,
)
from testing.blas_threads import limit_native_threads

limit_native_threads()

from testing.config import TrainConfig
from testing.features.build_features import NUMERIC_COLS, apply_train_test_split
from testing.features.schema import BAND_INPUT_COLS, CATEGORICAL_COLS
from testing.inference.confidence import apply_confidence_to_frame, build_analogue_lookup
from testing.model.dual_head_tf import frame_to_tf_inputs, load_dual_head_model
from testing.tensorflow_env import import_tensorflow

ART = MODEL_DIR / "real"


def _pilot() -> bool:
    return os.environ.get("FORECAST_PILOT", "1").strip().lower() not in {"0", "false", "no", "off"}


_PILOT = _pilot()
_SFX = "" if _PILOT else "_full"
SAMPLES = ART / ("samples_pilot.pkl" if _PILOT else "samples.pkl")
MODEL_PATH = ART / ("dual_head_pilot.keras" if _PILOT else "dual_head.keras")
ENC_PATH = ART / ("encoders_pilot.npz" if _PILOT else "encoders.npz")
OUT_JSON = ART / f"ensemble_improved_metrics{_SFX}.json"
OUT_BANDS = ART / f"ensemble_improved_bands{_SFX}.csv"
OUT_PARAMS = ART / f"ensemble_improved_params{_SFX}.json"
OUT_HIGH_MODEL = ART / f"lgbm_high_volume{_SFX}.joblib"
OUT_HIGH_MAPS = ART / f"lgbm_high_volume_maps{_SFX}.json"

CAT_COLS = list(CATEGORICAL_COLS) + list(BAND_INPUT_COLS.keys())
HIGH_PRIOR_MIN = 8.0  # train specialista su prior >= 8


def _metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    mae = float(np.abs(pred - y).mean())
    mape = float(np.mean(np.abs(pred - y) / np.maximum(y, 1.0)) * 100)
    wmape = float(np.abs(pred - y).sum() / max(float(y.sum()), 1.0) * 100)
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum((y - pred) ** 2) / ss) if ss > 0 else 0.0
    return {
        "mae": round(mae, 4),
        "mape": round(mape, 2),
        "wmape": round(wmape, 2),
        "r2": round(r2, 4),
        "n": int(len(y)),
        "mean_y": round(float(y.mean()), 4) if len(y) else 0.0,
    }


def _band_masks(y: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "y=0": y == 0,
        "y(0,2]": (y > 0) & (y <= 2),
        "y(2,5]": (y > 2) & (y <= 5),
        "y(5,15]": (y > 5) & (y <= 15),
        "y>15": y > 15,
        "ops_y>=3": y >= 3,
    }


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
        codes = out[col].astype(str).fillna("NA").map(mapping)
        out[col] = codes.fillna(-1).astype(int)
    for col in NUMERIC_COLS:
        if col not in out.columns:
            out[col] = 0.0
    out["sku_prior"] = (out["run_rate_14d"].astype(float) * 15.0).clip(lower=0.05)
    feature_cols = NUMERIC_COLS + ["sku_prior"] + CAT_COLS
    X = out[feature_cols].copy()
    for col in CAT_COLS:
        X[col] = X[col].astype("category")
    y = out["y_sku"].astype(float).to_numpy()
    return X, y, maps, feature_cols


def _val_mask(train_df: pd.DataFrame, n_val_as_of: int) -> np.ndarray:
    as_ofs = sorted(train_df["as_of"].unique())
    n_val = min(max(1, n_val_as_of), max(1, len(as_ofs) // 4))
    val_set = set(as_ofs[-n_val:])
    return train_df["as_of"].isin(val_set).to_numpy()


def train_lgbm_global(train_df: pd.DataFrame, cfg: TrainConfig):
    mask_val = _val_mask(train_df, cfg.n_val_as_of)
    tr = train_df.loc[~mask_val]
    va = train_df.loc[mask_val]
    X_tr, y_tr, maps, feats = _prepare_xy(tr, fit=True)
    X_va, y_va, _, _ = _prepare_xy(va, cat_maps=maps, fit=False)
    w = np.ones(len(tr), dtype=np.float32)
    prior = tr["run_rate_14d"].astype(float).to_numpy() * 15.0
    w *= (1.0 + np.log1p(np.clip(prior, 0, None))).astype(np.float32)
    if "is_new_sku" in tr.columns:
        w[tr["is_new_sku"].astype(float).to_numpy() >= 0.5] *= 1.4

    model = lgb.LGBMRegressor(
        objective="regression_l1",
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
    model.fit(
        X_tr,
        y_tr,
        sample_weight=w,
        eval_set=[(X_va, y_va)],
        eval_metric="l1",
        categorical_feature=CAT_COLS,
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    print(f"  LGBM global best_iteration={model.best_iteration_}")
    return model, maps, feats


def train_lgbm_high(train_df: pd.DataFrame, cfg: TrainConfig, cat_maps: dict, feats: list[str]):
    """Specialista coda alta: prior >= HIGH_PRIOR_MIN, target log1p."""
    prior_all = train_df["run_rate_14d"].astype(float) * float(cfg.horizon_sku_days)
    hi = train_df.loc[prior_all >= HIGH_PRIOR_MIN].copy()
    if len(hi) < 2000:
        print(f"  WARN high-volume sample piccolo: {len(hi):,}")
    mask_val = _val_mask(hi, max(2, cfg.n_val_as_of // 2))
    tr = hi.loc[~mask_val]
    va = hi.loc[mask_val]
    X_tr, y_tr, _, _ = _prepare_xy(tr, cat_maps=cat_maps, fit=False)
    X_va, y_va, _, _ = _prepare_xy(va, cat_maps=cat_maps, fit=False)
    y_tr_log = np.log1p(np.clip(y_tr, 0, None))
    y_va_log = np.log1p(np.clip(y_va, 0, None))
    # peso più forte sui volumi estremi
    w = (1.0 + np.log1p(np.clip(y_tr, 0, None))).astype(np.float32)

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=0.8,
        random_state=cfg.seed + 7,
        n_jobs=2,
        verbosity=-1,
    )
    model.fit(
        X_tr[feats],
        y_tr_log,
        sample_weight=w,
        eval_set=[(X_va[feats], y_va_log)],
        eval_metric="l1",
        categorical_feature=CAT_COLS,
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
    )
    print(
        f"  LGBM high-volume n_train={len(tr):,} n_val={len(va):,} "
        f"best_iteration={model.best_iteration_}"
    )
    return model


def _predict_high(model, X: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return np.maximum(np.expm1(model.predict(X[feats])), 0.0)


def main() -> None:
    if not SAMPLES.exists():
        raise SystemExit(f"Manca {SAMPLES}")

    cfg = TrainConfig()
    cfg.train_window_days = 270 if _PILOT else 365
    cfg.horizon_sku_days = 15
    np.random.seed(cfg.seed)

    print(f"=== 1) Load sample ({'pilota' if _PILOT else 'rete intera'}) ===")
    samples = pd.read_pickle(SAMPLES)
    train_df, test_df, cut, train_start = apply_train_test_split(samples, cfg)
    print(f"train={len(train_df):,} test={len(test_df):,} cut>={cut} start={train_start}")
    y = test_df["y_sku"].astype(float).to_numpy()
    prior_te = test_df["run_rate_14d"].astype(float).to_numpy() * cfg.horizon_sku_days
    naive = prior_te.copy()

    print(f"\n=== 2) Predizioni TF ({'pilota' if _PILOT else 'full'}) ===")
    tf = import_tensorflow()
    enc = np.load(ENC_PATH, allow_pickle=True)
    maps_tf = enc["maps"].item()
    num_mean = enc["num_mean"]
    num_std = enc["num_std"]
    model_tf = load_dual_head_model(MODEL_PATH)
    sku_out = model_tf.get_layer("y_sku").output
    sku_model = tf.keras.Model(inputs=model_tf.inputs, outputs=sku_out)
    analogue = build_analogue_lookup(train_df, cfg.horizon_sku_days)
    x_te = frame_to_tf_inputs(test_df, maps_tf, num_mean, num_std)
    pred_tf_raw = np.maximum(sku_model.predict(x_te, verbose=0).reshape(-1), 0.0)
    pred_tf = apply_confidence_to_frame(test_df, pred_tf_raw, analogue, cfg.horizon_sku_days)[
        "pred_final"
    ].to_numpy()

    # predizioni TF anche su coda validation per calibrazione
    mask_val = _val_mask(train_df, cfg.n_val_as_of)
    val_df = train_df.loc[mask_val].copy()
    x_va = frame_to_tf_inputs(val_df, maps_tf, num_mean, num_std)
    pred_tf_va_raw = np.maximum(sku_model.predict(x_va, verbose=0).reshape(-1), 0.0)
    pred_tf_va = apply_confidence_to_frame(val_df, pred_tf_va_raw, analogue, cfg.horizon_sku_days)[
        "pred_final"
    ].to_numpy()
    y_va = val_df["y_sku"].astype(float).to_numpy()
    prior_va = val_df["run_rate_14d"].astype(float).to_numpy() * cfg.horizon_sku_days
    naive_va = prior_va.copy()

    print("\n=== 3) Train LightGBM global + high-volume ===")
    lgb_model, lgb_maps, lgb_feats = train_lgbm_global(train_df, cfg)
    high_model = train_lgbm_high(train_df, cfg, lgb_maps, lgb_feats)

    X_te, _, _, _ = _prepare_xy(test_df, cat_maps=lgb_maps, fit=False)
    X_va_lgb, _, _, _ = _prepare_xy(val_df, cat_maps=lgb_maps, fit=False)

    pred_lgb_raw = np.maximum(lgb_model.predict(X_te[lgb_feats]), 0.0)
    pred_lgb = apply_confidence_to_frame(test_df, pred_lgb_raw, analogue, cfg.horizon_sku_days)[
        "pred_final"
    ].to_numpy()
    pred_high = _predict_high(high_model, X_te, lgb_feats)

    pred_lgb_va_raw = np.maximum(lgb_model.predict(X_va_lgb[lgb_feats]), 0.0)
    pred_lgb_va = apply_confidence_to_frame(val_df, pred_lgb_va_raw, analogue, cfg.horizon_sku_days)[
        "pred_final"
    ].to_numpy()
    pred_high_va = _predict_high(high_model, X_va_lgb, lgb_feats)

    print("\n=== 4) Calibrazione routing su validation ===")
    best = calibrate_route_params(
        prior_va,
        y_va,
        pred_tf_va,
        pred_lgb_va,
        naive_va,
        pred_high=pred_high_va,
        objective="composite",
    )
    print(f"  best params (composite): {json.dumps(best, indent=2)}")

    alt_ops = calibrate_route_params(
        prior_va, y_va, pred_tf_va, pred_lgb_va, naive_va, pred_high=pred_high_va, objective="ops"
    )
    alt_global = calibrate_route_params(
        prior_va, y_va, pred_tf_va, pred_lgb_va, naive_va, pred_high=pred_high_va, objective="global"
    )
    alt_high = calibrate_route_params(
        prior_va, y_va, pred_tf_va, pred_lgb_va, naive_va, pred_high=pred_high_va, objective="high"
    )

    print("\n=== 5) Valutazione test ===")
    # baseline storici
    route_mix_legacy = route_forecast(prior_te, pred_tf, pred_lgb, naive)
    route_naive_high = route_forecast(
        prior_te, pred_tf, pred_lgb, naive, high_mode="naive", high_tf_weight=0.0
    )
    route_improved = apply_calibrated_route(
        prior_te, pred_tf, pred_lgb, naive, best, pred_high=pred_high
    )
    route_alt_ops = apply_calibrated_route(
        prior_te, pred_tf, pred_lgb, naive, alt_ops, pred_high=pred_high
    )
    route_alt_global = apply_calibrated_route(
        prior_te, pred_tf, pred_lgb, naive, alt_global, pred_high=pred_high
    )
    route_alt_high = apply_calibrated_route(
        prior_te, pred_tf, pred_lgb, naive, alt_high, pred_high=pred_high
    )

    models = {
        "naive": naive,
        "tf_blend": pred_tf,
        "lgbm_blend": pred_lgb,
        "lgbm_high": pred_high,
        "route_mix_legacy": route_mix_legacy,
        "route_naive_high": route_naive_high,
        "route_improved": route_improved,
        "route_cal_ops": route_alt_ops,
        "route_cal_global": route_alt_global,
        "route_cal_high": route_alt_high,
    }

    global_m = {name: _metrics(pred, y) for name, pred in models.items()}
    for name, m in global_m.items():
        print(
            f"  {name:20} MAE={m['mae']:.3f}  wMAPE={m['wmape']:.1f}%  "
            f"MAPE={m['mape']:.1f}%  R2={m['r2']:.3f}"
        )

    print("\n=== 6) Metriche per fascia ===")
    band_rows = []
    for band, mask in _band_masks(y).items():
        if mask.sum() < 50:
            continue
        print(f"\n  [{band}] n={mask.sum():,} mean_y={y[mask].mean():.2f}")
        for name, pred in models.items():
            m = _metrics(pred[mask], y[mask])
            print(f"    {name:20} MAE={m['mae']:.3f} wMAPE={m['wmape']:.1f}%")
            band_rows.append({"band": band, "model": name, **m})

    # winner = min MAE globale tra route_*
    route_names = [n for n in models if n.startswith("route_")]
    winner = min(route_names, key=lambda n: global_m[n]["mae"])
    # se ops MAE migliore su un altro route, segnalalo
    ops_mask = y >= 3
    ops_mae = {n: float(np.abs(models[n][ops_mask] - y[ops_mask]).mean()) for n in route_names}
    winner_ops = min(ops_mae, key=ops_mae.get)

    pd.DataFrame(band_rows).to_csv(OUT_BANDS, index=False)
    param_by_route = {
        "route_improved": best,
        "route_cal_ops": alt_ops,
        "route_cal_global": alt_global,
        "route_cal_high": alt_high,
    }
    OUT_PARAMS.write_text(
        json.dumps(
            {
                "primary_composite": best,
                "alt_ops": alt_ops,
                "alt_global": alt_global,
                "alt_high": alt_high,
                "high_prior_min": HIGH_PRIOR_MIN,
                "winner_global": winner,
                "winner_ops": winner_ops,
                "deploy_params": param_by_route.get(winner_ops, best),
            },
            indent=2,
        )
    )
    joblib.dump(
        {"model": high_model, "features": lgb_feats, "log_target": True},
        OUT_HIGH_MODEL,
    )
    joblib.dump(
        {"model": lgb_model, "features": lgb_feats, "maps": lgb_maps, "log_target": False},
        ART / f"lgbm_global{_SFX}.joblib",
    )
    OUT_HIGH_MAPS.write_text(json.dumps({k: {str(a): b for a, b in v.items()} for k, v in lgb_maps.items()}))

    baseline_mae = global_m["route_mix_legacy"]["mae"]
    improved_mae = global_m[winner]["mae"]
    out = {
        "cut": str(cut),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "global": global_m,
        "winner_global": winner,
        "winner_ops": winner_ops,
        "ops_mae": {k: round(v, 4) for k, v in ops_mae.items()},
        "improvement_vs_route_mix_pct": round(
            (baseline_mae - improved_mae) / max(baseline_mae, 1e-9) * 100, 2
        ),
        "params_used": param_by_route.get(winner, best),
        "deploy_params": param_by_route.get(winner_ops, best),
        "lgbm_global_best_iteration": int(lgb_model.best_iteration_ or 0),
        "lgbm_high_best_iteration": int(high_model.best_iteration_ or 0),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWinner global: {winner} MAE={improved_mae:.4f} "
          f"(vs route_mix {baseline_mae:.4f}, Δ={out['improvement_vs_route_mix_pct']}%)")
    print(f"Winner ops:    {winner_ops} MAE={ops_mae[winner_ops]:.4f}")
    print(f"Saved → {OUT_JSON}")
    print(f"Bands  → {OUT_BANDS}")
    print(f"Params → {OUT_PARAMS}")


if __name__ == "__main__":
    main()
