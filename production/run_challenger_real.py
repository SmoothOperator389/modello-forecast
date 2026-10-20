#!/usr/bin/env python3
"""Confronto TF pilota vs LightGBM + metriche per fascia — dati reali.

Usa lo stesso sample del pilota:
  production/artifacts/real/samples_pilot.pkl

Uso:
  OMP_NUM_THREADS=1 .venv2/bin/python -m production.run_challenger_real
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from production.config import MODEL_DIR, PROJECT_ROOT
from testing.blas_threads import limit_native_threads

limit_native_threads()

from testing.config import TrainConfig
from testing.features.build_features import NUMERIC_COLS, apply_train_test_split
from testing.features.schema import BAND_INPUT_COLS, CATEGORICAL_COLS
from testing.inference.cdm_baseline import fit_cdm_baseline, predict_cdm_baseline
from testing.inference.confidence import apply_confidence_to_frame, build_analogue_lookup
from testing.tensorflow_env import import_tensorflow

ART = MODEL_DIR / "real"
SAMPLES = ART / "samples_pilot.pkl"
MODEL_PATH = ART / "dual_head_pilot.keras"
ENC_PATH = ART / "encoders_pilot.npz"
OUT_JSON = ART / "challenger_pilot_metrics.json"
OUT_BANDS = ART / "challenger_pilot_bands.csv"
OUT_IMP = ART / "challenger_pilot_importance.csv"

CAT_COLS = list(CATEGORICAL_COLS) + list(BAND_INPUT_COLS.keys())


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
        "mean_y": round(float(y.mean()), 4),
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


def train_lgbm(train_df: pd.DataFrame, cfg: TrainConfig):
    mask_val = _val_mask(train_df, cfg.n_val_as_of)
    tr = train_df.loc[~mask_val]
    va = train_df.loc[mask_val]
    X_tr, y_tr, maps, feats = _prepare_xy(tr, fit=True)
    X_va, y_va, _, _ = _prepare_xy(va, cat_maps=maps, fit=False)
    # peso su volume + SKU nuovi
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
    print(f"LightGBM best iteration: {model.best_iteration_}")
    return model, maps, feats


def main() -> None:
    if not SAMPLES.exists():
        raise SystemExit(f"Manca {SAMPLES}: esegui prima production.run_train_real con FORECAST_PILOT=1")

    cfg = TrainConfig()
    cfg.train_window_days = 270
    cfg.horizon_sku_days = 15
    np.random.seed(cfg.seed)

    print("=== 1) Load sample pilota ===")
    samples = pd.read_pickle(SAMPLES)
    train_df, test_df, cut, train_start = apply_train_test_split(samples, cfg)
    print(f"train={len(train_df):,} test={len(test_df):,} cut>={cut} start={train_start}")
    y = test_df["y_sku"].astype(float).to_numpy()
    naive = test_df["run_rate_14d"].astype(float).to_numpy() * cfg.horizon_sku_days

    print("\n=== 2) Baseline CDM ===")
    cdm = fit_cdm_baseline(train_df, cfg.horizon_sku_days)
    pred_cdm = predict_cdm_baseline(test_df, cdm, cfg.horizon_sku_days)
    analogue = build_analogue_lookup(train_df, cfg.horizon_sku_days)

    print("\n=== 3) Predizioni TF pilota ===")
    tf = import_tensorflow()
    from testing.model.dual_head_tf import frame_to_tf_inputs

    enc = np.load(ENC_PATH, allow_pickle=True)
    maps = enc["maps"].item()
    num_mean = enc["num_mean"]
    num_std = enc["num_std"]
    model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)
    # Evita Lambda rotta sulla testa categoria in deserialize: predici solo y_sku
    sku_out = model.get_layer("y_sku").output
    sku_model = tf.keras.Model(inputs=model.inputs, outputs=sku_out)
    x_te = frame_to_tf_inputs(test_df, maps, num_mean, num_std)
    pred_tf_raw = np.maximum(sku_model.predict(x_te, verbose=0).reshape(-1), 0.0)
    blend_tf = apply_confidence_to_frame(test_df, pred_tf_raw, analogue, cfg.horizon_sku_days)
    pred_tf = blend_tf["pred_final"].to_numpy()

    print("\n=== 4) Train LightGBM ===")
    lgb_model, lgb_maps, lgb_feats = train_lgbm(train_df, cfg)
    X_te, _, _, _ = _prepare_xy(test_df, cat_maps=lgb_maps, fit=False)
    pred_lgb_raw = np.maximum(lgb_model.predict(X_te[lgb_feats]), 0.0)
    blend_lgb = apply_confidence_to_frame(test_df, pred_lgb_raw, analogue, cfg.horizon_sku_days)
    pred_lgb = blend_lgb["pred_final"].to_numpy()

    models = {
        "naive": naive,
        "cdm": pred_cdm,
        "tf_raw": pred_tf_raw,
        "tf_blend": pred_tf,
        "lgbm_raw": pred_lgb_raw,
        "lgbm_blend": pred_lgb,
    }

    print("\n=== 5) Metriche globali ===")
    global_m = {name: _metrics(pred, y) for name, pred in models.items()}
    for name, m in global_m.items():
        print(
            f"  {name:12} MAE={m['mae']:.3f}  wMAPE={m['wmape']:.1f}%  "
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
            print(f"    {name:12} MAE={m['mae']:.3f} wMAPE={m['wmape']:.1f}%")
            band_rows.append({"band": band, "model": name, **m})

    pd.DataFrame(band_rows).to_csv(OUT_BANDS, index=False)
    imp = (
        pd.DataFrame({"feature": lgb_feats, "importance": lgb_model.feature_importances_})
        .sort_values("importance", ascending=False)
    )
    imp.to_csv(OUT_IMP, index=False)

    winner = min(
        ("tf_blend", global_m["tf_blend"]["mae"]),
        ("lgbm_blend", global_m["lgbm_blend"]["mae"]),
        key=lambda t: t[1],
    )[0]
    out = {
        "cut": str(cut),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "global": global_m,
        "winner_blend": winner,
        "lgbm_best_iteration": int(lgb_model.best_iteration_ or 0),
        "top_features": imp.head(15).to_dict(orient="records"),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWinner (blend MAE): {winner}")
    print(f"Saved → {OUT_JSON}")
    print(f"Bands  → {OUT_BANDS}")
    print("Top 10 feature LGBM:")
    print(imp.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
