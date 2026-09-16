#!/usr/bin/env python3
"""Challenger LightGBM vs rete TensorFlow — stesso split, stesse metriche.

Uso:
  cd ~/Desktop/modello-forecast
  OMP_NUM_THREADS=1 .venv2/bin/python -m testing.run_lgbm_challenger
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from testing.blas_threads import limit_native_threads

limit_native_threads()

from testing.config import DATA_DIR, TrainConfig, as_of_range
from testing.features.build_features import (
    NUMERIC_COLS,
    apply_train_test_split,
    build_training_frames,
    load_fact_calendar,
)
from testing.features.schema import BAND_INPUT_COLS, CATEGORICAL_COLS
from testing.inference.artefacts import ARTEFACTS_PATH, load_forecast_artefacts
from testing.inference.cdm_baseline import fit_cdm_baseline, predict_cdm_baseline
from testing.inference.confidence import apply_confidence_to_frame, build_analogue_lookup
from testing.tensorflow_env import import_tensorflow

CAT_COLS = list(CATEGORICAL_COLS) + list(BAND_INPUT_COLS.keys())
OUT_JSON = DATA_DIR / "challenger_lgbm_metrics.json"
OUT_IMP = DATA_DIR / "challenger_lgbm_importance.csv"
SAMPLES_CACHE = DATA_DIR / "challenger_samples.pkl"


def _metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mae = float(np.abs(pred - y).mean())
    mape = float(np.mean(np.abs(pred - y) / np.maximum(y, 1.0)) * 100)
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum((y - pred) ** 2) / ss) if ss > 0 else 0.0
    return {"mae": round(mae, 4), "mape": round(mape, 2), "r2": round(r2, 4)}


def _prepare_xy(df: pd.DataFrame, cat_maps: dict[str, dict[str, int]] | None = None, fit: bool = False):
    """Feature matrix: numeriche + categoriche come category (LightGBM native)."""
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
        # valori sconosciuti → -1 (LightGBM gestisce missing/unknown)
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


def _sample_weights(df: pd.DataFrame) -> np.ndarray:
    w = np.ones(len(df), dtype=np.float32)
    if "sku_share_hot" in df.columns and "sku_share_cold" in df.columns:
        sens = (df["sku_share_hot"].astype(float) - df["sku_share_cold"].astype(float)).abs()
        w = (1.0 + 1.8 * sens.clip(0.0, 1.0)).to_numpy(dtype=np.float32)
        if "sku_temp_fit" in df.columns:
            mismatch = df["sku_temp_fit"].astype(float) < -0.15
            w = np.where(mismatch.to_numpy(), w * 1.35, w).astype(np.float32)
    if "is_new_sku" in df.columns:
        w[df["is_new_sku"].astype(float) >= 0.5] *= 1.4
    return w


def _val_mask(train_df: pd.DataFrame, n_val_as_of: int) -> np.ndarray:
    as_ofs = sorted(train_df["as_of"].unique())
    n_val = min(max(1, n_val_as_of), max(1, len(as_ofs) // 4))
    val_set = set(as_ofs[-n_val:])
    return train_df["as_of"].isin(val_set).to_numpy()


def train_lgbm(train_df: pd.DataFrame, cfg: TrainConfig) -> tuple[lgb.LGBMRegressor, dict, list[str]]:
    mask_val = _val_mask(train_df, cfg.n_val_as_of)
    tr = train_df.loc[~mask_val]
    va = train_df.loc[mask_val]
    X_tr, y_tr, maps, feats = _prepare_xy(tr, fit=True)
    X_va, y_va, _, _ = _prepare_xy(va, cat_maps=maps, fit=False)
    w_tr = _sample_weights(tr)

    model = lgb.LGBMRegressor(
        objective="regression_l1",  # MAE
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=cfg.seed,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(
        X_tr,
        y_tr,
        sample_weight=w_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="l1",
        categorical_feature=CAT_COLS,
        callbacks=[
            lgb.early_stopping(60, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    print(f"LightGBM best iteration: {model.best_iteration_}")
    return model, maps, feats


def predict_lgbm(model: lgb.LGBMRegressor, df: pd.DataFrame, maps: dict, feats: list[str]) -> np.ndarray:
    X, _, _, _ = _prepare_xy(df, cat_maps=maps, fit=False)
    pred = model.predict(X[feats])
    return np.maximum(np.asarray(pred, dtype=float), 0.0)


def main() -> None:
    cfg = TrainConfig()
    np.random.seed(cfg.seed)

    model_path = DATA_DIR / "dual_head_testing.keras"
    if not model_path.exists() or not ARTEFACTS_PATH.exists():
        raise SystemExit(
            "Mancano dual_head_testing.keras / forecast_artefacts.pkl.\n"
            "Esegui prima: OMP_NUM_THREADS=1 .venv2/bin/python -m testing.run_train"
        )

    print("=== 1) Build sample rolling (stesso di run_train) ===")
    if SAMPLES_CACHE.exists():
        print(f"Cache sample: {SAMPLES_CACHE}")
        with SAMPLES_CACHE.open("rb") as f:
            samples = pickle.load(f)
    else:
        fact, calendar = load_fact_calendar()
        as_of_list = as_of_range(cfg.train_as_of_start, cfg.train_as_of_end, cfg.as_of_step_days)
        samples = build_training_frames(fact, calendar, cfg, as_of_list)
        with SAMPLES_CACHE.open("wb") as f:
            pickle.dump(samples, f, protocol=4)
        print(f"Cache salvata: {SAMPLES_CACHE}")
    train_df, test_df, cut, train_start = apply_train_test_split(samples, cfg)
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,} | cut>={cut} | start={train_start}")

    print("\n=== 2) Baseline CDM + analoghi ===")
    cdm_fitted = fit_cdm_baseline(train_df, cfg.horizon_sku_days)
    analogue_lookup = build_analogue_lookup(train_df, cfg.horizon_sku_days)
    pred_cdm = predict_cdm_baseline(test_df, cdm_fitted, cfg.horizon_sku_days)
    y_true = test_df["y_sku"].to_numpy(dtype=float)
    naive = test_df["run_rate_14d"].to_numpy(dtype=float) * cfg.horizon_sku_days

    print("\n=== 3) Train LightGBM challenger ===")
    lgb_model, lgb_maps, lgb_feats = train_lgbm(train_df, cfg)
    pred_lgb_raw = predict_lgbm(lgb_model, test_df, lgb_maps, lgb_feats)
    blend_lgb = apply_confidence_to_frame(test_df, pred_lgb_raw, analogue_lookup, cfg.horizon_sku_days)
    pred_lgb = blend_lgb["pred_final"].to_numpy()

    print("\n=== 4) Inferenza rete TensorFlow (modello già trainato) ===")
    tf_source = "live"
    pred_tf_raw = None
    pred_tf = None
    try:
        import_tensorflow()
        import tensorflow as tf
        from testing.model.dual_head_tf import predict as tf_predict

        artefacts = load_forecast_artefacts()
        try:
            tf_model = tf.keras.models.load_model(model_path, safe_mode=False)
        except TypeError:
            tf_model = tf.keras.models.load_model(model_path)
        pred_tf_raw, _ = tf_predict(
            tf_model,
            test_df,
            artefacts["maps"],
            artefacts["num_mean"],
            artefacts["num_std"],
        )
        blend_tf = apply_confidence_to_frame(test_df, pred_tf_raw, analogue_lookup, cfg.horizon_sku_days)
        pred_tf = blend_tf["pred_final"].to_numpy()
    except Exception as exc:  # noqa: BLE001 — Lambda legacy .keras su Mac
        tf_source = "report_metrics.json (ultimo train, stesso split)"
        print(f"Load rete fallito ({type(exc).__name__}): uso metriche salvate.")
        print(f"  Motivo tipico: Lambda stop_gradient senza output_shape nel .keras.")
        saved = json.loads((DATA_DIR / "report_metrics.json").read_text(encoding="utf-8"))
        pred_tf_raw = pred_tf = None
        tf_fallback = {
            "tf_raw": {
                "mae": float(saved.get("mae_tf_raw", saved.get("mae_tf"))),
                "mape": float(saved.get("mape_tf", 0)),
                "r2": float(saved.get("r2_raw", saved.get("r2", 0))),
            },
            "tf_blend": {
                "mae": float(saved.get("mae_tf")),
                "mape": float(saved.get("mape_tf", 0)),
                "r2": float(saved.get("r2", 0)),
            },
            "cold_mae": saved.get("mae_cold_sku_confidence"),
        }

    print("\n=== 5) Confronto (stesso test) ===")
    rows = {
        "naive": _metrics(naive, y_true),
        "cdm": _metrics(pred_cdm, y_true),
        "lgbm_raw": _metrics(pred_lgb_raw, y_true),
        "lgbm_blend": _metrics(pred_lgb, y_true),
    }
    if pred_tf is not None and pred_tf_raw is not None:
        rows["tf_raw"] = _metrics(pred_tf_raw, y_true)
        rows["tf_blend"] = _metrics(pred_tf, y_true)
    else:
        rows["tf_raw"] = tf_fallback["tf_raw"]
        rows["tf_blend"] = tf_fallback["tf_blend"]

    mae_cdm = rows["cdm"]["mae"]
    for _name, m in rows.items():
        lift = (mae_cdm - m["mae"]) / mae_cdm * 100 if mae_cdm else 0.0
        m["lift_vs_cdm_pct"] = round(lift, 2)

    cold = test_df["giorni_da_lancio"].astype(float) < 30
    cold_mask = cold.to_numpy()
    cold_stats: dict = {}
    if int(cold_mask.sum()) >= 50:
        cold_stats = {
            "n": int(cold_mask.sum()),
            "naive": _metrics(naive[cold_mask], y_true[cold_mask]),
            "cdm": _metrics(pred_cdm[cold_mask], y_true[cold_mask]),
            "lgbm_blend": _metrics(pred_lgb[cold_mask], y_true[cold_mask]),
        }
        if pred_tf is not None:
            cold_stats["tf_blend"] = _metrics(pred_tf[cold_mask], y_true[cold_mask])
        elif tf_fallback.get("cold_mae") is not None:
            cold_stats["tf_blend"] = {
                "mae": float(tf_fallback["cold_mae"]),
                "mape": None,
                "r2": None,
            }

    order = ["naive", "cdm", "tf_raw", "tf_blend", "lgbm_raw", "lgbm_blend"]
    labels = {
        "naive": "Naive run-rate×15",
        "cdm": "CDM regole",
        "tf_raw": "Rete grezza",
        "tf_blend": "Rete + blend",
        "lgbm_raw": "LightGBM grezzo",
        "lgbm_blend": "LightGBM + blend",
    }
    print(f"Fonte metriche rete: {tf_source}")
    print(f"{'modello':22} {'MAE':>8} {'MAPE%':>8} {'R²':>8} {'vs CDM':>10}")
    print("-" * 60)
    for k in order:
        m = rows[k]
        mape_s = f"{m['mape']:8.1f}" if m.get("mape") is not None else f"{'—':>8}"
        r2_s = f"{m['r2']:8.4f}" if m.get("r2") is not None else f"{'—':>8}"
        print(
            f"{labels[k]:22} {m['mae']:8.2f} {mape_s} {r2_s} "
            f"{m['lift_vs_cdm_pct']:+9.1f}%"
        )

    if cold_stats:
        print(f"\nSKU nuovi (<30g), n={cold_stats['n']:,}")
        for k in ("naive", "cdm", "tf_blend", "lgbm_blend"):
            if k not in cold_stats:
                continue
            print(f"  {labels.get(k, k):22} MAE={cold_stats[k]['mae']:.2f}")

    winner = "LightGBM" if rows["lgbm_blend"]["mae"] < rows["tf_blend"]["mae"] else "Rete TF"
    delta = abs(rows["lgbm_blend"]["mae"] - rows["tf_blend"]["mae"])
    print(f"\n→ Vincente sul test (MAE blend): {winner} (Δ MAE = {delta:.3f} pezzi)")

    # Feature importance
    imp = pd.DataFrame(
        {
            "feature": lgb_feats,
            "importance": lgb_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    imp.to_csv(OUT_IMP, index=False)
    print("\nTop 15 feature LightGBM:")
    print(imp.head(15).to_string(index=False))

    payload = {
        "cut_as_of": str(cut),
        "train_start": str(train_start) if train_start else None,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "tf_metrics_source": tf_source,
        "lgbm_best_iteration": int(lgb_model.best_iteration_ or 0),
        "metrics": rows,
        "cold_sku": cold_stats,
        "winner_blend": winner,
        "delta_mae_blend": round(delta, 4),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSalvato: {OUT_JSON}")
    print(f"Salvato: {OUT_IMP}")


if __name__ == "__main__":
    main()
