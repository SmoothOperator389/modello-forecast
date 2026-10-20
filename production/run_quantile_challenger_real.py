#!/usr/bin/env python3
"""Challenger avanzati: LGBM Tweedie + quantile P50/P80 + reconcile soft.

Confronta col routing ops già calibrato e salva modelli + metriche operative
(coverage P80, pinball, MAE).

Uso:
  OMP_NUM_THREADS=1 .venv2/bin/python -m production.run_quantile_challenger_real
  FORECAST_PILOT=0 OMP_NUM_THREADS=1 .venv2/bin/python -m production.run_quantile_challenger_real
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
from production.operational.routing import apply_calibrated_route
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
PARAMS_PATH = ART / f"ensemble_improved_params{_SFX}.json"
HIGH_MODEL_PATH = ART / f"lgbm_high_volume{_SFX}.joblib"

OUT_JSON = ART / f"quantile_challenger_metrics{_SFX}.json"
OUT_BANDS = ART / f"quantile_challenger_bands{_SFX}.csv"
OUT_MODELS = ART / f"lgbm_quantile_bundle{_SFX}.joblib"
OUT_MAPS = ART / f"lgbm_quantile_maps{_SFX}.json"

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
        "mean_y": round(float(y.mean()), 4) if len(y) else 0.0,
    }


def _pinball(pred: np.ndarray, y: np.ndarray, alpha: float) -> float:
    e = y - pred
    return float(np.mean(np.where(e >= 0, alpha * e, (alpha - 1.0) * e)))


def _quantile_ops(pred_p50: np.ndarray, pred_p80: np.ndarray, y: np.ndarray) -> dict:
    return {
        "mae_p50": round(float(np.abs(pred_p50 - y).mean()), 4),
        "coverage_p80": round(float(np.mean(y <= pred_p80) * 100), 2),
        "pinball_p50": round(_pinball(pred_p50, y, 0.5), 4),
        "pinball_p80": round(_pinball(pred_p80, y, 0.8), 4),
        "mean_p50": round(float(pred_p50.mean()), 4),
        "mean_p80": round(float(pred_p80.mean()), 4),
        "mean_y": round(float(y.mean()), 4),
    }


def _band_masks(y: np.ndarray) -> dict[str, np.ndarray]:
    return {
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


def _sample_weights(tr: pd.DataFrame, horizon: int) -> np.ndarray:
    w = np.ones(len(tr), dtype=np.float32)
    prior = tr["run_rate_14d"].astype(float).to_numpy() * horizon
    w *= (1.0 + np.log1p(np.clip(prior, 0, None))).astype(np.float32)
    if "is_new_sku" in tr.columns:
        w[tr["is_new_sku"].astype(float).to_numpy() >= 0.5] *= 1.4
    return w


def _fit_lgbm(
    objective: str,
    X_tr,
    y_tr,
    X_va,
    y_va,
    w,
    cfg: TrainConfig,
    *,
    alpha: float | None = None,
    tweedie_power: float | None = None,
    n_estimators: int = 2000,
):
    params: dict = {
        "objective": objective,
        "n_estimators": n_estimators,
        "learning_rate": 0.04,
        "num_leaves": 95,
        "min_child_samples": 50,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.2,
        "random_state": cfg.seed,
        "n_jobs": 2,
        "verbosity": -1,
    }
    if alpha is not None:
        params["alpha"] = alpha
    if tweedie_power is not None:
        params["tweedie_variance_power"] = tweedie_power
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr,
        y_tr,
        sample_weight=w,
        eval_set=[(X_va, y_va)],
        eval_metric="l1",
        categorical_feature=CAT_COLS,
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def soft_reconcile(
    df: pd.DataFrame,
    pred: np.ndarray,
    *,
    keys: tuple[str, ...] = ("pdv_id", "modello"),
    alpha: float = 0.25,
) -> np.ndarray:
    """Shrink pred verso media di gruppo (pdv×modello): riduce incoerenze long-tail."""
    out = df.copy()
    out["_pred"] = np.asarray(pred, dtype=float)
    cols = [k for k in keys if k in out.columns]
    if not cols:
        return out["_pred"].to_numpy()
    gmean = out.groupby(cols, observed=True)["_pred"].transform("mean")
    blended = (1.0 - alpha) * out["_pred"] + alpha * gmean
    return np.maximum(blended.to_numpy(), 0.0)


def main() -> None:
    if not SAMPLES.exists():
        raise SystemExit(f"Manca {SAMPLES}")
    if not PARAMS_PATH.exists():
        raise SystemExit(f"Manca {PARAMS_PATH}: esegui prima run_improve_ensemble_real")

    cfg = TrainConfig()
    cfg.train_window_days = 270 if _PILOT else 365
    cfg.horizon_sku_days = 15
    np.random.seed(cfg.seed)
    deploy = json.loads(PARAMS_PATH.read_text())["deploy_params"]

    print("=== 1) Load sample pilota ===")
    samples = pd.read_pickle(SAMPLES)
    train_df, test_df, cut, train_start = apply_train_test_split(samples, cfg)
    print(f"train={len(train_df):,} test={len(test_df):,} cut>={cut} start={train_start}")
    y = test_df["y_sku"].astype(float).to_numpy()
    prior_te = test_df["run_rate_14d"].astype(float).to_numpy() * cfg.horizon_sku_days
    naive = prior_te.copy()

    print("\n=== 2) Predizioni TF ===")
    tf = import_tensorflow()
    enc = np.load(ENC_PATH, allow_pickle=True)
    maps_tf = enc["maps"].item()
    num_mean, num_std = enc["num_mean"], enc["num_std"]
    model_tf = load_dual_head_model(MODEL_PATH)
    sku_model = tf.keras.Model(inputs=model_tf.inputs, outputs=model_tf.get_layer("y_sku").output)
    analogue = build_analogue_lookup(train_df, cfg.horizon_sku_days)
    x_te = frame_to_tf_inputs(test_df, maps_tf, num_mean, num_std)
    pred_tf = apply_confidence_to_frame(
        test_df,
        np.maximum(sku_model.predict(x_te, verbose=0).reshape(-1), 0.0),
        analogue,
        cfg.horizon_sku_days,
    )["pred_final"].to_numpy()

    print("\n=== 3) Train L1 / Tweedie / P50 / P80 ===")
    mask_val = _val_mask(train_df, cfg.n_val_as_of)
    tr = train_df.loc[~mask_val]
    va = train_df.loc[mask_val]
    X_tr, y_tr, maps, feats = _prepare_xy(tr, fit=True)
    X_va, y_va, _, _ = _prepare_xy(va, cat_maps=maps, fit=False)
    X_te, _, _, _ = _prepare_xy(test_df, cat_maps=maps, fit=False)
    w = _sample_weights(tr, cfg.horizon_sku_days)

    m_l1 = _fit_lgbm("regression_l1", X_tr[feats], y_tr, X_va[feats], y_va, w, cfg)
    print(f"  L1 best_iter={m_l1.best_iteration_}")
    m_tw = _fit_lgbm(
        "tweedie",
        X_tr[feats],
        y_tr,
        X_va[feats],
        y_va,
        w,
        cfg,
        tweedie_power=1.3,
    )
    print(f"  Tweedie best_iter={m_tw.best_iteration_}")
    m_p50 = _fit_lgbm("quantile", X_tr[feats], y_tr, X_va[feats], y_va, w, cfg, alpha=0.5)
    print(f"  P50 best_iter={m_p50.best_iteration_}")
    m_p80 = _fit_lgbm("quantile", X_tr[feats], y_tr, X_va[feats], y_va, w, cfg, alpha=0.8)
    print(f"  P80 best_iter={m_p80.best_iteration_}")

    def _blend_pred(raw: np.ndarray) -> np.ndarray:
        return apply_confidence_to_frame(test_df, np.maximum(raw, 0.0), analogue, cfg.horizon_sku_days)[
            "pred_final"
        ].to_numpy()

    pred_l1 = _blend_pred(m_l1.predict(X_te[feats]))
    pred_tw = _blend_pred(m_tw.predict(X_te[feats]))
    pred_p50 = _blend_pred(m_p50.predict(X_te[feats]))
    pred_p80 = np.maximum(m_p80.predict(X_te[feats]), pred_p50)  # monotonicità soft
    pred_p80 = _blend_pred(pred_p80)

    # high specialist (già trainato)
    if HIGH_MODEL_PATH.exists():
        bundle = joblib.load(HIGH_MODEL_PATH)
        hi_m, hi_feats = bundle["model"], bundle["features"]
        raw_hi = np.expm1(hi_m.predict(X_te[hi_feats])) if bundle.get("log_target") else hi_m.predict(X_te[hi_feats])
        pred_high = np.maximum(raw_hi, 0.0)
    else:
        pred_high = naive.copy()

    pred_tw_rec = soft_reconcile(test_df, pred_tw, alpha=0.25)
    pred_p50_rec = soft_reconcile(test_df, pred_p50, alpha=0.25)

    print("\n=== 4) Routing con slot LGBM sostituito ===")
    routes = {}
    for name, mid in [
        ("route_ops_l1", pred_l1),
        ("route_ops_tweedie", pred_tw),
        ("route_ops_p50", pred_p50),
        ("route_ops_tweedie_rec", pred_tw_rec),
        ("route_ops_p50_rec", pred_p50_rec),
    ]:
        routes[name] = apply_calibrated_route(
            prior_te, pred_tf, mid, naive, deploy, pred_high=pred_high
        )

    # punto + quantili per riassortimento: forecast = P50 route, qty con P80
    route_p50 = routes["route_ops_p50"]
    # P80 routed: stessa logica ma partendo da pred_p80 nel mid/low dove usiamo lgbm
    route_p80 = apply_calibrated_route(
        prior_te, pred_tf, pred_p80, naive, deploy, pred_high=np.maximum(pred_high, pred_p80)
    )
    # safety: p80 >= p50
    route_p80 = np.maximum(route_p80, route_p50)

    models = {
        "naive": naive,
        "tf_blend": pred_tf,
        "lgbm_l1": pred_l1,
        "lgbm_tweedie": pred_tw,
        "lgbm_p50": pred_p50,
        "lgbm_p80": pred_p80,
        "lgbm_tweedie_rec": pred_tw_rec,
        "lgbm_p50_rec": pred_p50_rec,
        **routes,
        "route_ops_p80": route_p80,
    }

    print("\n=== 5) Metriche globali / ops ===")
    global_m = {k: _metrics(v, y) for k, v in models.items()}
    ops = y >= 3
    ops_mae = {k: round(float(np.abs(v[ops] - y[ops]).mean()), 4) for k, v in models.items()}
    for name in [
        "naive",
        "lgbm_l1",
        "lgbm_tweedie",
        "lgbm_p50",
        "route_ops_l1",
        "route_ops_tweedie",
        "route_ops_p50",
        "route_ops_tweedie_rec",
        "route_ops_p50_rec",
    ]:
        m = global_m[name]
        print(
            f"  {name:24} MAE={m['mae']:.3f} ops={ops_mae[name]:.3f} "
            f"wMAPE={m['wmape']:.1f}%"
        )

    qops = {
        "raw": _quantile_ops(pred_p50, pred_p80, y),
        "routed": _quantile_ops(route_p50, route_p80, y),
        "ops_y>=3": _quantile_ops(route_p50[ops], route_p80[ops], y[ops]),
    }
    print("\n=== 6) Quantili (riassortimento) ===")
    print(json.dumps(qops, indent=2))

    print("\n=== 7) Bands ===")
    band_rows = []
    for band, mask in _band_masks(y).items():
        if mask.sum() < 50:
            continue
        print(f"\n  [{band}] n={mask.sum():,}")
        for name in [
            "naive",
            "lgbm_tweedie",
            "lgbm_p50",
            "route_ops_l1",
            "route_ops_tweedie",
            "route_ops_p50",
            "route_ops_tweedie_rec",
        ]:
            m = _metrics(models[name][mask], y[mask])
            print(f"    {name:24} MAE={m['mae']:.3f}")
            band_rows.append({"band": band, "model": name, **m})

    # winner point-forecast: min ops MAE tra route_*
    route_names = [n for n in models if n.startswith("route_ops_") and n != "route_ops_p80"]
    winner_ops = min(route_names, key=lambda n: ops_mae[n])
    winner_global = min(route_names, key=lambda n: global_m[n]["mae"])
    baseline_ops = ops_mae["route_ops_l1"]

    pd.DataFrame(band_rows).to_csv(OUT_BANDS, index=False)
    joblib.dump(
        {
            "l1": m_l1,
            "tweedie": m_tw,
            "p50": m_p50,
            "p80": m_p80,
            "features": feats,
            "deploy_params": deploy,
            "winner_ops": winner_ops,
            "reconcile_alpha": 0.25,
        },
        OUT_MODELS,
    )
    OUT_MAPS.write_text(json.dumps({k: {str(a): b for a, b in v.items()} for k, v in maps.items()}))

    out = {
        "cut": str(cut),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "global": global_m,
        "ops_mae": ops_mae,
        "quantile_ops": qops,
        "winner_ops": winner_ops,
        "winner_global": winner_global,
        "improvement_ops_vs_route_l1_pct": round(
            (baseline_ops - ops_mae[winner_ops]) / max(baseline_ops, 1e-9) * 100, 2
        ),
        "deploy_params": deploy,
        "best_iterations": {
            "l1": int(m_l1.best_iteration_ or 0),
            "tweedie": int(m_tw.best_iteration_ or 0),
            "p50": int(m_p50.best_iteration_ or 0),
            "p80": int(m_p80.best_iteration_ or 0),
        },
        "replenishment_note": (
            "Usa route P50 come previsione centrale; P80 per qty/safety. "
            "Coverage P80 ideale ~80%."
        ),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(
        f"\nWinner ops: {winner_ops} MAE_ops={ops_mae[winner_ops]:.4f} "
        f"(vs route_l1 {baseline_ops:.4f}, Δ={out['improvement_ops_vs_route_l1_pct']}%)"
    )
    print(f"Saved → {OUT_JSON}")
    print(f"Models → {OUT_MODELS}")


if __name__ == "__main__":
    main()
