#!/usr/bin/env python3
"""Esporta previsioni consultabili per la BI CDM (ultimo as_of di test).

Output:
  reports/data/bi_predictions.json   — righe SKU×PDV (compatto)
  reports/data/bi_rollups.json       — aggregati paese / negozio / categoria

Uso:
  OMP_NUM_THREADS=1 .venv2/bin/python -m production.run_export_bi_predictions
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from production.config import MODEL_DIR, PROJECT_ROOT
from production.operational.routing import apply_calibrated_route, soft_reconcile_frame
# replenishment logic inlined as vectorized _qty below
from testing.blas_threads import limit_native_threads

limit_native_threads()

from testing.config import TrainConfig
from testing.features.build_features import NUMERIC_COLS, apply_train_test_split
from testing.features.schema import BAND_INPUT_COLS, CATEGORICAL_COLS
from testing.inference.confidence import apply_confidence_to_frame, build_analogue_lookup
from testing.model.dual_head_tf import frame_to_tf_inputs
from testing.tensorflow_env import import_tensorflow

ART = MODEL_DIR / "real"
OUT_DIR = PROJECT_ROOT / "reports" / "data"
SAMPLES = ART / "samples_pilot.pkl"
MODEL_PATH = ART / "dual_head_pilot.keras"
ENC_PATH = ART / "encoders_pilot.npz"
BUNDLE = ART / "lgbm_quantile_bundle.joblib"
MAPS = ART / "lgbm_quantile_maps.json"
HIGH = ART / "lgbm_high_volume.joblib"
PARAMS = ART / "ensemble_improved_params.json"

CAT_COLS = list(CATEGORICAL_COLS) + list(BAND_INPUT_COLS.keys())


def _prepare_xy(df: pd.DataFrame, cat_maps: dict):
    out = df.copy()
    for col in CAT_COLS:
        if col not in out.columns:
            out[col] = "NA"
        mapping = cat_maps.get(col, {})
        codes = out[col].astype(str).fillna("NA").map(mapping)
        out[col] = codes.fillna(-1).astype(int)
    for col in NUMERIC_COLS:
        if col not in out.columns:
            out[col] = 0.0
    out["sku_prior"] = (out["run_rate_14d"].astype(float) * 15.0).clip(lower=0.05)
    feats = NUMERIC_COLS + ["sku_prior"] + CAT_COLS
    X = out[feats].copy()
    for col in CAT_COLS:
        X[col] = X[col].astype("category")
    return X, feats


def main() -> None:
    for p in (SAMPLES, BUNDLE, MAPS, HIGH, PARAMS, MODEL_PATH, ENC_PATH):
        if not Path(p).exists():
            raise SystemExit(f"Manca {p}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig()
    cfg.train_window_days = 270
    cfg.horizon_sku_days = 15

    print("=== Load samples ===")
    samples = pd.read_pickle(SAMPLES)
    train_df, test_df, cut, _ = apply_train_test_split(samples, cfg)
    last = test_df["as_of"].max()
    te = test_df.loc[test_df["as_of"] == last].copy()
    print(f"cut>={cut}  as_of={last}  rows={len(te):,}  pdv={te['pdv_id'].nunique()} sku={te['sku_id'].nunique()}")

    params_all = json.loads(PARAMS.read_text())
    deploy = params_all["deploy_params"]
    p80_scale = float(params_all.get("quantile", {}).get("p80_scale", 1.2742))

    print("=== Models ===")
    bundle = joblib.load(BUNDLE)
    maps = {k: {str(a): int(b) for a, b in v.items()} for k, v in json.loads(MAPS.read_text()).items()}
    feats = bundle["features"]
    X, _ = _prepare_xy(te, maps)
    analogue = build_analogue_lookup(train_df, cfg.horizon_sku_days)

    def blend(raw: np.ndarray) -> np.ndarray:
        return apply_confidence_to_frame(te, np.maximum(raw, 0.0), analogue, cfg.horizon_sku_days)[
            "pred_final"
        ].to_numpy()

    pred_tw = blend(bundle["tweedie"].predict(X[feats]))
    pred_p50 = blend(bundle["p50"].predict(X[feats]))
    pred_p80_raw = blend(np.maximum(bundle["p80"].predict(X[feats]), pred_p50))
    pred_tw_rec = soft_reconcile_frame(te, pred_tw, alpha=0.25)

    hi = joblib.load(HIGH)
    pred_high = np.maximum(np.expm1(hi["model"].predict(X[hi["features"]])), 0.0)

    tf = import_tensorflow()
    enc = np.load(ENC_PATH, allow_pickle=True)
    model_tf = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)
    sku_m = tf.keras.Model(inputs=model_tf.inputs, outputs=model_tf.get_layer("y_sku").output)
    x_te = frame_to_tf_inputs(te, enc["maps"].item(), enc["num_mean"], enc["num_std"])
    pred_tf = blend(sku_m.predict(x_te, verbose=0).reshape(-1))

    prior = te["run_rate_14d"].astype(float).to_numpy() * cfg.horizon_sku_days
    naive = prior.copy()
    pred = apply_calibrated_route(prior, pred_tf, pred_tw_rec, naive, deploy, pred_high=pred_high)
    pred_p80 = apply_calibrated_route(
        prior, pred_tf, pred_p80_raw, naive, deploy, pred_high=np.maximum(pred_high, pred_p80_raw)
    )
    pred_p80 = np.maximum(pred_p80, pred) * p80_scale

    y = te["y_sku"].astype(float).to_numpy()
    giac = te["giacenza"].astype(float).to_numpy() if "giacenza" in te.columns else np.zeros(len(te))
    transit = (
        te["transit_time_days"].astype(float).to_numpy()
        if "transit_time_days" in te.columns
        else np.full(len(te), 5.0)
    )
    immature = (
        (te["is_new_sku"].astype(float).to_numpy() >= 0.5)
        | (te["is_new_pdv"].astype(float).to_numpy() >= 0.5)
        if "is_new_sku" in te.columns
        else np.zeros(len(te), dtype=bool)
    )

    def _qty(forecast: np.ndarray, service: np.ndarray) -> np.ndarray:
        daily = service / 15.0
        need = service + daily * transit
        raw = np.maximum(0.0, need - giac)
        raw = np.where(immature, raw * 0.5, raw)
        return np.rint(raw).astype(int)

    qty = _qty(pred, pred)
    qty80 = _qty(pred, pred_p80)
    def s(col: str, default: str = "NA") -> pd.Series:
        if col not in te.columns:
            return pd.Series([default] * len(te))
        return te[col].astype(str).fillna(default)

    out = pd.DataFrame(
        {
            "as_of": str(last),
            "country": s("country"),
            "city": s("city"),
            "store": s("store_name"),
            "pdv_id": s("pdv_id"),
            "sku_id": s("sku_id"),
            "modello": s("modello"),
            "colore": s("colore"),
            "modtescol": (s("modello") + s("tessuto") + s("colore")),
            "tessuto": s("tessuto"),
            "classe": s("classe"),
            "tipologia": s("tipologia"),
            "categoria_id": s("categoria_id"),
            "pred": np.round(pred, 2),
            "pred_p80": np.round(pred_p80, 2),
            "naive": np.round(naive, 2),
            "y": np.round(y, 2),
            "err": np.round(np.abs(pred - y), 2),
            "giacenza": np.round(giac, 1),
            "qty": qty.astype(int),
            "qty_p80": qty80.astype(int),
            "new": immature.astype(int),
        }
    )

    # Compact columnar JSON (much smaller than records)
    cols = list(out.columns)
    payload = {
        "meta": {
            "as_of": str(last),
            "cut": str(cut),
            "n": int(len(out)),
            "n_pdv": int(out["pdv_id"].nunique()),
            "n_sku": int(out["sku_id"].nunique()),
            "n_country": int(out["country"].nunique()),
            "horizon_days": 15,
            "point_forecast": "route_ops_tweedie_rec",
            "p80_scale": p80_scale,
            "mae": round(float(np.abs(pred - y).mean()), 4),
            "mae_ops": round(float(np.abs(pred[y >= 3] - y[y >= 3]).mean()), 4) if (y >= 3).any() else None,
        },
        "cols": cols,
        "rows": out[cols].values.tolist(),
    }
    pred_path = OUT_DIR / "bi_predictions.json"
    pred_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    print(f"Saved {pred_path} ({pred_path.stat().st_size / 1e6:.1f} MB)")

    def rollup(df: pd.DataFrame, key: str) -> list[dict]:
        g = df.groupby(key, dropna=False)
        rows = []
        for k, part in g:
            rows.append(
                {
                    "key": str(k),
                    "n": int(len(part)),
                    "n_sku": int(part["sku_id"].nunique()),
                    "pred_sum": round(float(part["pred"].sum()), 1),
                    "y_sum": round(float(part["y"].sum()), 1),
                    "mae": round(float(part["err"].mean()), 3),
                    "qty_sum": int(part["qty"].sum()),
                    "qty_p80_sum": int(part["qty_p80"].sum()),
                }
            )
        rows.sort(key=lambda r: -r["qty_sum"])
        return rows

    rollups = {
        "meta": payload["meta"],
        "by_country": rollup(out, "country"),
        "by_pdv": rollup(out, "pdv_id"),
        "by_modtescol": rollup(out, "modtescol"),
        "by_tipologia": rollup(out, "tipologia"),
        "by_classe": rollup(out, "classe"),
        "filters": {
            "countries": sorted(out["country"].unique().tolist()),
            "modtescols": sorted(out["modtescol"].unique().tolist()),
            "pdvs": sorted(
                out.groupby("pdv_id")
                .agg(store=("store", "first"), city=("city", "first"), country=("country", "first"))
                .reset_index()
                .to_dict(orient="records"),
                key=lambda r: (r["country"], r["city"], r["pdv_id"]),
            ),
            "tipologie": sorted(out["tipologia"].unique().tolist()),
            "classi": sorted(out["classe"].unique().tolist()),
        },
    }
    # enrich pdv list
    rollups["filters"]["pdvs"] = [
        {
            "pdv_id": r["pdv_id"],
            "store": r["store"],
            "city": r["city"],
            "country": r["country"],
            "label": f"{r['pdv_id']} · {r['store']} ({r['city']}, {r['country']})",
        }
        for r in rollups["filters"]["pdvs"]
    ]
    roll_path = OUT_DIR / "bi_rollups.json"
    roll_path.write_text(json.dumps(rollups, ensure_ascii=False, indent=2))
    print(f"Saved {roll_path}")
    print(f"MAE={payload['meta']['mae']}  MAE_ops={payload['meta']['mae_ops']}")


if __name__ == "__main__":
    main()
