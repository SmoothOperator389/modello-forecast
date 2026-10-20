#!/usr/bin/env python3
"""Esporta snapshot BI storici (as_of del passato) con previsione vs reale.

Scrive in ``reports/data/holdout/``:
  - index.json          — elenco as_of disponibili
  - bi_YYYY-MM-DD.json  — stesso schema della BI forward + y / y_90

Non modifica ``bi_predictions.json`` (forward operativo).

Uso:
  FORECAST_PILOT=0 OMP_NUM_THREADS=1 .venv2/bin/python -m production.run_export_bi_holdout
  BI_HOLDOUT_N=8 …   # quanti as_of più recenti (default 12)
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from production.config import MODEL_DIR, PROJECT_ROOT
from production.operational.routing import apply_calibrated_route, soft_reconcile_frame
from production.run_train_real import load_real_tables
from testing.blas_threads import limit_native_threads

limit_native_threads()

from testing.config import TrainConfig
from testing.features.build_features import NUMERIC_COLS
from testing.features.schema import BAND_INPUT_COLS, CATEGORICAL_COLS
from testing.inference.confidence import apply_confidence_to_frame, build_analogue_lookup
from testing.model.dual_head_tf import frame_to_tf_inputs, load_dual_head_model
from testing.tensorflow_env import import_tensorflow

ART = MODEL_DIR / "real"
OUT = PROJECT_ROOT / "reports" / "data" / "holdout"
CAT_COLS = list(CATEGORICAL_COLS) + list(BAND_INPUT_COLS.keys())
H15 = 15
H90 = 90


def _pilot() -> bool:
    return os.environ.get("FORECAST_PILOT", "0").strip().lower() not in {"0", "false", "no", "off"}


def _n_holdout() -> int:
    try:
        return max(1, int(os.environ.get("BI_HOLDOUT_N", "12")))
    except ValueError:
        return 12


def _prepare_xy(df: pd.DataFrame, cat_maps: dict, horizon: int = 15):
    out = df.copy()
    for col in CAT_COLS:
        if col not in out.columns:
            out[col] = "NA"
        out[col] = out[col].astype(str).fillna("NA").map(cat_maps.get(col, {})).fillna(-1).astype(int)
    for col in NUMERIC_COLS:
        if col not in out.columns:
            out[col] = 0.0
    out["sku_prior"] = (out["run_rate_14d"].astype(float) * float(horizon)).clip(lower=0.05)
    feats = NUMERIC_COLS + ["sku_prior"] + CAT_COLS
    X = out[feats].copy()
    for col in CAT_COLS:
        X[col] = X[col].astype("category")
    return X, feats


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    mae = float(np.abs(p - y).mean()) if len(y) else 0.0
    wmape = float(np.abs(p - y).sum() / max(float(y.sum()), 1.0) * 100) if len(y) else 0.0
    return {
        "n": int(len(y)),
        "sum_y": round(float(y.sum()), 1),
        "sum_pred": round(float(p.sum()), 1),
        "mae": round(mae, 4),
        "wmape": round(wmape, 2),
        "bias": round(float((p - y).sum()), 1),
    }


def main() -> None:
    pilot = _pilot()
    sfx = "" if pilot else "_full"
    samples_path = ART / ("samples_pilot.pkl" if pilot else "samples.pkl")
    model_path = ART / ("dual_head_pilot.keras" if pilot else "dual_head.keras")
    enc_path = ART / ("encoders_pilot.npz" if pilot else "encoders.npz")
    bundle_path = ART / f"lgbm_quantile_bundle{sfx}.joblib"
    maps_path = ART / f"lgbm_quantile_maps{sfx}.json"
    high_path = ART / f"lgbm_high_volume{sfx}.joblib"
    params_path = ART / f"ensemble_improved_params{sfx}.json"

    for p in (samples_path, model_path, enc_path, bundle_path, maps_path, high_path, params_path):
        if not p.exists():
            raise SystemExit(f"Manca {p}")

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig()
    cfg.horizon_sku_days = H15

    print("=== Load samples ===")
    samples = pd.read_pickle(samples_path)
    samples["as_of"] = pd.to_datetime(samples["as_of"]).dt.date
    as_ofs = sorted(samples["as_of"].unique())
    # servono 15gg di futuro osservato → escludi as_of troppo recenti vs max fact
    print("=== Load fact (per y_90 + cutoff) ===")
    fact, _cal, _pdv = load_real_tables()
    fact["data"] = pd.to_datetime(fact["data"]).dt.date
    max_d = max(fact["data"])
    as_ofs_15 = [a for a in as_ofs if a + timedelta(days=H15) <= max_d]
    n_keep = _n_holdout()
    as_ofs_use = as_ofs_15[-n_keep:]
    print(f"as_of candidati 15gg: {len(as_ofs_15)} → esportati ultimi {len(as_ofs_use)}: {as_ofs_use[0]} → {as_ofs_use[-1]}")

    params_all = json.loads(params_path.read_text())
    deploy = params_all["deploy_params"]
    p80_scale = float(params_all.get("quantile", {}).get("p80_scale", 1.23))

    print("=== Load models (una volta) ===")
    bundle = joblib.load(bundle_path)
    maps15 = {k: {str(a): int(b) for a, b in v.items()} for k, v in json.loads(maps_path.read_text()).items()}
    feats15 = bundle["features"]
    hi = joblib.load(high_path)
    tf = import_tensorflow()
    enc = np.load(enc_path, allow_pickle=True)
    model_tf = load_dual_head_model(model_path)
    sku_m = tf.keras.Model(inputs=model_tf.inputs, outputs=model_tf.get_layer("y_sku").output)

    h90_bundle = h90_maps = None
    h90_dir = ART / "h90"
    if (h90_dir / "lgbm_bundle.joblib").exists():
        from production.inference_h90 import load_h90_bundle, predict_h90

        h90_bundle, h90_maps = load_h90_bundle(pilot=False)
        print("h90: OK")
    else:
        predict_h90 = None  # type: ignore
        print("h90: assente")

    # fact ridotto per y_90
    fact_y = fact[["data", "pdv_id", "sku_id", "vendite_qty"]].copy()
    fact_y["pdv_id"] = fact_y["pdv_id"].astype(str)
    fact_y["sku_id"] = fact_y["sku_id"].astype(str)

    index_entries = []
    for i, as_of in enumerate(as_ofs_use):
        print(f"\n=== [{i+1}/{len(as_ofs_use)}] as_of={as_of} ===", flush=True)
        te = samples[samples["as_of"] == as_of].copy()
        if te.empty:
            continue
        # analogue: solo passato
        hist = samples[samples["as_of"] < as_of]
        analogue = build_analogue_lookup(hist if len(hist) else te, H15)

        def blend(raw: np.ndarray, frame: pd.DataFrame = te) -> np.ndarray:
            return apply_confidence_to_frame(frame, np.maximum(raw, 0.0), analogue, H15)[
                "pred_final"
            ].to_numpy()

        X, _ = _prepare_xy(te, maps15, horizon=H15)
        pred_tw = blend(bundle["tweedie"].predict(X[feats15]))
        pred_p50 = blend(bundle["p50"].predict(X[feats15]))
        pred_p80_raw = blend(np.maximum(bundle["p80"].predict(X[feats15]), pred_p50))
        pred_tw_rec = soft_reconcile_frame(te, pred_tw, alpha=0.25)
        pred_high = np.maximum(np.expm1(hi["model"].predict(X[hi["features"]])), 0.0)
        x_te = frame_to_tf_inputs(te, enc["maps"].item(), enc["num_mean"], enc["num_std"])
        pred_tf = blend(sku_m.predict(x_te, verbose=0).reshape(-1))
        prior = te["run_rate_14d"].astype(float).to_numpy() * H15
        naive = prior.copy()
        pred = apply_calibrated_route(prior, pred_tf, pred_tw_rec, naive, deploy, pred_high=pred_high)
        pred_p80 = apply_calibrated_route(
            prior, pred_tf, pred_p80_raw, naive, deploy, pred_high=np.maximum(pred_high, pred_p80_raw)
        )
        pred_p80 = np.maximum(pred_p80, pred) * p80_scale
        y15 = te["y_sku"].astype(float).to_numpy()

        pred_90 = np.zeros(len(te), dtype=float)
        y90 = np.full(len(te), np.nan)
        has_y90 = as_of + timedelta(days=H90) <= max_d
        if predict_h90 is not None and h90_bundle is not None:
            pred_90 = predict_h90(te, bundle=h90_bundle, maps=h90_maps)
        if has_y90:
            c0, c1 = as_of + timedelta(days=1), as_of + timedelta(days=H90)
            sub = fact_y[(fact_y["data"] >= c0) & (fact_y["data"] <= c1)]
            g = sub.groupby(["pdv_id", "sku_id"], as_index=False, observed=True).agg(
                y_90=("vendite_qty", "sum")
            )
            tmp = te[["pdv_id", "sku_id"]].astype(str).merge(g, on=["pdv_id", "sku_id"], how="left")
            y90 = tmp["y_90"].fillna(0.0).to_numpy()

        giac = te["giacenza"].astype(float).to_numpy() if "giacenza" in te.columns else np.zeros(len(te))
        transit = (
            te["transit_time_days"].astype(float).to_numpy()
            if "transit_time_days" in te.columns
            else np.full(len(te), 5.0)
        )
        immature = np.zeros(len(te), dtype=bool)
        if "is_new_sku" in te.columns:
            immature |= te["is_new_sku"].astype(float).to_numpy() >= 0.5
        if "is_new_pdv" in te.columns:
            immature |= te["is_new_pdv"].astype(float).to_numpy() >= 0.5

        def _qty(service: np.ndarray) -> np.ndarray:
            daily = service / float(H15)
            need = service + daily * transit
            raw = np.maximum(0.0, need - giac)
            raw = np.where(immature, raw * 0.5, raw)
            return np.rint(raw).astype(int)

        qty = _qty(pred)
        qty80 = _qty(pred_p80)
        qty90 = np.maximum(np.rint(pred_90 - giac), 0).astype(int)

        def s(col: str, default: str = "NA") -> pd.Series:
            if col not in te.columns:
                return pd.Series([default] * len(te))
            return te[col].astype(str).fillna(default)

        out = pd.DataFrame(
            {
                "as_of": str(as_of),
                "country": s("country"),
                "city": s("city"),
                "store": s("store_name"),
                "pdv_id": s("pdv_id"),
                "sku_id": s("sku_id"),
                "modello": s("modello"),
                "tessuto": s("tessuto"),
                "colore": s("colore"),
                "modtescol": s("modello") + s("tessuto") + s("colore"),
                "classe": s("classe"),
                "tipologia": s("tipologia"),
                "categoria_id": s("categoria_id"),
                "pred": np.round(pred, 2),
                "pred_p80": np.round(pred_p80, 2),
                "pred_90": np.round(pred_90, 2),
                "naive": np.round(naive, 2),
                "y": np.round(y15, 2),
                "y_90": np.round(y90, 2) if has_y90 else None,
                "err": np.round(np.abs(pred - y15), 2),
                "err_90": np.round(np.abs(pred_90 - y90), 2) if has_y90 else None,
                "giacenza": np.round(giac, 1),
                "qty": qty,
                "qty_p80": qty80,
                "qty_90": qty90,
                "new": immature.astype(int),
            }
        )
        # None-safe for JSON
        if not has_y90:
            out["y_90"] = None
            out["err_90"] = None

        m15 = _metrics(y15, pred)
        m90 = _metrics(y90, pred_90) if has_y90 else None
        print(f"  rows={len(out):,}  MAE15={m15['mae']} wMAPE15={m15['wmape']}%", end="")
        if m90:
            print(f"  MAE90={m90['mae']} wMAPE90={m90['wmape']}%")
        else:
            print("  (no y_90)")

        cols = list(out.columns)
        rows = []
        for row in out.itertuples(index=False, name=None):
            rows.append([None if (isinstance(x, float) and x != x) else x for x in row])

        payload = {
            "meta": {
                "mode": "holdout",
                "as_of": str(as_of),
                "horizon_start": str(as_of + timedelta(days=1)),
                "horizon_end": str(as_of + timedelta(days=H15)),
                "horizon_days": H15,
                "horizon_90_days": H90 if has_y90 else None,
                "horizon_90_start": str(as_of + timedelta(days=1)) if has_y90 else None,
                "horizon_90_end": str(as_of + timedelta(days=H90)) if has_y90 else None,
                "n": int(len(out)),
                "n_pdv": int(out["pdv_id"].nunique()),
                "n_sku": int(out["sku_id"].nunique()),
                "n_country": int(out["country"].nunique()),
                "point_forecast": "route_ops_tweedie_rec",
                "point_forecast_90": "native_h90" if predict_h90 is not None else None,
                "modtescol_def": "modello + tessuto + colore (no taglia)",
                "mae": m15["mae"],
                "wmape": m15["wmape"],
                "mae_ops": round(float(np.abs(pred[y15 >= 3] - y15[y15 >= 3]).mean()), 4)
                if (y15 >= 3).any()
                else None,
                "metrics_15": m15,
                "metrics_90": m90,
                "note": "Holdout: previsione all'as_of confrontata con venduto reale successivo.",
            },
            "cols": cols,
            "rows": rows,
        }
        out_path = OUT / f"bi_{as_of.isoformat()}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
        print(f"  saved {out_path.name} ({out_path.stat().st_size/1e6:.1f} MB)")

        index_entries.append(
            {
                "as_of": str(as_of),
                "file": out_path.name,
                "horizon_start": payload["meta"]["horizon_start"],
                "horizon_end": payload["meta"]["horizon_end"],
                "horizon_90_end": payload["meta"]["horizon_90_end"],
                "has_y90": has_y90,
                "n": payload["meta"]["n"],
                "mae_15": m15["mae"],
                "wmape_15": m15["wmape"],
                "mae_90": m90["mae"] if m90 else None,
                "wmape_90": m90["wmape"] if m90 else None,
                "label": f"{as_of} → +15gg ({payload['meta']['horizon_start']}…{payload['meta']['horizon_end']})"
                + (f" · +90gg" if has_y90 else ""),
            }
        )

    index = {
        "generated": str(date.today()),
        "fact_max": str(max_d),
        "n": len(index_entries),
        "snapshots": index_entries,
        "note": "Seleziona un as_of in BI (modalità Storico) per confrontare previsione vs reale.",
    }
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))
    print(f"\nIndex: {OUT / 'index.json'} ({len(index_entries)} snapshot)")


if __name__ == "__main__":
    main()
