#!/usr/bin/env python3
"""Inferenza forward: previsione 15gg dopo l'ultima data vendite.

Diverso dall'holdout BI (che valuta as_of storici con y noto).
Qui as_of = max(data) in fact_daily → predice (as_of+1 … as_of+15).
y non esiste ancora (NaN).

Default: top 40 PDV + ensemble deploy (Tweedie/routing), coerente con la BI.
Rete intera: FORECAST_PILOT=0 (solo DualHead full + floor naive sugli alti).

Uso:
  OMP_NUM_THREADS=1 .venv2/bin/python -m production.run_infer_forward_real
  FORECAST_PILOT=0 OMP_NUM_THREADS=1 .venv2/bin/python -m production.run_infer_forward_real
"""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from production.config import MODEL_DIR, PROJECT_ROOT
from production.operational.explain_bi import build_why_text, lgbm_top_drivers, route_zone_label
from production.operational.routing import apply_calibrated_route, soft_reconcile_frame
from production.run_train_real import load_real_tables
from testing.blas_threads import limit_native_threads
limit_native_threads()

from testing.config import TrainConfig
from testing.features.build_features import NUMERIC_COLS, build_training_frames
from testing.features.schema import BAND_INPUT_COLS, CATEGORICAL_COLS
from testing.inference.confidence import apply_confidence_to_frame, build_analogue_lookup
from testing.model.dual_head_tf import frame_to_tf_inputs, load_dual_head_model
from testing.tensorflow_env import import_tensorflow

ART = MODEL_DIR / "real"
OUT_DIR = PROJECT_ROOT / "reports" / "data"
CAT_COLS = list(CATEGORICAL_COLS) + list(BAND_INPUT_COLS.keys())


def _pilot() -> bool:
    return os.environ.get("FORECAST_PILOT", "1").strip().lower() not in {"0", "false", "no", "off"}


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


def _qty(pred: np.ndarray, service: np.ndarray, giac: np.ndarray, transit: np.ndarray, immature: np.ndarray) -> np.ndarray:
    daily = service / 15.0
    need = service + daily * transit
    raw = np.maximum(0.0, need - giac)
    raw = np.where(immature, raw * 0.5, raw)
    return np.rint(raw).astype(int)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pilot = _pilot()
    print(f"=== Inferenza FORWARD ({'pilota 40 PDV + ensemble' if pilot else 'rete intera TF'}) ===")

    fact, calendar, pdv = load_real_tables()
    last_sale = max(pd.to_datetime(fact["data"]).dt.date)
    as_of = last_sale
    horizon_end = as_of + timedelta(days=15)
    print(f"Ultima vendita: {last_sale}")
    print(f"as_of (fine storico): {as_of}")
    print(f"Orizzonte previsto: {as_of + timedelta(days=1)} → {horizon_end}")

    if pilot:
        vol = fact.groupby("pdv_id", observed=True)["vendite_qty"].sum().sort_values(ascending=False)
        keep = set(vol.head(40).index.astype(str))
        fact = fact[fact["pdv_id"].astype(str).isin(keep)].copy()
        pdv = pdv[pdv["pdv_id"].astype(str).isin(keep)].copy()
        print(f"Pilota: {len(keep)} PDV · fact={len(fact):,}")

    cfg = TrainConfig()
    cfg.horizon_sku_days = 15
    print("=== Build feature (require_target=False) ===")
    feat = build_training_frames(fact, calendar, cfg, [as_of], pdv_master=pdv, require_target=False)
    if feat.empty:
        raise SystemExit("Nessuna riga feature: controlla fact/as_of")
    print(f"righe={len(feat):,}  sku={feat['sku_id'].nunique():,}  pdv={feat['pdv_id'].nunique()}")

    # Analogue lookup: usa sample storico se c'è, altrimenti lo stesso feat
    samples_path = ART / ("samples_pilot.pkl" if pilot else "samples.pkl")
    if samples_path.exists():
        hist = pd.read_pickle(samples_path)
        # solo as_of precedenti all'inferenza
        hist = hist[hist["as_of"] < as_of] if "as_of" in hist.columns else hist
        analogue = build_analogue_lookup(hist if len(hist) else feat, cfg.horizon_sku_days)
    else:
        analogue = build_analogue_lookup(feat, cfg.horizon_sku_days)

    prior = feat["run_rate_14d"].astype(float).to_numpy() * cfg.horizon_sku_days
    naive = prior.copy()

    def blend(raw: np.ndarray) -> np.ndarray:
        return apply_confidence_to_frame(feat, np.maximum(raw, 0.0), analogue, cfg.horizon_sku_days)[
            "pred_final"
        ].to_numpy()

    tf = import_tensorflow()
    if pilot:
        model_path, enc_path = ART / "dual_head_pilot.keras", ART / "encoders_pilot.npz"
    else:
        model_path, enc_path = ART / "dual_head.keras", ART / "encoders.npz"
    enc = np.load(enc_path, allow_pickle=True)
    model_tf = load_dual_head_model(model_path)
    sku_m = tf.keras.Model(inputs=model_tf.inputs, outputs=model_tf.get_layer("y_sku").output)
    x = frame_to_tf_inputs(feat, enc["maps"].item(), enc["num_mean"], enc["num_std"])
    pred_tf = blend(sku_m.predict(x, verbose=0).reshape(-1))

    explain_labs: list[list[str]] = [[] for _ in range(len(feat))]
    explain_dels: list[list[float]] = [[] for _ in range(len(feat))]
    route_msgs: list[str] = ["motore TF"] * len(feat)
    low_thr, high_thr = 2.0, 18.0

    _sfx = "" if pilot else "_full"
    bundle_path = ART / f"lgbm_quantile_bundle{_sfx}.joblib"
    if bundle_path.exists():
        print(f"=== Ensemble deploy (Tweedie + routing){' rete intera' if not pilot else ''} ===")
        bundle = joblib.load(bundle_path)
        maps = {
            k: {str(a): int(b) for a, b in v.items()}
            for k, v in json.loads((ART / f"lgbm_quantile_maps{_sfx}.json").read_text()).items()
        }
        params = json.loads((ART / f"ensemble_improved_params{_sfx}.json").read_text())
        deploy = params["deploy_params"]
        p80_scale = float(params.get("quantile", {}).get("p80_scale", 1.2742))
        low_thr = float(deploy.get("low_thr", 2.0))
        high_thr = float(deploy.get("high_thr", 18.0))
        X, feats = _prepare_xy(feat, maps)
        pred_tw = blend(bundle["tweedie"].predict(X[feats]))
        pred_p50 = blend(bundle["p50"].predict(X[feats]))
        pred_p80_raw = blend(np.maximum(bundle["p80"].predict(X[feats]), pred_p50))
        pred_tw_rec = soft_reconcile_frame(feat, pred_tw, alpha=0.25)
        hi = joblib.load(ART / f"lgbm_high_volume{_sfx}.joblib")
        pred_high = np.maximum(np.expm1(hi["model"].predict(X[hi["features"]])), 0.0)
        pred = apply_calibrated_route(prior, pred_tf, pred_tw_rec, naive, deploy, pred_high=pred_high)
        pred_p80 = apply_calibrated_route(
            prior, pred_tf, pred_p80_raw, naive, deploy, pred_high=np.maximum(pred_high, pred_p80_raw)
        )
        pred_p80 = np.maximum(pred_p80, pred) * p80_scale
        engine = "route_ops_tweedie_rec"
        print("=== Explain (contributi Tweedie) ===")
        explain_labs, explain_dels = lgbm_top_drivers(bundle["tweedie"], X, feats, top_k=4)
        route_msgs = [route_zone_label(float(p), low_thr, high_thr)[1] for p in prior]
    else:
        print("=== TF + floor naive high ===")
        from production.operational.routing import floor_toward_naive

        pred = floor_toward_naive(pred_tf, naive, prior, prior_thr=15.0, floor_ratio=0.9)
        pred_p80 = np.maximum(pred * 1.27, pred)
        p80_scale = 1.27
        engine = "tf_full_floor_naive" if not pilot else "tf_pilot_floor_naive"
        route_msgs = [
            "volumi alti → floor verso naive" if float(p) > 15 else "motore DualHead TF"
            for p in prior
        ]
        for i in range(len(feat)):
            labs, dels = [], []
            rr = float(feat["run_rate_14d"].iloc[i]) if "run_rate_14d" in feat.columns else 0.0
            labs.append("ritmo vendite 14gg")
            dels.append(round(rr * 15.0, 2))
            tr = float(feat["trend_7v14"].iloc[i]) if "trend_7v14" in feat.columns else 0.0
            if abs(tr) > 0.05:
                labs.append("trend 7 vs 14gg")
                dels.append(round(tr * max(float(naive[i]), 1.0) * 0.3, 2))
            explain_labs[i] = labs[:4]
            explain_dels[i] = dels[:4]

    giac = feat["giacenza"].astype(float).to_numpy() if "giacenza" in feat.columns else np.zeros(len(feat))
    transit = (
        feat["transit_time_days"].astype(float).to_numpy()
        if "transit_time_days" in feat.columns
        else np.full(len(feat), 5.0)
    )
    immature = np.zeros(len(feat), dtype=bool)
    if "is_new_sku" in feat.columns:
        immature |= feat["is_new_sku"].astype(float).to_numpy() >= 0.5
    if "is_new_pdv" in feat.columns:
        immature |= feat["is_new_pdv"].astype(float).to_numpy() >= 0.5
    qty = _qty(pred, pred, giac, transit, immature)
    qty80 = _qty(pred, pred_p80, giac, transit, immature)

    # Orizzonte 90gg: modello nativo isolato (non modifica il 15gg)
    pred_90 = np.zeros(len(feat), dtype=float)
    qty_90 = np.zeros(len(feat), dtype=int)
    h90_meta: dict = {}
    try:
        from production.inference_h90 import predict_h90, HORIZON as H90, ART_H90

        if (ART_H90 / "lgbm_bundle.joblib").exists():
            print("=== Native h90 (3 mesi) ===")
            pred_90 = predict_h90(feat)
            qty_90 = np.maximum(np.rint(pred_90 - giac), 0).astype(int)
            h90_meta = {
                "horizon_90_days": H90,
                "horizon_90_start": str(as_of + timedelta(days=1)),
                "horizon_90_end": str(as_of + timedelta(days=H90)),
                "point_forecast_90": "native_h90",
            }
            print(f"  pred_90 sum={float(pred_90.sum()):.0f}")
        else:
            print("=== h90 saltato: manca artifacts/real/h90/lgbm_bundle.joblib ===")
    except Exception as exc:  # noqa: BLE001 — BI forward non deve fallire se h90 assente
        print(f"=== h90 non disponibile: {exc} ===")

    why_texts = [
        build_why_text(
            pred=float(pred[i]),
            naive=float(naive[i]),
            route_msg=route_msgs[i],
            driver_labels=explain_labs[i],
            driver_deltas=explain_dels[i],
            giacenza=float(giac[i]),
            is_new=bool(immature[i]),
        )
        for i in range(len(feat))
    ]
    # serializza top-3 driver come stringhe compatte "label|delta;..."
    def _pack(labs: list[str], dels: list[float]) -> str:
        return ";".join(f"{a}|{b}" for a, b in zip(labs[:3], dels[:3]))

    drivers_pack = [_pack(explain_labs[i], explain_dels[i]) for i in range(len(feat))]

    def s(col: str, default: str = "NA") -> pd.Series:
        if col not in feat.columns:
            return pd.Series([default] * len(feat))
        return feat[col].astype(str).fillna(default)

    out = pd.DataFrame(
        {
            "as_of": str(as_of),
            "horizon_start": str(as_of + timedelta(days=1)),
            "horizon_end": str(horizon_end),
            "country": s("country"),
            "city": s("city"),
            "store": s("store_name"),
            "pdv_id": s("pdv_id"),
            "sku_id": s("sku_id"),
            "modello": s("modello"),
            "tessuto": s("tessuto"),
            "colore": s("colore"),
            "modtescol": (s("modello") + s("tessuto") + s("colore")),
            "classe": s("classe"),
            "tipologia": s("tipologia"),
            "categoria_id": s("categoria_id"),
            "pred": np.round(pred, 2),
            "pred_p80": np.round(pred_p80, 2),
            "pred_90": np.round(pred_90, 2),
            "naive": np.round(naive, 2),
            "y": np.nan,  # futuro: non ancora osservato
            "err": np.nan,
            "giacenza": np.round(giac, 1),
            "qty": qty.astype(int),
            "qty_p80": qty80.astype(int),
            "qty_90": qty_90.astype(int),
            "new": immature.astype(int),
            "why": why_texts,
            "drivers": drivers_pack,
        }
    )

    bi_cols = [
        "as_of", "country", "city", "store", "pdv_id", "sku_id",
        "modello", "tessuto", "colore", "modtescol",
        "classe", "tipologia", "categoria_id",
        "pred", "pred_p80", "pred_90", "naive", "y", "err",
        "giacenza", "qty", "qty_p80", "qty_90", "new", "why", "drivers",
    ]
    # JSON browser-safe: niente NaN
    records = out[bi_cols].astype(object).where(out[bi_cols].notna(), None)
    # force None for remaining float nan
    rows = []
    for row in records.itertuples(index=False, name=None):
        rows.append([None if (isinstance(x, float) and (x != x)) else x for x in row])
    payload = {
        "meta": {
            "mode": "forward",
            "as_of": str(as_of),
            "horizon_start": str(as_of + timedelta(days=1)),
            "horizon_end": str(horizon_end),
            "n": int(len(out)),
            "n_pdv": int(out["pdv_id"].nunique()),
            "n_sku": int(out["sku_id"].nunique()),
            "n_country": int(out["country"].nunique()),
            "horizon_days": 15,
            "point_forecast": engine,
            "p80_scale": p80_scale,
            "pilot": pilot,
            "mae": None,
            "mae_ops": None,
            "modtescol_def": "modello + tessuto + colore (no taglia)",
            "note": "Previsione operativa: y non disponibile (15gg successivi all'ultima vendita).",
            **h90_meta,
        },
        "cols": bi_cols,
        "rows": rows,
    }
    pred_path = OUT_DIR / "bi_predictions.json"
    pred_path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    # anche copia dedicata
    (OUT_DIR / "bi_predictions_forward.json").write_text(pred_path.read_text())
    print(f"Saved {pred_path} ({pred_path.stat().st_size/1e6:.1f} MB)")

    def rollup(df: pd.DataFrame, key: str) -> list[dict]:
        rows = []
        for k, part in df.groupby(key, dropna=False):
            rows.append(
                {
                    "key": str(k),
                    "n": int(len(part)),
                    "n_sku": int(part["sku_id"].nunique()),
                    "pred_sum": round(float(part["pred"].sum()), 1),
                    "pred_90_sum": round(float(part["pred_90"].sum()), 1),
                    "y_sum": None,
                    "mae": None,
                    "qty_sum": int(part["qty"].sum()),
                    "qty_p80_sum": int(part["qty_p80"].sum()),
                    "qty_90_sum": int(part["qty_90"].sum()),
                }
            )
        rows.sort(key=lambda r: -r["pred_90_sum"])
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
            "pdvs": [
                {
                    "pdv_id": r["pdv_id"],
                    "store": r["store"],
                    "city": r["city"],
                    "country": r["country"],
                    "label": f"{r['pdv_id']} · {r['store']} ({r['city']}, {r['country']})",
                }
                for r in sorted(
                    out.groupby("pdv_id")
                    .agg(store=("store", "first"), city=("city", "first"), country=("country", "first"))
                    .reset_index()
                    .to_dict(orient="records"),
                    key=lambda r: (r["country"], r["city"], r["pdv_id"]),
                )
            ],
            "tipologie": sorted(out["tipologia"].unique().tolist()),
            "classi": sorted(out["classe"].unique().tolist()),
        },
    }
    (OUT_DIR / "bi_rollups.json").write_text(json.dumps(rollups, ensure_ascii=False, indent=2))
    print(f"pred_sum={out['pred'].sum():.0f}  qty_sum={out['qty'].sum()}  engine={engine}")
    print(f"Orizzonte CDM: {payload['meta']['horizon_start']} → {payload['meta']['horizon_end']}")


if __name__ == "__main__":
    main()
