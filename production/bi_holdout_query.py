"""Genera / recupera snapshot holdout per la BI (anche on-demand).

Usato da ``reports/serve_bi.py`` e dallo script di export batch.
"""

from __future__ import annotations

import json
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from production.config import MODEL_DIR, PROJECT_ROOT

ART = MODEL_DIR / "real"
OUT = PROJECT_ROOT / "reports" / "data" / "holdout"
FEAT_DAILY = OUT / "feat_daily"
H15, H90 = 15, 90

_lock = threading.Lock()
_state: dict[str, Any] = {"ready": False}


def _parse_date(s: str) -> date:
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


def list_sample_as_ofs() -> list[date]:
    ensure_models()
    return list(_state["as_ofs"])


def ensure_models() -> None:
    if _state.get("ready"):
        return
    with _lock:
        if _state.get("ready"):
            return
        from testing.blas_threads import limit_native_threads

        limit_native_threads()
        from testing.config import TrainConfig
        from testing.features.build_features import NUMERIC_COLS
        from testing.features.schema import BAND_INPUT_COLS, CATEGORICAL_COLS
        from testing.inference.confidence import apply_confidence_to_frame, build_analogue_lookup
        from testing.model.dual_head_tf import frame_to_tf_inputs, load_dual_head_model
        from testing.tensorflow_env import import_tensorflow
        from production.operational.routing import apply_calibrated_route, soft_reconcile_frame
        from production.run_train_real import load_real_tables

        samples = pd.read_pickle(ART / "samples.pkl")
        samples["as_of"] = pd.to_datetime(samples["as_of"]).dt.date
        as_ofs = sorted(samples["as_of"].unique())

        params = json.loads((ART / "ensemble_improved_params_full.json").read_text())
        bundle = joblib.load(ART / "lgbm_quantile_bundle_full.joblib")
        maps = {
            k: {str(a): int(b) for a, b in v.items()}
            for k, v in json.loads((ART / "lgbm_quantile_maps_full.json").read_text()).items()
        }
        hi = joblib.load(ART / "lgbm_high_volume_full.joblib")
        tf = import_tensorflow()
        enc = np.load(ART / "encoders.npz", allow_pickle=True)
        model_tf = load_dual_head_model(ART / "dual_head.keras")
        sku_m = tf.keras.Model(inputs=model_tf.inputs, outputs=model_tf.get_layer("y_sku").output)

        h90_bundle = h90_maps = None
        try:
            from production.inference_h90 import load_h90_bundle

            h90_bundle, h90_maps = load_h90_bundle(pilot=False)
        except Exception:
            pass

        fact, calendar, pdv = load_real_tables()
        fact["data"] = pd.to_datetime(fact["data"]).dt.date
        calendar = calendar.copy()
        calendar["data"] = pd.to_datetime(calendar["data"]).dt.date
        fact_y = fact[["data", "pdv_id", "sku_id", "vendite_qty"]].copy()
        fact_y["pdv_id"] = fact_y["pdv_id"].astype(str)
        fact_y["sku_id"] = fact_y["sku_id"].astype(str)
        max_d = max(fact_y["data"])
        min_as_of = as_ofs[0] if as_ofs else min(fact_y["data"])

        _state.update(
            {
                "ready": True,
                "samples": samples,
                "as_ofs": as_ofs,
                "params": params,
                "bundle": bundle,
                "maps": maps,
                "hi": hi,
                "enc": enc,
                "sku_m": sku_m,
                "h90_bundle": h90_bundle,
                "h90_maps": h90_maps,
                "fact": fact,
                "calendar": calendar,
                "pdv": pdv,
                "fact_y": fact_y,
                "max_d": max_d,
                "min_as_of": min_as_of,
                "daily_as_of": True,
                "NUMERIC_COLS": NUMERIC_COLS,
                "CAT_COLS": list(CATEGORICAL_COLS) + list(BAND_INPUT_COLS.keys()),
                "apply_confidence_to_frame": apply_confidence_to_frame,
                "build_analogue_lookup": build_analogue_lookup,
                "frame_to_tf_inputs": frame_to_tf_inputs,
                "apply_calibrated_route": apply_calibrated_route,
                "soft_reconcile_frame": soft_reconcile_frame,
                "TrainConfig": TrainConfig,
            }
        )


def clamp_as_of(target: date) -> date:
    """as_of giornaliero: qualsiasi giorno nello storico utile (non solo scaglioni 14gg)."""
    ensure_models()
    min_a: date = _state["min_as_of"]
    max_d: date = _state["max_d"]
    # serve almeno 14gg di storico prima + tipicamente y dopo; max = ultimo giorno fact
    if target < min_a:
        return min_a
    if target > max_d:
        return max_d
    return target


def nearest_as_of(target: date, *, before: bool = True) -> date:
    """Compat: con daily_as_of=True non snappa più alla griglia 14gg."""
    ensure_models()
    if _state.get("daily_as_of"):
        t = clamp_as_of(target)
        if before and t > target:
            # clamp ha alzato al minimo
            return t
        return t
    as_ofs: list[date] = _state["as_ofs"]
    max_d: date = _state["max_d"]
    usable = [a for a in as_ofs if a + timedelta(days=H15) <= max_d]
    if not usable:
        raise ValueError("Nessun as_of con 15gg di reale disponibile")
    if before:
        cands = [a for a in usable if a <= target]
        if not cands:
            return usable[0]
        return cands[-1]
    return min(usable, key=lambda a: abs((a - target).days))


def _frame_for_as_of(as_of: date) -> tuple[pd.DataFrame, str]:
    """Ritorna (te, source) con source in {precomputed, daily_ondemand}."""
    ensure_models()
    samples: pd.DataFrame = _state["samples"]
    te = samples[samples["as_of"] == as_of].copy()
    if not te.empty:
        return te, "precomputed"

    FEAT_DAILY.mkdir(parents=True, exist_ok=True)
    feat_path = FEAT_DAILY / f"feat_{as_of.isoformat()}.pkl"
    if feat_path.exists():
        te = pd.read_pickle(feat_path)
        te["as_of"] = pd.to_datetime(te["as_of"]).dt.date
        return te, "daily_ondemand_cache"

    from testing.features.build_features import build_training_frames

    cfg = _state["TrainConfig"]()
    cfg.horizon_sku_days = H15
    te = build_training_frames(
        _state["fact"],
        _state["calendar"],
        cfg,
        [as_of],
        pdv_master=_state["pdv"],
        require_target=False,
    )
    if te.empty:
        raise ValueError(
            f"as_of {as_of}: nessuna riga feature (serve storico vendite nelle 2 settimane precedenti)"
        )
    te["as_of"] = as_of
    te.to_pickle(feat_path)
    return te, "daily_ondemand"


def _prepare_xy(df: pd.DataFrame, cat_maps: dict, horizon: int = 15):
    NUMERIC_COLS = _state["NUMERIC_COLS"]
    CAT_COLS = _state["CAT_COLS"]
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
    return {
        "n": int(len(y)),
        "sum_y": round(float(y.sum()), 1),
        "sum_pred": round(float(p.sum()), 1),
        "mae": round(float(np.abs(p - y).mean()) if len(y) else 0.0, 4),
        "wmape": round(float(np.abs(p - y).sum() / max(float(y.sum()), 1.0) * 100) if len(y) else 0.0, 2),
        "bias": round(float((p - y).sum()), 1),
    }


def generate_as_of_snapshot(as_of: date, *, cache: bool = True) -> dict:
    """Predizione all'as_of vs y_15 (e y_90 se possibile). Cache su disco.

    as_of può essere **qualsiasi giorno** nello storico: se non è nella griglia
    biweekly dei samples, le feature si costruiscono on-demand e si cachano.
    """
    ensure_models()
    as_of = clamp_as_of(as_of)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"bi_{as_of.isoformat()}.json"
    if cache and path.exists():
        return json.loads(path.read_text())

    with _lock:
        samples: pd.DataFrame = _state["samples"]
        te, source = _frame_for_as_of(as_of)
        te = te.copy()
        te["pdv_id"] = te["pdv_id"].astype(str)
        te["sku_id"] = te["sku_id"].astype(str)

        hist = samples[samples["as_of"] < as_of]
        analogue = _state["build_analogue_lookup"](hist if len(hist) else te, H15)
        bundle, maps, hi = _state["bundle"], _state["maps"], _state["hi"]
        feats = bundle["features"]
        deploy = _state["params"]["deploy_params"]
        p80_scale = float(_state["params"].get("quantile", {}).get("p80_scale", 1.23))

        def blend(raw: np.ndarray) -> np.ndarray:
            return _state["apply_confidence_to_frame"](te, np.maximum(raw, 0.0), analogue, H15)[
                "pred_final"
            ].to_numpy()

        X, _ = _prepare_xy(te, maps, horizon=H15)
        pred_tw = blend(bundle["tweedie"].predict(X[feats]))
        pred_p50 = blend(bundle["p50"].predict(X[feats]))
        pred_p80_raw = blend(np.maximum(bundle["p80"].predict(X[feats]), pred_p50))
        pred_tw_rec = _state["soft_reconcile_frame"](te, pred_tw, alpha=0.25)
        pred_high = np.maximum(np.expm1(hi["model"].predict(X[hi["features"]])), 0.0)
        enc = _state["enc"]
        x_te = _state["frame_to_tf_inputs"](te, enc["maps"].item(), enc["num_mean"], enc["num_std"])
        pred_tf = blend(_state["sku_m"].predict(x_te, verbose=0).reshape(-1))
        prior = te["run_rate_14d"].astype(float).to_numpy() * H15
        naive = prior.copy()
        pred = _state["apply_calibrated_route"](
            prior, pred_tf, pred_tw_rec, naive, deploy, pred_high=pred_high
        )
        pred_p80 = _state["apply_calibrated_route"](
            prior, pred_tf, pred_p80_raw, naive, deploy, pred_high=np.maximum(pred_high, pred_p80_raw)
        )
        pred_p80 = np.maximum(pred_p80, pred) * p80_scale

        max_d: date = _state["max_d"]
        fact_y: pd.DataFrame = _state["fact_y"]
        has_y15 = as_of + timedelta(days=H15) <= max_d
        if "y_sku" in te.columns and te["y_sku"].notna().any() and has_y15:
            y15 = te["y_sku"].astype(float).fillna(0.0).to_numpy()
        elif has_y15:
            c0, c1 = as_of + timedelta(days=1), as_of + timedelta(days=H15)
            sub = fact_y[(fact_y["data"] >= c0) & (fact_y["data"] <= c1)]
            g = sub.groupby(["pdv_id", "sku_id"], as_index=False, observed=True).agg(
                y_15=("vendite_qty", "sum")
            )
            tmp = te[["pdv_id", "sku_id"]].merge(g, on=["pdv_id", "sku_id"], how="left")
            y15 = tmp["y_15"].fillna(0.0).to_numpy()
        else:
            y15 = np.zeros(len(te), dtype=float)

        pred_90 = np.zeros(len(te), dtype=float)
        if _state["h90_bundle"] is not None:
            from production.inference_h90 import predict_h90

            pred_90 = predict_h90(te, bundle=_state["h90_bundle"], maps=_state["h90_maps"])

        has_y90 = as_of + timedelta(days=H90) <= max_d
        y90 = np.full(len(te), np.nan)
        if has_y90:
            c0, c1 = as_of + timedelta(days=1), as_of + timedelta(days=H90)
            sub = fact_y[(fact_y["data"] >= c0) & (fact_y["data"] <= c1)]
            g = sub.groupby(["pdv_id", "sku_id"], as_index=False, observed=True).agg(
                y_90=("vendite_qty", "sum")
            )
            tmp = te[["pdv_id", "sku_id"]].merge(g, on=["pdv_id", "sku_id"], how="left")
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
                "y": np.round(y15, 2) if has_y15 else None,
                "y_90": np.round(y90, 2) if has_y90 else None,
                "err": np.round(np.abs(pred - y15), 2) if has_y15 else None,
                "err_90": np.round(np.abs(pred_90 - y90), 2) if has_y90 else None,
                "giacenza": np.round(giac, 1),
                "qty": _qty(pred),
                "qty_p80": _qty(pred_p80),
                "qty_90": np.maximum(np.rint(pred_90 - giac), 0).astype(int),
                "new": immature.astype(int),
            }
        )
        if not has_y15:
            out["y"] = None
            out["err"] = None
        if not has_y90:
            out["y_90"] = None
            out["err_90"] = None

        m15 = _metrics(y15, pred) if has_y15 else None
        m90 = _metrics(y90, pred_90) if has_y90 else None
        cols = list(out.columns)
        rows = []
        for row in out.itertuples(index=False, name=None):
            rows.append([None if (isinstance(x, float) and x != x) else x for x in row])

        payload = {
            "meta": {
                "mode": "holdout",
                "as_of": str(as_of),
                "as_of_source": source,
                "daily_as_of": True,
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
                "point_forecast_90": "native_h90" if _state["h90_bundle"] is not None else None,
                "modtescol_def": "modello + tessuto + colore (no taglia)",
                "mae": (m15 or {}).get("mae"),
                "wmape": (m15 or {}).get("wmape"),
                "metrics_15": m15,
                "metrics_90": m90,
                "note": (
                    f"Holdout as_of {as_of} ({source}). "
                    + ("Feature on-demand giornaliere." if "daily" in source else "Sample precomputato.")
                ),
            },
            "cols": cols,
            "rows": rows,
        }
        if cache:
            path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
            _refresh_index()
        return payload


def _actuals_window(d0: date, d1: date) -> pd.DataFrame:
    ensure_models()
    fact_y: pd.DataFrame = _state["fact_y"]
    sub = fact_y[(fact_y["data"] >= d0) & (fact_y["data"] <= d1)]
    return sub.groupby(["pdv_id", "sku_id"], as_index=False, observed=True).agg(
        y_window=("vendite_qty", "sum")
    )


def build_range_payload(
    d_from: date,
    d_to: date,
    *,
    as_of: date | None = None,
) -> dict:
    """Forbice manuale [from, to]: pred all'as_of, y = venduto nella forbice.

    - as_of default = nearest sample ≤ from-1
    - oppure as_of scelto manualmente (snappato al sample più vicino ≤ richiesto)
    - pred scalata: se giorni≈90 usa pred_90, altrimenti pred_15 × (days/15)
    """
    if d_to < d_from:
        raise ValueError("La data fine deve essere ≥ data inizio")
    ensure_models()
    max_d: date = _state["max_d"]
    if d_to > max_d:
        raise ValueError(f"Data fine oltre lo storico disponibile ({max_d})")
    if d_from > max_d:
        raise ValueError(f"Data inizio oltre lo storico disponibile ({max_d})")

    as_of_manual = as_of is not None
    if as_of is not None:
        as_of = clamp_as_of(as_of)
    else:
        as_of = clamp_as_of(d_from - timedelta(days=1))
    base = generate_as_of_snapshot(as_of, cache=True)
    cols = list(base["cols"])
    idx = {c: i for i, c in enumerate(cols)}
    days = (d_to - d_from).days + 1

    # scegli base pred
    use_90 = days >= 60 and base["meta"].get("horizon_90_end")
    if use_90:
        scale = days / float(H90)
        pred_key = "pred_90"
    else:
        scale = days / float(H15)
        pred_key = "pred"

    act = _actuals_window(d_from, d_to)
    act_map = {(str(r.pdv_id), str(r.sku_id)): float(r.y_window) for r in act.itertuples(index=False)}

    # rebuild rows
    new_cols = list(cols)
    # ensure y_window / pred_window / err_window
    for c in ("pred_window", "y_window", "err_window"):
        if c not in new_cols:
            new_cols.append(c)

    new_rows = []
    ys, ps = [], []
    for r in base["rows"]:
        d = {c: r[idx[c]] for c in cols}
        key = (str(d["pdv_id"]), str(d["sku_id"]))
        p = float(d.get(pred_key) or 0.0) * scale
        y = float(act_map.get(key, 0.0))
        d["pred_window"] = round(p, 2)
        d["y_window"] = round(y, 2)
        d["err_window"] = round(abs(p - y), 2)
        # per compatibilità KPI esistenti: y/pred puntano alla forbice
        d["pred"] = d["pred_window"]
        d["y"] = d["y_window"]
        d["err"] = d["err_window"]
        # pred_90 resta il nativo 90 se c'era; y_90 solo se forbice ≈90
        if use_90:
            d["pred_90"] = d["pred_window"]
            d["y_90"] = d["y_window"]
            d["err_90"] = d["err_window"]
        ys.append(y)
        ps.append(p)
        new_rows.append([d.get(c) for c in new_cols])

    m = _metrics(np.asarray(ys), np.asarray(ps))
    meta = dict(base["meta"])
    meta.update(
        {
            "mode": "holdout",
            "selection": "manual_range",
            "range_from": str(d_from),
            "range_to": str(d_to),
            "range_days": days,
            "as_of": str(as_of),
            "as_of_snapped": False,
            "as_of_manual": as_of_manual,
            "as_of_source": base["meta"].get("as_of_source"),
            "daily_as_of": True,
            "sample_as_ofs": [str(a) for a in _state["as_ofs"]],
            "pred_basis": pred_key,
            "pred_scale": round(scale, 4),
            "horizon_start": str(d_from),
            "horizon_end": str(d_to),
            "horizon_days": days,
            "mae": m["mae"],
            "wmape": m["wmape"],
            "metrics_15": m if not use_90 else base["meta"].get("metrics_15"),
            "metrics_window": m,
            "metrics_90": m if use_90 else base["meta"].get("metrics_90"),
            "note": (
                f"Forbice manuale {d_from}→{d_to} ({days}gg). "
                + (
                    f"as_of manuale → {as_of}. "
                    if as_of_manual
                    else f"as_of automatico (giorno prima di Dal) → {as_of}. "
                )
                + f"Fonte feature: {base['meta'].get('as_of_source')}. "
                + f"Pred scalata da {pred_key} × {scale:.3f}. Reale = venduto nella forbice."
            ),
        }
    )
    if use_90:
        meta["horizon_90_start"] = str(d_from)
        meta["horizon_90_end"] = str(d_to)
        meta["horizon_90_days"] = days

    return {"meta": meta, "cols": new_cols, "rows": new_rows}


def resolve_query_skus(query: str) -> tuple[list[str], dict[str, str]]:
    """Risolve SKU / modtescol / modello+colore → lista sku_id + info."""
    ensure_models()
    q = str(query or "").strip().upper()
    if not q:
        raise ValueError("Indica uno SKU o un codice modello (es. 020H3LI0KU)")
    samples: pd.DataFrame = _state["samples"]
    sub = samples[["sku_id", "modello", "tessuto", "colore"]].drop_duplicates().copy()
    for c in ("sku_id", "modello", "tessuto", "colore"):
        sub[c] = sub[c].astype(str)
    sub["modtescol"] = sub["modello"] + sub["tessuto"] + sub["colore"]
    sub["modcol"] = sub["modello"] + sub["colore"]
    hit = sub[
        (sub["sku_id"].str.upper() == q)
        | (sub["modtescol"].str.upper() == q)
        | (sub["modcol"].str.upper() == q)
        | (sub["sku_id"].str.upper().str.contains(q, regex=False))
        | (sub["modtescol"].str.upper().str.contains(q, regex=False))
        | (sub["modcol"].str.upper().str.contains(q, regex=False))
    ]
    if hit.empty:
        raise ValueError(f"Nessuno SKU trovato per «{query}» nei sample")
    skus = sorted(hit["sku_id"].unique().tolist())
    info = {
        "query": q,
        "n_sku": str(len(skus)),
        "modello": ",".join(sorted(hit["modello"].unique())[:8]),
        "colore": ",".join(sorted(hit["colore"].unique())[:8]),
        "modtescol": ",".join(sorted(hit["modtescol"].unique())[:8]),
    }
    return skus, info


def _first_as_of_with_skus(skus: set[str], *, on_or_before: date | None = None) -> date | None:
    ensure_models()
    samples: pd.DataFrame = _state["samples"]
    sku_set = {str(s) for s in skus}
    as_ofs: list[date] = list(_state["as_ofs"])
    if on_or_before is not None:
        as_ofs = [a for a in as_ofs if a <= on_or_before]
    for a in reversed(as_ofs):
        te = samples.loc[samples["as_of"] == a, "sku_id"].astype(str)
        if te.isin(sku_set).any():
            return a
    return None


def _next_as_of_with_skus(skus: set[str], *, on_or_after: date) -> date | None:
    ensure_models()
    samples: pd.DataFrame = _state["samples"]
    sku_set = {str(s) for s in skus}
    for a in _state["as_ofs"]:
        if a < on_or_after:
            continue
        te = samples.loc[samples["as_of"] == a, "sku_id"].astype(str)
        if te.isin(sku_set).any():
            return a
    return None


def _sku_has_hist(sku_set: set[str], as_of: date) -> bool:
    """True se lo SKU ha vendite nelle 2 settimane fino ad as_of (serve per feature)."""
    ensure_models()
    fact_y: pd.DataFrame = _state["fact_y"]
    w0 = as_of - timedelta(days=13)
    sub = fact_y[
        (fact_y["data"] >= w0)
        & (fact_y["data"] <= as_of)
        & (fact_y["sku_id"].isin(sku_set))
        & (fact_y["vendite_qty"] > 0)
    ]
    return not sub.empty


def _resolve_as_of_for_skus(
    sku_set: set[str],
    *,
    preferred: date | None,
    d_to: date,
    query: str,
) -> tuple[date, str, bool]:
    """Ritorna (as_of, note, manuale). as_of giornaliero (non più snappato a 14gg).

    - preferred: giorno scelto dall'utente
    - default: fine forbice (d_to) — forecast dopo le vendite osservate
    """
    ensure_models()

    if preferred is not None:
        as_of = clamp_as_of(preferred)
        if not _sku_has_hist(sku_set, as_of):
            raise ValueError(
                f"as_of {as_of}: «{query}» senza vendite nelle 2 settimane precedenti "
                f"(feature non costruibili). Scegli un giorno dopo il debutto dello SKU."
            )
        note = f"as_of scelto manualmente: {as_of}."
        if as_of != preferred:
            note += f" (richiesto {preferred}, limitato allo storico disponibile)"
        return as_of, note, True

    as_of = clamp_as_of(d_to)
    if not _sku_has_hist(sku_set, as_of):
        # cerca il primo giorno ≥ d_to con storico, o l'ultimo ≤ d_to
        max_d: date = _state["max_d"]
        found = None
        d = d_to
        while d <= max_d:
            if _sku_has_hist(sku_set, d):
                found = d
                break
            d += timedelta(days=1)
        if found is None:
            d = d_to - timedelta(days=1)
            min_a: date = _state["min_as_of"]
            while d >= min_a:
                if _sku_has_hist(sku_set, d):
                    found = d
                    break
                d -= timedelta(days=1)
        if found is None:
            raise ValueError(f"«{query}» senza storico vendite sufficiente per un as_of")
        as_of = found
        return (
            as_of,
            f"as_of automatico: {as_of} (primo giorno utile rispetto alla fine forbice {d_to}).",
            False,
        )
    return (
        as_of,
        f"as_of automatico = fine forbice ({as_of}): forecast dai giorni successivi.",
        False,
    )


def build_sku_window_forecast(
    query: str,
    d_from: date,
    d_to: date,
    *,
    as_of: date | None = None,
) -> dict:
    """Vendite reali in [from,to] + forecast 15g / 1m / 2m / 3m per SKU o modtescol.

    - Reale = venduto nella forbice scelta (periodo di osservazione)
    - as_of = scelto dall'utente, oppure primo sample **≥ fine forbice** con lo SKU
    - Orizzonti da as_of+1: 15 (nativo), 30/60 (pro-rata da pred_90), 90 (nativo h90)
    """
    if d_to < d_from:
        raise ValueError("La data fine deve essere ≥ data inizio")
    ensure_models()
    max_d: date = _state["max_d"]
    if d_from > max_d:
        raise ValueError(f"Data inizio oltre lo storico disponibile ({max_d})")

    skus, info = resolve_query_skus(query)
    sku_set = set(skus)

    as_of, as_of_note, as_of_manual = _resolve_as_of_for_skus(
        sku_set, preferred=as_of, d_to=d_to, query=query
    )

    base = generate_as_of_snapshot(as_of, cache=True)
    cols = list(base["cols"])
    idx = {c: i for i, c in enumerate(cols)}
    rows_all = base["rows"]
    rows_f = [r for r in rows_all if str(r[idx["sku_id"]]) in sku_set]
    if not rows_f:
        raise ValueError(f"Snapshot {as_of} senza righe per «{query}»")

    # venduto reale nella forbice osservata
    fact_y: pd.DataFrame = _state["fact_y"]
    act = fact_y[
        (fact_y["data"] >= d_from)
        & (fact_y["data"] <= min(d_to, max_d))
        & (fact_y["sku_id"].isin(sku_set))
    ]
    act_key = act.groupby(["pdv_id", "sku_id"], as_index=False, observed=True).agg(
        y_obs=("vendite_qty", "sum")
    )
    act_map = {(str(r.pdv_id), str(r.sku_id)): float(r.y_obs) for r in act_key.itertuples(index=False)}
    by_day = (
        act.groupby("data", as_index=False)
        .agg(y=("vendite_qty", "sum"))
        .sort_values("data")
    )
    actuals_daily = [{"date": str(r.data), "y": round(float(r.y), 2)} for r in by_day.itertuples(index=False)]

    # y per orizzonti post as_of (se coperto dallo storico)
    def _y_map(h_days: int) -> dict[tuple[str, str], float]:
        c0, c1 = as_of + timedelta(days=1), as_of + timedelta(days=h_days)
        if c0 > max_d:
            return {}
        c1 = min(c1, max_d)
        sub = fact_y[
            (fact_y["data"] >= c0) & (fact_y["data"] <= c1) & (fact_y["sku_id"].isin(sku_set))
        ]
        g = sub.groupby(["pdv_id", "sku_id"], as_index=False, observed=True).agg(
            y_h=("vendite_qty", "sum")
        )
        return {(str(r.pdv_id), str(r.sku_id)): float(r.y_h) for r in g.itertuples(index=False)}

    y15_map = _y_map(H15)
    y30_map = _y_map(30)
    y60_map = _y_map(60)
    y90_map = _y_map(H90)

    has_h90 = any(float(r[idx.get("pred_90", -1)] or 0) != 0 for r in rows_f) if "pred_90" in idx else False

    # Predizioni native 30/60 se i bundle esistono (altrimenti fallback ritmo 15g)
    native_30: dict[tuple[str, str], float] = {}
    native_60: dict[tuple[str, str], float] = {}
    try:
        from production.inference_horizon_native import has_horizon_bundle, predict_horizon

        te_feat, _src = _frame_for_as_of(as_of)
        te_feat = te_feat.copy()
        te_feat["pdv_id"] = te_feat["pdv_id"].astype(str)
        te_feat["sku_id"] = te_feat["sku_id"].astype(str)
        te_feat = te_feat[te_feat["sku_id"].isin(sku_set)]
        if not te_feat.empty and has_horizon_bundle(30):
            p30n = predict_horizon(te_feat, 30)
            for i_row, (pid, sid) in enumerate(zip(te_feat["pdv_id"], te_feat["sku_id"])):
                native_30[(str(pid), str(sid))] = float(p30n[i_row])
        if not te_feat.empty and has_horizon_bundle(60):
            p60n = predict_horizon(te_feat, 60)
            for i_row, (pid, sid) in enumerate(zip(te_feat["pdv_id"], te_feat["sku_id"])):
                native_60[(str(pid), str(sid))] = float(p60n[i_row])
    except Exception:
        native_30, native_60 = {}, {}

    def _scale_pred(pred15: float, pred90: float, days: int, key: tuple[str, str] | None = None) -> float:
        """15/90 nativi; 30/60 nativi se disponibili, altrimenti ritmo 15g."""
        if days == H15:
            return float(pred15)
        if days == H90:
            return float(pred90) if has_h90 and pred90 is not None else float(pred15) * (H90 / float(H15))
        if days == 30 and key is not None and key in native_30:
            return float(native_30[key])
        if days == 60 and key is not None and key in native_60:
            return float(native_60[key])
        from_15 = float(pred15) * (days / float(H15))
        if has_h90 and pred90 is not None and float(pred90) > 0:
            from_90 = float(pred90) * (days / float(H90))
            return float(min(from_15, from_90)) if from_90 < from_15 else float(from_15)
        return from_15

    new_cols = list(cols)
    for c in (
        "y_obs",
        "pred_15",
        "pred_30",
        "pred_60",
        "y_30",
        "y_60",
        "err_30",
        "err_60",
    ):
        if c not in new_cols:
            new_cols.append(c)

    new_rows = []
    sums = {
        "y_obs": 0.0,
        "pred_15": 0.0,
        "pred_30": 0.0,
        "pred_60": 0.0,
        "pred_90": 0.0,
        "y_15": 0.0,
        "y_30": 0.0,
        "y_60": 0.0,
        "y_90": 0.0,
    }
    for r in rows_f:
        d = {c: r[idx[c]] for c in cols}
        key = (str(d["pdv_id"]), str(d["sku_id"]))
        p15 = float(d.get("pred") or 0.0)
        p90 = float(d.get("pred_90") or 0.0)
        p30 = _scale_pred(p15, p90, 30, key=key)
        p60 = _scale_pred(p15, p90, 60, key=key)
        y_obs = float(act_map.get(key, 0.0))
        y15 = float(y15_map.get(key, 0.0)) if y15_map else None
        y30 = float(y30_map.get(key, 0.0)) if y30_map else None
        y60 = float(y60_map.get(key, 0.0)) if y60_map else None
        y90 = float(y90_map.get(key, 0.0)) if y90_map else None

        d["y_obs"] = round(y_obs, 2)
        d["pred_15"] = round(p15, 2)
        d["pred_30"] = round(p30, 2)
        d["pred_60"] = round(p60, 2)
        d["pred"] = d["pred_15"]
        d["y"] = d["y_obs"]  # KPI/filtri: reale nella forbice osservata
        d["err"] = round(abs(p15 - (y15 if y15 is not None else 0.0)), 2) if y15_map else None
        if y30 is not None:
            d["y_30"] = round(y30, 2)
            d["err_30"] = round(abs(p30 - y30), 2)
        else:
            d["y_30"] = None
            d["err_30"] = None
        if y60 is not None:
            d["y_60"] = round(y60, 2)
            d["err_60"] = round(abs(p60 - y60), 2)
        else:
            d["y_60"] = None
            d["err_60"] = None
        if y90 is not None and d.get("y_90") is None:
            d["y_90"] = round(y90, 2)
            d["err_90"] = round(abs(p90 - y90), 2)

        sums["y_obs"] += y_obs
        sums["pred_15"] += p15
        sums["pred_30"] += p30
        sums["pred_60"] += p60
        sums["pred_90"] += p90
        if y15 is not None:
            sums["y_15"] += y15
        if y30 is not None:
            sums["y_30"] += y30
        if y60 is not None:
            sums["y_60"] += y60
        if y90 is not None:
            sums["y_90"] += y90
        new_rows.append([d.get(c) for c in new_cols])

    def _hz(days: int, pred_sum: float, y_sum: float, has_y: bool) -> dict:
        start = as_of + timedelta(days=1)
        end = as_of + timedelta(days=days)
        out = {
            "days": days,
            "start": str(start),
            "end": str(end),
            "pred": round(pred_sum, 1),
            "source": (
                "native_15"
                if days == H15
                else (
                    "native_h90"
                    if days == H90 and has_h90
                    else (
                        "native_h30"
                        if days == 30 and native_30
                        else (
                            "native_h60"
                            if days == 60 and native_60
                            else ("h15_prorata" if days in (30, 60) else "h90_prorata")
                        )
                    )
                )
            ),
        }
        if has_y and end <= max_d:
            out["y"] = round(y_sum, 1)
            out["wmape"] = round(
                float(abs(pred_sum - y_sum) / max(y_sum, 1.0) * 100) if y_sum or pred_sum else 0.0,
                2,
            )
            out["bias"] = round(pred_sum - y_sum, 1)
        return out

    has_y15 = bool(y15_map) and (as_of + timedelta(days=H15) <= max_d)
    has_y30 = bool(y30_map) and (as_of + timedelta(days=30) <= max_d)
    has_y60 = bool(y60_map) and (as_of + timedelta(days=60) <= max_d)
    has_y90 = bool(y90_map) and (as_of + timedelta(days=H90) <= max_d)

    horizons = {
        "d15": _hz(15, sums["pred_15"], sums["y_15"], has_y15),
        "d30": _hz(30, sums["pred_30"], sums["y_30"], has_y30),
        "d60": _hz(60, sums["pred_60"], sums["y_60"], has_y60),
        "d90": _hz(90, sums["pred_90"], sums["y_90"], has_y90),
    }

    meta = {
        "mode": "sku_window",
        "query": info["query"],
        "query_info": info,
        "skus": skus,
        "n_sku": len(skus),
        "selection": "sku_window",
        "range_from": str(d_from),
        "range_to": str(d_to),
        "range_days": (d_to - d_from).days + 1,
        "as_of": str(as_of),
        "as_of_note": as_of_note,
        "as_of_manual": as_of_manual,
        "as_of_source": base["meta"].get("as_of_source"),
        "daily_as_of": True,
        "sample_as_ofs": [str(a) for a in _state["as_ofs"]],
        "horizon_start": str(as_of + timedelta(days=1)),
        "horizon_end": str(as_of + timedelta(days=H15)),
        "horizon_90_start": str(as_of + timedelta(days=1)),
        "horizon_90_end": str(as_of + timedelta(days=H90)),
        "horizon_days": H15,
        "horizon_90_days": H90,
        "n": len(new_rows),
        "n_pdv": len({r[idx["pdv_id"]] for r in rows_f}),
        "n_sku_rows": len(skus),
        "n_sku": len(skus),
        "n_country": len({r[idx["country"]] for r in rows_f}) if "country" in idx else 0,
        "point_forecast": "route_ops_tweedie_rec",
        "point_forecast_90": "native_h90" if has_h90 else None,
        "modtescol_def": "modello + tessuto + colore (no taglia)",
        "actuals": {
            "from": str(d_from),
            "to": str(min(d_to, max_d)),
            "sum_y": round(sums["y_obs"], 1),
            "n_days_with_sales": len(actuals_daily),
            "daily": actuals_daily,
        },
        "horizons": horizons,
        "mae": None,
        "wmape": horizons["d15"].get("wmape"),
        "metrics_15": horizons["d15"],
        "metrics_90": horizons["d90"],
        "note": (
            f"Osservate vendite reali {d_from}→{d_to} ({sums['y_obs']:.0f} pz), "
            f"poi forecast all'as_of {as_of} → 15g/1m/2m/3m"
            + (f". {as_of_note}" if as_of_note else "")
        ),
    }
    return {"meta": meta, "cols": new_cols, "rows": new_rows}


def _refresh_index() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    snaps = []
    for p in sorted(OUT.glob("bi_*.json")):
        try:
            meta = json.loads(p.read_text()).get("meta", {})
        except Exception:
            continue
        if meta.get("selection") == "manual_range":
            continue
        as_of = meta.get("as_of")
        if not as_of:
            continue
        snaps.append(
            {
                "as_of": as_of,
                "file": p.name,
                "horizon_start": meta.get("horizon_start"),
                "horizon_end": meta.get("horizon_end"),
                "horizon_90_end": meta.get("horizon_90_end"),
                "has_y90": bool(meta.get("horizon_90_end")),
                "n": meta.get("n"),
                "mae_15": (meta.get("metrics_15") or {}).get("mae", meta.get("mae")),
                "wmape_15": (meta.get("metrics_15") or {}).get("wmape", meta.get("wmape")),
                "label": f"{as_of} → +15gg ({meta.get('horizon_start')}…{meta.get('horizon_end')})"
                + (" · +90gg" if meta.get("horizon_90_end") else ""),
            }
        )
    ensure_models()
    index = {
        "generated": str(date.today()),
        "fact_max": str(_state["max_d"]),
        "sample_as_ofs": [str(a) for a in _state["as_ofs"]],
        "n": len(snaps),
        "snapshots": snaps,
        "min_as_of": str(_state["as_ofs"][0]) if _state["as_ofs"] else None,
        "max_as_of": str(_state["as_ofs"][-1]) if _state["as_ofs"] else None,
        "note": "Scegli dal/al manualmente in BI; as_of viene agganciato al sample più vicino.",
    }
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))
