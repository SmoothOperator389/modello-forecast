#!/usr/bin/env python3
"""
TESTING — genera dati fake (volume alto) e addestra DualHead TensorFlow ottimizzato.

Uso:
  cd ~/Desktop/modello-forecast
  OMP_NUM_THREADS=1 .venv/bin/python -m testing.run_train
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from testing.config import (
    DATA_DIR,
    FakeDataConfig,
    PRODUCTION_N_PDV,
    PRODUCTION_N_SKU,
    TrainConfig,
    as_of_range,
)
from testing.inference.artefacts import save_forecast_artefacts
from testing.fake_data.generator import write_all
from testing.features.build_features import (
    NUMERIC_COLS,
    apply_train_test_split,
    build_training_frames,
    load_fact_calendar,
)
from testing.features.sku_weather import add_season_labels
from testing.inference.confidence import apply_confidence_to_frame, build_analogue_lookup
from testing.inference.cdm_baseline import fit_cdm_baseline, predict_cdm_baseline
from testing.inference.explain import explain_focus_frame
from testing.bi_dashboard import write_bi_dashboard
from testing.report_builder import write_report
from testing.tensorflow_env import import_tensorflow


def _in_virtualenv() -> bool:
    return sys.prefix != sys.base_prefix or bool(os.environ.get("VIRTUAL_ENV"))


def _check_venv() -> None:
    if _in_virtualenv():
        return
    print(
        "ATTENZIONE: non stai usando un virtualenv.\n"
        f"  Python attuale: {sys.executable}\n"
        "  Usa: cd ~/Desktop/modello-forecast && .venv/bin/python -m testing.run_train\n"
        "  Oppure crea un venv: ./scripts/setup_venv_mac.sh",
        file=sys.stderr,
    )


def suggest_replenishment(
    forecast: float,
    giacenza: float,
    transit: float,
    horizon: int,
    confidence: float = 1.0,
) -> dict:
    daily = forecast / max(horizon, 1)
    need = forecast + daily * transit
    raw = max(0.0, need - giacenza)
    # Prudenza operativa: qty ridotta se poca fiducia (SKU/PDV giovane)
    raw *= max(0.45, min(1.0, 0.45 + 0.55 * confidence))
    qty = float(np.rint(raw))
    return {
        "vendite_previste_15gg": round(forecast, 1),
        "giacenza": giacenza,
        "qty_suggerita": qty,
        "azione": "monitora" if qty == 0 else "valuta_riassortimento",
    }


def apply_replenishment(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Versione vettoriale di ``suggest_replenishment`` (lista CDM intera)."""
    forecast = df["pred_final"].astype(float)
    giacenza = df["giacenza"].astype(float)
    if "transit_time_days" in df.columns:
        transit = df["transit_time_days"].astype(float).fillna(5.0)
    else:
        transit = pd.Series(5.0, index=df.index)
    if "confidence_score" in df.columns:
        conf = df["confidence_score"].astype(float).clip(0.0, 1.0)
    else:
        conf = pd.Series(1.0, index=df.index)
    daily = forecast / max(horizon, 1)
    need = forecast + daily * transit
    raw = (need - giacenza).clip(lower=0.0)
    raw = raw * (0.45 + 0.55 * conf).clip(lower=0.45, upper=1.0)
    qty = np.rint(raw)
    out = df.copy()
    out["vendite_previste_15gg"] = forecast.round(1)
    out["qty_suggerita"] = qty
    out["azione"] = np.where(qty == 0, "monitora", "valuta_riassortimento")
    return out


def main() -> None:
    _check_venv()
    cfg = TrainConfig()
    np.random.seed(cfg.seed)

    fact_path = DATA_DIR / "fact_daily.csv"
    tickets_path = DATA_DIR / "fact_tickets.csv"
    regen = os.environ.get("FORECAST_REGEN_DATA", "").strip().lower() in {"1", "true", "yes", "on"}
    if fact_path.exists() and tickets_path.exists() and not regen:
        print("=== 1) Dataset già presente — skip generazione ===")
        print("    (FORECAST_REGEN_DATA=1 per rigenerare)")
    else:
        print("=== 1) Generazione dataset fake (testing, CSV) ===")
        write_all(FakeDataConfig())
    meta = pd.read_csv(DATA_DIR / "dataset_meta.csv")
    print(meta.to_string(index=False))

    print("\n=== 2) Build sample rolling ===")
    fact, calendar = load_fact_calendar()
    as_of_list = as_of_range(cfg.train_as_of_start, cfg.train_as_of_end, cfg.as_of_step_days)

    samples = build_training_frames(fact, calendar, cfg, as_of_list)
    print(f"Righe sample rolling (tutti gli as_of): {len(samples)}")

    train_df, test_df, cut, train_start = apply_train_test_split(samples, cfg)
    n_as_of_train = train_df["as_of"].nunique()
    if train_start is not None:
        print(
            f"Finestra training: ultimi {cfg.train_window_days} gg "
            f"(as_of >= {train_start}) → {n_as_of_train} as_of, {len(train_df)} righe"
        )
    else:
        print(f"Finestra training: tutto lo storico → {n_as_of_train} as_of, {len(train_df)} righe")
    print(f"Test: {len(test_df)} | cut as_of>={cut}")
    print(
        f"Validation interna: ultimi {cfg.n_val_as_of} as_of del train "
        f"(early stopping patience={cfg.early_stopping_patience})"
    )

    tf = import_tensorflow()
    from testing.model.dual_head_tf import build_dual_head_model, fit_encoders, load_encoder_vocab, predict, train_model

    tf.random.set_seed(cfg.seed)

    # Encoder: train + anagrafica catalogo (no label test). UNK solo per entità davvero nuove.
    catalog_vocab = load_encoder_vocab(DATA_DIR)
    maps = fit_encoders(train_df, catalog_vocab)
    num_mean = train_df[NUMERIC_COLS].astype(float).mean().to_numpy(copy=True)
    num_std = train_df[NUMERIC_COLS].astype(float).std().to_numpy(copy=True)
    num_std[num_std == 0] = 1.0

    model = build_dual_head_model(
        n_numeric=len(NUMERIC_COLS),
        maps=maps,
    )
    print(f"\n=== 3) Training TensorFlow — parametri: {model.count_params():,} ===")
    model, history = train_model(model, train_df, maps, num_mean, num_std, cfg)

    # --- Metriche test (modello grezzo) ---
    pred_s, _ = predict(model, test_df, maps, num_mean, num_std)
    y_true = test_df["y_sku"].to_numpy()
    naive = test_df["run_rate_14d"].to_numpy() * cfg.horizon_sku_days

    cdm_fitted = fit_cdm_baseline(train_df, cfg.horizon_sku_days)
    pred_cdm = predict_cdm_baseline(test_df, cdm_fitted, cfg.horizon_sku_days)

    # --- Confidence score + blend analoghi (post-modello, PDF CDM) ---
    analogue_lookup = build_analogue_lookup(train_df, cfg.horizon_sku_days)
    test_blend = apply_confidence_to_frame(test_df, pred_s, analogue_lookup, cfg.horizon_sku_days)
    test_blend["pred_cdm"] = pred_cdm
    pred_final = test_blend["pred_final"].to_numpy()

    def _metrics(pred: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
        mae = float(np.abs(pred - y).mean())
        mape = float(np.mean(np.abs(pred - y) / np.maximum(y, 1.0)) * 100)
        ss = np.sum((y - y.mean()) ** 2)
        r2 = float(1.0 - np.sum((y - pred) ** 2) / ss) if ss > 0 else 0.0
        return mae, mape, r2

    mae, mape, r2 = _metrics(pred_final, y_true)
    mae_raw, mape_raw, r2_raw = _metrics(pred_s, y_true)
    mae_cdm, mape_cdm, r2_cdm = _metrics(pred_cdm, y_true)
    mae_naive = float(np.abs(naive - y_true).mean())
    mape_naive = float(np.mean(np.abs(naive - y_true) / np.maximum(y_true, 1.0)) * 100)
    improvement = (mae_cdm - mae) / mae_cdm * 100 if mae_cdm else 0.0
    improvement_raw = (mae_cdm - mae_raw) / mae_cdm * 100 if mae_cdm else 0.0
    improvement_naive = (mae_naive - mae) / mae_naive * 100 if mae_naive else 0.0

    # Metriche cold start (SKU < 30 gg)
    cold = test_blend["giorni_da_lancio"].astype(float) < 30
    cold_mask = cold.to_numpy()
    if cold.sum() >= 50:
        mae_cold, _, _ = _metrics(pred_final[cold_mask], y_true[cold_mask])
        mae_cold_raw, _, _ = _metrics(pred_s[cold_mask], y_true[cold_mask])
        mae_cold_cdm, _, _ = _metrics(pred_cdm[cold_mask], y_true[cold_mask])
        mae_cold_nv = float(np.abs(naive[cold_mask] - y_true[cold_mask]).mean())
    else:
        mae_cold = mae_cold_raw = mae_cold_cdm = mae_cold_nv = None

    print(f"\n{'='*60}")
    print(f"=== 4) RISULTATI TEST (ultime {len(test_df)} righe) ===")
    print(f"{'='*60}")
    print(f"  MAE  rete neurale (finale) : {mae:.2f} pezzi")
    print(f"  MAE  modello CDM originale : {mae_cdm:.2f} pezzi  (regole: coeff stagionali, cluster 0.70/1.00/1.30, proxy PDV)")
    print(f"  MAE  rete grezza           : {mae_raw:.2f} pezzi  ({improvement_raw:+.1f}% vs CDM)")
    print(f"  MAE  naive run-rate×15     : {mae_naive:.2f} pezzi")
    print(f"  Miglioramento vs CDM       : {improvement:+.1f}%")
    print(f"  MAPE rete / CDM / naive    : {mape:.1f}% / {mape_cdm:.1f}% / {mape_naive:.1f}%")
    print(f"  R² rete                    : {r2:.4f}")
    if mae_cold is not None:
        print(
            f"  MAE SKU nuovi (<30g)       : rete={mae_cold:.2f} | CDM={mae_cold_cdm:.2f} | "
            f"naive={mae_cold_nv:.2f}"
        )

    def _seg(name: str, mask: np.ndarray) -> dict | None:
        if int(mask.sum()) < 80:
            return None
        mae_m = float(np.abs(pred_final[mask] - y_true[mask]).mean())
        mae_r = float(np.abs(pred_s[mask] - y_true[mask]).mean())
        mae_c = float(np.abs(pred_cdm[mask] - y_true[mask]).mean())
        mae_n = float(np.abs(naive[mask] - y_true[mask]).mean())
        imp = (mae_c - mae_m) / mae_c * 100 if mae_c else 0.0
        print(
            f"  MAE {name:22} : rete={mae_m:.2f}  CDM={mae_c:.2f}  "
            f"naive={mae_n:.2f}  ({imp:+.1f}% vs CDM)  n={int(mask.sum()):,}"
        )
        return {
            "mae": round(mae_m, 4),
            "mae_raw": round(mae_r, 4),
            "mae_cdm": round(mae_c, 4),
            "mae_naive": round(mae_n, 4),
            "improvement_pct": round(imp, 2),
            "n": int(mask.sum()),
        }

    # Etichette fake solo per QA (il modello non le vede). In produzione Valido non ci sono.
    stag_map: dict[str, str] = {}
    products_path = DATA_DIR / "products.csv"
    if products_path.exists():
        prod = pd.read_csv(products_path)
        if "sku_id" in prod.columns and "stagione" in prod.columns:
            stag_map = (
                prod.drop_duplicates("sku_id")
                .set_index("sku_id")["stagione"]
                .astype(str)
                .to_dict()
            )
    stag = test_blend["sku_id"].astype(str).map(stag_map).fillna("")
    seg_estate = _seg("capi ESTATE", (stag == "ESTATE").to_numpy())
    seg_inverno = _seg("capi INVERNO", (stag == "INVERNO").to_numpy())
    seg_cont = _seg("capi CONTINUATIVO", (stag == "CONTINUATIVO").to_numpy())
    meteo_mask = stag.isin(["ESTATE", "INVERNO"]).to_numpy()
    seg_meteo = _seg("meteo (est+inv)", meteo_mask)
    print(f"{'='*60}")

    out_dir = Path(__file__).resolve().parent / "data" / "output"
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    (out_dir / "report_metrics.json").write_text(
        json.dumps(
            {
                "mae_tf": round(mae, 4),
                "mae_tf_raw": round(mae_raw, 4),
                "mae_cdm": round(mae_cdm, 4),
                "mae_naive": round(mae_naive, 4),
                "mape_tf": round(mape, 2),
                "mape_cdm": round(mape_cdm, 2),
                "mape_naive": round(mape_naive, 2),
                "r2": round(r2, 4),
                "r2_raw": round(r2_raw, 4),
                "r2_cdm": round(r2_cdm, 4),
                "improvement_pct": round(improvement, 2),
                "improvement_pct_raw": round(improvement_raw, 2),
                "improvement_pct_naive": round(improvement_naive, 2),
                "mae_cold_sku_confidence": round(mae_cold, 4) if mae_cold else None,
                "mae_cold_sku_raw": round(mae_cold_raw, 4) if mae_cold_raw else None,
                "mae_cold_cdm": round(mae_cold_cdm, 4) if mae_cold_cdm else None,
                "mae_cold_naive": round(mae_cold_nv, 4) if mae_cold_nv else None,
                "seg_estate": seg_estate,
                "seg_inverno": seg_inverno,
                "seg_continuativo": seg_cont,
                "seg_meteo": seg_meteo,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "horizon_sku_days": cfg.horizon_sku_days,
                "train_window_days": cfg.train_window_days,
                "production_n_pdv": PRODUCTION_N_PDV,
                "production_n_sku": PRODUCTION_N_SKU,
                "n_as_of_train": int(train_df["as_of"].nunique()),
                "n_val_as_of": cfg.n_val_as_of,
                "early_stopping_patience": cfg.early_stopping_patience,
                "train_start": str(train_start) if train_start else None,
                "cut_as_of": str(cut),
                "n_params": int(model.count_params()),
                "epochs_run": len(history.get("loss", [])),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    test_sample = test_df.sample(n=min(500, len(test_df)), random_state=cfg.seed).copy()
    test_sample = apply_confidence_to_frame(
        test_sample,
        predict(model, test_sample, maps, num_mean, num_std)[0],
        analogue_lookup,
        cfg.horizon_sku_days,
    )
    test_sample["pred_cdm"] = predict_cdm_baseline(test_sample, cdm_fitted, cfg.horizon_sku_days)
    test_sample[["pdv_id", "sku_id", "y_sku", "pred_final", "pred_model", "pred_cdm", "confidence_score", "run_rate_14d", "giacenza"]].to_csv(
        out_dir / "test_sample_predictions.csv", index=False
    )

    last = test_df["as_of"].max()
    focus = apply_replenishment(test_blend[test_blend["as_of"] == last].copy(), cfg.horizon_sku_days)
    focus = add_season_labels(focus)
    sku_prof_cols = [
        c
        for c in [
            "sku_id",
            "sku_share_hot",
            "sku_share_cold",
            "sku_temp_vendita",
            "sku_meteo_giorni",
            "stagione_stimata",
        ]
        if c in focus.columns
    ]
    if sku_prof_cols and "sku_id" in sku_prof_cols:
        (
            focus[sku_prof_cols]
            .drop_duplicates("sku_id")
            .to_csv(out_dir / "sku_meteo_profilo.csv", index=False)
        )

    prio_cols = [
        "as_of",
        "pdv_id",
        "sku_id",
        "y_sku",
        "pred_final",
        "pred_model",
        "pred_cdm",
        "confidence_score",
        "is_new_pdv",
        "vendite_previste_15gg",
        "giacenza",
        "qty_suggerita",
        "azione",
        "stagione_stimata",
    ]
    priorities = (
        focus[[c for c in prio_cols if c in focus.columns]]
        .rename(columns={"y_sku": "y_vera", "pred_final": "pred_tf"})
        .copy()
    )
    priorities["pred_tf"] = priorities["pred_tf"].round(1)
    priorities["pred_model"] = priorities["pred_model"].round(1)
    priorities["pred_cdm"] = priorities["pred_cdm"].round(1)
    priorities["confidence_score"] = priorities["confidence_score"].round(3)
    priorities["is_new_pdv"] = priorities["is_new_pdv"].astype(int)
    priorities = priorities.sort_values("qty_suggerita", ascending=False)
    priorities.to_csv(out_dir / "priorita_riassortimento.csv", index=False)

    interesting = focus[
        ((focus["pdv_id"] == "MIL_01") & (focus["sku_id"].isin(["SKU_A", "SKU_Z", "SKU_J", "SKU_W"])))
        | ((focus["pdv_id"] == "ROM_NEW") & (focus["sku_id"] == "SKU_A"))
    ].copy()

    out = interesting[
        [
            "as_of",
            "pdv_id",
            "sku_id",
            "y_sku",
            "pred_final",
            "pred_model",
            "pred_analogue",
            "pred_cdm",
            "confidence_score",
            "run_rate_14d",
            "is_new_pdv",
            "vendite_previste_15gg",
            "giacenza",
            "qty_suggerita",
            "azione",
        ]
    ].rename(columns={"y_sku": "y_vera", "pred_final": "pred_tf"}).copy()
    out["pred_tf"] = out["pred_tf"].round(1)
    out["pred_model"] = out["pred_model"].round(1)
    out["pred_analogue"] = out["pred_analogue"].round(1)
    out["pred_cdm"] = out["pred_cdm"].round(1)
    out["confidence_score"] = out["confidence_score"].round(3)
    out["run_rate_14d"] = out["run_rate_14d"].round(3)
    out["is_new_pdv"] = out["is_new_pdv"].astype(int)

    # Spiegazioni locali (driver + narrativa) sui casi didattici
    print("\n=== 5b) Spiegazioni previsione (driver feature) ===")
    explanations = explain_focus_frame(
        interesting,
        model,
        maps,
        num_mean,
        num_std,
        train_df,
        predict,
        horizon=cfg.horizon_sku_days,
    )
    out["as_of"] = pd.to_datetime(out["as_of"]).dt.strftime("%Y-%m-%d")
    explanations["as_of"] = pd.to_datetime(explanations["as_of"]).dt.strftime("%Y-%m-%d")
    out = out.merge(explanations, on=["as_of", "pdv_id", "sku_id"], how="left")
    out.to_csv(out_dir / "focus_casi_tf.csv", index=False)
    explanations.to_csv(out_dir / "focus_casi_spiegazioni.csv", index=False)
    for _, er in explanations.iterrows():
        d1 = er.get("driver_1", "—")
        print(f"  {er['pdv_id']}/{er['sku_id']}: {d1}")
    model.save(out_dir / "dual_head_testing.keras")
    art_path = save_forecast_artefacts(
        maps=maps,
        num_mean=num_mean,
        num_std=num_std,
        analogue_lookup=analogue_lookup,
        cdm_fitted=cdm_fitted,
        explain_ref=train_df.sample(n=min(2000, len(train_df)), random_state=cfg.seed),
    )

    print("\n=== 5) Casi didattici (TensorFlow + operativo) ===")
    cols_show = [c for c in out.columns if c != "spiegazione_testuale"]
    print(out[cols_show].to_string(index=False))
    print(f"\nArtefatti: {out_dir / 'dual_head_testing.keras'}")
    print(f"Artefatti inferenza: {art_path}")

    print("\n=== 6) Report visivo produzione ===")
    report_path = write_report()
    bi_path = write_bi_dashboard()
    print(f"Report HTML: {report_path}")
    print(f"Dashboard CDM: {bi_path}")
    print(f"Apri: open {bi_path}")


if __name__ == "__main__":
    main()
