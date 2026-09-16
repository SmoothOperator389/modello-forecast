#!/usr/bin/env python3
"""Ciclo operativo 15 giorni — pacchetto previsioni per il CDM.

Non riaddestra. Carica il modello già trainato e produce la lista
PDV×SKU da portare in riunione (domanda attesa 15 gg + qty suggerita).

Uso:
  cd ~/Desktop/modello-forecast
  OMP_NUM_THREADS=1 .venv2/bin/python -m testing.run_forecast
  OMP_NUM_THREADS=1 .venv2/bin/python -m testing.run_forecast --as-of 2026-08-16
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from testing.config import DATA_DIR, TrainConfig, as_of_range
from testing.features.build_features import (
    NUMERIC_COLS,
    apply_train_test_split,
    build_training_frames,
    load_fact_calendar,
)
from testing.features.sku_weather import add_season_labels
from testing.inference.artefacts import ARTEFACTS_PATH, load_forecast_artefacts, save_forecast_artefacts
from testing.inference.cdm_baseline import fit_cdm_baseline, predict_cdm_baseline
from testing.inference.confidence import apply_confidence_to_frame, build_analogue_lookup
from testing.inference.explain import explain_focus_frame
from testing.bi_dashboard import write_bi_dashboard
from testing.report_builder import write_report
from testing.run_train import apply_replenishment
from testing.tensorflow_env import import_tensorflow


def _parse_as_of(raw: str | None, fact: pd.DataFrame) -> date:
    if raw:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    return max(fact["data"])


def _ensure_artefacts(cfg: TrainConfig, fact: pd.DataFrame, calendar: pd.DataFrame) -> dict:
    if ARTEFACTS_PATH.exists():
        print(f"Artefatti inferenza: {ARTEFACTS_PATH}")
        return load_forecast_artefacts()

    print("Artefatti inferenza assenti — li ricostruisco dallo stesso dataset del train (una tantum, senza retrain).")
    from testing.model.dual_head_tf import fit_encoders, load_encoder_vocab

    as_of_list = as_of_range(cfg.train_as_of_start, cfg.train_as_of_end, cfg.as_of_step_days)
    samples = build_training_frames(fact, calendar, cfg, as_of_list, require_target=True)
    train_df, _, _, _ = apply_train_test_split(samples, cfg)
    catalog_vocab = load_encoder_vocab(DATA_DIR)
    maps = fit_encoders(train_df, catalog_vocab)
    num_mean = train_df[NUMERIC_COLS].astype(float).mean().to_numpy(copy=True)
    num_std = train_df[NUMERIC_COLS].astype(float).std().to_numpy(copy=True)
    num_std[num_std == 0] = 1.0
    analogue_lookup = build_analogue_lookup(train_df, cfg.horizon_sku_days)
    cdm_fitted = fit_cdm_baseline(train_df, cfg.horizon_sku_days)
    path = save_forecast_artefacts(
        maps=maps,
        num_mean=num_mean,
        num_std=num_std,
        analogue_lookup=analogue_lookup,
        cdm_fitted=cdm_fitted,
        explain_ref=train_df.sample(n=min(2000, len(train_df)), random_state=cfg.seed),
    )
    print(f"Salvati in {path}")
    return load_forecast_artefacts()


def _cdm_pack(focus: pd.DataFrame, horizon: int, as_of: date) -> pd.DataFrame:
    horizon_from = as_of + timedelta(days=1)
    horizon_to = as_of + timedelta(days=horizon)
    products_path = DATA_DIR / "products.csv"
    products = pd.read_csv(products_path) if products_path.exists() else pd.DataFrame()
    pdv = pd.read_csv(DATA_DIR / "pdv_master.csv") if (DATA_DIR / "pdv_master.csv").exists() else pd.DataFrame()

    cols = [
        "as_of",
        "pdv_id",
        "sku_id",
        "pred_final",
        "pred_model",
        "pred_cdm",
        "confidence_score",
        "is_new_pdv",
        "run_rate_14d",
        "vendite_previste_15gg",
        "giacenza",
        "qty_suggerita",
        "azione",
        "stagione_stimata",
    ]
    if "y_sku" in focus.columns:
        cols.insert(3, "y_sku")
    pack = focus[[c for c in cols if c in focus.columns]].copy()
    pack = pack.rename(columns={"pred_final": "pred_tf", "y_sku": "y_vera"})
    pack["as_of"] = pd.to_datetime(pack["as_of"]).dt.strftime("%Y-%m-%d")
    pack.insert(1, "orizzonte_da", horizon_from.isoformat())
    pack.insert(2, "orizzonte_a", horizon_to.isoformat())
    pack["pred_tf"] = pack["pred_tf"].round(1)
    if "pred_model" in pack.columns:
        pack["pred_model"] = pack["pred_model"].round(1)
    pack["pred_cdm"] = pack["pred_cdm"].round(1)
    pack["delta_vs_cdm"] = (pack["pred_tf"] - pack["pred_cdm"]).round(1)
    pack["confidence_score"] = pack["confidence_score"].round(3)
    if "is_new_pdv" in pack.columns:
        pack["is_new_pdv"] = pack["is_new_pdv"].astype(int)
    if "run_rate_14d" in pack.columns:
        pack["run_rate_14d"] = pack["run_rate_14d"].round(3)

    if not products.empty:
        keep = [c for c in ["sku_id", "nome", "categoria_id", "modello"] if c in products.columns]
        pack = pack.merge(products[keep].drop_duplicates("sku_id"), on="sku_id", how="left")
    if not pdv.empty:
        keep = [c for c in ["pdv_id", "country", "city"] if c in pdv.columns]
        pack = pack.merge(pdv[keep].drop_duplicates("pdv_id"), on="pdv_id", how="left")

    return pack.sort_values("qty_suggerita", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Previsioni 15 giorni per il CDM (senza retrain).")
    parser.add_argument(
        "--as-of",
        default=None,
        help="Data di chiusura (YYYY-MM-DD). Default: ultimo giorno in fact_daily.",
    )
    args = parser.parse_args()

    cfg = TrainConfig()
    model_path = DATA_DIR / "dual_head_testing.keras"
    if not model_path.exists():
        raise SystemExit(f"Manca {model_path} — esegui prima: OMP_NUM_THREADS=1 .venv2/bin/python -m testing.run_train")

    print("=== 1) Load dati ===")
    fact, calendar = load_fact_calendar()
    as_of = _parse_as_of(args.as_of, fact)
    horizon_to = as_of + timedelta(days=cfg.horizon_sku_days)
    print(f"as_of (chiusura): {as_of}")
    print(f"orizzonte CDM:    {as_of + timedelta(days=1)} → {horizon_to} ({cfg.horizon_sku_days} gg)")

    if as_of not in set(calendar["data"]):
        raise SystemExit(f"as_of {as_of} non è nel calendario. Controlla calendar.csv.")

    print("\n=== 2) Artefatti encoder / analoghi / CDM ===")
    artefacts = _ensure_artefacts(cfg, fact, calendar)
    maps = artefacts["maps"]
    num_mean = artefacts["num_mean"]
    num_std = artefacts["num_std"]
    analogue_lookup = artefacts["analogue_lookup"]
    cdm_fitted = artefacts["cdm_fitted"]
    explain_ref = artefacts["explain_ref"]

    print("\n=== 3) Feature sulla sola chiusura (niente vendite future in input) ===")
    feat = build_training_frames(
        fact, calendar, cfg, [as_of], require_target=False
    )
    if feat.empty:
        raise SystemExit(
            f"Nessuna riga per as_of={as_of}. Servono almeno 14 giorni di vendite fino a quella data."
        )
    print(f"Righe PDV×SKU: {len(feat):,}")

    print("\n=== 4) Inferenza rete + blend + confronto CDM ===")
    import_tensorflow()
    import tensorflow as tf

    from testing.model.dual_head_tf import predict

    model = tf.keras.models.load_model(model_path)
    pred_s, _ = predict(model, feat, maps, num_mean, num_std)
    blend = apply_confidence_to_frame(feat, pred_s, analogue_lookup, cfg.horizon_sku_days)
    blend["pred_cdm"] = predict_cdm_baseline(blend, cdm_fitted, cfg.horizon_sku_days)
    blend = apply_replenishment(blend, cfg.horizon_sku_days)
    blend = add_season_labels(blend)
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
        if c in blend.columns
    ]
    if "sku_id" in sku_prof_cols:
        blend[sku_prof_cols].drop_duplicates("sku_id").to_csv(
            DATA_DIR / "sku_meteo_profilo.csv", index=False
        )

    pack = _cdm_pack(blend, cfg.horizon_sku_days, as_of)
    dated_name = f"previsioni_cdm_{as_of.isoformat()}.csv"
    pack.to_csv(DATA_DIR / dated_name, index=False)
    pack.to_csv(DATA_DIR / "previsioni_cdm.csv", index=False)

    # Dashboard / report riusano questi file
    dash_cols = [
        c
        for c in [
            "as_of",
            "pdv_id",
            "sku_id",
            "y_vera",
            "pred_tf",
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
        if c in pack.columns
    ]
    pack[dash_cols].to_csv(DATA_DIR / "priorita_riassortimento.csv", index=False)

    didactic = blend[
        ((blend["pdv_id"] == "MIL_01") & (blend["sku_id"].isin(["SKU_A", "SKU_Z", "SKU_J", "SKU_W"])))
        | ((blend["pdv_id"] == "ROM_NEW") & (blend["sku_id"] == "SKU_A"))
    ]
    top_n = blend.sort_values("qty_suggerita", ascending=False).head(40)
    explain_src = pd.concat([didactic, top_n]).drop_duplicates(["pdv_id", "sku_id"])

    print(f"\n=== 5) Spiegazioni (top qty + casi didattici, {len(explain_src)} righe) ===")
    explanations = explain_focus_frame(
        explain_src,
        model,
        maps,
        num_mean,
        num_std,
        explain_ref,
        predict,
        horizon=cfg.horizon_sku_days,
    )
    focus_out = explain_src.copy()
    focus_out["as_of"] = pd.to_datetime(focus_out["as_of"]).dt.strftime("%Y-%m-%d")
    explanations["as_of"] = pd.to_datetime(explanations["as_of"]).dt.strftime("%Y-%m-%d")
    focus_out = focus_out.merge(explanations, on=["as_of", "pdv_id", "sku_id"], how="left")
    focus_out.to_csv(DATA_DIR / "focus_casi_tf.csv", index=False)
    explanations.to_csv(DATA_DIR / "focus_casi_spiegazioni.csv", index=False)

    n_valuta = int((pack["azione"] == "valuta_riassortimento").sum())
    n_monitora = int((pack["azione"] == "monitora").sum())
    qty_tot = float(pack["qty_suggerita"].sum())
    meta = {
        "as_of": as_of.isoformat(),
        "orizzonte_da": (as_of + timedelta(days=1)).isoformat(),
        "orizzonte_a": horizon_to.isoformat(),
        "n_righe": int(len(pack)),
        "n_valuta_riassortimento": n_valuta,
        "n_monitora": n_monitora,
        "qty_suggerita_totale": round(qty_tot, 0),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    (DATA_DIR / "forecast_run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n=== 6) Pacchetto CDM ===")
    print(f"  Righe:              {len(pack):,}")
    print(f"  valuta_riassortimento: {n_valuta:,}")
    print(f"  monitora:           {n_monitora:,}")
    print(f"  qty totale:         {qty_tot:,.0f}")
    print(f"  CSV:                {DATA_DIR / 'previsioni_cdm.csv'}")
    print(f"  CSV datato:         {DATA_DIR / dated_name}")

    print("\n=== 7) Dashboard ===")
    report_path = write_report()
    bi_path = write_bi_dashboard()
    print(f"Report:    {report_path}")
    print(f"Dashboard: {bi_path}")
    print(f"Apri:      open {bi_path}")


if __name__ == "__main__":
    main()
