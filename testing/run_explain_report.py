#!/usr/bin/env python3
"""Rigenera spiegazioni + report SENZA riaddestrare (riusa .keras e dataset).

Uso:
  cd ~/Desktop/modello-forecast
  OMP_NUM_THREADS=1 .venv2/bin/python -m testing.run_explain_report
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from testing.config import DATA_DIR, TrainConfig
from testing.features.build_features import (
    NUMERIC_COLS,
    apply_train_test_split,
    build_training_frames,
    load_fact_calendar,
)
from testing.inference.artefacts import ARTEFACTS_PATH, load_forecast_artefacts
from testing.inference.confidence import apply_confidence_to_frame, build_analogue_lookup
from testing.inference.explain import explain_focus_frame
from testing.model.dual_head_tf import fit_encoders, load_encoder_vocab, predict
from testing.bi_dashboard import write_bi_dashboard
from testing.report_builder import write_report
from testing.tensorflow_env import import_tensorflow
from testing.run_train import suggest_replenishment


def main() -> None:
    cfg = TrainConfig()
    model_path = DATA_DIR / "dual_head_testing.keras"
    if not model_path.exists():
        raise SystemExit(f"Manca {model_path} — esegui prima run_train")

    print("=== 1) Load dati + sample (solo ultimi as_of utili) ===")
    fact, calendar = load_fact_calendar()
    # Solo as_of recenti per velocità (train window + test)
    as_of_list = []
    d = date(2026, 4, 17)
    while d <= date(2026, 8, 10):
        as_of_list.append(d)
        d += timedelta(days=cfg.as_of_step_days)

    samples = build_training_frames(fact, calendar, cfg, as_of_list)
    train_df, test_df, cut, _ = apply_train_test_split(samples, cfg)
    print(f"Train {len(train_df)} | Test {len(test_df)} | cut {cut}")

    print("=== 2) Load modello ===")
    import_tensorflow()
    model = tf.keras.models.load_model(model_path)
    if ARTEFACTS_PATH.exists():
        art = load_forecast_artefacts()
        maps = art["maps"]
        num_mean = art["num_mean"]
        num_std = art["num_std"]
        analogue_lookup = art["analogue_lookup"]
        print(f"Encoder da {ARTEFACTS_PATH}")
    else:
        maps = fit_encoders(train_df, load_encoder_vocab(DATA_DIR))
        num_mean = train_df[NUMERIC_COLS].astype(float).mean().to_numpy(copy=True)
        num_std = train_df[NUMERIC_COLS].astype(float).std().to_numpy(copy=True)
        num_std[num_std == 0] = 1.0
        analogue_lookup = build_analogue_lookup(train_df, cfg.horizon_sku_days)
        print("Encoder ricostruiti solo sul train (artefatti assenti)")

    pred_s, _ = predict(model, test_df, maps, num_mean, num_std)
    test_blend = apply_confidence_to_frame(test_df, pred_s, analogue_lookup, cfg.horizon_sku_days)

    last = test_blend["as_of"].max()
    focus = test_blend[test_blend["as_of"] == last].copy()
    interesting = focus[
        ((focus["pdv_id"] == "MIL_01") & (focus["sku_id"].isin(["SKU_A", "SKU_Z", "SKU_J", "SKU_W"])))
        | ((focus["pdv_id"] == "ROM_NEW") & (focus["sku_id"] == "SKU_A"))
    ].copy()

    rows = []
    for _, r in interesting.iterrows():
        conf = float(r.get("confidence_score", 0.5))
        op = suggest_replenishment(
            float(r["pred_final"]),
            float(r["giacenza"]),
            float(r.get("transit_time_days", 5)),
            cfg.horizon_sku_days,
            conf,
        )
        rows.append(
            {
                "as_of": r["as_of"],
                "pdv_id": r["pdv_id"],
                "sku_id": r["sku_id"],
                "y_vera": r["y_sku"],
                "pred_tf": round(float(r["pred_final"]), 1),
                "pred_model": round(float(r["pred_model"]), 1),
                "pred_analogue": round(float(r.get("pred_analogue", 0.0)), 1),
                "confidence_score": round(conf, 3),
                "run_rate_14d": round(float(r["run_rate_14d"]), 3),
                "is_new_pdv": int(r["is_new_pdv"]),
                **op,
            }
        )
    out = pd.DataFrame(rows)

    print("=== 3) Spiegazioni ===")
    explanations = explain_focus_frame(
        interesting, model, maps, num_mean, num_std, train_df, predict, cfg.horizon_sku_days
    )
    out["as_of"] = pd.to_datetime(out["as_of"]).dt.strftime("%Y-%m-%d")
    explanations["as_of"] = pd.to_datetime(explanations["as_of"]).dt.strftime("%Y-%m-%d")
    out = out.merge(explanations, on=["as_of", "pdv_id", "sku_id"], how="left")
    out.to_csv(DATA_DIR / "focus_casi_tf.csv", index=False)
    explanations.to_csv(DATA_DIR / "focus_casi_spiegazioni.csv", index=False)

    for _, er in explanations.iterrows():
        print(f"  {er['pdv_id']}/{er['sku_id']}: {er.get('driver_1', '—')}")
        print(f"    {er['spiegazione_testuale'][:160]}...")

    path = write_report()
    bi_path = write_bi_dashboard()
    print(f"\nReport: {path}")
    print(f"Dashboard CDM: {bi_path}")
    print(f"Apri: open {bi_path}")


if __name__ == "__main__":
    main()
