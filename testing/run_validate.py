#!/usr/bin/env python3
"""
TESTING — valida generazione dati + feature rolling SENZA TensorFlow.

Usa questo se TF crasha (conflitto protobuf/pyarrow su Mac).
  cd ~/Desktop/modello-forecast
  .venv/bin/python -m testing.run_validate
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from testing.config import DATA_DIR, FakeDataConfig, TrainConfig
from testing.fake_data.generator import write_all
from testing.features.build_features import NUMERIC_COLS, build_training_frames, load_fact_calendar


def main() -> None:
    print("=== 1) Generazione dataset fake (CSV, no pyarrow) ===")
    write_all(FakeDataConfig())
    meta = pd.read_csv(DATA_DIR / "dataset_meta.csv")
    print(meta.to_string(index=False))

    cfg = TrainConfig()
    fact, calendar = load_fact_calendar()
    as_of_list = []
    d = date(2025, 1, 15)
    while d <= date(2026, 8, 10):
        as_of_list.append(d)
        d += timedelta(days=cfg.as_of_step_days)

    print("\n=== 2) Build sample rolling ===")
    samples = build_training_frames(fact, calendar, cfg, as_of_list)
    print(f"Righe rolling: {len(samples)}")
    print(f"PDV unici: {samples['pdv_id'].nunique()} | SKU unici: {samples['sku_id'].nunique()}")
    print(f"Colonne numeriche: {len(NUMERIC_COLS)}")

    cut = sorted(samples["as_of"].unique())[-1]
    last = samples[samples["as_of"] == cut]
    focus = last[
        ((last["pdv_id"] == "MIL_01") & (last["sku_id"].isin(["SKU_A", "SKU_Z", "SKU_J", "SKU_W"])))
        | ((last["pdv_id"] == "ROM_NEW") & (last["sku_id"] == "SKU_A"))
    ][["pdv_id", "sku_id", "run_rate_14d", "is_new_pdv", "mq_espositivi_band", "weather_regime", "y_sku"]]
    print(f"\n=== 3) Casi didattici (as_of={cut}) ===")
    print(focus.to_string(index=False))

    naive = last["run_rate_14d"].to_numpy() * cfg.horizon_sku_days
    mae_naive = np.abs(naive - last["y_sku"].to_numpy()).mean()
    print(f"\nBaseline naive MAE (tutto test): {mae_naive:.2f}")
    print("\nOK — pipeline dati validata. Per training TF: .venv/bin/python -m testing.run_train")


if __name__ == "__main__":
    main()
