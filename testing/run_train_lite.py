#!/usr/bin/env python3
"""
TESTING lite — valida dati fake + feature rolling SENZA TensorFlow.

Usare quando TF non gira nell'ambiente (es. mutex lock su Mac agent sandbox).
Equivalente funzionale al pre-check di run_train.py.

Uso:
  cd ~/Desktop/modello-forecast
  .venv/bin/python -m testing.run_train_lite
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from testing.config import DATA_DIR, FakeDataConfig, TrainConfig
from testing.fake_data.generator import write_all
from testing.features.build_features import (
    NUMERIC_COLS,
    apply_train_test_split,
    build_training_frames,
    load_fact_calendar,
)


def suggest_replenishment(forecast: float, giacenza: float, transit: float, horizon: int, immature: bool) -> dict:
    daily = forecast / max(horizon, 1)
    need = forecast + daily * transit
    raw = max(0.0, need - giacenza)
    if immature:
        raw *= 0.5
    qty = float(np.rint(raw))
    return {
        "vendite_previste_15gg": round(forecast, 1),
        "giacenza": giacenza,
        "qty_suggerita": qty,
        "azione": "monitora" if qty == 0 else "valuta_riassortimento",
    }


def main() -> None:
    cfg = TrainConfig()

    print("=== 1) Dataset fake (rigenera se assente o senza scontrini) ===")
    if not (DATA_DIR / "fact_daily.csv").exists() or not (DATA_DIR / "fact_tickets.csv").exists():
        write_all(FakeDataConfig())
    meta = pd.read_csv(DATA_DIR / "dataset_meta.csv")
    print(meta.to_string(index=False))

    print("\n=== 2) Sample rolling (no TensorFlow) ===")
    fact, calendar = load_fact_calendar()
    as_of_list = []
    d = date(2025, 1, 15)
    while d <= date(2026, 8, 10):
        as_of_list.append(d)
        d += timedelta(days=cfg.as_of_step_days)

    samples = build_training_frames(fact, calendar, cfg, as_of_list)
    print(f"Righe rolling (tutti gli as_of): {len(samples)}")
    print(f"Colonne numeriche: {len(NUMERIC_COLS)}")
    print(f"as_of distinti: {samples['as_of'].nunique()}")

    train_df, test_df, cut, train_start = apply_train_test_split(samples, cfg)
    if train_start is not None:
        print(
            f"Finestra training: {cfg.train_window_days} gg "
            f"(as_of >= {train_start}) → {len(train_df)} righe train"
        )
    else:
        print(f"Finestra training: tutto lo storico → {len(train_df)} righe train")
    print(f"Test: {len(test_df)} | cut as_of>={cut}")

    last = test_df["as_of"].max()
    focus = test_df[test_df["as_of"] == last].copy()

    # baseline naive
    focus["pred_naive"] = focus["run_rate_14d"] * cfg.horizon_sku_days
    mae_naive = np.abs(focus["pred_naive"] - focus["y_sku"]).mean()
    print(f"\nMAE naive (ultimo as_of, tutte le righe): {mae_naive:.2f}")

    interesting = focus[
        ((focus["pdv_id"] == "MIL_01") & (focus["sku_id"].isin(["SKU_A", "SKU_Z", "SKU_J", "SKU_W"])))
        | ((focus["pdv_id"] == "ROM_NEW") & (focus["sku_id"] == "SKU_A"))
    ].copy()

    rows = []
    for _, r in interesting.iterrows():
        immature = (r["has_history"] < 0.5) or (r["is_new_pdv"] > 0.5)
        op = suggest_replenishment(
            forecast=float(r["pred_naive"]),
            giacenza=float(r["giacenza"]),
            transit=float(r.get("transit_time_days", 5)),
            horizon=cfg.horizon_sku_days,
            immature=bool(immature),
        )
        rows.append(
            {
                "as_of": r["as_of"],
                "pdv_id": r["pdv_id"],
                "sku_id": r["sku_id"],
                "y_vera": r["y_sku"],
                "pred_naive": round(float(r["pred_naive"]), 1),
                "mq_espositivi_band": int(r["mq_espositivi_band"]),
                "weather_regime": int(r["weather_regime"]),
                "is_new_pdv": int(r["is_new_pdv"]),
                **op,
            }
        )

    out = pd.DataFrame(rows)
    out_path = Path(__file__).resolve().parent / "data" / "output" / "focus_casi_lite.csv"
    out.to_csv(out_path, index=False)

    print("\n=== 3) Casi didattici (baseline naive + operativo) ===")
    print(out.to_string(index=False))
    print(f"\nSalvato: {out_path}")
    print("\nPer training TensorFlow completo:")
    print("  Docker (consigliato su Mac): ./scripts/run_train_docker.sh")
    print("  Locale (Python 3.11+ venv):  ./scripts/setup_venv_mac.sh")
    print("    poi: OMP_NUM_THREADS=1 .venv/bin/python -m testing.run_train")


if __name__ == "__main__":
    main()
