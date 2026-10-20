#!/usr/bin/env python3
"""
PRODUCTION — primo training DualHead su dati Valido reali (data/real).

- Vendite/giacenza: fact_daily.csv
- DNA SKU: products.csv (modello/colore/taglia/tessuto + campagna/classe dove c'è)
- PDV: pdv_master.csv
- Meteo: profilo simulato mensile (no API) — come testing di default

Uso:
  cd ~/Desktop/modello-forecast
  OMP_NUM_THREADS=1 .venv2/bin/python -m production.run_train_real

Pilota più leggero (consigliato la prima volta):
  FORECAST_PILOT=1 OMP_NUM_THREADS=1 .venv2/bin/python -m production.run_train_real
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from production.config import MODEL_DIR, PROJECT_ROOT
from testing.config import TrainConfig, as_of_range
from testing.features.build_features import (
    NUMERIC_COLS,
    apply_train_test_split,
    build_training_frames,
)
from testing.inference.cdm_baseline import fit_cdm_baseline, predict_cdm_baseline
from testing.inference.confidence import apply_confidence_to_frame, build_analogue_lookup
from testing.tensorflow_env import import_tensorflow

REAL_DIR = PROJECT_ROOT / "data" / "real"
ART_DIR = MODEL_DIR / "real"
ART_DIR.mkdir(parents=True, exist_ok=True)

SKU_ATTRS = [
    "modello",
    "tessuto",
    "colore",
    "campagna",
    "classe",
    "sclasse",
    "ssclasse",
    "tipologia",
    "nome",
    "categoria_id",
    "data_lancio_sku",
]


def _pilot() -> bool:
    return os.environ.get("FORECAST_PILOT", "").strip().lower() in {"1", "true", "yes", "on"}


def load_real_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("=== 1) Load data/real ===")
    usecols = [
        "data",
        "country",
        "pdv_id",
        "sku_id",
        "vendite_qty",
        "giacenza_qty",
        "ora",
        "prezzo_listino",
        "prezzo_pagato",
        "sconto_pct",
        "is_in_sconto",
        "data_apertura",
    ]
    fact = pd.read_csv(
        REAL_DIR / "fact_daily.csv",
        usecols=usecols,
        parse_dates=["data", "data_apertura"],
        dtype={
            "country": "category",
            "pdv_id": "category",
            "sku_id": "category",
        },
        low_memory=False,
    )
    fact["data"] = fact["data"].dt.date
    fact["data_apertura"] = fact["data_apertura"].dt.date
    # Giacenza negativa Valido: per feature operative clip a 0
    fact["giacenza_qty"] = fact["giacenza_qty"].astype("float32").clip(lower=0.0)
    fact["vendite_qty"] = fact["vendite_qty"].astype("float32")

    products = pd.read_csv(REAL_DIR / "products.csv", dtype=str)
    for c in SKU_ATTRS:
        if c not in products.columns:
            products[c] = "NA"
    products["data_lancio_sku"] = pd.to_datetime(products["data_lancio_sku"], errors="coerce").dt.date
    products["data_lancio_sku"] = products["data_lancio_sku"].fillna(date(2024, 1, 1))

    # Sovrascrive attributi stub dell'ingest con DNA aggiornato
    drop_attrs = [c for c in SKU_ATTRS if c in fact.columns]
    fact = fact.drop(columns=drop_attrs, errors="ignore")
    fact["sku_id"] = fact["sku_id"].astype(str)
    products["sku_id"] = products["sku_id"].astype(str)
    fact = fact.merge(products[["sku_id", *SKU_ATTRS]], on="sku_id", how="left")
    for c in SKU_ATTRS:
        if c == "data_lancio_sku":
            fact[c] = pd.to_datetime(fact[c], errors="coerce").dt.date
            fact[c] = fact[c].fillna(date(2024, 1, 1))
        else:
            fact[c] = fact[c].astype(object).where(fact[c].notna(), "NA").astype(str)
            fact[c] = fact[c].replace({"": "NA", "nan": "NA", "None": "NA", "<NA>": "NA"})

    fact["sku_id"] = fact["sku_id"].astype(str).astype("category")
    fact["pdv_id"] = fact["pdv_id"].astype(str).astype("category")
    fact["country"] = fact["country"].astype(str).astype("category")
    fact["categoria_id"] = fact["categoria_id"].astype(str).astype("category")

    from production.connectors.calendar import load_calendar

    countries = sorted(fact["country"].astype(str).unique())
    calendar = load_calendar(
        start=date(2024, 1, 1),
        end=date(2026, 12, 31),
        countries=countries,
    )
    need = [
        "data",
        "country",
        "mese",
        "sin_anno",
        "cos_anno",
        "is_festivo",
        "is_saldi",
        "giorni_da_inizio_collezione",
    ]
    calendar = calendar[need].copy()
    print(f"  calendar rows={len(calendar):,} countries={calendar['country'].nunique()}")
    print(
        "  festivi/paese (sample):",
        calendar.groupby("country", observed=True)["is_festivo"].sum().sort_values(ascending=False).head(8).to_dict(),
    )

    pdv = pd.read_csv(REAL_DIR / "pdv_master.csv", parse_dates=["data_apertura"])
    pdv["data_apertura"] = pdv["data_apertura"].dt.date
    if "transit_time_days" not in pdv.columns:
        pdv["transit_time_days"] = 5.0
    pdv["transit_time_days"] = pdv["transit_time_days"].fillna(5.0)
    for col in ("mq_espositivi", "mq_stock", "mq_totali", "numero_fatturazioni"):
        if col not in pdv.columns:
            pdv[col] = 0.0
        pdv[col] = pd.to_numeric(pdv[col], errors="coerce").fillna(0.0)
    if "commerciale_riferimento" not in pdv.columns:
        pdv["commerciale_riferimento"] = "NA"
    pdv["commerciale_riferimento"] = pdv["commerciale_riferimento"].fillna("NA").astype(str)
    if "city" not in pdv.columns:
        pdv["city"] = "NA"
    pdv["city"] = pdv["city"].fillna("NA").astype(str)

    print(
        f"  fact={len(fact):,}  sku={fact['sku_id'].nunique():,}  "
        f"pdv={fact['pdv_id'].nunique():,}  "
        f"date {fact['data'].min()} → {fact['data'].max()}"
    )
    return fact, calendar, pdv


def maybe_pilot_filter(
    fact: pd.DataFrame,
    pdv: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, TrainConfig, list[date]]:
    cfg = TrainConfig()
    # Date reali: target fino a ~2026-08-18 → ultimo as_of utile ~2026-08-03
    cfg.train_as_of_end = date(2026, 8, 3)
    cfg.as_of_step_days = 14
    cfg.epochs = 50
    cfg.batch_size = 2048
    cfg.train_window_days = 365

    if _pilot():
        print("=== PILOT mode (FORECAST_PILOT=1) ===")
        # Top PDV per volume vendite: segnale forte, runtime contenuto
        vol = (
            fact.groupby("pdv_id", observed=True)["vendite_qty"]
            .sum()
            .sort_values(ascending=False)
        )
        keep_pdv = set(vol.head(40).index.astype(str))
        fact = fact[fact["pdv_id"].astype(str).isin(keep_pdv)].copy()
        pdv = pdv[pdv["pdv_id"].astype(str).isin(keep_pdv)].copy()
        cfg.train_as_of_start = date(2025, 10, 1)
        cfg.as_of_step_days = 14
        cfg.train_window_days = 270
        cfg.epochs = 40
        cfg.batch_size = 1024
        print(f"  PDV pilota={len(keep_pdv)}  fact filtrato={len(fact):,}")
    else:
        cfg.train_as_of_start = date(2025, 6, 1)

    as_ofs = as_of_range(cfg.train_as_of_start, cfg.train_as_of_end, cfg.as_of_step_days)
    # Serve storico 28gg prima del primo as_of
    hist_start = as_ofs[0] - timedelta(days=28)
    fact = fact[fact["data"] >= hist_start].copy()
    print(f"  as_of={len(as_ofs)} ({as_ofs[0]} → {as_ofs[-1]})  fact window={len(fact):,}")
    return fact, pdv, cfg, as_ofs


def main() -> None:
    if sys.prefix == sys.base_prefix and not os.environ.get("VIRTUAL_ENV"):
        print("Usa .venv2/bin/python -m production.run_train_real", file=sys.stderr)

    fact, calendar, pdv = load_real_tables()
    fact, pdv, cfg, as_ofs = maybe_pilot_filter(fact, pdv)
    np.random.seed(cfg.seed)

    print("\n=== 2) Build sample rolling ===")
    samples = build_training_frames(fact, calendar, cfg, as_ofs, pdv_master=pdv)
    print(f"  righe sample={len(samples):,}  as_of distinti={samples['as_of'].nunique()}")
    if samples.empty:
        raise SystemExit("Nessun sample: controlla date / filtri")

    samples_path = ART_DIR / ("samples_pilot.pkl" if _pilot() else "samples.pkl")
    samples.to_pickle(samples_path)
    print(f"  salvato {samples_path} ({samples_path.stat().st_size / 1e6:.1f} MB)")

    train_df, test_df, cut, train_start = apply_train_test_split(samples, cfg)
    print(
        f"  train={len(train_df):,} (from {train_start})  "
        f"test={len(test_df):,} (as_of>={cut})"
    )

    tf = import_tensorflow()
    from testing.model.dual_head_tf import (
        build_dual_head_model,
        fit_encoders,
        load_encoder_vocab,
        predict,
        train_model,
    )

    tf.random.set_seed(cfg.seed)
    # Vocab da anagrafica reale
    catalog_vocab = load_encoder_vocab(REAL_DIR)
    maps = fit_encoders(train_df, catalog_vocab)
    num_mean = train_df[NUMERIC_COLS].astype(float).mean().to_numpy(copy=True)
    num_std = train_df[NUMERIC_COLS].astype(float).std().to_numpy(copy=True)
    num_std[num_std == 0] = 1.0

    model = build_dual_head_model(n_numeric=len(NUMERIC_COLS), maps=maps)
    print(f"\n=== 3) Training TF — params={model.count_params():,} ===")
    model, history = train_model(model, train_df, maps, num_mean, num_std, cfg)

    pred_s, _ = predict(model, test_df, maps, num_mean, num_std)
    y_true = test_df["y_sku"].to_numpy()
    naive = test_df["run_rate_14d"].to_numpy() * cfg.horizon_sku_days
    cdm = fit_cdm_baseline(train_df, cfg.horizon_sku_days)
    pred_cdm = predict_cdm_baseline(test_df, cdm, cfg.horizon_sku_days)
    analogue = build_analogue_lookup(train_df, cfg.horizon_sku_days)
    blended = apply_confidence_to_frame(test_df, pred_s, analogue, cfg.horizon_sku_days)
    pred_final = blended["pred_final"].to_numpy()

    def _mae(p: np.ndarray) -> float:
        return float(np.abs(p - y_true).mean())

    mae = _mae(pred_final)
    mae_raw = _mae(pred_s)
    mae_cdm = _mae(pred_cdm)
    mae_nv = _mae(naive)
    mape = float(np.mean(np.abs(pred_final - y_true) / np.maximum(y_true, 1.0)) * 100)
    ss = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum((pred_final - y_true) ** 2) / ss) if ss > 0 else 0.0
    imp_cdm = (mae_cdm - mae) / mae_cdm * 100 if mae_cdm else 0.0

    metrics = {
        "pilot": _pilot(),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_pdv": int(fact["pdv_id"].nunique()),
        "n_sku": int(fact["sku_id"].nunique()),
        "mae_final": round(mae, 4),
        "mae_raw": round(mae_raw, 4),
        "mae_cdm": round(mae_cdm, 4),
        "mae_naive": round(mae_nv, 4),
        "mape_final": round(mape, 2),
        "r2_final": round(r2, 4),
        "improvement_vs_cdm_pct": round(imp_cdm, 2),
        "as_of_start": str(as_ofs[0]),
        "as_of_end": str(as_ofs[-1]),
        "cut_test": str(cut),
    }
    print("\n=== 4) RISULTATI TEST ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    model_path = ART_DIR / ("dual_head_pilot.keras" if _pilot() else "dual_head.keras")
    enc_path = ART_DIR / ("encoders_pilot.npz" if _pilot() else "encoders.npz")
    model.save(model_path)
    np.savez(enc_path, maps=maps, num_mean=num_mean, num_std=num_std, allow_pickle=True)
    (ART_DIR / ("metrics_pilot.json" if _pilot() else "metrics.json")).write_text(
        json.dumps(metrics, indent=2)
    )
    hist_obj = history.history if hasattr(history, "history") else history
    pd.DataFrame(hist_obj).to_csv(
        ART_DIR / ("history_pilot.csv" if _pilot() else "history.csv"), index=False
    )
    print(f"\nArtefacts → {ART_DIR}")
    print(f"  model={model_path.name}")


if __name__ == "__main__":
    main()
