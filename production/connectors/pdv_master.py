"""Lettura master PDV da file interni team (CSV manuale)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from production.config import RunConfig


def load_pdv_master(path: Path | None = None, cfg: RunConfig | None = None) -> pd.DataFrame:
    cfg = cfg or RunConfig()
    path = path or cfg.pdv_master_path
    if not path.exists():
        raise FileNotFoundError(
            f"Master PDV non trovato: {path}\n"
            "Compilare data/manual/pdv_master.csv con i campi della riunione."
        )
    df = pd.read_csv(path, parse_dates=["data_apertura"])
    df["data_apertura"] = df["data_apertura"].dt.date
    if "transit_time_days" not in df.columns:
        if "transit_time" in df.columns:
            df["transit_time_days"] = df["transit_time"]
        else:
            raise ValueError("Manca transit_time o transit_time_days nel master PDV")
    if "mq_totali" not in df.columns:
        df["mq_totali"] = df["mq_espositivi"] + df["mq_stock"]
    return df
