"""Calendario commerciale + festività da file team."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from production.config import RunConfig


def _expand_commercial(commercial: pd.DataFrame) -> pd.DataFrame:
    """Espande intervalli data_inizio/data_fine in righe giornaliere per evento."""
    commercial = commercial.copy()
    commercial["data_inizio"] = pd.to_datetime(commercial["data_inizio"]).dt.date
    commercial["data_fine"] = pd.to_datetime(commercial["data_fine"]).dt.date

    rows = []
    for _, r in commercial.iterrows():
        d = r["data_inizio"]
        evento = str(r["evento"])
        valore = r["valore"]
        while d <= r["data_fine"]:
            rows.append({"data": d, "country": r["country"], "evento": evento, "valore": valore})
            d += timedelta(days=1)
    if not rows:
        return pd.DataFrame(columns=["data", "country"])
    long = pd.DataFrame(rows)
    # pivot: una colonna per tipo evento
    return long.pivot_table(
        index=["data", "country"],
        columns="evento",
        values="valore",
        aggfunc="last",
    ).reset_index()


def load_calendar(cfg: RunConfig | None = None, start: date | None = None, end: date | None = None) -> pd.DataFrame:
    """
    Costruisce calendario giornaliero per country.
    Schema output allineato a build_features:
      data, country, mese, sin_anno, cos_anno, is_festivo, is_saldi, giorni_da_inizio_collezione
    """
    cfg = cfg or RunConfig()
    start = start or date(2024, 1, 1)
    end = end or cfg.as_of

    holidays = pd.read_csv(cfg.calendar_holidays_path, parse_dates=["data"])
    holidays["data"] = holidays["data"].dt.date
    holidays["is_festivo"] = 1

    commercial_raw = pd.read_csv(cfg.calendar_commercial_path)
    commercial_exp = _expand_commercial(commercial_raw)

    rows = []
    d = start
    while d <= end:
        week = int(d.strftime("%W"))
        angle = 2 * np.pi * week / 52.0
        for country in sorted(holidays["country"].unique()):
            rows.append({"data": d, "country": country})
        d += timedelta(days=1)

    cal = pd.DataFrame(rows)
    cal = cal.merge(holidays[["data", "country", "is_festivo"]], on=["data", "country"], how="left")
    cal["is_festivo"] = cal["is_festivo"].fillna(0).astype(int)

    if not commercial_exp.empty:
        cal = cal.merge(commercial_exp, on=["data", "country"], how="left")

    if "is_saldi" not in cal.columns:
        cal["is_saldi"] = 0
    else:
        cal["is_saldi"] = cal["is_saldi"].fillna(0).astype(int)

    if "collezione_start" in cal.columns:
        cal["giorni_da_inizio_collezione"] = cal.apply(
            lambda r: (r["data"] - pd.to_datetime(r["collezione_start"]).date()).days
            if pd.notna(r.get("collezione_start"))
            else 0,
            axis=1,
        )
    else:
        cal["giorni_da_inizio_collezione"] = 0

    cal["mese"] = cal["data"].apply(lambda x: x.month)
    cal["settimana_anno"] = cal["data"].apply(lambda x: int(x.strftime("%W")))
    cal["sin_anno"] = cal["settimana_anno"].apply(lambda w: float(np.sin(2 * np.pi * w / 52.0)))
    cal["cos_anno"] = cal["settimana_anno"].apply(lambda w: float(np.cos(2 * np.pi * w / 52.0)))

    return cal


def calendar_for_date(cal: pd.DataFrame, as_of: date, country: str) -> pd.Series | None:
    row = cal[(cal["data"] == as_of) & (cal["country"] == country)]
    if row.empty:
        return None
    return row.iloc[0]
