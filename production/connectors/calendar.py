"""Calendario commerciale + festività per paese.

Festività: generate da ``holidays`` (offline, gratis) per tutti i country ISO3 del fact.
Override/aggiunte manuali: ``data/manual/calendar_holidays.csv``.
Saldi / collezione: ``data/manual/calendar_commercial.csv``.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from production.config import MANUAL_DATA_DIR, PROJECT_ROOT, RunConfig
from production.connectors.holidays_calendar import (
    build_holidays_frame,
    to_iso3,
    write_holidays_csv,
)


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
            rows.append({"data": d, "country": str(r["country"]).strip().upper(), "evento": evento, "valore": valore})
            d += timedelta(days=1)
    if not rows:
        return pd.DataFrame(columns=["data", "country"])
    long = pd.DataFrame(rows)
    return long.pivot_table(
        index=["data", "country"],
        columns="evento",
        values="valore",
        aggfunc="last",
    ).reset_index()


def _load_holidays(countries: list[str], start: date, end: date, cfg: RunConfig) -> pd.DataFrame:
    """Unisce festività auto (holidays lib) + file manuale."""
    auto = build_holidays_frame(countries, start, end)
    # cache CSV aggiornato per audit
    write_holidays_csv(
        PROJECT_ROOT / "data" / "real" / "calendar_holidays_auto.csv",
        countries,
        start,
        end,
    )
    manual_path = cfg.calendar_holidays_path
    parts = []
    if not auto.empty:
        parts.append(auto[["country", "data", "nome_festivo"]])
    if manual_path.exists():
        man = pd.read_csv(manual_path, parse_dates=["data"])
        man["data"] = man["data"].dt.date
        man["country"] = man["country"].map(to_iso3)
        countries_set = set(countries)
        man = man[man["country"].isin(countries_set)]
        if not man.empty:
            parts.append(man[["country", "data", "nome_festivo"]])
    if not parts:
        return pd.DataFrame(columns=["country", "data", "nome_festivo", "is_festivo"])
    hol = pd.concat(parts, ignore_index=True).drop_duplicates(["country", "data"])
    hol["is_festivo"] = 1
    return hol


def load_calendar(
    cfg: RunConfig | None = None,
    start: date | None = None,
    end: date | None = None,
    countries: list[str] | None = None,
) -> pd.DataFrame:
    """
    Calendario giornaliero per country (ISO3 Valido).
    Schema:
      data, country, mese, sin_anno, cos_anno, is_festivo, is_saldi, giorni_da_inizio_collezione
    """
    cfg = cfg or RunConfig()
    start = start or date(2024, 1, 1)
    end = end or date(2026, 12, 31)

    if countries is None:
        pdv_path = PROJECT_ROOT / "data" / "real" / "pdv_master.csv"
        if pdv_path.exists():
            countries = (
                pd.read_csv(pdv_path, usecols=["country"])["country"]
                .astype(str)
                .str.strip()
                .str.upper()
                .unique()
                .tolist()
            )
        else:
            countries = ["ITA", "FRA", "DEU", "ESP", "GBR"]

    countries = sorted({str(c).strip().upper() for c in countries if str(c).strip()})
    holidays = _load_holidays(countries, start, end, cfg)

    commercial_raw = pd.read_csv(cfg.calendar_commercial_path)
    # Accetta IT o ITA nei file manuali → sempre ISO3 Valido
    commercial_raw["country"] = commercial_raw["country"].map(to_iso3)
    commercial_exp = _expand_commercial(commercial_raw)

    rows = []
    d = start
    while d <= end:
        for country in countries:
            rows.append({"data": d, "country": country})
        d += timedelta(days=1)
    cal = pd.DataFrame(rows)

    cal = cal.merge(
        holidays[["data", "country", "is_festivo"]],
        on=["data", "country"],
        how="left",
    )
    cal["is_festivo"] = cal["is_festivo"].fillna(0).astype(int)

    if not commercial_exp.empty:
        cal = cal.merge(commercial_exp, on=["data", "country"], how="left")

    if "is_saldi" not in cal.columns:
        cal["is_saldi"] = 0
    else:
        cal["is_saldi"] = pd.to_numeric(cal["is_saldi"], errors="coerce").fillna(0).astype(int)

    if "collezione_start" in cal.columns:
        def _giorni(r):
            v = r.get("collezione_start")
            if pd.isna(v):
                return 0
            try:
                start_c = pd.to_datetime(v).date()
            except Exception:
                return 0
            return (r["data"] - start_c).days

        cal["giorni_da_inizio_collezione"] = cal.apply(_giorni, axis=1)
    else:
        cal["giorni_da_inizio_collezione"] = 0

    cal["mese"] = cal["data"].map(lambda x: x.month)
    cal["settimana_anno"] = cal["data"].map(lambda x: int(x.strftime("%W")))
    cal["sin_anno"] = np.sin(2 * np.pi * cal["settimana_anno"] / 52.0)
    cal["cos_anno"] = np.cos(2 * np.pi * cal["settimana_anno"] / 52.0)

    # salva snapshot usabile dal train
    out_path = PROJECT_ROOT / "data" / "real" / "calendar_full.csv"
    cal.to_csv(out_path, index=False)
    return cal


def calendar_for_date(cal: pd.DataFrame, as_of: date, country: str) -> pd.Series | None:
    row = cal[(cal["data"] == as_of) & (cal["country"] == str(country).upper())]
    if row.empty:
        return None
    return row.iloc[0]


if __name__ == "__main__":
    cal = load_calendar()
    print(cal.groupby("country")["is_festivo"].sum().sort_values(ascending=False).head(20))
    print("rows", len(cal), "→ data/real/calendar_full.csv")
