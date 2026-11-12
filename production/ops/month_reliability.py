"""Policy Confronta mesi: Jul/Aug NON AFFIDABILE senza Export luglio.

P0: non attendere July export. Luglio e agosto restano esclusi dalle
metriche «fair» di Confronta / scoreboard finché non arriva un export affidabile.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

# Mesi calendariali (1–12) con buco dati / export non affidabile.
NON_AFFIDABILE_MONTHS: frozenset[int] = frozenset({7, 8})  # luglio, agosto

NON_AFFIDABILE_LABEL = "NON AFFIDABILE"
NON_AFFIDABILE_REASON = (
    "Export luglio assente / as_of poisoned (Jul holes): "
    "mese escluso dal confronto fair fino a backfill."
)


def _to_date(v) -> date | None:
    if v is None:
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    s = str(v)[:10]
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def is_non_affidabile_month(month: int) -> bool:
    return int(month) in NON_AFFIDABILE_MONTHS


def is_non_affidabile_as_of(as_of) -> bool:
    d = _to_date(as_of)
    if d is None:
        return False
    return d.month in NON_AFFIDABILE_MONTHS


def month_status(year: int, month: int) -> dict:
    bad = is_non_affidabile_month(month)
    return {
        "year": int(year),
        "month": int(month),
        "affidabile": not bad,
        "label": NON_AFFIDABILE_LABEL if bad else "OK",
        "reason": NON_AFFIDABILE_REASON if bad else None,
        "include_in_fair_compare": not bad,
    }


def filter_fair_as_ofs(as_ofs: Iterable) -> list:
    """Tieni solo as_of fuori da Jul/Aug per metriche Confronta fair."""
    out = []
    for a in as_ofs:
        if not is_non_affidabile_as_of(a):
            out.append(a)
    return out


def annotate_month_row(row: dict) -> dict:
    """Arricchisce una riga Confronta mesi con flag affidabilità."""
    out = dict(row)
    m = out.get("month") or out.get("mese")
    y = out.get("year") or out.get("anno")
    if m is None and out.get("as_of"):
        d = _to_date(out["as_of"])
        if d:
            m, y = d.month, d.year
    if m is None:
        out.setdefault("affidabile", True)
        return out
    st = month_status(int(y or 0), int(m))
    out["affidabile"] = st["affidabile"]
    out["affidabilita_label"] = st["label"]
    out["affidabilita_reason"] = st["reason"]
    out["include_in_fair_compare"] = st["include_in_fair_compare"]
    if not st["affidabile"]:
        out["wmape_fair"] = None
        out["note"] = NON_AFFIDABILE_LABEL
    return out
