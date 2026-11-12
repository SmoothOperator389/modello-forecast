"""Esclusione as_of «poisoned» (Jul holes) dal rebuild samples SKU.

Default: tutti gli as_of in luglio/agosto + eventuali date esplicite via env
``POISONED_AS_OF_EXTRA=YYYY-MM-DD,YYYY-MM-DD``.
"""

from __future__ import annotations

import os
from datetime import date, datetime

import pandas as pd

from production.ops.month_reliability import is_non_affidabile_as_of


def _parse_extra() -> set[date]:
    raw = os.environ.get("POISONED_AS_OF_EXTRA", "").strip()
    out: set[date] = set()
    if not raw:
        return out
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(date.fromisoformat(part[:10]))
    return out


def is_poisoned_as_of(as_of) -> bool:
    if is_non_affidabile_as_of(as_of):
        return True
    if isinstance(as_of, datetime):
        d = as_of.date()
    elif isinstance(as_of, date):
        d = as_of
    else:
        d = date.fromisoformat(str(as_of)[:10])
    return d in _parse_extra()


def filter_samples_exclude_poisoned(samples: pd.DataFrame) -> pd.DataFrame:
    s = samples.copy()
    s["as_of"] = pd.to_datetime(s["as_of"]).dt.date
    before = len(s)
    mask = ~s["as_of"].map(is_poisoned_as_of)
    out = s.loc[mask].copy()
    dropped = before - len(out)
    out.attrs["poisoned_dropped_rows"] = int(dropped)
    out.attrs["poisoned_dropped_as_ofs"] = int(
        s.loc[~mask, "as_of"].nunique() if dropped else 0
    )
    return out
