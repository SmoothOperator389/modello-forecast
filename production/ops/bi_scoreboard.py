"""BI scoreboard: CDM top-80 headline, catalog secondario, naive opzionale.

Produce un payload JSON consumabile dalla BI (:8765) senza imporre un layout
card-heavy — solo metriche ordinate.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from production.ops.month_reliability import filter_fair_as_ofs, is_non_affidabile_as_of


def _wmape(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.abs(p - y).sum() / max(float(y.sum()), 1.0) * 100)


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.abs(p - y).mean())


def cdm_top80_slice(df: pd.DataFrame, *, y_col: str = "y", sku_col: str = "sku_id") -> pd.DataFrame:
    """Top 80 SKU per volume reale (CDM priority)."""
    vol = df.groupby(sku_col, observed=True)[y_col].sum().sort_values(ascending=False)
    top = set(vol.head(80).index.astype(str))
    out = df[df[sku_col].astype(str).isin(top)].copy()
    out.attrs["n_sku_top80"] = len(top)
    out.attrs["volume_share"] = float(vol.head(80).sum() / max(float(vol.sum()), 1.0))
    return out


def build_scoreboard(
    df: pd.DataFrame,
    *,
    y_col: str = "y",
    pred_cdm_col: str = "pred_cdm",
    pred_model_col: str = "pred",
    pred_naive_col: str | None = "pred_naive",
    as_of_col: str = "as_of",
    sku_col: str = "sku_id",
    include_naive: bool = True,
) -> dict[str, Any]:
    """Ritorna scoreboard con headline CDM top-80 + catalog + naive."""
    work = df.copy()
    if as_of_col in work.columns:
        work[as_of_col] = pd.to_datetime(work[as_of_col]).dt.date
        fair_as = set(filter_fair_as_ofs(work[as_of_col].unique()))
        work = work[work[as_of_col].isin(fair_as)]

    def _block(sub: pd.DataFrame, label: str) -> dict[str, Any]:
        y = sub[y_col].astype(float).to_numpy()
        block: dict[str, Any] = {
            "label": label,
            "n": int(len(sub)),
            "sum_y": round(float(y.sum()), 1),
        }
        if pred_model_col in sub.columns:
            p = sub[pred_model_col].astype(float).to_numpy()
            block["model"] = {
                "mae": round(_mae(y, p), 4),
                "wmape": round(_wmape(y, p), 2),
            }
        if pred_cdm_col in sub.columns:
            p = sub[pred_cdm_col].astype(float).to_numpy()
            block["cdm"] = {
                "mae": round(_mae(y, p), 4),
                "wmape": round(_wmape(y, p), 2),
            }
        if include_naive and pred_naive_col and pred_naive_col in sub.columns:
            p = sub[pred_naive_col].astype(float).to_numpy()
            block["naive"] = {
                "mae": round(_mae(y, p), 4),
                "wmape": round(_wmape(y, p), 2),
            }
        return block

    top = cdm_top80_slice(work, y_col=y_col, sku_col=sku_col)
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "policy": {
            "jul_aug": "NON AFFIDABILE — esclusi dal fair compare",
            "headline": "CDM top-80",
            "secondary": "catalog (tutti gli SKU in scope)",
        },
        "headline_cdm_top80": _block(top, "CDM top-80"),
        "catalog": _block(work, "catalog"),
        "meta": {
            "n_sku_top80": top.attrs.get("n_sku_top80"),
            "top80_volume_share": round(float(top.attrs.get("volume_share") or 0), 4),
            "excluded_non_affidabile_rows": int(
                df[as_of_col].map(is_non_affidabile_as_of).sum()
            )
            if as_of_col in df.columns
            else 0,
        },
    }
    return payload


def write_scoreboard(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
