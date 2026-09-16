"""Layer operativo riassortimento — stessa importanza del forecast."""

from __future__ import annotations

import numpy as np


def suggest_replenishment(
    forecast_sku: float,
    giacenza: float,
    transit_days: float,
    horizon_days: int,
    is_immature: bool,
) -> dict:
    """
    Domanda attesa → qty suggerita → review umana.
    is_immature: SKU nuovo o PDV nuovo → prudenza (no proxy manuali).
    """
    daily = forecast_sku / max(horizon_days, 1)
    need = forecast_sku + daily * transit_days
    raw = max(0.0, need - giacenza)
    if is_immature:
        raw *= 0.5
    qty = float(np.rint(raw))
    return {
        "vendite_previste_15gg": round(forecast_sku, 1),
        "giacenza": giacenza,
        "qty_suggerita": qty,
        "azione": "monitora" if qty == 0 else "valuta_riassortimento",
    }
