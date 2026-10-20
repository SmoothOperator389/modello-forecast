"""Layer operativo riassortimento — stessa importanza del forecast."""

from __future__ import annotations

import numpy as np


def suggest_replenishment(
    forecast_sku: float,
    giacenza: float,
    transit_days: float,
    horizon_days: int,
    is_immature: bool,
    *,
    forecast_p80: float | None = None,
    service_level: str = "p50",
) -> dict:
    """
    Domanda attesa → qty suggerita → review umana.

    - forecast_sku: previsione centrale (P50 / media / route ops)
    - forecast_p80: quantile alto per buffer servizio (opzionale)
    - service_level: "p50" | "p80" — quale forecast guida il fabbisogno
    - is_immature: SKU nuovo o PDV nuovo → prudenza (no proxy manuali)
    """
    central = float(forecast_sku)
    upper = float(forecast_p80) if forecast_p80 is not None else central
    upper = max(upper, central)
    demand = upper if service_level == "p80" else central

    daily = demand / max(horizon_days, 1)
    need = demand + daily * transit_days
    raw = max(0.0, need - giacenza)
    if is_immature:
        raw *= 0.5
    qty = float(np.rint(raw))
    safety = max(0.0, upper - central)
    return {
        "vendite_previste_15gg": round(central, 1),
        "vendite_p80_15gg": round(upper, 1),
        "safety_buffer": round(safety, 1),
        "service_level": service_level,
        "giacenza": giacenza,
        "qty_suggerita": qty,
        "azione": "monitora" if qty == 0 else "valuta_riassortimento",
    }
