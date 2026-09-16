#!/usr/bin/env python3
"""
Pipeline PRODUCTION — scheletro su cui lavorerai con dati reali.

Flusso:
  Valido (DB) + CSV PDV + API meteo + calendario
    → feature table (rolling, solo reali)
    → modello TensorFlow
    → suggerimento riassortimento
    → export per review umana

Oggi i connettori Valido/meteo sollevano NotImplementedError finché non li implementi.
"""

from __future__ import annotations

from production.config import RunConfig
from production.connectors.calendar import load_calendar
from production.connectors.pdv_master import load_pdv_master
from production.connectors.valido import fetch_sales
from production.features.build_features import build_feature_table
from production.operational.replenishment import suggest_replenishment


def run_inference(cfg: RunConfig | None = None) -> None:
    cfg = cfg or RunConfig()
    print("=== PRODUCTION pipeline (scheletro) ===")
    print(f"as_of: {cfg.as_of}")

    # 1) Fonti reali (CSV team — ok)
    pdv = load_pdv_master(cfg=cfg)
    print(f"PDV caricati da CSV: {len(pdv)}")
    calendar = load_calendar(cfg)
    print(f"Calendario: {len(calendar)} righe | country: {sorted(calendar['country'].unique())}")

    # 2) Valido + meteo — ancora da implementare
    raise NotImplementedError(
        "Prossimo passo: implementare fetch_sales() in production/connectors/valido.py. "
        "API meteo Open-Meteo: production.connectors.weather_api (già collegata)."
    )

    # 2) Feature (rolling)
    # feat = build_feature_table(fact, calendar, cfg.as_of, cfg.run_rate_window, pdv)

    # 3) Inferenza modello (dopo training)
    # model = keras.models.load_model(...)

    # 4) Operativo
    # for _, row in feat.iterrows():
    #     immature = row["has_history"] < 0.5 or row["is_new_pdv"] > 0.5
    #     suggest_replenishment(...)


if __name__ == "__main__":
    try:
        run_inference()
    except NotImplementedError as e:
        print(f"\n[OK scheletro] {e}")
        raise SystemExit(0)
