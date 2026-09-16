"""
Connettore Valido (DB vendite / movimenti magazzino).

TODO: sostituire il corpo con query reali quando avete accesso read-only.
Vedi docs/DATA_CONTRACTS.md per schema campi.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from production.config import ValidoConfig

# Export Valido (csv_MIRKONE / DB)
VALIDO_TIPODOC_VENDITA = "100"
VALIDO_TIPODOC_CARICO = "47"
VALIDO_TIPODOC_CARICO_WOMEN = "51"  # Fattura Women's clothing — variante carico
VALIDO_TIPODOC_AZZERAMENTO = "20"

# Movimenti necessari per ricostruire la giacenza (oltre alle vendite 100).
VALIDO_TIPODOC_GIACENZA = frozenset({
    VALIDO_TIPODOC_VENDITA,
    VALIDO_TIPODOC_CARICO,
    VALIDO_TIPODOC_CARICO_WOMEN,
    VALIDO_TIPODOC_AZZERAMENTO,
})

# Export ufficiale: ; senza header — vedi docs/DATA_CONTRACTS.md / data/raw/export24kk.path
VALIDO_EXPORT_COLUMNS = (
    "pdv_id",       # IDAnagrafica
    "country",      # iso3
    "data",
    "sku_id",       # IDArticolo
    "tipodocumento",
    "doc_descrizione",
    "qta",
    "prezzo_pubblico",
    "pagato_effettivo",
    "giacenza",
)


def fetch_sales(
    start: date,
    end: date,
    cfg: ValidoConfig | None = None,
) -> pd.DataFrame:
    """
    Fact vendite giornaliero (solo domanda).

    Filtro: ``tipodocumento == VALIDO_TIPODOC_VENDITA`` (100).
    Per la giacenza usare ``fetch_stock_movements()`` — servono anche 47 e 20.
    """
    raise NotImplementedError(
        "Collegare Valido: implementare fetch_sales() con query SQL reale. "
        "Vedi docs/DATA_CONTRACTS.md"
    )


def fetch_stock_movements(
    start: date,
    end: date,
    cfg: ValidoConfig | None = None,
) -> pd.DataFrame:
    """
    Movimenti per giacenza: vendite (100), carichi (47, 51), azzeramenti (20).

    Colonne attese in uscita:
      data, country, pdv_id, sku_id, tipodocumento, qta_movimento, giacenza_dopo
    """
    raise NotImplementedError(
        "Collegare Valido: implementare fetch_stock_movements() — "
        f"tipodocumento in {sorted(VALIDO_TIPODOC_GIACENZA)}. "
        "Vedi docs/DATA_CONTRACTS.md"
    )


def check_connection(cfg: ValidoConfig | None = None) -> bool:
    """Ping DB — da implementare."""
    return False
