"""Schema feature condiviso — ambiente TESTING."""

from __future__ import annotations

# Colonne prodotto allineate a Valido (referenza = modello + tessuto + colore)
SKU_ATTR_COLS = [
    "modello",
    "tessuto",
    "colore",
    "campagna",
    "classe",
    "sclasse",
    "ssclasse",
    "tipologia",
    # niente `stagione`: in produzione Valido i capi non sono taggati.
    # La rete impara estate/inverno da data + temperatura delle vendite dello SKU.
]

# Embedding categorici (inclusi PDV/SKU e attributi prodotto)
CATEGORICAL_COLS = [
    "country",
    "pdv_id",
    "sku_id",
    "categoria_id",
    "commerciale_riferimento",
    *SKU_ATTR_COLS,
]

# Input speciali già codificati a intero (band / regime)
BAND_INPUT_COLS = {
    "mq_espositivi_band": "mq_espo_b",
    "mq_stock_band": "mq_stock_b",
    "weather_regime": "weather_reg",
}

# Dimensioni embedding default per cardinalità tipica
EMB_DIMS: dict[str, int] = {
    "country": 6,
    "pdv_id": 8,
    "sku_id": 12,
    "categoria_id": 4,
    "commerciale_riferimento": 4,
    "modello": 6,
    "tessuto": 4,
    "colore": 6,
    "campagna": 4,
    "classe": 4,
    "sclasse": 4,
    "ssclasse": 4,
    "tipologia": 3,
    "mq": 4,
    "weather": 4,
}
