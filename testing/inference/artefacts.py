"""Artefatti di inferenza: encoder, scaler, analoghi, baseline CDM.

Il file .keras da solo non basta: gli embedding sono indici, non stringhe PDV/SKU.
Senza queste mappe una run operativa produrrebbe previsioni sbagliate.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from testing.config import DATA_DIR

ARTEFACTS_PATH = DATA_DIR / "forecast_artefacts.pkl"


def save_forecast_artefacts(
    maps: dict[str, dict[str, int]],
    num_mean: np.ndarray,
    num_std: np.ndarray,
    analogue_lookup: dict,
    cdm_fitted: dict,
    explain_ref: Any,
    path: Path | None = None,
) -> Path:
    out = path or ARTEFACTS_PATH
    payload = {
        "maps": maps,
        "num_mean": np.asarray(num_mean),
        "num_std": np.asarray(num_std),
        "analogue_lookup": analogue_lookup,
        "cdm_fitted": cdm_fitted,
        "explain_ref": explain_ref,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(payload, f, protocol=4)
    return out


def load_forecast_artefacts(path: Path | None = None) -> dict:
    src = path or ARTEFACTS_PATH
    if not src.exists():
        raise FileNotFoundError(
            f"Manca {src}. Esegui prima il training (`python -m testing.run_train`) "
            "oppure una forecast: al primo giro ricostruisce e salva gli artefatti."
        )
    with src.open("rb") as f:
        return pickle.load(f)
