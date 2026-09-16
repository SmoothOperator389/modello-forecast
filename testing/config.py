"""Configurazione ambiente TESTING (solo dati fake)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "output"


@dataclass
class FakeDataConfig:
    start: date = date(2024, 1, 1)
    end: date = date(2026, 8, 31)
    countries: tuple[str, ...] = ("IT", "FR", "DE", "AT", "ES", "UK")
    # Rete Subdued ~150 negozi × 500+ SKU
    pdvs_per_country: int = 25
    n_skus: int = 500
    seed: int = 42


# Riferimento operativo rete (report / allineamento CDM)
PRODUCTION_N_PDV = 150
PRODUCTION_N_SKU = 500


@dataclass
class TrainConfig:
    horizon_sku_days: int = 15
    run_rate_window: int = 14
    epochs: int = 80
    batch_size: int = 1024
    lr: float = 3e-3
    seed: int = 42
    # Densità campioni in training (più as_of = più dati). Non è la cadenza CDM.
    as_of_step_days: int = 7
    # Ciclo operativo: una previsione ogni 15 giorni da portare al CDM.
    forecast_cycle_days: int = 15
    train_as_of_start: date = date(2025, 1, 15)
    train_as_of_end: date = date(2026, 8, 10)
    # Finestra mobile sul training (giorni di as_of).
    # 365 = un anno civile: inverno + estate precedente per imparare il meteo.
    # 0 = tutto lo storico (fallback se pochi negozi).
    train_window_days: int = 365
    # Validation: 4 chiusure = compromesso (2 rumorose, 8 tolgono troppo train).
    n_val_as_of: int = 4
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 0.01


def as_of_range(start: date, end: date, step_days: int) -> list[date]:
    out: list[date] = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=step_days)
    return out
