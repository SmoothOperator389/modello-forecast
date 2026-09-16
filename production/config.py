"""Configurazione ambiente PRODUCTION (codice vero — connettori da completare)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
MANUAL_DATA_DIR = PROJECT_ROOT / "data" / "manual"
MODEL_DIR = ROOT / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ValidoConfig:
    """Connessione DB Valido — da compilare con IT."""
    host: str = "TODO_VALIDO_HOST"
    database: str = "TODO_VALIDO_DB"
    user: str = "TODO_USER"
    password_env: str = "VALIDO_DB_PASSWORD"  # leggere da env, mai hardcodare


@dataclass
class WeatherConfig:
    """Open-Meteo — nessuna API key. Override URL solo se usi un proxy aziendale."""
    archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    geocoding_url: str = "https://geocoding-api.open-meteo.com/v1/search"
    api_key_env: str = "WEATHER_API_KEY"  # non richiesto da Open-Meteo


@dataclass
class RunConfig:
    as_of: date = date.today()
    horizon_sku_days: int = 15
    run_rate_window: int = 14
    pdv_master_path: Path = MANUAL_DATA_DIR / "pdv_master.csv"
    calendar_commercial_path: Path = MANUAL_DATA_DIR / "calendar_commercial.csv"
    calendar_holidays_path: Path = MANUAL_DATA_DIR / "calendar_holidays.csv"
