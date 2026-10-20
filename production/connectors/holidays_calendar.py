"""Festività nazionali per paese — libreria offline ``holidays`` (gratis, no API key).

I paesi Valido sono in ISO3 (ITA, ESP, …). Mappiamo a ISO2 supportati da ``holidays``.
Per paesi non coperti: nessuna riga (is_festivo resta 0), oppure override manuale
in ``data/manual/calendar_holidays.csv``.

Alternativa online (non usata di default): Nager.Date
  https://date.nager.at/api/v3/PublicHolidays/{year}/{ISO2}
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import holidays as hol
import pandas as pd

# ISO3 Valido → codice country di ``holidays`` / Open-Meteo
ISO3_TO_ISO2: dict[str, str] = {
    "ITA": "IT",
    "FRA": "FR",
    "DEU": "DE",
    "ESP": "ES",
    "GBR": "GB",
    "AUT": "AT",
    "BEL": "BE",
    "NLD": "NL",
    "CHE": "CH",
    "IRL": "IE",
    "LUX": "LU",
    "SWE": "SE",
    "NOR": "NO",
    "USA": "US",
    "CHN": "CN",
    "KOR": "KR",
    "SGP": "SG",
    "THA": "TH",
    "ARE": "AE",
    "AND": "AD",
    # Libano: non in holidays standard → skip automatico
    "LBN": "",
}

# ISO2 → ISO3 (file manuali calendar_commercial / holidays)
ISO2_TO_ISO3: dict[str, str] = {v: k for k, v in ISO3_TO_ISO2.items() if v}


def to_iso3(country: str) -> str:
    """Normalizza country a ISO3 Valido (accetta IT o ITA)."""
    c = str(country).strip().upper()
    if len(c) == 2:
        return ISO2_TO_ISO3.get(c, c)
    return c

# Alias città IT/FR/DE → nome geocode Open-Meteo più stabile
CITY_ALIASES: dict[str, str] = {
    "milano": "Milan",
    "roma": "Rome",
    "napoli": "Naples",
    "venezia": "Venice",
    "torino": "Turin",
    "firenze": "Florence",
    "genova": "Genoa",
    "padova": "Padua",
    "parigi": "Paris",
    "lione": "Lyon",
    "marsiglia": "Marseille",
    "nizza": "Nice",
    "tolosa": "Toulouse",
    "lille": "Lille",
    "bordeaux": "Bordeaux",
    "strasburgo": "Strasbourg",
    "berlino": "Berlin",
    "berlino 2": "Berlin",
    "berlino kudamm": "Berlin",
    "monaco": "Munich",
    "amburgo": "Hamburg",
    "colonia": "Cologne",
    "francoforte": "Frankfurt",
    "duesseldorf": "Dusseldorf",
    "stoccarda": "Stuttgart",
    "norimberg": "Nuremberg",
    "barcellona": "Barcelona",
    "barcellon": "Barcelona",
    "barcellona 3": "Barcelona",
    "madrid": "Madrid",
    "madrid 3": "Madrid",
    "madrid2": "Madrid",
    "valencia": "Valencia",
    "siviglia": "Seville",
    "zaragozza": "Zaragoza",
    "malaga": "Malaga",
    "bilbao": "Bilbao",
    "la coruna": "A Coruna",
    "palma de maiorca": "Palma",
    "san sebastian": "San Sebastian",
    "londra": "London",
    "londra4": "London",
    "london": "London",
    "birmingham": "Birmingham",
    "dublino": "Dublin",
    "bruxelles": "Brussels",
    "anversa": "Antwerp",
    "bruges": "Bruges",
    "gand": "Ghent",
    "zurigo": "Zurich",
    "basilea": "Basel",
    "losanna": "Lausanne",
    "lugano": "Lugano",
    "vienna": "Vienna",
    "stoccolma": "Stockholm",
    "lussemburgo": "Luxembourg",
    "aix en provence": "Aix-en-Provence",
    "p.to rotondo": "Porto Rotondo",
    "porto rotondo": "Porto Rotondo",
    "amsterdam 2": "Amsterdam",
    "vintage bcn": "Barcelona",
    "vintage roma": "Rome",
    "aia": "The Hague",
    "den haag": "The Hague",
}


def to_iso2(country: str) -> str:
    c = str(country).strip().upper()
    if len(c) == 2:
        return c
    return ISO3_TO_ISO2.get(c, c[:2] if len(c) >= 2 else "")


def normalize_city(city: str) -> str:
    if city is None or (isinstance(city, float) and pd.isna(city)):
        return ""
    raw = str(city).strip()
    if not raw or raw.lower() in {"nan", "none", "e-commerce", "online", "outlet"}:
        return ""
    return CITY_ALIASES.get(raw.lower(), raw)


def build_holidays_frame(
    countries: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Una riga per festivo: country (ISO3 originale), data, nome_festivo."""
    years = list(range(start.year, end.year + 1))
    rows: list[dict] = []
    for country in sorted(set(str(c).strip().upper() for c in countries if str(c).strip())):
        iso2 = to_iso2(country)
        if not iso2:
            continue
        try:
            cal = hol.country_holidays(iso2, years=years)
        except NotImplementedError:
            continue
        for d, name in sorted(cal.items()):
            if start <= d <= end:
                rows.append(
                    {
                        "country": country,
                        "data": d,
                        "nome_festivo": str(name),
                        "iso2": iso2,
                    }
                )
    return pd.DataFrame(rows)


def write_holidays_csv(
    path: Path,
    countries: list[str],
    start: date = date(2024, 1, 1),
    end: date = date(2026, 12, 31),
) -> pd.DataFrame:
    df = build_holidays_frame(countries, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[["country", "data", "nome_festivo"]].copy()
    out.to_csv(path, index=False)
    return out
