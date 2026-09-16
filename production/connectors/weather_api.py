"""Meteo da Open-Meteo (storico + orario) — nessuna API key.

Documentazione: https://open-meteo.com/
- Giornaliero: temperatura media/min/max, pioggia
- Orario: temperatura e precipitazione al momento della vendita
  (quando Valido espone data+ora scontrino)
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from production.config import MODEL_DIR, WeatherConfig

CACHE_DIR = MODEL_DIR / "weather_cache"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Coordinate note (testing + rete EU) — evita una geocode per ogni chiamata
CITY_COORDS: dict[str, tuple[float, float, str]] = {
    "Milan": (45.4642, 9.1900, "Europe/Rome"),
    "Rome": (41.9028, 12.4964, "Europe/Rome"),
    "Naples": (40.8518, 14.2681, "Europe/Rome"),
    "Paris": (48.8566, 2.3522, "Europe/Paris"),
    "Lyon": (45.7640, 4.8357, "Europe/Paris"),
    "Marseille": (43.2965, 5.3698, "Europe/Paris"),
    "Berlin": (52.5200, 13.4050, "Europe/Berlin"),
    "Munich": (48.1351, 11.5820, "Europe/Berlin"),
    "Hamburg": (53.5511, 9.9937, "Europe/Berlin"),
    "Vienna": (48.2082, 16.3738, "Europe/Vienna"),
    "Salzburg": (47.8095, 13.0550, "Europe/Vienna"),
    "Graz": (47.0707, 15.4395, "Europe/Vienna"),
    "Madrid": (40.4168, -3.7038, "Europe/Madrid"),
    "Barcelona": (41.3874, 2.1686, "Europe/Madrid"),
    "Valencia": (39.4699, -0.3763, "Europe/Madrid"),
    "London": (51.5074, -0.1278, "Europe/London"),
    "Manchester": (53.4808, -2.2426, "Europe/London"),
    "Birmingham": (52.4862, -1.8904, "Europe/London"),
}

COUNTRY_TZ = {
    "IT": "Europe/Rome",
    "FR": "Europe/Paris",
    "DE": "Europe/Berlin",
    "AT": "Europe/Vienna",
    "ES": "Europe/Madrid",
    "UK": "Europe/London",
}

# Orario tipico negozio abbigliamento (locale) se manca l'ora scontrino
STORE_HOUR_START = 10
STORE_HOUR_END = 20  # escluso


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return CACHE_DIR / f"{safe}.json"


def _http_get_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "modello-forecast-cdm/1.0 (open-meteo)"},
    )
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"Open-Meteo non raggiungibile: {url}\n{last_err}") from last_err


def geocode_city(city: str, country: str | None = None) -> tuple[float, float, str]:
    """Ritorna (lat, lon, timezone). Cache su disco."""
    if city in CITY_COORDS:
        return CITY_COORDS[city]
    cache = _cache_path(f"geo_{city}_{country or ''}")
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        return float(d["lat"]), float(d["lon"]), str(d["tz"])

    params = {"name": city, "count": 1, "language": "it", "format": "json"}
    if country:
        params["country"] = country
    url = GEOCODE_URL + "?" + urllib.parse.urlencode(params)
    payload = _http_get_json(url)
    results = payload.get("results") or []
    if not results:
        tz = COUNTRY_TZ.get(country or "", "Europe/Rome")
        raise ValueError(f"Città non geocodificata: {city} ({country})")
    r0 = results[0]
    lat, lon = float(r0["latitude"]), float(r0["longitude"])
    tz = str(r0.get("timezone") or COUNTRY_TZ.get(country or "", "Europe/Rome"))
    cache.write_text(json.dumps({"lat": lat, "lon": lon, "tz": tz}), encoding="utf-8")
    return lat, lon, tz


def _open_meteo_payload(
    lat: float,
    lon: float,
    start: date,
    end: date,
    tz: str,
    *,
    daily: str | None = None,
    hourly: str | None = None,
) -> dict:
    """Archive per il passato, forecast per oggi/futuro — unisce se la finestra attraversa oggi."""
    yesterday = date.today() - timedelta(days=1)
    chunks: list[dict] = []

    def _call(base: str, a: date, b: date) -> dict:
        params: dict[str, str] = {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "start_date": a.isoformat(),
            "end_date": b.isoformat(),
            "timezone": tz,
        }
        if daily:
            params["daily"] = daily
        if hourly:
            params["hourly"] = hourly
        url = base + "?" + urllib.parse.urlencode(params)
        return _http_get_json(url)

    if start <= yesterday:
        chunks.append(_call(ARCHIVE_URL, start, min(end, yesterday)))
    if end >= date.today():
        chunks.append(_call(FORECAST_URL, max(start, date.today()), end))
    if not chunks:
        raise RuntimeError("Finestra meteo vuota")
    if len(chunks) == 1:
        return chunks[0]
    out = chunks[0]
    for extra in chunks[1:]:
        for block in ("daily", "hourly"):
            if block not in extra:
                continue
            if block not in out:
                out[block] = extra[block]
                continue
            for k, vals in extra[block].items():
                out[block].setdefault(k, [])
                out[block][k] = list(out[block][k]) + list(vals)
    return out


def fetch_daily_weather(
    city: str,
    start: date,
    end: date,
    country: str | None = None,
) -> pd.DataFrame:
    """Una riga per giorno: temp media/min/max, pioggia."""
    lat, lon, tz = geocode_city(city, country)
    cache = _cache_path(f"daily_{city}_{start.isoformat()}_{end.isoformat()}")
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
    else:
        raw = _open_meteo_payload(
            lat,
            lon,
            start,
            end,
            tz,
            daily="temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum",
        )
        cache.write_text(json.dumps(raw), encoding="utf-8")

    daily = raw.get("daily") or {}
    times = daily.get("time") or []
    mean_t = daily.get("temperature_2m_mean") or [None] * len(times)
    max_t = daily.get("temperature_2m_max") or [None] * len(times)
    min_t = daily.get("temperature_2m_min") or [None] * len(times)
    rain = daily.get("precipitation_sum") or daily.get("rain_sum") or [0] * len(times)
    rows = []
    for i, t in enumerate(times):
        mm = float(rain[i] or 0.0)
        temp = mean_t[i]
        if temp is None:
            continue
        rows.append(
            {
                "data": date.fromisoformat(t[:10]),
                "city": city,
                "country": country or "",
                "temp_media": float(temp),
                "temp_max": float(max_t[i] if max_t[i] is not None else temp),
                "temp_min": float(min_t[i] if min_t[i] is not None else temp),
                "pioggia_mm": mm,
                "is_pioggia": int(mm >= 1.0),
            }
        )
    return pd.DataFrame(rows)


def fetch_hourly_weather(
    city: str,
    start: date,
    end: date,
    country: str | None = None,
) -> pd.DataFrame:
    """Una riga per ora locale: temperatura e pioggia al momento della vendita."""
    lat, lon, tz = geocode_city(city, country)
    cache = _cache_path(f"hourly_{city}_{start.isoformat()}_{end.isoformat()}")
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
    else:
        raw = _open_meteo_payload(
            lat,
            lon,
            start,
            end,
            tz,
            hourly="temperature_2m,precipitation",
        )
        cache.write_text(json.dumps(raw), encoding="utf-8")

    hourly = raw.get("hourly") or {}
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []
    precs = hourly.get("precipitation") or []
    rows = []
    for t, temp, prec in zip(times, temps, precs):
        if temp is None:
            continue
        dt = datetime.fromisoformat(t)
        rows.append(
            {
                "data_ora": dt,
                "data": dt.date(),
                "ora": dt.hour,
                "city": city,
                "temp_ora": float(temp),
                "pioggia_mm_ora": float(prec or 0.0),
            }
        )
    return pd.DataFrame(rows)


def weather_at_sale(
    city: str,
    when: datetime | date,
    country: str | None = None,
    hour: int | None = None,
) -> dict:
    """Meteo nel momento della vendita.

    - Se ``when`` è datetime (o ``hour`` è valorizzato): temperatura di quell'ora.
    - Se c'è solo la data: media nelle ore di apertura negozio (10–20).
    """
    if isinstance(when, datetime):
        d = when.date()
        h = when.hour
    else:
        d = when
        h = hour
    hw = fetch_hourly_weather(city, d, d, country)
    if hw.empty:
        daily = fetch_daily_weather(city, d, d, country)
        if daily.empty:
            raise ValueError(f"Nessun meteo per {city} il {d}")
        r = daily.iloc[0]
        return {
            "data": d,
            "ora": h,
            "city": city,
            "temp_media": float(r["temp_media"]),
            "temp_ora": float(r["temp_media"]),
            "pioggia_mm": float(r["pioggia_mm"]),
            "is_pioggia": int(r["is_pioggia"]),
            "fonte": "open-meteo-daily",
        }
    if h is not None:
        hit = hw[hw["ora"] == int(h)]
        if hit.empty:
            hit = hw
        row = hit.iloc[0]
        return {
            "data": d,
            "ora": int(h),
            "city": city,
            "temp_media": float(hw["temp_ora"].mean()),
            "temp_ora": float(row["temp_ora"]),
            "pioggia_mm": float(hw["pioggia_mm_ora"].sum()),
            "is_pioggia": int(float(row["pioggia_mm_ora"]) >= 0.2),
            "fonte": "open-meteo-hourly",
        }
    shop = hw[(hw["ora"] >= STORE_HOUR_START) & (hw["ora"] < STORE_HOUR_END)]
    use = shop if not shop.empty else hw
    return {
        "data": d,
        "ora": None,
        "city": city,
        "temp_media": float(use["temp_ora"].mean()),
        "temp_ora": float(use["temp_ora"].mean()),
        "pioggia_mm": float(hw["pioggia_mm_ora"].sum()),
        "is_pioggia": int((hw["pioggia_mm_ora"] >= 0.2).any()),
        "fonte": "open-meteo-store-hours",
    }


def fetch_weather_window(
    country: str,
    city: str,
    start: date,
    end: date,
    cfg: WeatherConfig | None = None,
) -> pd.DataFrame:
    """Finestra giornaliera — stesso schema usato dalle feature (temp_media, is_pioggia)."""
    _ = cfg or WeatherConfig()
    df = fetch_daily_weather(city, start, end, country)
    if df.empty:
        raise RuntimeError(f"Open-Meteo: nessuna riga per {city} {start}–{end}")
    return df


def fetch_weather_for_pdv_master(
    as_of: date,
    window_days: int,
    pdv_master: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregato meteo per pdv_id sulla finestra [as_of-window+1, as_of]."""
    from production.connectors.pdv_master import load_pdv_master

    master = pdv_master if pdv_master is not None else load_pdv_master()
    start = as_of - timedelta(days=window_days - 1)
    city_frames: dict[tuple[str, str], pd.DataFrame] = {}
    parts = []
    for _, row in master.iterrows():
        key = (str(row["country"]), str(row["city"]))
        if key not in city_frames:
            city_frames[key] = fetch_weather_window(key[0], key[1], start, as_of)
        w = city_frames[key].copy()
        w["pdv_id"] = row["pdv_id"]
        parts.append(w)
    weather = pd.concat(parts, ignore_index=True)
    agg = weather.groupby("pdv_id", as_index=False).agg(
        temp_media=("temp_media", "mean"),
        giorni_pioggia=("is_pioggia", "sum"),
        pioggia_mm_tot=("pioggia_mm", "sum"),
        temp_max=("temp_max", "max"),
        temp_min=("temp_min", "min"),
    )
    hour_rows = []
    for _, row in master.iterrows():
        key = (str(row["country"]), str(row["city"]))
        h = fetch_hourly_weather(key[1], start, as_of, key[0])
        shop = h[(h["ora"] >= STORE_HOUR_START) & (h["ora"] < STORE_HOUR_END)] if not h.empty else h
        pm = h[(h["ora"] >= 14) & (h["ora"] < 18)] if not h.empty else h
        daily_mean = float(city_frames[key]["temp_media"].mean())
        hour_rows.append(
            {
                "pdv_id": row["pdv_id"],
                "temp_orario_negozio": float(shop["temp_ora"].mean()) if not shop.empty else daily_mean + 1.2,
                "temp_pomeriggio": float(pm["temp_ora"].mean()) if not pm.empty else daily_mean + 3.5,
            }
        )
    return agg.merge(pd.DataFrame(hour_rows), on="pdv_id", how="left")


if __name__ == "__main__":
    today = date.today()
    start = today - timedelta(days=3)
    print("Probe Open-Meteo — Milan, ultimi 3 giorni + ora 15:00")
    daily = fetch_weather_window("IT", "Milan", start, today - timedelta(days=1))
    print(daily.to_string(index=False))
    sample_dt = datetime.now().replace(hour=15, minute=0, second=0, microsecond=0) - timedelta(days=20)
    sample = weather_at_sale("Milan", sample_dt, "IT")
    print("Vendita esempio:", sample)
