"""
Generatore dati fake per ambiente TESTING — fast fashion teenager.

Catalogo da negozio abbigliamento 13–19 anni (Bershka / Subdued / Pull&Bear):
capi estivi, invernali, continuativi e accessori (collane, bracciali, orecchini…).
Vendite con stagionalità e meteo realistici così il modello ha un segnale meteo da imparare.
Grain come Valido: ``fact_tickets.csv`` ha data+ora di ogni pezzo venduto.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from testing.config import DATA_DIR, FakeDataConfig
from testing.features.store_features import daily_weather

CITIES = {
    "IT": ["Milan", "Rome", "Naples"],
    "FR": ["Paris", "Lyon", "Marseille"],
    "DE": ["Berlin", "Munich", "Hamburg"],
    "AT": ["Vienna", "Salzburg", "Graz"],
    "ES": ["Madrid", "Barcelona", "Valencia"],
    "UK": ["London", "Manchester", "Birmingham"],
}

COMMERCIALI = ["Rossi", "Bianchi", "Verdi", "Neri", "Russo"]
CATEGORIE = ["MAG", "ACC", "SHO", "OUT"]

MODELLI = ["SUGAR", "REBEL", "GLOW", "NOVA", "FLASH", "MOON", "CORE", "ICON"]
COLORI_APPAREL = ["NERO", "BIANCO", "ROSA", "LILLA", "BEIGE", "BLU", "ROSSO", "VERDE", "PRINT"]
COLORI_GIOIELLI = ["ARGENTO", "ORO", "NERO", "ROSA", "PERLA", "COLORATO"]
COLORI_SHOES = ["NERO", "BIANCO", "ROSA", "BEIGE", "ARGENTO"]

# t_ref, k_hot (temp > ref), k_cold (temp < ref), rain_mult
# k_hot > 0 → il caldo alza le vendite; k_cold > 0 → il freddo alza le vendite
WEATHER_COEF: dict[str, tuple[float, float, float, float]] = {
    "heat_up_strong": (24.0, 0.090, -0.070, 0.68),  # costumi, occhiali
    "heat_up": (20.0, 0.055, -0.045, 0.78),  # crop, shorts, sandali
    "cold_up_strong": (10.0, -0.070, 0.085, 1.22),  # piumini
    "cold_up": (12.0, -0.050, 0.060, 1.12),  # maglioni, stivali, sciarpe
    "cold_up_mild": (14.0, -0.028, 0.040, 1.06),  # felpe, bomber
    "mild": (18.0, 0.008, 0.008, 0.92),  # jeans, sneaker
    "impulse": (18.0, 0.012, -0.008, 0.70),  # gioielli: pioggia taglia il traffico
}

SEASONAL_BY_STAGIONE = {
    "ESTATE": [0.32, 0.38, 0.55, 0.90, 1.20, 1.50, 1.60, 1.42, 1.05, 0.65, 0.40, 0.30],
    "INVERNO": [1.50, 1.38, 1.08, 0.65, 0.40, 0.28, 0.22, 0.28, 0.75, 1.20, 1.45, 1.60],
    "CONTINUATIVO": [0.88, 0.88, 0.96, 1.05, 1.10, 1.12, 1.08, 0.95, 1.05, 1.08, 1.15, 1.28],
}

# (nome, classe, sclasse, ssclasse, cat, stagione, weather_profile, tessuti, pmin, pmax, weight)
ARCHETYPES: list[tuple] = [
    # --- Abbigliamento estate ---
    ("Crop top", "ABBIGLIAMENTO", "TOP", "CROP", "MAG", "ESTATE", "heat_up", ["COTONE", "JERSEY"], 7.99, 16.99, 22),
    ("T-shirt grafica", "ABBIGLIAMENTO", "TOP", "TSHIRT", "MAG", "ESTATE", "heat_up", ["COTONE", "JERSEY"], 9.99, 19.99, 24),
    ("Canotta", "ABBIGLIAMENTO", "TOP", "TANK", "MAG", "ESTATE", "heat_up", ["COTONE", "JERSEY"], 6.99, 14.99, 12),
    ("Camicia crop", "ABBIGLIAMENTO", "TOP", "SHIRT", "MAG", "ESTATE", "heat_up", ["COTONE", "LINO"], 14.99, 24.99, 8),
    ("Shorts denim", "ABBIGLIAMENTO", "BOTTOM", "SHORT", "MAG", "ESTATE", "heat_up", ["DENIM", "COTONE"], 16.99, 29.99, 14),
    ("Gonna mini", "ABBIGLIAMENTO", "BOTTOM", "SKIRT", "MAG", "ESTATE", "heat_up", ["COTONE", "DENIM"], 12.99, 24.99, 12),
    ("Vestito estate", "ABBIGLIAMENTO", "DRESS", "MINI", "MAG", "ESTATE", "heat_up", ["COTONE", "JERSEY", "LINO"], 14.99, 29.99, 12),
    ("Costume", "ABBIGLIAMENTO", "SWIM", "BIKINI", "MAG", "ESTATE", "heat_up_strong", ["LYCRA"], 12.99, 24.99, 8),
    # --- Abbigliamento inverno ---
    ("Felpa hoodie", "ABBIGLIAMENTO", "TOP", "HOODIE", "MAG", "INVERNO", "cold_up_mild", ["FELPA", "COTONE"], 19.99, 34.99, 16),
    ("Maglione knit", "ABBIGLIAMENTO", "TOP", "KNIT", "MAG", "INVERNO", "cold_up", ["MAGLIA", "PILE"], 16.99, 32.99, 14),
    ("Cardigan", "ABBIGLIAMENTO", "TOP", "CARDIGAN", "MAG", "INVERNO", "cold_up", ["MAGLIA"], 18.99, 34.99, 8),
    ("Leggings invernali", "ABBIGLIAMENTO", "BOTTOM", "LEGGING", "MAG", "INVERNO", "cold_up_mild", ["JERSEY"], 9.99, 19.99, 8),
    # --- Continuativo ---
    ("Jeans", "ABBIGLIAMENTO", "BOTTOM", "JEANS", "MAG", "CONTINUATIVO", "mild", ["DENIM"], 22.99, 39.99, 20),
    ("Cargo", "ABBIGLIAMENTO", "BOTTOM", "PANT", "MAG", "CONTINUATIVO", "mild", ["COTONE"], 19.99, 34.99, 10),
    ("Body / top basic", "ABBIGLIAMENTO", "TOP", "BODY", "MAG", "CONTINUATIVO", "mild", ["JERSEY", "COTONE"], 7.99, 16.99, 10),
    # --- Outerwear ---
    ("Piumino", "ABBIGLIAMENTO", "OUTER", "PUFFER", "OUT", "INVERNO", "cold_up_strong", ["NYLON"], 39.99, 79.99, 12),
    ("Cappotto", "ABBIGLIAMENTO", "OUTER", "COAT", "OUT", "INVERNO", "cold_up", ["LANA", "MISTO"], 49.99, 89.99, 6),
    ("Bomber", "ABBIGLIAMENTO", "OUTER", "BOMBER", "OUT", "INVERNO", "cold_up_mild", ["NYLON", "ECOPELLE"], 29.99, 59.99, 10),
    ("Parka", "ABBIGLIAMENTO", "OUTER", "PARKA", "OUT", "INVERNO", "cold_up", ["NYLON"], 39.99, 69.99, 6),
    # --- Calzature ---
    ("Sneaker", "CALZATURE", "SNEAKER", "LOW", "SHO", "CONTINUATIVO", "mild", ["SINTETICO", "TELA"], 24.99, 49.99, 16),
    ("Sandali", "CALZATURE", "SANDAL", "FLAT", "SHO", "ESTATE", "heat_up", ["SINTETICO"], 12.99, 29.99, 12),
    ("Stivaletti", "CALZATURE", "BOOT", "ANKLE", "SHO", "INVERNO", "cold_up", ["ECOPELLE"], 29.99, 54.99, 10),
    ("Ciabatte", "CALZATURE", "SLIDE", "FLAT", "SHO", "ESTATE", "heat_up", ["SINTETICO"], 9.99, 19.99, 6),
    # --- Accessori gioielli / small ---
    ("Collana", "ACCESSORI", "GIOIELLI", "COLLANA", "ACC", "CONTINUATIVO", "impulse", ["METALLO"], 4.99, 14.99, 16),
    ("Bracciale", "ACCESSORI", "GIOIELLI", "BRACCIALE", "ACC", "CONTINUATIVO", "impulse", ["METALLO"], 3.99, 12.99, 14),
    ("Orecchini", "ACCESSORI", "GIOIELLI", "ORECCHINI", "ACC", "CONTINUATIVO", "impulse", ["METALLO"], 3.99, 11.99, 16),
    ("Anello", "ACCESSORI", "GIOIELLI", "ANELLO", "ACC", "CONTINUATIVO", "impulse", ["METALLO"], 2.99, 9.99, 10),
    ("Fermacapelli", "ACCESSORI", "HAIR", "CLIP", "ACC", "CONTINUATIVO", "impulse", ["PLASTICA", "METALLO"], 2.99, 8.99, 10),
    ("Borsa", "ACCESSORI", "BAG", "SHOULDER", "ACC", "CONTINUATIVO", "mild", ["ECOPELLE", "SINTETICO"], 14.99, 34.99, 10),
    ("Cintura", "ACCESSORI", "BELT", "CLASSIC", "ACC", "CONTINUATIVO", "mild", ["ECOPELLE"], 9.99, 19.99, 6),
    ("Cappellino", "ACCESSORI", "HAT", "CAP", "ACC", "ESTATE", "heat_up", ["COTONE"], 9.99, 16.99, 6),
    ("Berretto", "ACCESSORI", "HAT", "BEANIE", "ACC", "INVERNO", "cold_up", ["MAGLIA"], 7.99, 14.99, 6),
    ("Sciarpa", "ACCESSORI", "SCARF", "KNIT", "ACC", "INVERNO", "cold_up", ["MAGLIA", "PILE"], 9.99, 19.99, 6),
    ("Occhiali da sole", "ACCESSORI", "SUNGLASSES", "AVIATOR", "ACC", "ESTATE", "heat_up_strong", ["PLASTICA"], 7.99, 16.99, 8),
    ("Calze pack", "ACCESSORI", "SOCKS", "CREW", "ACC", "CONTINUATIVO", "mild", ["COTONE"], 4.99, 9.99, 6),
    ("Phone charm", "ACCESSORI", "GIOIELLI", "CHARM", "ACC", "CONTINUATIVO", "impulse", ["PLASTICA", "METALLO"], 3.99, 9.99, 6),
]


def _rng(cfg: FakeDataConfig) -> np.random.Generator:
    return np.random.default_rng(cfg.seed)


def generate_pdv_master(cfg: FakeDataConfig) -> pd.DataFrame:
    rng = _rng(cfg)
    rows = []
    idx = 0
    for country in cfg.countries:
        cities = CITIES[country]
        for j in range(cfg.pdvs_per_country):
            idx += 1
            mq_espo = int(rng.integers(70, 150))
            mq_stock = int(rng.integers(20, 55))
            is_new = rng.random() < 0.2
            if is_new:
                opened = cfg.end - timedelta(days=int(rng.integers(10, 90)))
            else:
                opened = cfg.start + timedelta(days=int(rng.integers(0, 400)))
            rows.append(
                {
                    "pdv_id": f"{country}_{idx:03d}",
                    "country": country,
                    "city": cities[j % len(cities)],
                    "commerciale_riferimento": COMMERCIALI[rng.integers(0, len(COMMERCIALI))],
                    "mq_espositivi": mq_espo,
                    "mq_stock": mq_stock,
                    "mq_totali": mq_espo + mq_stock,
                    "numero_fatturazioni": int(rng.integers(40, 450)),
                    "transit_time_days": int(rng.integers(3, 9)),
                    "data_apertura": opened,
                    "pdv_strategico": int(rng.random() < 0.15),
                }
            )
    didactic = pd.DataFrame(
        [
            {
                "pdv_id": "MIL_01", "country": "IT", "city": "Milan",
                "commerciale_riferimento": "Rossi",
                "mq_espositivi": 120, "mq_stock": 40, "mq_totali": 160,
                "numero_fatturazioni": 320, "transit_time_days": 5,
                "data_apertura": date(2024, 3, 1), "pdv_strategico": 1,
            },
            {
                "pdv_id": "ROM_NEW", "country": "IT", "city": "Rome",
                "commerciale_riferimento": "Bianchi",
                "mq_espositivi": 90, "mq_stock": 30, "mq_totali": 120,
                "numero_fatturazioni": 40, "transit_time_days": 5,
                "data_apertura": date(2026, 7, 20), "pdv_strategico": 0,
            },
        ]
    )
    base = pd.DataFrame(rows)
    base = base[~base["pdv_id"].isin(didactic["pdv_id"])]
    return pd.concat([didactic, base], ignore_index=True)


def _colors_for(sclasse: str) -> list[str]:
    if sclasse == "GIOIELLI":
        return COLORI_GIOIELLI
    if sclasse in {"SNEAKER", "SANDAL", "BOOT", "SLIDE"}:
        return COLORI_SHOES
    return COLORI_APPAREL


def _campagna_for(stagione: str, launch: date, rng: np.random.Generator) -> str:
    yy = launch.year % 100
    if stagione == "ESTATE":
        return f"SS{yy:02d}"
    if stagione == "INVERNO":
        return f"FW{yy:02d}"
    return rng.choice(["CORE", f"SS{yy:02d}", f"FW{yy:02d}"])


def _tipologia_for(stagione: str, rng: np.random.Generator) -> str:
    if stagione == "CONTINUATIVO":
        return rng.choice(["continuativo", "carryover", "continuativo"])
    return rng.choice(["stagionale", "stagionale", "limited"])


def _row_from_archetype(
    sku_id: str,
    arch: tuple,
    launch: date,
    strength: float,
    rng: np.random.Generator,
) -> dict:
    nome, classe, sclasse, ssclasse, cat, stagione, wprof, tessuti, pmin, pmax, _w = arch
    modello = rng.choice(MODELLI)
    colore = rng.choice(_colors_for(sclasse))
    return {
        "sku_id": sku_id,
        "nome": f"{nome} {modello}",
        "data_lancio_sku": launch,
        "strength": strength,
        "prezzo_base": round(float(rng.uniform(pmin, pmax)), 2),
        "modello": modello,
        "tessuto": rng.choice(tessuti),
        "colore": colore,
        "campagna": _campagna_for(stagione, launch, rng),
        "classe": classe,
        "sclasse": sclasse,
        "ssclasse": ssclasse,
        "tipologia": _tipologia_for(stagione, rng),
        "categoria_id": cat,
        "stagione": stagione,
        "weather_profile": wprof,
    }


def generate_products(cfg: FakeDataConfig) -> pd.DataFrame:
    rng = _rng(cfg)
    weights = np.array([a[-1] for a in ARCHETYPES], dtype=float)
    weights /= weights.sum()
    span = (cfg.end - cfg.start).days - 60
    rows = []
    for i in range(cfg.n_skus):
        arch = ARCHETYPES[int(rng.choice(len(ARCHETYPES), p=weights))]
        launch = cfg.start + timedelta(days=int(rng.integers(0, max(span, 1))))
        strength = float(rng.uniform(0.45, 2.1))
        rows.append(_row_from_archetype(f"SKU_{i:03d}", arch, launch, strength, rng))

    extra = [
        {
            "sku_id": "SKU_A",
            "nome": "Crop top SUGAR",
            "data_lancio_sku": date(2026, 2, 1),
            "strength": 2.15,
            "prezzo_base": 12.99,
            "modello": "SUGAR",
            "tessuto": "JERSEY",
            "colore": "ROSA",
            "campagna": "SS26",
            "classe": "ABBIGLIAMENTO",
            "sclasse": "TOP",
            "ssclasse": "CROP",
            "tipologia": "stagionale",
            "categoria_id": "MAG",
            "stagione": "ESTATE",
            "weather_profile": "heat_up",
        },
        {
            "sku_id": "SKU_Z",
            "nome": "Piumino REBEL",
            "data_lancio_sku": date(2026, 8, 8),
            "strength": 1.7,
            "prezzo_base": 59.99,
            "modello": "REBEL",
            "tessuto": "NYLON",
            "colore": "NERO",
            "campagna": "FW26",
            "classe": "ABBIGLIAMENTO",
            "sclasse": "OUTER",
            "ssclasse": "PUFFER",
            "tipologia": "stagionale",
            "categoria_id": "OUT",
            "stagione": "INVERNO",
            "weather_profile": "cold_up_strong",
        },
        {
            "sku_id": "SKU_W",
            "nome": "Piumino ICON",
            "data_lancio_sku": date(2025, 10, 1),
            "strength": 1.85,
            "prezzo_base": 64.99,
            "modello": "ICON",
            "tessuto": "NYLON",
            "colore": "NERO",
            "campagna": "FW25",
            "classe": "ABBIGLIAMENTO",
            "sclasse": "OUTER",
            "ssclasse": "PUFFER",
            "tipologia": "stagionale",
            "categoria_id": "OUT",
            "stagione": "INVERNO",
            "weather_profile": "cold_up_strong",
        },
        {
            "sku_id": "SKU_J",
            "nome": "Collana GLOW",
            "data_lancio_sku": date(2025, 3, 15),
            "strength": 1.9,
            "prezzo_base": 7.99,
            "modello": "GLOW",
            "tessuto": "METALLO",
            "colore": "ARGENTO",
            "campagna": "CORE",
            "classe": "ACCESSORI",
            "sclasse": "GIOIELLI",
            "ssclasse": "COLLANA",
            "tipologia": "continuativo",
            "categoria_id": "ACC",
            "stagione": "CONTINUATIVO",
            "weather_profile": "impulse",
        },
    ]
    base = pd.DataFrame(rows)
    extra_df = pd.DataFrame(extra)
    base = base[~base["sku_id"].isin(extra_df["sku_id"])]
    return pd.concat([extra_df, base], ignore_index=True)


# Orario negozio 10–20, come scontrino Valido (un pezzo = una riga con data+ora).
_STORE_HOURS = np.arange(10, 20)
_HOUR_PROFILES = np.array(
    [
        [2.0, 4.0, 8.0, 7.0, 5.0, 6.0, 9.0, 10.0, 8.0, 5.0],  # default: pranzo + dopo scuola
        [1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 12.0, 14.0, 10.0, 6.0],  # gioielli: pomeriggio/sera
        [3.0, 6.0, 10.0, 9.0, 6.0, 5.0, 6.0, 5.0, 4.0, 2.0],  # inverno: mattina/pranzo
        [1.0, 2.0, 4.0, 5.0, 8.0, 10.0, 11.0, 10.0, 8.0, 4.0],  # estate: pomeriggio
    ],
    dtype=float,
)
_HOUR_PROFILES = _HOUR_PROFILES / _HOUR_PROFILES.sum(axis=1, keepdims=True)
_WEEKEND_HOUR_BOOST = np.array([1.15, 1.25, 1.20, 1.10, 1.00, 1.05, 1.15, 1.25, 1.30, 1.15])
_SS_JEWEL = frozenset({"COLLANA", "BRACCIALE", "ORECCHINI", "ANELLO", "CHARM", "CLIP"})
_SS_WINTER = frozenset({"PUFFER", "COAT", "PARKA", "BOMBER", "KNIT", "CARDIGAN", "BEANIE", "SCARF", "ANKLE"})
_SS_SUMMER = frozenset({"CROP", "TSHIRT", "TANK", "SHORT", "SKIRT", "MINI", "BIKINI", "FLAT", "AVIATOR"})
_TICKET_COLS = [
    "data_ora",
    "data",
    "ora",
    "country",
    "pdv_id",
    "sku_id",
    "categoria_id",
    "vendite_qty",
    "giacenza_qty",
    "prezzo_listino",
    "prezzo_pagato",
    "sconto_pct",
    "is_in_sconto",
]


def _hour_profile_id(ssclasse: np.ndarray) -> np.ndarray:
    s = np.asarray(ssclasse, dtype=object).astype(str)
    out = np.zeros(len(s), dtype=np.int8)
    out[np.isin(s, list(_SS_JEWEL))] = 1
    out[np.isin(s, list(_SS_WINTER))] = 2
    out[np.isin(s, list(_SS_SUMMER))] = 3
    return out


def _sample_sale_clock(
    rng: np.random.Generator,
    n: int,
    profile_id: np.ndarray,
    weekend: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ora/minuto/secondo in fascia 10:00–19:59, profilo per tipo SKU."""
    hours = np.empty(n, dtype=np.int16)
    for pid in range(_HOUR_PROFILES.shape[0]):
        mask = profile_id == pid
        k = int(mask.sum())
        if k == 0:
            continue
        w = _HOUR_PROFILES[pid].copy()
        if weekend:
            w = w * _WEEKEND_HOUR_BOOST
            w = w / w.sum()
        hours[mask] = rng.choice(_STORE_HOURS, size=k, p=w)
    minutes = rng.integers(0, 60, size=n, dtype=np.int16)
    seconds = rng.integers(0, 60, size=n, dtype=np.int16)
    return hours, minutes, seconds


def _tickets_from_daily_qty(
    rng: np.random.Generator,
    d: date,
    g: pd.DataFrame,
    qty: np.ndarray,
    giac: np.ndarray,
    prezzo_listino: np.ndarray,
    prezzo_pagato: np.ndarray,
    sconto_pct: np.ndarray,
    is_in_sconto: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Una riga per pezzo venduto (come Valido: data+ora del singolo prodotto).

    Ritorna anche ``ora_media`` per PDV×SKU del giorno (NaN se qty=0).
    """
    ora_media = np.full(len(g), np.nan, dtype=float)
    sold = qty > 0
    n_units = int(qty[sold].sum()) if sold.any() else 0
    if n_units == 0:
        return pd.DataFrame(columns=_TICKET_COLS), ora_media

    rep = np.repeat(np.flatnonzero(sold), qty[sold])
    pid = _hour_profile_id(g["ssclasse"].to_numpy()[rep])
    hours, minutes, seconds = _sample_sale_clock(rng, n_units, pid, d.weekday() >= 5)
    ora = hours.astype(float) + minutes.astype(float) / 60.0 + seconds.astype(float) / 3600.0
    sums = np.zeros(len(g), dtype=float)
    np.add.at(sums, rep, ora)
    ora_media[sold] = sums[sold] / qty[sold].astype(float)

    offset_sec = (
        hours.astype(np.int64) * 3600
        + minutes.astype(np.int64) * 60
        + seconds.astype(np.int64)
    )
    data_ora = np.datetime64(d, "s") + offset_sec.astype("timedelta64[s]")
    tickets = pd.DataFrame(
        {
            "data_ora": data_ora,
            "data": d,
            "ora": np.round(ora, 4),
            "country": g["country"].to_numpy()[rep],
            "pdv_id": g["pdv_id"].to_numpy()[rep],
            "sku_id": g["sku_id"].to_numpy()[rep],
            "categoria_id": g["categoria_id"].to_numpy()[rep],
            "vendite_qty": np.ones(n_units, dtype=np.int16),
            "giacenza_qty": giac[rep],
            "prezzo_listino": prezzo_listino[rep],
            "prezzo_pagato": np.round(prezzo_pagato[rep], 2),
            "sconto_pct": np.round(sconto_pct[rep], 1),
            "is_in_sconto": is_in_sconto[rep],
        }
    )
    return tickets, ora_media


def _weather_mult(temps: np.ndarray, rains: np.ndarray, profiles: np.ndarray) -> np.ndarray:
    t_ref = np.empty(len(profiles), dtype=float)
    k_hot = np.empty(len(profiles), dtype=float)
    k_cold = np.empty(len(profiles), dtype=float)
    rain_m = np.empty(len(profiles), dtype=float)
    default = WEATHER_COEF["mild"]
    for i, p in enumerate(profiles):
        coef = WEATHER_COEF.get(str(p), default)
        t_ref[i], k_hot[i], k_cold[i], rain_m[i] = coef
    delta = temps - t_ref
    temp_m = np.where(delta >= 0, 1.0 + k_hot * delta, 1.0 + k_cold * (-delta))
    temp_m = np.clip(temp_m, 0.22, 2.5)
    return temp_m * np.where(rains, rain_m, 1.0)


def generate_daily_fact(
    cfg: FakeDataConfig,
    pdv_master: pd.DataFrame,
    products: pd.DataFrame,
    out_path: Path | None = None,
    tickets_path: Path | None = None,
) -> pd.DataFrame:
    """Genera scontrini Valido-like (data+ora per pezzo) e il fact giornaliero per il rolling."""
    rng = _rng(cfg)
    stores = pdv_master.copy()
    stores["data_apertura"] = pd.to_datetime(stores["data_apertura"]).dt.date
    products = products.copy()
    products["data_lancio_sku"] = pd.to_datetime(products["data_lancio_sku"]).dt.date

    grid = stores.assign(_k=1).merge(products.assign(_k=1), on="_k").drop(columns="_k")

    dates: list[date] = []
    d = cfg.start
    while d <= cfg.end:
        dates.append(d)
        d += timedelta(days=1)

    chunks: list[pd.DataFrame] = []
    first_write = True
    n_rows = 0
    n_tickets = 0
    if tickets_path is not None:
        pd.DataFrame(columns=_TICKET_COLS).to_csv(tickets_path, index=False)

    for i_d, d in enumerate(dates):
        g = grid[(grid["data_apertura"] <= d) & (grid["data_lancio_sku"] <= d)]
        if g.empty:
            continue

        pdv_age = (pd.to_datetime(d) - pd.to_datetime(g["data_apertura"])).dt.days.to_numpy()
        sku_age = (pd.to_datetime(d) - pd.to_datetime(g["data_lancio_sku"])).dt.days.to_numpy()
        traffic = 0.35 + 0.65 * (g["numero_fatturazioni"].to_numpy(dtype=float) / 400.0).clip(max=1.0)
        new_pdv = 0.35 + 0.65 * np.clip(pdv_age / 60.0, 0, 1)
        sku_factor = 0.3 + 0.7 * np.clip(sku_age / 14.0, 0, 1)

        month = d.month
        seasonal = g["stagione"].map(
            {s: SEASONAL_BY_STAGIONE[s][month - 1] for s in SEASONAL_BY_STAGIONE}
        ).to_numpy(dtype=float)
        gift = np.where((month == 12) & (g["sclasse"].to_numpy() == "GIOIELLI"), 1.4, 1.0)

        dow = d.weekday()
        weekend_mult = 1.25 if dow == 5 else 1.35 if dow == 6 else 1.0

        temps_map: dict[str, float] = {}
        rain_map: dict[str, bool] = {}
        for country, cities in CITIES.items():
            for city in cities:
                w = daily_weather(country, city, d)
                temps_map[city] = float(w["temp_media"])
                rain_map[city] = bool(w["is_pioggia"])
        temps = g["city"].map(temps_map).to_numpy(dtype=float)
        rains = g["city"].map(rain_map).to_numpy(dtype=bool)
        weather_m = _weather_mult(temps, rains, g["weather_profile"].to_numpy())

        years_since_start = (d - cfg.start).days / 365.0
        trend = 1.0 + 0.05 * years_since_start

        is_saldi_period = month in (1, 7) and d.day <= 15
        saldi = 1.4 if is_saldi_period else 1.0

        prezzo_listino = g["prezzo_base"].to_numpy(dtype=float)
        promo_prob = 0.35 if is_saldi_period else 0.12
        promo_roll = np.array(
            [hash((pid, sid, d.isoformat())) % 1000 / 1000.0 for pid, sid in zip(g["pdv_id"], g["sku_id"])]
        )
        is_promo = promo_roll < promo_prob
        sconto_pct = np.where(
            is_promo,
            rng.uniform(15, 45, size=len(g)) if is_saldi_period else rng.uniform(10, 35, size=len(g)),
            0.0,
        )
        prezzo_pagato = prezzo_listino * (1.0 - sconto_pct / 100.0)
        is_in_sconto = (sconto_pct >= 5.0).astype(int)
        discount_mult = 1.0 + 0.025 * sconto_pct

        lam = (
            g["strength"].to_numpy(dtype=float)
            * traffic
            * new_pdv
            * sku_factor
            * seasonal
            * gift
            * weekend_mult
            * weather_m
            * trend
            * saldi
            * discount_mult
        )

        qty = rng.poisson(np.maximum(lam, 0.02))
        giac = np.maximum(0, 25 - qty + rng.integers(-4, 5, size=len(qty)))
        tickets, ora_media = _tickets_from_daily_qty(
            rng,
            d,
            g,
            qty,
            giac,
            prezzo_listino,
            prezzo_pagato,
            sconto_pct,
            is_in_sconto,
        )
        chunk = pd.DataFrame(
            {
                "data": d,
                "country": g["country"].values,
                "pdv_id": g["pdv_id"].values,
                "sku_id": g["sku_id"].values,
                "categoria_id": g["categoria_id"].values,
                "vendite_qty": qty,
                "giacenza_qty": giac,
                "ora": np.round(ora_media, 4),
                "prezzo_listino": prezzo_listino,
                "prezzo_pagato": prezzo_pagato,
                "sconto_pct": np.round(sconto_pct, 1),
                "is_in_sconto": is_in_sconto,
                "modello": g["modello"].values,
                "tessuto": g["tessuto"].values,
                "colore": g["colore"].values,
                "campagna": g["campagna"].values,
                "classe": g["classe"].values,
                "sclasse": g["sclasse"].values,
                "ssclasse": g["ssclasse"].values,
                "tipologia": g["tipologia"].values,
                "stagione": g["stagione"].values,
                "nome": g["nome"].values,
                "data_apertura": g["data_apertura"].values,
                "data_lancio_sku": g["data_lancio_sku"].values,
            }
        )

        if tickets_path is not None and not tickets.empty:
            tickets.to_csv(tickets_path, mode="a", header=False, index=False)
            n_tickets += len(tickets)
        if out_path is not None:
            chunk.to_csv(out_path, mode="w" if first_write else "a", header=first_write, index=False)
            first_write = False
            n_rows += len(chunk)
            if (i_d + 1) % 60 == 0 or d == dates[-1]:
                print(
                    f"  fact {d.isoformat()}  giorni {i_d+1}/{len(dates)}  "
                    f"giornaliero {n_rows:,}  scontrini {n_tickets:,}",
                    flush=True,
                )
        else:
            chunks.append(chunk)

    if out_path is not None:
        print(f"  fact_daily: {n_rows:,} righe → {out_path}")
        if tickets_path is not None:
            print(f"  fact_tickets: {n_tickets:,} vendite (1 pezzo = 1 riga data+ora) → {tickets_path}")
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def generate_calendar(cfg: FakeDataConfig) -> pd.DataFrame:
    rows = []
    d = cfg.start
    while d <= cfg.end:
        week = int(d.strftime("%W"))
        angle = 2 * np.pi * week / 52.0
        for country in cfg.countries:
            rows.append(
                {
                    "data": d,
                    "country": country,
                    "mese": d.month,
                    "settimana_anno": week,
                    "sin_anno": float(np.sin(angle)),
                    "cos_anno": float(np.cos(angle)),
                    "is_festivo": int(d.month == 8 and d.day == 15),
                    "is_saldi": int(d.month in (1, 7) and d.day <= 15),
                    "giorni_da_inizio_collezione": (d - date(d.year, 7, 1)).days,
                }
            )
        d += timedelta(days=1)
    return pd.DataFrame(rows)


def write_all(cfg: FakeDataConfig | None = None, out_dir: Path | None = None) -> dict[str, Path]:
    cfg = cfg or FakeDataConfig()
    out_dir = out_dir or DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    pdv = generate_pdv_master(cfg)
    products = generate_products(cfg)
    calendar = generate_calendar(cfg)

    paths = {
        "pdv_master": out_dir / "pdv_master.csv",
        "products": out_dir / "products.csv",
        "fact": out_dir / "fact_daily.csv",
        "tickets": out_dir / "fact_tickets.csv",
        "calendar": out_dir / "calendar.csv",
    }
    pdv.to_csv(paths["pdv_master"], index=False)
    products.to_csv(paths["products"], index=False)
    calendar.to_csv(paths["calendar"], index=False)
    generate_daily_fact(cfg, pdv, products, out_path=paths["fact"], tickets_path=paths["tickets"])

    n_fact = 0
    with paths["fact"].open() as f:
        for n_fact, _ in enumerate(f):
            pass
    n_tickets = 0
    if paths["tickets"].exists():
        with paths["tickets"].open() as f:
            for n_tickets, _ in enumerate(f):
                pass
    n_fact = max(0, n_fact)  # header included in count
    meta = pd.DataFrame(
        [
            {"metric": "n_pdv", "value": len(pdv)},
            {"metric": "n_sku", "value": len(products)},
            {"metric": "n_fact_rows", "value": max(0, n_fact - 1)},
            {"metric": "n_ticket_rows", "value": max(0, n_tickets - 1) if n_tickets else 0},
            {"metric": "grain_vendite", "value": "scontrino_data_ora"},
            {"metric": "date_min", "value": str(cfg.start)},
            {"metric": "date_max", "value": str(cfg.end)},
            {"metric": "catalogo", "value": "fast_fashion_teenager"},
        ]
    )
    meta.to_csv(out_dir / "dataset_meta.csv", index=False)
    return paths


if __name__ == "__main__":
    paths = write_all()
    print("Dataset fake scritto in:", DATA_DIR)
    for k, p in paths.items():
        print(f"  {k}: {p.name}")
