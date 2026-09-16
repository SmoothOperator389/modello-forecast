"""Costruzione dataset rolling per TESTING — versione ottimizzata."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from testing.config import DATA_DIR, TrainConfig
from testing.features.schema import SKU_ATTR_COLS
from testing.features.sku_weather import attach_sku_weather_response, build_sku_sale_weather_cumsum
from testing.features.store_features import attach_store_weather, load_pdv_master, ora_picco_vendita, temp_at_sale_hour

SCONTO_SOGLIA_PCT = 5.0


def _ensure_discount_columns(fact: pd.DataFrame) -> pd.DataFrame:
    """Deriva colonne sconto se mancanti (es. dati Valido: prezzo_pubblico / pagato_effettivo)."""
    out = fact.copy()
    if "prezzo_listino" not in out.columns:
        if "prezzo_pubblico" in out.columns:
            out["prezzo_listino"] = out["prezzo_pubblico"].astype(float)
        else:
            out["prezzo_listino"] = out["prezzo_pagato"].astype(float)
    if "sconto_pct" not in out.columns:
        listino = out["prezzo_listino"].astype(float).clip(lower=0.01)
        pagato = out["prezzo_pagato"].astype(float)
        out["sconto_pct"] = ((1.0 - pagato / listino) * 100.0).clip(0.0, 90.0)
    if "is_in_sconto" not in out.columns:
        out["is_in_sconto"] = (out["sconto_pct"] >= SCONTO_SOGLIA_PCT).astype(int)
    return out


def _aggregate_sku_attrs(hist_14: pd.DataFrame) -> pd.DataFrame:
    keys = ["country", "pdv_id", "sku_id", "categoria_id"]
    agg = {col: (col, "last") for col in SKU_ATTR_COLS if col in hist_14.columns}
    if not agg:
        return pd.DataFrame(columns=keys)
    return hist_14.groupby(keys, as_index=False).agg(**agg)


def _fill_entity_defaults(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "commerciale_riferimento" not in out.columns:
        out["commerciale_riferimento"] = "NA"
    else:
        out["commerciale_riferimento"] = out["commerciale_riferimento"].fillna("NA").astype(str)
    if "transit_time_days" not in out.columns:
        if "transit_time" in out.columns:
            out["transit_time_days"] = out["transit_time"].astype(float)
        else:
            out["transit_time_days"] = 5.0
    else:
        out["transit_time_days"] = out["transit_time_days"].fillna(5.0).astype(float)
    for col in SKU_ATTR_COLS:
        if col not in out.columns:
            out[col] = "NA"
        else:
            out[col] = out[col].fillna("NA").astype(str)
    return out


def _aggregate_discount_features(hist_14: pd.DataFrame) -> pd.DataFrame:
    hist = _ensure_discount_columns(hist_14)
    return (
        hist.groupby(["country", "pdv_id", "sku_id", "categoria_id"], as_index=False)
        .agg(
            prezzo_listino=("prezzo_listino", "last"),
            prezzo_pagato=("prezzo_pagato", "last"),
            sconto_pct=("sconto_pct", "last"),
            sconto_pct_medio_14d=("sconto_pct", "mean"),
            giorni_in_sconto_14d=("is_in_sconto", "mean"),
        )
    )


NUMERIC_COLS = [
    "run_rate_14d",
    "run_rate_7d",
    "run_rate_28d",
    "trend_7v14",
    "trend_14v28",
    "vendite_volatilita_14d",
    "max_giornaliero_14d",
    "giorni_vendita_su_14",
    "has_history",
    "giorni_da_lancio",
    "log_giorni_da_lancio",
    "pdv_age_days",
    "log_pdv_age_days",
    "is_new_pdv",
    "is_new_sku",
    "giacenza",
    "giacenza_su_run_rate",
    "prezzo_listino",
    "prezzo_pagato",
    "rapporto_prezzo_pagato",
    "is_in_sconto",
    "sconto_pct",
    "sconto_pct_medio_14d",
    "giorni_in_sconto_14d",
    "sconto_impatto_stimato",
    "run_rate_x_sconto",
    "mq_espositivi",
    "mq_stock",
    "mq_totali",
    "numero_fatturazioni",
    "log_fatturazioni",
    "transit_time_days",
    "temp_media",
    "giorni_pioggia",
    "temp_orario_negozio",
    "temp_pomeriggio",
    "ora_picco",
    "temp_ora_vendita",
    "sku_temp_vendita",
    "sku_share_hot",
    "sku_share_cold",
    "sku_temp_gap",
    "sku_temp_fit",
    "run_rate_x_sku_temp",
    "pioggia_x_gioielli",
    "mese",
    "sin_anno",
    "cos_anno",
    "sin_mese",
    "cos_mese",
    "day_of_week",
    "is_weekend",
    "is_festivo",
    "is_saldi",
    "giorni_da_inizio_collezione",
    "run_rate_x_mq",
    "run_rate_x_fatt",
]


def load_fact_calendar() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carica il fact di training.

    Fonte Valido-like: ``fact_tickets.csv`` (data+ora per pezzo). Il rolling usa
    ``fact_daily.csv`` (aggregato, con ``ora`` media reale del giorno).
    """
    fact = pd.read_csv(DATA_DIR / "fact_daily.csv", parse_dates=["data", "data_apertura", "data_lancio_sku"])
    calendar = pd.read_csv(DATA_DIR / "calendar.csv", parse_dates=["data"])
    fact["data"] = pd.to_datetime(fact["data"]).dt.date
    fact["data_apertura"] = pd.to_datetime(fact["data_apertura"]).dt.date
    fact["data_lancio_sku"] = pd.to_datetime(fact["data_lancio_sku"]).dt.date
    calendar["data"] = pd.to_datetime(calendar["data"]).dt.date
    if "data_ora" in fact.columns and "ora" not in fact.columns:
        fact["ora"] = pd.to_datetime(fact["data_ora"]).dt.hour + pd.to_datetime(fact["data_ora"]).dt.minute / 60.0
    return fact, calendar


def build_training_frames(
    fact: pd.DataFrame,
    calendar: pd.DataFrame,
    cfg: TrainConfig,
    as_of_list: list[date],
    pdv_master: pd.DataFrame | None = None,
    require_target: bool = True,
) -> pd.DataFrame:
    """Costruisce una riga per PDV×SKU×as_of.

    ``require_target=True`` (training): tiene solo as_of con vendite nei 15 gg successivi.
    ``require_target=False`` (ciclo CDM): usa solo lo storico fino ad as_of; y può mancare.
    """
    master = pdv_master if pdv_master is not None else load_pdv_master()
    # Produzione: i capi non sono taggati ESTATE/INVERNO. Il fact fake può
    # ancora avere la colonna per la dashboard QA — la rete non la vede.
    fact = fact.drop(columns=["stagione"], errors="ignore")
    sku_wx = build_sku_sale_weather_cumsum(fact, master)
    samples = []

    for as_of in as_of_list:
        w7_start = as_of - timedelta(days=6)
        w14_start = as_of - timedelta(days=13)
        w28_start = as_of - timedelta(days=27)
        y_end = as_of + timedelta(days=cfg.horizon_sku_days)

        hist_28 = fact[(fact["data"] >= w28_start) & (fact["data"] <= as_of)]
        hist_14 = hist_28[hist_28["data"] >= w14_start]
        hist_7 = hist_28[hist_28["data"] >= w7_start]
        future = fact[(fact["data"] > as_of) & (fact["data"] <= y_end)]
        if hist_14.empty:
            continue
        if require_target and future.empty:
            continue

        hist_14 = _ensure_discount_columns(hist_14)
        if "data_ora" in hist_14.columns and "ora" not in hist_14.columns:
            hist_14 = hist_14.copy()
            ts = pd.to_datetime(hist_14["data_ora"])
            hist_14["ora"] = ts.dt.hour + ts.dt.minute / 60.0

        agg14 = dict(
            vendite_14=("vendite_qty", "sum"),
            giorni_con_vendita_14=("vendite_qty", lambda s: int((s > 0).sum())),
            std_14=("vendite_qty", "std"),
            max_14=("vendite_qty", "max"),
            giacenza=("giacenza_qty", "last"),
            data_apertura=("data_apertura", "first"),
            data_lancio_sku=("data_lancio_sku", "first"),
        )
        g14 = hist_14.groupby(["country", "pdv_id", "sku_id", "categoria_id"], as_index=False).agg(**agg14)
        g14["std_14"] = g14["std_14"].fillna(0.0)
        if "ora" in hist_14.columns:
            sold_ora = hist_14.loc[hist_14["vendite_qty"] > 0, ["country", "pdv_id", "sku_id", "categoria_id", "ora", "vendite_qty"]].copy()
            if not sold_ora.empty:
                sold_ora["_ora_mass"] = sold_ora["ora"].astype(float) * sold_ora["vendite_qty"].astype(float)
                g_ora = sold_ora.groupby(["country", "pdv_id", "sku_id", "categoria_id"], as_index=False).agg(
                    _ora_mass=("_ora_mass", "sum"),
                    _qty_ora=("vendite_qty", "sum"),
                )
                g_ora["ora_media_14d"] = g_ora["_ora_mass"] / g_ora["_qty_ora"].clip(lower=1)
                g14 = g14.merge(
                    g_ora[["country", "pdv_id", "sku_id", "categoria_id", "ora_media_14d"]],
                    on=["country", "pdv_id", "sku_id", "categoria_id"],
                    how="left",
                )
        g_disc = _aggregate_discount_features(hist_14)
        g_sku = _aggregate_sku_attrs(hist_14)

        sold = hist_14[hist_14["vendite_qty"] > 0]
        g_net = hist_14.groupby(["country", "sku_id"], as_index=False).agg(
            units_14_net=("vendite_qty", "sum"),
        )
        if not sold.empty:
            g_active = sold.groupby(["country", "sku_id"], as_index=False).agg(
                pdv_active_14=("pdv_id", "nunique"),
            )
            g_net = g_net.merge(g_active, on=["country", "sku_id"], how="left")
        else:
            g_net["pdv_active_14"] = 0

        g7 = (
            hist_7.groupby(["country", "pdv_id", "sku_id", "categoria_id"], as_index=False)
            .agg(vendite_7=("vendite_qty", "sum"))
        )

        g28 = (
            hist_28.groupby(["country", "pdv_id", "sku_id", "categoria_id"], as_index=False)
            .agg(vendite_28=("vendite_qty", "sum"))
        )

        m = g14.merge(g7, on=["country", "pdv_id", "sku_id", "categoria_id"], how="left")
        m = m.merge(g28, on=["country", "pdv_id", "sku_id", "categoria_id"], how="left")
        m = m.merge(g_disc, on=["country", "pdv_id", "sku_id", "categoria_id"], how="left")
        if not g_sku.empty:
            m = m.merge(g_sku, on=["country", "pdv_id", "sku_id", "categoria_id"], how="left")
        m = m.merge(g_net, on=["country", "sku_id"], how="left")
        m["units_14_net"] = m["units_14_net"].fillna(m["vendite_14"])
        m["pdv_active_14"] = m["pdv_active_14"].fillna(0.0)
        if not future.empty:
            y = future.groupby(["country", "pdv_id", "sku_id"], as_index=False).agg(y_sku=("vendite_qty", "sum"))
            how = "inner" if require_target else "left"
            m = m.merge(y, on=["country", "pdv_id", "sku_id"], how=how)
        else:
            m["y_sku"] = np.nan

        m["vendite_7"] = m["vendite_7"].fillna(0)
        m["vendite_28"] = m["vendite_28"].fillna(m["vendite_14"])

        m["run_rate_7d"] = m["vendite_7"] / 7.0
        m["run_rate_14d"] = m["vendite_14"] / 14.0
        m["run_rate_28d"] = m["vendite_28"] / 28.0

        m["trend_7v14"] = (m["run_rate_7d"] - m["run_rate_14d"]) / (m["run_rate_14d"] + 0.1)
        m["trend_14v28"] = (m["run_rate_14d"] - m["run_rate_28d"]) / (m["run_rate_28d"] + 0.1)

        m["vendite_volatilita_14d"] = m["std_14"]
        m["max_giornaliero_14d"] = m["max_14"]
        m["giorni_vendita_su_14"] = m["giorni_con_vendita_14"] / 14.0

        m["has_history"] = (m["giorni_con_vendita_14"] > 0).astype(float)
        m["giorni_da_lancio"] = m["data_lancio_sku"].apply(lambda d: (as_of - d).days)
        m["log_giorni_da_lancio"] = np.log1p(np.maximum(m["giorni_da_lancio"], 0))
        m["pdv_age_days"] = m["data_apertura"].apply(lambda d: (as_of - d).days)
        m["log_pdv_age_days"] = np.log1p(np.maximum(m["pdv_age_days"], 0))
        m["is_new_pdv"] = (m["pdv_age_days"] < 60).astype(float)
        m["is_new_sku"] = (m["giorni_da_lancio"] < 30).astype(float)

        m["giacenza_su_run_rate"] = m["giacenza"] / (m["run_rate_14d"] * cfg.horizon_sku_days + 1.0)

        m["sconto_pct"] = m["sconto_pct"].fillna(0.0)
        m["sconto_pct_medio_14d"] = m["sconto_pct_medio_14d"].fillna(0.0)
        m["giorni_in_sconto_14d"] = m["giorni_in_sconto_14d"].fillna(0.0)
        m["prezzo_listino"] = m["prezzo_listino"].fillna(m["prezzo_pagato"]).fillna(0.0)
        m["prezzo_pagato"] = m["prezzo_pagato"].fillna(m["prezzo_listino"])
        m["rapporto_prezzo_pagato"] = (m["prezzo_pagato"] / m["prezzo_listino"].clip(lower=0.01)).clip(0.1, 1.0)
        m["is_in_sconto"] = (m["sconto_pct"] >= SCONTO_SOGLIA_PCT).astype(float)
        # Impatto stimato: quanto le vendite recenti sono «spinte» dallo sconto
        m["sconto_impatto_stimato"] = m["run_rate_14d"] * (m["sconto_pct_medio_14d"] / 100.0)
        m["run_rate_x_sconto"] = m["run_rate_14d"] * (1.0 + m["sconto_pct_medio_14d"] / 100.0)

        cal = calendar[calendar["data"] == as_of]
        m = m.merge(
            cal[["country", "mese", "sin_anno", "cos_anno", "is_festivo", "is_saldi", "giorni_da_inizio_collezione"]],
            on="country",
            how="left",
        )

        m["sin_mese"] = np.sin(2 * np.pi * m["mese"] / 12.0)
        m["cos_mese"] = np.cos(2 * np.pi * m["mese"] / 12.0)
        m["day_of_week"] = as_of.weekday()
        m["is_weekend"] = float(as_of.weekday() >= 5)

        m = attach_store_weather(m, as_of, cfg.run_rate_window, master)
        m = _fill_entity_defaults(m)

        m["log_fatturazioni"] = np.log1p(m["numero_fatturazioni"].astype(float))
        m["run_rate_x_mq"] = m["run_rate_14d"] * m["mq_espositivi"] / 100.0
        m["run_rate_x_fatt"] = m["run_rate_14d"] * m["log_fatturazioni"]

        m["pioggia_x_gioielli"] = m["giorni_pioggia"].astype(float) * (
            m["sclasse"].astype(str) == "GIOIELLI"
        ).astype(float)
        m["ora_picco"] = ora_picco_vendita(m)
        if "temp_pomeriggio" not in m.columns:
            m["temp_pomeriggio"] = m["temp_media"].astype(float) + 3.5
        if "temp_orario_negozio" not in m.columns:
            m["temp_orario_negozio"] = m["temp_media"].astype(float) + 1.2
        m["temp_ora_vendita"] = temp_at_sale_hour(m["temp_media"], m["temp_pomeriggio"], m["ora_picco"])
        m = attach_sku_weather_response(m, sku_wx, as_of)

        if m["y_sku"].notna().any():
            cat_y = (
                m.groupby(["pdv_id", "categoria_id"], as_index=False)["y_sku"]
                .sum()
                .rename(columns={"y_sku": "y_cat"})
            )
            m = m.merge(cat_y, on=["pdv_id", "categoria_id"], how="left")
        else:
            m["y_cat"] = np.nan
        m["as_of"] = as_of
        samples.append(m)

    return pd.concat(samples, ignore_index=True)


def apply_train_test_split(
    samples: pd.DataFrame,
    cfg: TrainConfig,
    n_test_as_of: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, date, date | None]:
    """
    Split temporale + finestra mobile sul training.

    - Test = ultimi ``n_test_as_of`` valori di ``as_of``
    - Train = as_of precedenti, limitati a ``cfg.train_window_days``
      rispetto all'ultimo as_of di train (prima del cut).

    Ritorna (train_df, test_df, cut, train_start).
    ``train_start`` è None se non c'è finestra (usa tutto lo storico).
    """
    as_ofs = sorted(samples["as_of"].unique())
    if len(as_ofs) <= n_test_as_of:
        raise ValueError(
            f"Troppi pochi as_of ({len(as_ofs)}) per split con n_test_as_of={n_test_as_of}"
        )

    cut = as_ofs[-n_test_as_of]
    test_df = samples[samples["as_of"] >= cut].copy()
    train_pool = samples[samples["as_of"] < cut].copy()

    train_start: date | None = None
    window = int(getattr(cfg, "train_window_days", 0) or 0)
    if window > 0 and not train_pool.empty:
        last_train_as_of = train_pool["as_of"].max()
        train_start = last_train_as_of - timedelta(days=window - 1)
        train_df = train_pool[train_pool["as_of"] >= train_start].copy()
    else:
        train_df = train_pool

    if train_df.empty:
        raise ValueError(
            f"Train vuoto dopo finestra mobile (train_window_days={window}, cut={cut})"
        )

    return train_df, test_df, cut, train_start
