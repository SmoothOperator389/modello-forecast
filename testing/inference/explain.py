"""Spiegazioni locali delle previsioni — driver di feature + narrativa in italiano.

Approccio: ablazione a gruppi (sostituire un blocco di feature con la media del train)
e misurare quanto cambia la previsione. Non è SHAP (troppo pesante), ma è
interpretabile per Produzione/CDM e non richiede dipendenze extra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from testing.features.build_features import NUMERIC_COLS
from testing.features.schema import SKU_ATTR_COLS


# Gruppi leggibili per Produzione (chiave → colonne numeriche o flag logici)
FEATURE_GROUPS: dict[str, list[str]] = {
    "andamento_vendite": [
        "run_rate_14d",
        "run_rate_7d",
        "run_rate_28d",
        "trend_7v14",
        "trend_14v28",
        "vendite_volatilita_14d",
        "max_giornaliero_14d",
        "giorni_vendita_su_14",
        "has_history",
        "run_rate_x_mq",
        "run_rate_x_fatt",
    ],
    "maturita_sku_pdv": [
        "giorni_da_lancio",
        "log_giorni_da_lancio",
        "pdv_age_days",
        "log_pdv_age_days",
        "is_new_pdv",
        "is_new_sku",
    ],
    "stock_e_capacita": [
        "giacenza",
        "giacenza_su_run_rate",
        "mq_espositivi",
        "mq_stock",
        "mq_totali",
        "transit_time_days",
    ],
    "prezzo_e_sconto": [
        "prezzo_listino",
        "prezzo_pagato",
        "rapporto_prezzo_pagato",
        "is_in_sconto",
        "sconto_pct",
        "sconto_pct_medio_14d",
        "giorni_in_sconto_14d",
        "sconto_impatto_stimato",
        "run_rate_x_sconto",
    ],
    "traffico_negozio": [
        "numero_fatturazioni",
        "log_fatturazioni",
    ],
    "meteo_e_calendario": [
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
    ],
}

GROUP_LABELS_IT: dict[str, str] = {
    "andamento_vendite": "andamento vendite recenti",
    "maturita_sku_pdv": "età SKU / negozio (cold start)",
    "stock_e_capacita": "giacenza e capacità negozio",
    "prezzo_e_sconto": "prezzo e promozioni",
    "traffico_negozio": "traffico / fatturazioni",
    "meteo_e_calendario": "meteo e calendario",
    "profilo_prodotto": "profilo prodotto (modello/tessuto/colore…)",
    "identita_negozio": "profilo negozio / paese",
    "analoghi_cold_start": "SKU analoghi (cold start)",
}

# Ablazione categorica: colonne da sostituire con moda del train
CAT_GROUPS: dict[str, list[str]] = {
    "profilo_prodotto": ["categoria_id", *SKU_ATTR_COLS],
    "identita_negozio": ["country", "pdv_id", "commerciale_riferimento"],
}


@dataclass
class Driver:
    group: str
    label: str
    delta: float  # pezzi: >0 alza la previsione, <0 la abbassa
    direction: str  # "alza" | "abbassa"


def _safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _train_means(train_df: pd.DataFrame) -> dict[str, float]:
    means: dict[str, float] = {}
    for col in NUMERIC_COLS:
        if col in train_df.columns:
            means[col] = float(train_df[col].astype(float).mean())
    return means


def _train_modes(train_df: pd.DataFrame) -> dict[str, str]:
    modes: dict[str, str] = {}
    for cols in CAT_GROUPS.values():
        for col in cols:
            if col in train_df.columns:
                modes[col] = str(train_df[col].astype(str).mode().iloc[0])
    return modes


def local_drivers(
    row: pd.Series,
    model,
    maps: dict,
    num_mean: np.ndarray,
    num_std: np.ndarray,
    train_means: dict[str, float],
    train_modes: dict[str, str],
    predict_fn,
    top_k: int = 4,
) -> list[Driver]:
    """Quanto ogni gruppo di feature sposta la previsione rispetto al caso reale."""
    base_df = pd.DataFrame([row])
    base_pred = float(predict_fn(model, base_df, maps, num_mean, num_std)[0][0])

    impacts: list[Driver] = []

    for group, cols in FEATURE_GROUPS.items():
        use = _safe_cols(base_df, cols)
        if not use:
            continue
        ablated = base_df.copy()
        for c in use:
            ablated[c] = ablated[c].astype(float)
            ablated.loc[:, c] = float(train_means.get(c, 0.0))
        pred_a = float(predict_fn(model, ablated, maps, num_mean, num_std)[0][0])
        # contributo = quanto la feature reale alza/abbassa rispetto alla media
        delta = base_pred - pred_a
        if abs(delta) < 0.05:
            continue
        impacts.append(
            Driver(
                group=group,
                label=GROUP_LABELS_IT[group],
                delta=delta,
                direction="alza" if delta > 0 else "abbassa",
            )
        )

    for group, cols in CAT_GROUPS.items():
        use = _safe_cols(base_df, cols)
        if not use:
            continue
        ablated = base_df.copy()
        for c in use:
            if c in train_modes:
                ablated[c] = ablated[c].astype(object)
                ablated.loc[:, c] = train_modes[c]
        pred_a = float(predict_fn(model, ablated, maps, num_mean, num_std)[0][0])
        delta = base_pred - pred_a
        if abs(delta) < 0.05:
            continue
        impacts.append(
            Driver(
                group=group,
                label=GROUP_LABELS_IT[group],
                delta=delta,
                direction="alza" if delta > 0 else "abbassa",
            )
        )

    impacts.sort(key=lambda d: abs(d.delta), reverse=True)
    return impacts[:top_k]


def narrative_for_row(
    row: pd.Series,
    drivers: list[Driver],
    pred_final: float,
    pred_model: float,
    pred_analogue: float | None,
    confidence: float,
    horizon: int = 15,
) -> str:
    """Testo parlato per Produzione/CDM."""
    pdv = str(row.get("pdv_id", "?"))
    sku = str(row.get("sku_id", "?"))
    rr = float(row.get("run_rate_14d", 0.0))
    naive = rr * horizon
    days = float(row.get("giorni_da_lancio", 0.0))
    is_new_pdv = float(row.get("is_new_pdv", 0.0)) >= 0.5
    is_new_sku = days < 30
    sconto = float(row.get("sconto_pct", 0.0))
    temp = float(row.get("temp_media", 20.0))
    giac = float(row.get("giacenza", 0.0))

    parts: list[str] = []
    parts.append(
        f"Per {pdv} / {sku} il sistema stima <strong>{pred_final:.1f} pezzi</strong> "
        f"nei prossimi {horizon} giorni (run-rate recente ≈ {naive:.1f})."
    )

    if is_new_pdv or is_new_sku:
        chi = "negozio nuovo" if is_new_pdv else "SKU giovane"
        if pred_analogue is not None:
            parts.append(
                f"Il caso è in <strong>cold start</strong> ({chi}, "
                f"{days:.0f} gg dal lancio, confidence {confidence:.0%}): "
                f"mescola rete ({pred_model:.1f}) e analoghi ({pred_analogue:.1f}) "
                f"→ {pred_final:.1f} pezzi."
            )
        else:
            parts.append(
                f"Il caso è in <strong>cold start</strong> ({chi}): "
                f"pesa di più il profilo degli analoghi (confidence {confidence:.0%})."
            )
    else:
        parts.append(
            f"SKU/negozio maturi (confidence storico {confidence:.0%}): "
            f"prevale la rete neurale ({pred_model:.1f} pezzi)."
        )

    if drivers:
        bullets = []
        for d in drivers:
            sign = "+" if d.delta > 0 else "−"
            bullets.append(
                f"<strong>{d.label}</strong> {d.direction} la stima di circa "
                f"{sign}{abs(d.delta):.1f} pezzi rispetto a un profilo medio"
            )
        parts.append("Driver principali: " + "; ".join(bullets) + ".")

    # Pattern logici contestuali (anche se non top driver)
    extras: list[str] = []
    if sconto >= 10:
        extras.append(
            f"c'è una promo attiva (~{sconto:.0f}% di sconto): "
            f"il modello separa spinta promo da domanda strutturale"
        )
    ssclasse = str(row.get("ssclasse", "") or "")
    tipo = ssclasse.lower() if ssclasse and ssclasse != "NA" else "prodotto"
    affinity = float(row.get("sku_share_hot", 0) or 0) - float(row.get("sku_share_cold", 0) or 0)
    if temp >= 26:
        if affinity > 0.15:
            extras.append(
                f"temperatura alta ({temp:.0f}°C): questo SKU ha venduto soprattutto col caldo ({tipo})"
            )
        elif affinity < -0.15:
            extras.append(
                f"temperatura alta ({temp:.0f}°C): questo SKU ha venduto soprattutto col freddo ({tipo}), stima in calo"
            )
        else:
            extras.append(f"temperatura alta ({temp:.0f}°C): effetto meteo sul mix negozio")
    elif temp <= 10:
        if affinity < -0.15:
            extras.append(
                f"temperatura bassa ({temp:.0f}°C): questo SKU ha venduto soprattutto col freddo ({tipo})"
            )
        elif affinity > 0.15:
            extras.append(
                f"temperatura bassa ({temp:.0f}°C): questo SKU ha venduto soprattutto col caldo ({tipo}), stima in calo"
            )
        else:
            extras.append(f"temperatura bassa ({temp:.0f}°C): effetto stagionale sul mix")
    rain_days = float(row.get("giorni_pioggia", 0) or 0)
    if rain_days >= 2 and str(row.get("classe", "")) == "ACCESSORI":
        extras.append(
            f"pioggia frequente ({rain_days:.0f} gg/14): i gioielli/impulse soffrono il calo di traffico"
        )
    if giac > pred_final * 1.2:
        extras.append(
            f"giacenza ({giac:.0f}) già copre la domanda attesa → tipicamente «monitora»"
        )
    elif giac < pred_final * 0.5:
        extras.append(
            f"giacenza bassa ({giac:.0f}) rispetto alla domanda → tipicamente «valuta riassortimento»"
        )

    if extras:
        parts.append("Contesto: " + "; ".join(extras) + ".")

    return " ".join(parts)


def explain_focus_frame(
    focus_df: pd.DataFrame,
    model,
    maps: dict,
    num_mean: np.ndarray,
    num_std: np.ndarray,
    train_df: pd.DataFrame,
    predict_fn,
    horizon: int = 15,
) -> pd.DataFrame:
    """Aggiunge colonne driver_1..n e spiegazione_testuale ai casi focus."""
    means = _train_means(train_df)
    modes = _train_modes(train_df)

    rows = []
    for _, r in focus_df.iterrows():
        drivers = local_drivers(
            r, model, maps, num_mean, num_std, means, modes, predict_fn, top_k=4
        )
        pred_final = float(r.get("pred_final", r.get("pred_tf", 0.0)))
        pred_model = float(r.get("pred_model", pred_final))
        pred_analogue = float(r["pred_analogue"]) if "pred_analogue" in r and pd.notna(r["pred_analogue"]) else None
        conf = float(r.get("confidence_score", r.get("confidence_history", 0.5)))
        text = narrative_for_row(r, drivers, pred_final, pred_model, pred_analogue, conf, horizon)

        out = {
            "as_of": r.get("as_of"),
            "pdv_id": r.get("pdv_id"),
            "sku_id": r.get("sku_id"),
            "spiegazione_testuale": text,
        }
        for i, d in enumerate(drivers, start=1):
            out[f"driver_{i}"] = d.label
            out[f"driver_{i}_delta"] = round(d.delta, 2)
            out[f"driver_{i}_dir"] = d.direction
        rows.append(out)

    return pd.DataFrame(rows)
