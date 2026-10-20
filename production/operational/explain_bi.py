"""Spiegazioni locali per BI CDM — contributi LightGBM + zona di routing."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_LABELS_IT: dict[str, str] = {
    "run_rate_14d": "ritmo vendite 14gg",
    "run_rate_7d": "ritmo vendite 7gg",
    "run_rate_28d": "ritmo vendite 28gg",
    "sku_prior": "prior (run-rate × 15gg)",
    "trend_7v14": "trend 7 vs 14gg",
    "trend_14v28": "trend 14 vs 28gg",
    "vendite_volatilita_14d": "volatilità vendite",
    "max_giornaliero_14d": "picco giornaliero 14gg",
    "giorni_vendita_su_14": "giorni con vendita / 14",
    "has_history": "storico disponibile",
    "giorni_da_lancio": "giorni dal lancio SKU",
    "log_giorni_da_lancio": "età SKU (log)",
    "pdv_age_days": "età negozio (giorni)",
    "log_pdv_age_days": "età negozio (log)",
    "is_new_pdv": "negozio nuovo",
    "is_new_sku": "SKU nuovo",
    "giacenza": "giacenza",
    "giacenza_su_run_rate": "giacenza vs ritmo",
    "prezzo_listino": "prezzo listino",
    "prezzo_pagato": "prezzo pagato",
    "rapporto_prezzo_pagato": "rapporto prezzo",
    "is_in_sconto": "in sconto",
    "sconto_pct": "% sconto",
    "sconto_pct_medio_14d": "sconto medio 14gg",
    "giorni_in_sconto_14d": "giorni in sconto",
    "sconto_impatto_stimato": "impatto sconto",
    "run_rate_x_sconto": "ritmo × sconto",
    "mq_espositivi": "mq espositivi",
    "mq_stock": "mq stock",
    "mq_totali": "mq totali",
    "numero_fatturazioni": "fatturazioni",
    "log_fatturazioni": "fatturazioni (log)",
    "transit_time_days": "giorni di transit",
    "temp_media": "temperatura media",
    "giorni_pioggia": "giorni di pioggia",
    "temp_orario_negozio": "temp. orario negozio",
    "temp_pomeriggio": "temp. pomeriggio",
    "ora_picco": "ora di picco",
    "temp_ora_vendita": "temp. ora vendita",
    "sku_temp_vendita": "temp. tipica SKU",
    "sku_share_hot": "quota vendite col caldo",
    "sku_share_cold": "quota vendite col freddo",
    "sku_temp_gap": "gap temperatura SKU",
    "sku_temp_fit": "fit temperatura SKU",
    "run_rate_x_sku_temp": "ritmo × fit temp.",
    "pioggia_x_gioielli": "pioggia × accessori",
    "mese": "mese",
    "sin_anno": "stagionalità (sin)",
    "cos_anno": "stagionalità (cos)",
    "sin_mese": "ciclo mese (sin)",
    "cos_mese": "ciclo mese (cos)",
    "day_of_week": "giorno settimana",
    "is_weekend": "weekend",
    "is_festivo": "festivo",
    "is_saldi": "saldi",
    "giorni_da_inizio_collezione": "gg da inizio collezione",
    "sku_id": "identità SKU",
    "pdv_id": "identità negozio",
    "modello": "modello",
    "colore": "colore",
    "tessuto": "tessuto",
    "campagna": "campagna",
    "classe": "classe",
    "sclasse": "sottoclasse",
    "ssclasse": "microclasse",
    "tipologia": "tipologia",
    "categoria_id": "categoria",
    "country": "paese",
    "commerciale_riferimento": "commerciale",
}


def route_zone_label(prior: float, low_thr: float, high_thr: float) -> tuple[str, str]:
    if prior <= low_thr:
        return "bassi", "volumi bassi → motore Tweedie (domanda intermittente)"
    if prior <= high_thr:
        return "medi", "volumi medi → mix specialista high-volume + Tweedie"
    return "alti", "coda alta → mix con floor verso il naive (anti-sottostima)"


def lgbm_top_drivers(
    model,
    X: pd.DataFrame,
    feature_names: list[str],
    *,
    top_k: int = 4,
) -> tuple[list[list[str]], list[list[float]]]:
    """Per ogni riga: top feature labels e contributi firmati (pezzi)."""
    contrib = model.predict(X[feature_names], pred_contrib=True)
    contrib = np.asarray(contrib, dtype=float)
    # ultima colonna = bias
    feat_part = contrib[:, : len(feature_names)]
    labels_out: list[list[str]] = []
    deltas_out: list[list[float]] = []
    for i in range(feat_part.shape[0]):
        idx = np.argsort(-np.abs(feat_part[i]))[:top_k]
        labs, dels = [], []
        for j in idx:
            if abs(feat_part[i, j]) < 0.02:
                continue
            name = feature_names[j]
            labs.append(FEATURE_LABELS_IT.get(name, name))
            dels.append(round(float(feat_part[i, j]), 2))
        labels_out.append(labs)
        deltas_out.append(dels)
    return labels_out, deltas_out


def build_why_text(
    *,
    pred: float,
    naive: float,
    route_msg: str,
    driver_labels: list[str],
    driver_deltas: list[float],
    giacenza: float,
    is_new: bool,
    horizon: int = 15,
) -> str:
    parts = [
        f"Stima {pred:.1f} pezzi / {horizon}gg (run-rate naive ≈ {naive:.1f}).",
        f"Routing: {route_msg}.",
    ]
    if driver_labels:
        bits = []
        for lab, d in zip(driver_labels, driver_deltas):
            arrow = "alza" if d >= 0 else "abbassa"
            bits.append(f"{lab} {arrow} (~{d:+.1f})")
        parts.append("Feature più influenti: " + "; ".join(bits) + ".")
    if is_new:
        parts.append("Caso nuovo (SKU/PDV): più prudenza sul riassortimento.")
    if giacenza > pred * 1.2:
        parts.append(f"Giacenza {giacenza:.0f} già copre la domanda → tipicamente monitora.")
    elif giacenza < pred * 0.5 and pred >= 2:
        parts.append(f"Giacenza bassa ({giacenza:.0f}) vs domanda → valuta riassortimento.")
    return " ".join(parts)
