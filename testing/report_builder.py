"""Genera report HTML per il reparto produzione (linguaggio semplice, grafici SVG)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from testing.config import DATA_DIR, PRODUCTION_N_PDV, PRODUCTION_N_SKU
from testing.features.build_features import NUMERIC_COLS

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"


def _svg_bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    y_label: str,
    colors: list[str] | None = None,
    width: int = 520,
    height: int = 280,
) -> str:
    if not values:
        return ""
    colors = colors or ["#2563eb"] * len(values)
    max_v = max(values) * 1.2 or 1
    bar_w = min(80, (width - 80) / max(len(values), 1) - 16)
    gap = 16
    start_x = 60
    bars = []
    for i, (lab, val, col) in enumerate(zip(labels, values, colors)):
        x = start_x + i * (bar_w + gap)
        h = (val / max_v) * (height - 80)
        y = height - 50 - h
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{col}" rx="4"/>')
        bars.append(f'<text x="{x + bar_w/2:.1f}" y="{y - 6}" text-anchor="middle" font-size="11" fill="#0f172a">{val:.2f}</text>')
        lines = lab.split("\n")
        for j, line in enumerate(lines):
            bars.append(
                f'<text x="{x + bar_w/2:.1f}" y="{height - 28 + j*13}" text-anchor="middle" font-size="10" fill="#64748b">{line}</text>'
            )
    return f"""
    <div class="chart-svg">
      <div class="chart-title">{title}</div>
      <svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="{title}">
        <line x1="50" y1="{height-50}" x2="{width-20}" y2="{height-50}" stroke="#cbd5e1" stroke-width="1"/>
        <line x1="50" y1="30" x2="50" y2="{height-50}" stroke="#cbd5e1" stroke-width="1"/>
        <text x="12" y="20" font-size="10" fill="#64748b" transform="rotate(-90 12 20)">{y_label}</text>
        {''.join(bars)}
      </svg>
    </div>"""


def _svg_line_chart(epochs: list[int], values: list[float], title: str, y_label: str) -> str:
    if not values:
        return ""
    w, h = 560, 260
    pad_l, pad_b, pad_t = 50, 40, 30
    max_v = max(values) * 1.1 or 1
    min_v = min(values) * 0.9
    span = max(max_v - min_v, 0.01)
    pts = []
    for i, (ep, val) in enumerate(zip(epochs, values)):
        x = pad_l + (i / max(len(epochs) - 1, 1)) * (w - pad_l - 20)
        y = pad_t + (1 - (val - min_v) / span) * (h - pad_t - pad_b)
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)
    return f"""
    <div class="chart-svg">
      <div class="chart-title">{title}</div>
      <svg viewBox="0 0 {w} {h}" width="100%" height="{h}" role="img" aria-label="{title}">
        <polyline points="{poly}" fill="none" stroke="#2563eb" stroke-width="2.5"/>
        <line x1="{pad_l}" y1="{h-pad_b}" x2="{w-20}" y2="{h-pad_b}" stroke="#cbd5e1"/>
        <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{h-pad_b}" stroke="#cbd5e1"/>
        <text x="8" y="18" font-size="10" fill="#64748b">{y_label}</text>
        <text x="{pad_l}" y="{h-12}" font-size="10" fill="#64748b">Epoca 1</text>
        <text x="{w-40}" y="{h-12}" font-size="10" fill="#64748b" text-anchor="end">Epoca {epochs[-1]}</text>
      </svg>
    </div>"""


def _svg_grouped_bars(focus: pd.DataFrame, pdv: pd.DataFrame) -> str:
    if focus.empty:
        return ""
    city = pdv.set_index("pdv_id")["city"]
    labels = [f"{city.get(r.pdv_id, r.pdv_id)} / {r.sku_id}" for r in focus.itertuples()]
    series = [
        ("Vendite reali", [r.y_vera for r in focus.itertuples()], "#16a34a"),
        ("Rete neurale", [r.pred_tf for r in focus.itertuples()], "#2563eb"),
        (
            "Modello CDM",
            [getattr(r, "pred_cdm", 0.0) or 0.0 for r in focus.itertuples()],
            "#94a3b8",
        ),
        ("Giacenza", [r.giacenza for r in focus.itertuples()], "#f59e0b"),
    ]
    w, h = 600, 300
    max_v = max(max(s[1]) for s in series) * 1.15 or 1
    group_w = (w - 80) / max(len(labels), 1)
    bar_w = group_w / 5
    parts = [
        f'<div class="chart-svg"><div class="chart-title">Confronto: reale · rete · modello CDM originale · giacenza</div>',
        f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}">',
        f'<line x1="50" y1="{h-40}" x2="{w-20}" y2="{h-40}" stroke="#cbd5e1"/>',
    ]
    for gi, lab in enumerate(labels):
        gx = 60 + gi * group_w
        for si, (sname, vals, col) in enumerate(series):
            val = vals[gi]
            bh = (val / max_v) * (h - 90)
            x = gx + si * (bar_w + 2)
            y = h - 40 - bh
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{col}" rx="3"/>')
        parts.append(f'<text x="{gx + group_w/2:.1f}" y="{h-18}" text-anchor="middle" font-size="9" fill="#64748b">{lab[:22]}</text>')
    legend = "".join(
        f'<span class="legend-item"><i style="background:{col}"></i>{name}</span>' for name, _, col in series
    )
    parts.append("</svg>")
    parts.append(f'<div class="legend">{legend}</div></div>')
    return "".join(parts)


def _section_stagionalita() -> str:
    return """
  <section>
    <h2>Stagionalità, meteo, temperatura = feature (su ogni riga)</h2>
    <p>Non sono un «sotto-modello» e non sono il KPI «capi estate/inverno».
    Su <strong>ogni</strong> snapshot PDV×SKU il modello riceve tre blocchi di input:</p>
    <ul>
      <li><strong>Calendario:</strong> <code>mese</code>, <code>sin_mese</code>/<code>cos_mese</code>,
      <code>sin_anno</code>/<code>cos_anno</code>, weekend, saldi, festivi, giorni da inizio collezione.</li>
      <li><strong>Meteo città del PDV (ultimi 14 gg):</strong> <code>temp_media</code>,
      <code>temp_orario_negozio</code>, <code>temp_pomeriggio</code>, <code>temp_ora_vendita</code>
      (temperatura all'ora media degli scontrini), <code>giorni_pioggia</code>, embedding del regime.</li>
      <li><strong>Profilo SKU dalle vendite:</strong> temperatura media a cui quello SKU ha venduto,
      quota pezzi al caldo/freddo, e quanto il meteo attuale ci si allinea
      (<code>sku_temp_vendita</code>, <code>sku_share_hot</code>/<code>sku_share_cold</code>, <code>sku_temp_fit</code>).
      Nessun tag ESTATE/INVERNO sul prodotto — come in produzione Valido.</li>
    </ul>
    <p class="note">Il vecchio modello CDM usava <em>coefficienti mese-su-mese calcolati a mano</em>.
    Qui mese e temperatura sono feature: la rete impara l'effetto dai dati.</p>

    <div class="concept-grid">
      <div class="concept-card">
        <h3>Mese dell'anno</h3>
        <p>Il modello riceve <code>mese</code>, <code>sin_mese</code> e <code>cos_mese</code>:
        una rappresentazione ciclica che collega dicembre a gennaio (non tratta dicembre come «lontano» da gennaio).</p>
      </div>
      <div class="concept-card">
        <h3>Pattern per tipo prodotto</h3>
        <p><strong>Estate</strong> (crop, shorts, sandali) vende di più con il caldo.<br>
        <strong>Inverno</strong> (piumini, maglioni, stivali) vende di più col freddo.<br>
        <strong>Continuativo</strong> (jeans, sneaker) è più stabile; il meteo pesa lo stesso, ma di meno.</p>
      </div>
      <div class="concept-card">
        <h3>Weekend e saldi</h3>
        <p><code>is_weekend</code> e <code>is_saldi</code> segnalano sabato/domenica e periodi promozionali
        (prime 2 settimane di gennaio e luglio).</p>
      </div>
      <div class="concept-card">
        <h3>Collezione</h3>
        <p><code>giorni_da_inizio_collezione</code> misura quanto siamo avanti nella stagione corrente.</p>
      </div>
    </div>

    <table class="season-table">
      <thead><tr><th>Tipo prodotto (attributo SKU)</th><th>Gen–Mar</th><th>Apr–Giu</th><th>Lug–Set</th><th>Ott–Dic</th></tr></thead>
      <tbody>
        <tr><td>ESTATE (crop, shorts, sandali)</td><td class="low">Basso</td><td class="high">Alto</td><td class="high">Picco</td><td class="low">Basso</td></tr>
        <tr><td>INVERNO (piumini, maglioni, stivali)</td><td class="high">Picco</td><td class="mid">Medio</td><td class="low">Basso</td><td class="high">Alto</td></tr>
        <tr><td>CONTINUATIVO (jeans, sneaker)</td><td class="mid">Medio</td><td class="mid">Medio</td><td class="mid">Medio</td><td class="mid">Medio</td></tr>
        <tr><td>Gioielli / impulse</td><td class="mid">Medio</td><td class="mid">Medio</td><td class="mid">Medio</td><td class="high">Natale</td></tr>
      </tbody>
    </table>
    <p class="note"><strong>Per Produzione:</strong> a parità di vendite recenti, lo stesso <code>temp_media</code>
    alza un crop a luglio e tiene basso un piumino. Non è un filtro sul report: è una feature in input.</p>
  </section>"""


def _section_sconto() -> str:
    return """
  <section>
    <h2>Sconto prodotto — prezzo listino vs prezzo pagato</h2>
    <p>Il modello distingue tra <strong>periodo saldi</strong> (calendario generale) e <strong>sconto sul singolo SKU</strong>
    (promo negozio / articolo). Usa i campi Valido <code>prezzo_pubblico</code> e <code>pagato_effettivo</code>.</p>

    <div class="flow">
      <div class="flow-step"><span class="flow-num">1</span><strong>Prezzo listino</strong><br><small>prezzo_pubblico</small></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><span class="flow-num">2</span><strong>Prezzo pagato</strong><br><small>pagato_effettivo</small></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><span class="flow-num">3</span><strong>% sconto</strong><br><small>(1 − pagato/listino) × 100</small></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><span class="flow-num">4</span><strong>Impatto vendite</strong><br><small>il modello impara l'elasticità</small></div>
    </div>

    <table>
      <thead><tr><th>Feature</th><th>Cosa misura</th><th>Perché conta</th></tr></thead>
      <tbody>
        <tr><td><code>is_in_sconto</code></td><td>Prodotto in promo oggi (sconto ≥ 5%)</td><td>Segnala promo attiva sullo SKU</td></tr>
        <tr><td><code>sconto_pct</code></td><td>Sconto % attuale</td><td>Quanto è aggressiva la promo</td></tr>
        <tr><td><code>sconto_pct_medio_14d</code></td><td>Sconto medio ultimi 14 giorni</td><td>Promo prolungata vs flash</td></tr>
        <tr><td><code>giorni_in_sconto_14d</code></td><td>Quota giorni in promo / 14</td><td>Continuità della promozione</td></tr>
        <tr><td><code>rapporto_prezzo_pagato</code></td><td>pagato / listino (0.7 = −30%)</td><td>Prezzo reale percepito dal cliente</td></tr>
        <tr><td><code>sconto_impatto_stimato</code></td><td>run_rate × sconto medio</td><td>Quanto le vendite recenti sono «spinte» dallo sconto</td></tr>
        <tr><td><code>run_rate_x_sconto</code></td><td>Interazione vendite × sconto</td><td>Il modello separa domanda vera da effetto promo</td></tr>
      </tbody>
    </table>

    <h3 class="sub">Sconto vs periodo saldi</h3>
    <table>
      <thead><tr><th>Segnale</th><th>Fonte</th><th>Esempio</th></tr></thead>
      <tbody>
        <tr><td><code>is_saldi</code></td><td>Calendario paese</td><td>Prime 2 settimane gennaio / luglio — tutto il negozio</td></tr>
        <tr><td><code>is_in_sconto</code></td><td>Prezzo SKU</td><td>Promo su singolo articolo, anche fuori saldi</td></tr>
      </tbody>
    </table>

    <p class="note"><strong>Per Produzione:</strong> se un prodotto vende molto perché è in sconto al 40%,
    il modello non «extrapola» quella cifra come domanda normale: usa le feature sconto per capire quanto
    dell'andamento è promo e quanto è domanda strutturale. La baseline naive invece vede solo il picco e lo ripete.</p>
  </section>"""


def _section_meteo() -> str:
    return """
  <section>
    <h2>Meteo — temperatura e pioggia nel negozio</h2>
    <p>Per ogni negozio il modello legge il meteo degli <strong>ultimi 14 giorni nella città del PDV</strong>
    e lo mette in input come feature (insieme al calendario). Vale per tutti gli SKU, non solo per crop e piumini.
    In testing è simulato, oppure da <strong>Open-Meteo</strong> con <code>FORECAST_WEATHER_API=1</code>.
    In produzione: Open-Meteo (storico + orario, senza API key) sulla <code>city</code> del negozio.</p>

    <div class="flow">
      <div class="flow-step"><span class="flow-num">1</span><strong>Dati meteo</strong><br><small>temp. media, giorni di pioggia, mm pioggia</small></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><span class="flow-num">2</span><strong>Regime</strong><br><small>freddo / mite / caldo / pioggia</small></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><span class="flow-num">3</span><strong>Embedding meteo</strong><br><small>il modello impara l'effetto sulle vendite</small></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><span class="flow-num">4</span><strong>Previsione aggiustata</strong><br><small>es. caldo → più crop/sandali, meno piumini</small></div>
    </div>

    <table>
      <thead><tr><th>Regime</th><th>Condizione</th><th>Effetto tipico</th></tr></thead>
      <tbody>
        <tr><td><span class="pill pill-cold">Freddo</span></td><td>Temp. media ≤ 10°C</td><td>Piumini, maglioni, stivali, sciarpe in salita; capi estivi in calo</td></tr>
        <tr><td><span class="pill pill-mild">Mite</span></td><td>10°C &lt; temp. &lt; 26°C</td><td>Domanda nella media; jeans/sneaker più stabili</td></tr>
        <tr><td><span class="pill pill-hot">Caldo</span></td><td>Temp. media ≥ 26°C</td><td>Crop, shorts, sandali, occhiali, costumi in salita; inverno in calo</td></tr>
        <tr><td><span class="pill pill-rain">Pioggia</span></td><td>≥ 2 giorni di pioggia / 14 gg</td><td>Gioielli/impulse in calo (meno traffico); piumini/parka in lieve salita</td></tr>
      </tbody>
    </table>

    <p><strong>Feature numeriche usate:</strong> <code>temp_media</code>, <code>temp_orario_negozio</code> (10–20),
    <code>temp_pomeriggio</code> (14–17), <code>ora_picco</code>, <code>temp_ora_vendita</code>
    (temperatura all'ora tipica di vendita dello SKU), <code>giorni_pioggia</code>, embedding del regime.</p>
    <p class="note"><strong>In produzione:</strong> Open-Meteo per <strong>city</strong> del PDV (es. Milan, Paris, Berlin).
    Due negozi nella stessa città condividono lo stesso meteo. Le vendite testing hanno
    <strong>data e ora scontrino</strong> come Valido (<code>fact_tickets.csv</code>);
    <code>temp_ora_vendita</code> usa l'ora media reale dei pezzi venduti.</p>
  </section>"""


def _section_embedding_cold_start() -> str:
    return """
  <section>
    <h2>SKU nuovi — embedding attributi + confidence score</h2>
    <p>Per uno <strong>SKU nuovo</strong> lo <code>sku_id</code> non ha storico (embedding vuoto).
    Il modello si appoggia agli <strong>attributi condivisi</strong> (modello, tessuto, colore, campagna, classe…)
    imparati dagli SKU solidi con lo stesso profilo prodotto.</p>

    <div class="flow">
      <div class="flow-step"><span class="flow-num">1</span><strong>Attributi prodotto</strong><br><small>embedding condivisi con SKU maturi</small></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><span class="flow-num">2</span><strong>Analoghi</strong><br><small>mediana vendite SKU solidi simili</small></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><span class="flow-num">3</span><strong>Confidence</strong><br><small>4 segnali maturità</small></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><span class="flow-num">4</span><strong>Blend finale</strong><br><small>analoghi ↔ storico</small></div>
    </div>

    <h3 class="sub">Quattro segnali di maturità (PDF CDM)</h3>
    <table>
      <thead><tr><th>Segnale</th><th>Feature</th><th>Cosa misura</th></tr></thead>
      <tbody>
        <tr><td>Settimane osservate</td><td><code>giorni_da_lancio / 7</code></td><td>Da quanto il SKU è in assortimento</td></tr>
        <tr><td>Unità vendute (rete)</td><td><code>units_14_net</code></td><td>Pezzi venduti ultimi 14 gg su tutti i PDV</td></tr>
        <tr><td>PDV con vendita</td><td><code>pdv_active_14</code></td><td>Quanti negozi vendono questo SKU</td></tr>
        <tr><td>Stabilità run rate</td><td><code>volatilità / run_rate</code></td><td>Vendite regolari vs casuali</td></tr>
      </tbody>
    </table>

    <h3 class="sub">Blend post-modello</h3>
    <p><code>pred_finale = (1 − w) × analoghi + w × storico</code><br>
    <strong>w</strong> (peso storico) cresce con la maturità: ~10% a 3 giorni, ~40% a 21 giorni, ~85% a 45 giorni (PDF CDM).</p>
    <ul>
      <li><strong>Analoghi:</strong> mediana vendite 15 gg di SKU solidi con stesso modello/tessuto/colore + ancora categoria</li>
      <li><strong>Storico:</strong> mix run rate (naive) e modello neurale — più modello man mano che il SKU matura</li>
      <li><strong>Qty riassortimento:</strong> ridotta se <code>confidence_score</code> basso</li>
    </ul>

    <table>
      <thead><tr><th>Età SKU</th><th>Peso analoghi</th><th>Peso storico</th><th>Comportamento</th></tr></thead>
      <tbody>
        <tr><td>~3 giorni</td><td>~90%</td><td>~10%</td><td>Profilo prodotto + categoria dominano</td></tr>
        <tr><td>~21 giorni</td><td>~60%</td><td>~40%</td><td>Transizione</td></tr>
        <tr><td>~45+ giorni</td><td>~15%</td><td>~85%</td><td>Run rate + modello dominano</td></tr>
      </tbody>
    </table>

    <p class="note"><strong>Importante:</strong> non copiamo lo <code>sku_id</code> di un altro prodotto.
    Gli analoghi derivano da <strong>attributi Valido condivisi</strong> e vendite osservate su SKU solidi simili.</p>
  </section>"""


def _section_architettura(metrics: dict) -> str:
    n_params = metrics.get("n_params", "—")
    epochs = metrics.get("epochs_run", "—")
    return f"""
  <section>
    <h2>Come funziona il modello — panoramica</h2>
    <p>Rete neurale <strong>dual-head</strong> (TensorFlow/Keras): un encoder condiviso, due uscite separate.</p>

    <div class="arch-diagram">
      <div class="arch-col">
        <div class="arch-box arch-input">Input<br><small>{len(NUMERIC_COLS)} feature numeriche</small></div>
        <div class="arch-box arch-input">Embedding negozio<br><small>country, PDV, commerciale, mq, meteo</small></div>
        <div class="arch-box arch-input">Embedding prodotto<br><small>SKU, modello, tessuto, colore, campagna, classe…</small></div>
      </div>
      <div class="arch-arrow">↓</div>
      <div class="arch-box arch-core">Encoder condiviso<br><small>3 blocchi residuali · BatchNorm · {n_params:,} parametri</small></div>
      <div class="arch-split">
        <div class="arch-box arch-out">Testa SKU<br><small>previsione 15 giorni</small></div>
        <div class="arch-box arch-out">Testa Categoria<br><small>previsione mensile</small></div>
      </div>
      <div class="arch-arrow">↓</div>
      <div class="arch-box arch-op">Layer operativo<br><small>giacenza → qty suggerita → azione</small></div>
    </div>

    <ul>
      <li><strong>Negozi nuovi / SKU nuovi:</strong> embedding dedicato «sconosciuto» (non visti in train); analoghi solo se lo SKU è giovane.</li>
      <li><strong>Commerciale di riferimento:</strong> embedding dedicato per cluster vendita umano.</li>
      <li><strong>Transit time:</strong> feature numerica nel modello + uso operativo per qty suggerita.</li>
      <li><strong>Attributi prodotto Valido:</strong> modello, tessuto, colore, campagna, classe, sclasse, ssclasse, tipologia.</li>
      <li><strong>Early stopping:</strong> validation su più chiusure (non 2 date); si ferma se non migliora (ultima run: {epochs} epoche).</li>
      <li><strong>Huber loss:</strong> robusta ai picchi anomali (saldi, eventi).</li>
    </ul>
  </section>"""


def _section_testing_vs_produzione(meta_d: dict, metrics: dict) -> str:
    n_pdv_test = meta_d.get("n_pdv", "—")
    n_sku = meta_d.get("n_sku", "—")
    win = int(metrics.get("train_window_days") or 0)
    n_train = metrics.get("n_train", "—")
    n_as_of = metrics.get("n_as_of_train", "—")
    if isinstance(n_train, int):
        n_train_txt = f"{n_train:,}"
    else:
        n_train_txt = str(n_train)
    win_txt = f"{win} giorni" if win > 0 else "tutto lo storico"

    return f"""
  <section>
    <h2>Testing vs Produzione — scala rete Subdued</h2>
    <p>Il report usa dati dimostrativi generati in ambiente <strong>testing</strong>,
    calibrati sulla rete reale Subdued/CDM (<strong>{PRODUCTION_N_PDV} negozi</strong>,
    <strong>{PRODUCTION_N_SKU}+ SKU</strong>).</p>

    <table>
      <thead><tr><th>Aspetto</th><th>Testing (questa run)</th><th>Produzione (target)</th></tr></thead>
      <tbody>
        <tr><td>Negozi (PDV)</td><td><strong>{n_pdv_test}</strong></td><td><strong>{PRODUCTION_N_PDV}</strong></td></tr>
        <tr><td>SKU</td><td><strong>{n_sku}</strong></td><td><strong>{PRODUCTION_N_SKU}+</strong> (catalogo Valido)</td></tr>
        <tr><td>Fonte dati</td><td>CSV fake con stagionalità</td><td>Valido + DWH + meteo API</td></tr>
        <tr><td>Feature rolling</td><td>7 / 14 / 28 gg passati</td><td>Identico</td></tr>
        <tr><td>Training rolling</td><td><code>train_window_days={win}</code> ({win_txt})</td><td>90–180 gg (da calibrare)</td></tr>
        <tr><td>Righe train (run corrente)</td><td>{n_train_txt} ({n_as_of} as_of)</td><td>~{PRODUCTION_N_PDV}×{PRODUCTION_N_SKU}×settimane finestra</td></tr>
        <tr><td>Refresh</td><td>Simulato su storico</td><td>Quindicinale SKU · mensile categoria (PDF CDM)</td></tr>
      </tbody>
    </table>

    <p class="note"><strong>Perché 150 negozi × 500 SKU cambia tutto:</strong> con pochi SKU (42) e finestra 90 gg
    avevamo ~70k righe train. Con {PRODUCTION_N_SKU} SKU la stessa finestra produce
    <strong>~1M righe</strong> in produzione — volume adatto al rolling training settimanale.
    Default testing: <code>n_skus={PRODUCTION_N_SKU}</code>, <code>train_window_days=90</code>.</p>

    <h3 class="sub">Allineamento documento CDM (Forecast Categoria &amp; SKU)</h3>
    <ul>
      <li>Dual-head categoria + SKU · embedding PDV/commerciale/attributi prodotto</li>
      <li>Feature rolling + training rolling · cold start con qty prudente</li>
      <li>Prossimi step: confidence score continuo, reconciliation somma SKU≈categoria, WAPE per bucket maturità</li>
    </ul>
  </section>"""


def _section_rolling(metrics: dict | None = None) -> str:
    metrics = metrics or {}
    win = int(metrics.get("train_window_days") or 0)
    n_as_of = metrics.get("n_as_of_train", "—")
    train_start = metrics.get("train_start") or "tutto lo storico"
    n_train = metrics.get("n_train", "—")
    if isinstance(n_train, int):
        n_train_txt = f"{n_train:,}"
    else:
        n_train_txt = str(n_train)
    win_label = f"{win} giorni" if win > 0 else "tutto lo storico (train_window_days = 0)"

    return f"""
  <section>
    <h2>Due tipi di «rolling» — non confonderli</h2>
    <p>Nel progetto «rolling» significa due cose diverse. Entrambe servono; non sono la stessa cosa.</p>

    <div class="concept-grid">
      <div class="concept-card">
        <h3>1. Feature rolling (sempre attivo)</h3>
        <p>Per ogni data di taglio <code>as_of</code> le feature usano <strong>solo vendite passate</strong>
        (7 / 14 / 28 giorni). Il target sono i <strong>15 giorni futuri</strong>.
        Zero leakage: il modello non «guarda» il futuro.</p>
      </div>
      <div class="concept-card">
        <h3>2. Training rolling (finestra mobile)</h3>
        <p>Opzionale: ogni ri-addestramento può usare solo gli <code>as_of</code> recenti
        (<code>train_window_days</code>) invece di tutto lo storico.
        Serve a non «portarsi dietro» pattern di collezioni / promo finite.</p>
      </div>
    </div>

    <h3 class="sub">Feature rolling — schema temporale</h3>
    <p>Ogni settimana il sistema si comporta come se fosse <strong>«oggi»</strong> (<code>as_of</code>):
    usa solo il passato per le X e i 15 giorni dopo come Y.</p>

    <div class="rolling-timeline" role="img" aria-label="Schema temporale feature rolling">
      <div class="rolling-labels">
        <span class="rolling-label rolling-label-past">Passato (feature)</span>
        <span class="rolling-label rolling-label-cut">as_of</span>
        <span class="rolling-label rolling-label-future">Futuro (target)</span>
      </div>
      <div class="rolling-bar">
        <div class="rolling-seg rolling-w28" title="28 giorni">28 gg</div>
        <div class="rolling-seg rolling-w14" title="14 giorni">14 gg</div>
        <div class="rolling-seg rolling-w7" title="7 giorni">7 gg</div>
        <div class="rolling-cut" title="Data di taglio">▲</div>
        <div class="rolling-seg rolling-horizon" title="Orizzonte previsione">15 gg → y</div>
      </div>
      <div class="rolling-legend">
        <span><i class="swatch swatch-28"></i>Run rate 28d, trend 14v28</span>
        <span><i class="swatch swatch-14"></i>Run rate 14d, sconto, volatilità, meteo</span>
        <span><i class="swatch swatch-7"></i>Run rate 7d, trend 7v14</span>
        <span><i class="swatch swatch-y"></i>Target: somma vendite (y_sku)</span>
      </div>
    </div>

    <table>
      <thead><tr><th>Finestra</th><th>Periodo</th><th>Cosa produce</th></tr></thead>
      <tbody>
        <tr><td><code>hist_28</code></td><td>[as_of − 27, as_of]</td><td>Run rate 28 giorni, trend 14 vs 28</td></tr>
        <tr><td><code>hist_14</code></td><td>[as_of − 13, as_of]</td><td>Run rate 14d, sconto medio, volatilità</td></tr>
        <tr><td><code>hist_7</code></td><td>[as_of − 6, as_of]</td><td>Run rate 7d, trend recente 7 vs 14</td></tr>
        <tr><td><code>future</code></td><td>(as_of, as_of + 15]</td><td><strong>y_sku</strong> — vendite da prevedere (label)</td></tr>
      </tbody>
    </table>

    <h3 class="sub">Training rolling — finestra mobile</h3>
    <p><strong>Importante:</strong> i 15 giorni sono l'<em>orizzonte di previsione</em>, non la lunghezza del dataset di training.
    Addestrare solo sugli «ultimi 15 giorni» darebbe 1–2 snapshot settimanali → troppo poco per ~370k parametri.</p>

    <div class="flow">
      <div class="flow-step"><span class="flow-num">A</span><strong>Storico completo</strong><br><small>tutti gli as_of generati</small></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><span class="flow-num">B</span><strong>Cut test</strong><br><small>ultime 3 settimane = holdout</small></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><span class="flow-num">C</span><strong>Finestra train</strong><br><small>{win_label}</small></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><span class="flow-num">D</span><strong>Ri-addestra</strong><br><small>da zero ogni settimana</small></div>
    </div>

    <table>
      <thead><tr><th>Parametro</th><th>Valore (ultima run)</th><th>Significato</th></tr></thead>
      <tbody>
        <tr><td><code>train_window_days</code></td><td><strong>{win if win else 0}</strong></td><td>0 = tutto lo storico; N = solo ultimi N giorni di as_of</td></tr>
        <tr><td>Inizio finestra train</td><td>{train_start}</td><td>Primo as_of usato per l'addestramento</td></tr>
        <tr><td>as_of nel train</td><td>{n_as_of}</td><td>Quante «settimane» di sample</td></tr>
        <tr><td>Righe train</td><td>{n_train_txt}</td><td>PDV × SKU × as_of nella finestra</td></tr>
      </tbody>
    </table>

    <table>
      <thead><tr><th>Scelta finestra</th><th>Pro</th><th>Contro</th></tr></thead>
      <tbody>
        <tr><td><strong>Tutto lo storico</strong> (default testing)</td><td>Impara stagionalità annuale, saldi, trend lunghi</td><td>Può trascinare regimi vecchi se il business cambia</td></tr>
        <tr><td><strong>90–180 giorni</strong></td><td>Più reattivo al presente</td><td>Con pochi PDV/SKU rischia overfitting (visto in test)</td></tr>
        <tr><td><strong>Solo 15 giorni</strong></td><td>—</td><td>Troppo poco: non usarlo come dataset di training</td></tr>
      </tbody>
    </table>

    <div class="concept-grid">
      <div class="concept-card">
        <h3>Modello CDM originale</h3>
        <p>Regole Subdued: run-rate 14gg × coeff stagionale (mediana+cap) × cluster 0.70/1.00/1.30
        × proxy PDV nuovo. È il confronto da battere al tavolo CDM.</p>
      </div>
      <div class="concept-card">
        <h3>Riassortimento</h3>
        <p><code>need = forecast + (forecast/15) × transit</code><br>
        <code>qty = max(0, need − giacenza)</code> · PDV/SKU nuovi: × 0,5</p>
      </div>
      <div class="concept-card">
        <h3>Cosa non succede</h3>
        <p>Il modello <strong>non</strong> ri-addestra sulle previsioni sbagliate del run precedente:
        usa sempre le <strong>vendite reali</strong> Valido come target.</p>
      </div>
      <div class="concept-card">
        <h3>In produzione</h3>
        <p>Ogni lunedì: as_of = oggi, feature da passato, train (finestra o full),
        forecast 15 gg, qty suggerita. Config: <code>train_window_days</code>.</p>
      </div>
    </div>

    <p class="note"><strong>Per Produzione:</strong> feature rolling = niente leakage (sempre).
    Training rolling = opzionale, da calibrare quando ci sono abbastanza negozi/SKU.
    I 15 giorni restano l'orizzonte di previsione, non la lunghezza del training.</p>
  </section>"""


def _badge(azione: str) -> str:
    if azione == "valuta_riassortimento":
        return '<span class="badge badge-ok">Valuta riassortimento</span>'
    return '<span class="badge badge-wait">Monitora</span>'


def _row_table(df: pd.DataFrame, pdv: pd.DataFrame) -> str:
    if df.empty:
        return "<p><em>Nessuna riga disponibile.</em></p>"

    city = pdv.set_index("pdv_id")["city"]
    rows = []
    for r in df.itertuples():
        rows.append(
            f"<tr>"
            f"<td>{city.get(r.pdv_id, r.pdv_id)}<br><small>{r.pdv_id}</small></td>"
            f"<td>{r.sku_id}</td>"
            f"<td class='num'>{r.vendite_previste_15gg:.1f}</td>"
            f"<td class='num'>{r.giacenza:.0f}</td>"
            f"<td class='num'><strong>{r.qty_suggerita:.0f}</strong></td>"
            f"<td>{_badge(r.azione)}</td>"
            f"</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Negozio</th><th>SKU</th><th>Vendite previste<br><small>prossimi 15 gg</small></th>"
        "<th>Giacenza</th><th>Qty suggerita</th><th>Azione</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _section_spiegazioni(focus: pd.DataFrame, pdv: pd.DataFrame) -> str:
    """Perché la previsione — narrativa + driver (non confronto naive)."""
    if focus.empty or "spiegazione_testuale" not in focus.columns:
        return (
            "<section><h2>Perché questa previsione</h2>"
            "<p><em>Esegui <code>run_train</code> (con spiegazioni) per popolare i casi.</em></p>"
            "</section>"
        )

    city = pdv.set_index("pdv_id")["city"] if "city" in pdv.columns else pd.Series(dtype=str)
    cards = []
    for r in focus.itertuples():
        luogo = city.get(r.pdv_id, r.pdv_id) if len(city) else r.pdv_id
        drivers_html = []
        for i in range(1, 5):
            lab = getattr(r, f"driver_{i}", None)
            if lab is None or (isinstance(lab, float) and pd.isna(lab)):
                continue
            delta = getattr(r, f"driver_{i}_delta", 0)
            direction = getattr(r, f"driver_{i}_dir", "")
            color = "#16a34a" if direction == "alza" else "#dc2626"
            arrow = "↑" if direction == "alza" else "↓"
            drivers_html.append(
                f'<li><span style="color:{color};font-weight:700">{arrow}</span> '
                f"{lab} <small>({delta:+.1f} pezzi)</small></li>"
            )
        drivers_block = (
            "<ul class='driver-list'>" + "".join(drivers_html) + "</ul>"
            if drivers_html
            else ""
        )
        azione = getattr(r, "azione", "")
        qty = getattr(r, "qty_suggerita", 0)
        pred = getattr(r, "pred_tf", getattr(r, "vendite_previste_15gg", 0))
        cards.append(
            f"""
            <div class="explain-card">
              <div class="explain-head">
                <strong>{luogo}</strong> · {r.sku_id}
                <span class="pill">{_badge(azione) if azione else ""}</span>
              </div>
              <div class="explain-metrics">
                Previsione <strong>{pred:.1f}</strong> pezzi / 15 gg
                · Giacenza {getattr(r, 'giacenza', 0):.0f}
                · Qty suggerita <strong>{qty:.0f}</strong>
              </div>
              <p class="explain-text">{getattr(r, 'spiegazione_testuale', '')}</p>
              {f'<div class="explain-drivers"><div class="chart-title">Cosa ha pesato di più</div>{drivers_block}</div>' if drivers_block else ''}
            </div>
            """
        )

    return f"""
    <section>
      <h2>Perché questa previsione</h2>
      <p>Qui non confrontiamo «noi vs baseline»: spieghiamo <strong>i pattern</strong> che il modello
      ha usato (andamento vendite, cold start, promo, meteo, capacità negozio…).
      I driver misurano quanto ogni blocco di informazioni <em>alza o abbassa</em> la stima
      rispetto a un profilo medio.</p>
      {''.join(cards)}
    </section>
    """


def _html_segment_table(metrics: dict) -> str:
    """MAE per attributo prodotto — non è la feature meteo."""
    rows = []
    for key, label in (
        ("seg_estate", "Prodotti ESTATE (crop, sandali…)"),
        ("seg_inverno", "Prodotti INVERNO (piumini, maglioni…)"),
        ("seg_continuativo", "Prodotti CONTINUATIVO (jeans, sneaker…)"),
    ):
        s = metrics.get(key) or {}
        if not s.get("mae"):
            continue
        rows.append(
            "<tr>"
            f"<td>{label}</td>"
            f"<td class='num'>{float(s['mae']):.2f}</td>"
            f"<td class='num'>{float(s.get('mae_cdm', s.get('mae_naive', 0))):.2f}</td>"
            f"<td class='num'>{float(s.get('improvement_pct', 0)):+.0f}%</td>"
            f"<td class='num'>{int(s.get('n', 0)):,}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return (
        "<h3 class='sub'>Errore per tipo prodotto (attributo SKU)</h3>"
        "<p>Questo è solo uno <strong>spezzamento del test</strong>, non un input del modello. "
        "Meteo e calendario sono già nelle feature di tutte le righe, jeans compresi.</p>"
        "<table><thead><tr>"
        "<th>Attributo <code>stagione</code></th><th>MAE rete</th>"
        "<th>MAE modello CDM</th><th>Lift vs CDM</th><th>N. righe test</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def build_html(
    meta: pd.DataFrame,
    metrics: dict,
    history: pd.DataFrame,
    focus: pd.DataFrame,
    priorities: pd.DataFrame,
    pdv: pd.DataFrame,
    generated_at: str | None = None,
) -> str:
    generated_at = generated_at or datetime.now().strftime("%d/%m/%Y %H:%M")
    meta_d = dict(zip(meta["metric"], meta["value"]))
    mae_tf = float(metrics["mae_tf"])
    mae_cdm = float(metrics.get("mae_cdm", metrics.get("mae_naive", mae_tf)))
    mae_naive = float(metrics.get("mae_naive", mae_cdm))
    improvement = float(
        metrics.get(
            "improvement_pct",
            (mae_cdm - mae_tf) / mae_cdm * 100 if mae_cdm else 0,
        )
    )
    mape_tf = float(metrics.get("mape_tf", 0))
    r2 = float(metrics.get("r2", 0))
    delta = mae_cdm - mae_tf
    delta_txt = f"{delta:+.2f} pezzi ({improvement:+.1f}%)"
    better = "migliore" if mae_tf < mae_cdm else "simile"
    mae_cold = metrics.get("mae_cold_sku_confidence")
    cold_card = (
        f'<div class="card"><div class="label">SKU nuovi</div><div class="value">{float(mae_cold):.1f}</div>'
        f'<div class="hint">errore pezzi / 15 gg</div></div>'
        if mae_cold is not None
        else ""
    )
    lead_card = (
        f'<div class="card highlight"><div class="label">Errore rete (tutti gli SKU)</div>'
        f'<div class="value">{mae_tf:.1f}</div>'
        f'<div class="hint">{improvement:+.0f}% vs modello CDM ({mae_cdm:.1f} pezzi)</div></div>'
    )
    cdm_card = (
        f'<div class="card"><div class="label">Modello CDM originale</div>'
        f'<div class="value">{mae_cdm:.1f}</div>'
        f'<div class="hint">regole Subdued · stesso test</div></div>'
    )

    chart_acc = _svg_bar_chart(
        ["Rete\nneurale", "Modello CDM\n(originale)"],
        [mae_tf, mae_cdm],
        "Accuratezza previsioni SKU — 15 giorni",
        "Errore medio (pezzi)",
        ["#2563eb", "#94a3b8"],
    )
    chart_train = ""
    if not history.empty and "y_sku_mae" in history.columns:
        chart_train = _svg_line_chart(
            list(range(1, len(history) + 1)),
            history["y_sku_mae"].tolist(),
            "Come migliora il modello durante l'apprendimento",
            "Errore SKU (MAE)",
        )
    chart_focus = _svg_grouped_bars(focus, pdv)

    prio_show = priorities[priorities["qty_suggerita"] > 0].head(20) if not priorities.empty else priorities
    n_riassort = int((priorities["azione"] == "valuta_riassortimento").sum()) if not priorities.empty else 0
    n_monitora = int((priorities["azione"] == "monitora").sum()) if not priorities.empty else 0

    return f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Report Domanda Attesa — Subdued/CDM</title>
  <style>
    :root {{
      --bg: #f8fafc; --card: #fff; --text: #0f172a; --muted: #64748b; --border: #e2e8f0;
    }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0; background: var(--bg); color: var(--text); line-height: 1.5; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 24px 20px 48px; }}
    h1 {{ font-size: 1.75rem; margin: 0 0 6px; }}
    .subtitle {{ color: var(--muted); font-size: 0.95rem; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; margin: 20px 0 28px; }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }}
    .card .label {{ font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.03em; }}
    .card .value {{ font-size: 1.55rem; font-weight: 700; margin-top: 4px; }}
    .card .hint {{ font-size: 0.8rem; color: var(--muted); margin-top: 4px; }}
    section {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px;
               padding: 20px 22px; margin-bottom: 18px; }}
    h2 {{ font-size: 1.12rem; margin: 0 0 12px; }}
    p {{ margin: 0 0 10px; }}
    ul {{ margin: 8px 0 0 18px; }}
    li {{ margin-bottom: 6px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
    th, td {{ border-bottom: 1px solid var(--border); padding: 10px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f1f5f9; font-weight: 600; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    small {{ color: var(--muted); }}
    .badge {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }}
    .badge-ok {{ background: #dcfce7; color: #166534; }}
    .badge-wait {{ background: #f1f5f9; color: #475569; }}
    .chart-svg {{ margin: 12px 0; padding: 12px; border: 1px solid var(--border); border-radius: 8px; background: #fafbfc; }}
    .chart-title {{ font-size: 0.9rem; font-weight: 600; margin-bottom: 8px; color: #334155; }}
    .legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px; font-size: 0.82rem; color: var(--muted); }}
    .legend-item i {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 5px; vertical-align: -2px; }}
    .note {{ background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 12px 14px; font-size: 0.9rem; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .grid3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 14px 0; }}
    h3.sub {{ font-size: 1rem; margin: 18px 0 8px; color: #334155; }}
    .concept-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 14px 0; }}
    .concept-card {{ background: #f8fafc; border: 1px solid var(--border); border-radius: 8px; padding: 14px; }}
    .concept-card h3 {{ font-size: 0.92rem; margin: 0 0 8px; }}
    .concept-card ul {{ margin: 6px 0 0 16px; font-size: 0.88rem; }}
    .flow {{ display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin: 16px 0; }}
    .flow-step {{ background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 10px 14px; font-size: 0.88rem; flex: 1; min-width: 140px; }}
    .flow-num {{ display: inline-block; background: #2563eb; color: #fff; width: 22px; height: 22px; border-radius: 50%;
                  text-align: center; line-height: 22px; font-size: 0.75rem; font-weight: 700; margin-right: 6px; }}
    .flow-arrow {{ color: var(--muted); font-size: 1.2rem; font-weight: 700; }}
    .pill {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }}
    .pill-cold {{ background: #dbeafe; color: #1e40af; }}
    .pill-mild {{ background: #dcfce7; color: #166534; }}
    .pill-hot {{ background: #fee2e2; color: #991b1b; }}
    .pill-rain {{ background: #e0e7ff; color: #3730a3; }}
    .season-table td.high {{ background: #dcfce7; font-weight: 600; }}
    .season-table td.mid {{ background: #fef9c3; }}
    .season-table td.low {{ background: #f1f5f9; color: var(--muted); }}
    .arch-diagram {{ text-align: center; margin: 16px 0; }}
    .arch-col {{ display: flex; gap: 10px; justify-content: center; margin-bottom: 8px; }}
    .arch-box {{ display: inline-block; padding: 10px 16px; border-radius: 8px; font-size: 0.88rem; margin: 4px; }}
    .arch-input {{ background: #f1f5f9; border: 1px solid var(--border); }}
    .arch-core {{ background: #dbeafe; border: 1px solid #93c5fd; font-weight: 600; }}
    .arch-out {{ background: #dcfce7; border: 1px solid #86efac; }}
    .arch-op {{ background: #fef3c7; border: 1px solid #fcd34d; }}
    .arch-arrow {{ color: var(--muted); font-size: 1.1rem; margin: 4px 0; }}
    .arch-split {{ display: flex; gap: 12px; justify-content: center; }}
    .rolling-timeline {{ margin: 18px 0; padding: 16px; border: 1px solid var(--border); border-radius: 8px; background: #fafbfc; }}
    .rolling-labels {{ display: flex; justify-content: space-between; font-size: 0.78rem; color: var(--muted); margin-bottom: 8px; padding: 0 4px; }}
    .rolling-label-cut {{ color: #dc2626; font-weight: 700; }}
    .rolling-bar {{ display: flex; align-items: stretch; height: 44px; border-radius: 6px; overflow: hidden; position: relative; border: 1px solid var(--border); }}
    .rolling-seg {{ display: flex; align-items: center; justify-content: center; font-size: 0.72rem; font-weight: 600; color: #fff; }}
    .rolling-w28 {{ flex: 28; background: #64748b; }}
    .rolling-w14 {{ flex: 14; background: #475569; }}
    .rolling-w7 {{ flex: 7; background: #334155; }}
    .rolling-horizon {{ flex: 15; background: #16a34a; border-left: 3px dashed #15803d; }}
    .rolling-cut {{ position: absolute; left: calc(100% * 49 / 64); top: -4px; transform: translateX(-50%); color: #dc2626; font-size: 1rem; line-height: 1; z-index: 2; }}
    .rolling-legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; font-size: 0.78rem; color: var(--muted); }}
    .swatch {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 4px; vertical-align: -2px; }}
    .swatch-28 {{ background: #64748b; }}
    .swatch-14 {{ background: #475569; }}
    .swatch-7 {{ background: #334155; }}
    .swatch-y {{ background: #16a34a; }}
    code {{ background: #f1f5f9; padding: 1px 5px; border-radius: 4px; font-size: 0.85em; }}
    .card.highlight {{ border-color: #2563eb; background: #eff6ff; }}
    .explain-card {{ border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; margin: 14px 0; background: #fafbfc; }}
    .explain-head {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 6px; }}
    .explain-metrics {{ font-size: 0.9rem; color: var(--muted); margin-bottom: 10px; }}
    .explain-text {{ font-size: 0.95rem; margin: 0 0 10px; }}
    .explain-drivers {{ margin-top: 8px; padding-top: 8px; border-top: 1px dashed var(--border); }}
    .driver-list {{ margin: 6px 0 0 4px; padding-left: 18px; font-size: 0.9rem; }}
    .driver-list li {{ margin-bottom: 4px; }}
    footer {{ color: var(--muted); font-size: 0.82rem; margin-top: 24px; text-align: center; }}
    @media (max-width: 760px) {{ .grid2, .grid3 {{ grid-template-columns: 1fr; }} .flow {{ flex-direction: column; }} }}
    @media print {{ body {{ background: #fff; }} section {{ break-inside: avoid; }} }}
  </style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Report Domanda Attesa &amp; Riassortimento</h1>
    <p class="subtitle">Generato il {generated_at} · Testing scalato a {meta_d.get("n_pdv", "—")} PDV
    (rete reale: {PRODUCTION_N_PDV} negozi)</p>
  </header>

  <div class="cards">
    {lead_card}
    {cdm_card}
    <div class="card"><div class="label">Domanda spiegata</div><div class="value">{r2*100:.0f}%</div><div class="hint">R² sul periodo di test</div></div>
    {cold_card}
    <div class="card"><div class="label">Negozi / SKU</div><div class="value">{meta_d.get("n_pdv", "—")} / {meta_d.get("n_sku", "—")}</div><div class="hint">rete reale: {PRODUCTION_N_PDV} PDV</div></div>
  </div>

  <section>
    <h2>In sintesi</h2>
    <p>Il sistema stima <strong>quanti pezzi di ogni SKU verranno venduti nei prossimi 15 giorni</strong> per negozio
    e propone una <strong>quantità di riassortimento</strong> in base alla giacenza.
    Calendario, temperatura e pioggia sono <strong>feature di input</strong> su ogni riga (come il run-rate), non un filtro del report.</p>
    <ul>
      <li><strong>Valuta riassortimento</strong> — fabbisogno stimato: il team verifica e decide l'ordine.</li>
      <li><strong>Monitora</strong> — stock sufficiente o negozio/prodotto «giovane»: nessun ordine suggerito.</li>
      <li>Ogni proposta include una <strong>spiegazione</strong>: quali pattern (vendite, cold start, promo, meteo…) hanno pesato.</li>
      <li>Solo <strong>vendite reali passate</strong> entrano nelle feature (feature rolling).</li>
    </ul>
    <p class="note" style="margin-top:14px"><strong>Nota:</strong> dati dimostrativi. In produzione: Valido + CSV PDV + meteo.</p>
  </section>

  {_section_spiegazioni(focus, pdv)}

  <section>
    <h2>Priorità riassortimento</h2>
    <p>Top 20 per quantità suggerita (solo proposte &gt; 0). Da valutare: <strong>{n_riassort}</strong> · In monitoraggio: <strong>{n_monitora}</strong></p>
    {_row_table(prio_show, pdv) if not prio_show.empty else "<p><em>Esegui run_train per popolare la lista completa.</em></p>"}
  </section>

  <section>
    <h2>Confronto con il modello CDM originale</h2>
    <p>Sul periodo di test la rete è <strong>{better}</strong> del modello a regole Subdued
    (coeff stagionali mediana+cap, cluster 0.70/1.00/1.30, proxy PDV): {delta_txt}.
    R² rete = <strong>{r2*100:.0f}%</strong>.</p>
    <p class="note">Questo è il confronto da portare al tavolo CDM. Il naive
    <code>run_rate × 15</code> resta un secondo controllo (MAE {mae_naive:.2f}).</p>
    {_html_segment_table(metrics)}
    <div class="grid2">{chart_acc}{chart_train}</div>
  </section>

  {_section_testing_vs_produzione(meta_d, metrics)}

  {_section_rolling(metrics)}

  {_section_architettura(metrics)}

  {_section_stagionalita()}

  {_section_sconto()}

  {_section_meteo()}

  {_section_embedding_cold_start()}

  <section>
    <h2>Casi esempio (numeri)</h2>
    <p>Stessi casi della sezione spiegazioni, in forma tabellare.</p>
    {chart_focus}
    {_row_table(focus, pdv)}
  </section>

  <section>
    <h2>Glossario</h2>
    <table>
      <tr><th>Termine</th><th>Per Produzione</th></tr>
      <tr><td>Vendite previste 15 gg</td><td>Pezzi attesi nel prossimo fortnight.</td></tr>
      <tr><td>Giacenza</td><td>Stock oggi in negozio.</td></tr>
      <tr><td>Qty suggerita</td><td>Ordine indicativo (decisione sempre umana).</td></tr>
      <tr><td>MAE</td><td>Errore medio in pezzi — più basso = più preciso.</td></tr>
      <tr><td>MAPE</td><td>Errore percentuale — utile per confrontare SKU con volumi diversi.</td></tr>
      <tr><td>R²</td><td>Quanto il modello spiega la variabilità (0–1, più alto = meglio).</td></tr>
      <tr><td>Modello CDM originale</td><td>Regole Subdued: run-rate × coeff stagionale (mediana+cap) × cluster 0.70/1.00/1.30 × proxy PDV — confronto ufficiale.</td></tr>
      <tr><td>Baseline naive</td><td>Solo media vendite ultimi 14 giorni × 15 — secondo controllo minimo.</td></tr>
      <tr><td>as_of</td><td>Data di taglio: «oggi» — passato = feature, futuro = target (15 gg).</td></tr>
      <tr><td>Feature rolling</td><td>Finestre 7/14/28 gg prima di as_of — niente leakage.</td></tr>
      <tr><td>Training rolling</td><td>Ri-addestramento su finestra mobile (default 90 gg) — non trascinare regimi vecchi.</td></tr>
      <tr><td>train_window_days</td><td>Quanti giorni di as_of tenere nel train (non confondere con l'orizzonte 15 gg).</td></tr>
      <tr><td>Run rate</td><td>Media pezzi/giorno su finestra passata (7, 14 o 28 giorni).</td></tr>
      <tr><td>Transit time</td><td>Giorni di lead time — usato nel calcolo qty suggerita.</td></tr>
      <tr><td>Embedding</td><td>Profilo numerico appreso per negozio/SKU — gestisce il cold start.</td></tr>
      <tr><td>Sconto %</td><td>Differenza tra prezzo listino e pagato — quanto la promo spinge le vendite.</td></tr>
      <tr><td>Cold start</td><td>Negozio o SKU senza storico — previsione prudente, qty dimezzata.</td></tr>
    </table>
  </section>

  <footer>
    modello-forecast · TensorFlow · {meta_d.get("date_min", "—")} → {meta_d.get("date_max", "—")}
    · {int(metrics.get("n_train", 0)):,} training · {int(metrics.get("n_test", 0)):,} test
  </footer>
</div>
</body>
</html>"""


def load_report_inputs(data_dir: Path = DATA_DIR) -> dict:
    meta = pd.read_csv(data_dir / "dataset_meta.csv")
    pdv = pd.read_csv(data_dir / "pdv_master.csv")
    focus = pd.read_csv(data_dir / "focus_casi_tf.csv") if (data_dir / "focus_casi_tf.csv").exists() else pd.DataFrame()
    priorities = (
        pd.read_csv(data_dir / "priorita_riassortimento.csv")
        if (data_dir / "priorita_riassortimento.csv").exists()
        else pd.DataFrame()
    )
    history = (
        pd.read_csv(data_dir / "training_history.csv") if (data_dir / "training_history.csv").exists() else pd.DataFrame()
    )
    metrics_path = data_dir / "report_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return {
        "meta": meta,
        "pdv": pdv,
        "focus": focus,
        "priorities": priorities,
        "history": history,
        "metrics": metrics,
    }


def write_report(output_path: Path | None = None, data_dir: Path = DATA_DIR) -> Path:
    inputs = load_report_inputs(data_dir)
    if not inputs["metrics"]:
        raise FileNotFoundError(
            f"Mancano artefatti in {data_dir}. Esegui: .venv/bin/python -m testing.run_train"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or REPORT_DIR / "report_produzione.html"
    html = build_html(
        meta=inputs["meta"],
        metrics=inputs["metrics"],
        history=inputs["history"],
        focus=inputs["focus"],
        priorities=inputs["priorities"],
        pdv=inputs["pdv"],
    )
    out.write_text(html, encoding="utf-8")
    return out
