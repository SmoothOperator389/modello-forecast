#!/usr/bin/env python3
"""PDF: flusso dato grezzo → layer/neuroni → output (Dual-head)."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parents[1] / "reports" / "CDM_flusso_dato_neuroni.pdf"
W, H = A4

INK = (0.08, 0.12, 0.17)
MUTED = (0.36, 0.42, 0.48)
LINE = (0.66, 0.70, 0.76)
NN = (0.11, 0.31, 0.54)
NNF = (0.86, 0.92, 0.97)
RAW = (0.54, 0.35, 0.17)
RAWF = (0.97, 0.94, 0.89)
OUTC = (0.04, 0.43, 0.31)
OUTF = (0.87, 0.95, 0.92)
WHITE = (1, 1, 1)


def box(c, x, y, w, h, title, lines=None, fill=NNF, stroke=NN, lw=1.2, title_sz=9):
    c.setStrokeColorRGB(*stroke)
    c.setFillColorRGB(*fill)
    c.setLineWidth(lw)
    c.roundRect(x, y, w, h, 3.5, fill=1, stroke=1)
    c.setFillColorRGB(*stroke if stroke != LINE else INK)
    c.setFont("Helvetica-Bold", title_sz)
    c.drawCentredString(x + w / 2, y + h - 12, title)
    if lines:
        c.setFillColorRGB(*MUTED)
        c.setFont("Helvetica", 7.2)
        ty = y + h - 24
        for ln in lines:
            c.drawCentredString(x + w / 2, ty, ln)
            ty -= 9.5


def arrow_v(c, x, y0, y1):
    c.setStrokeColorRGB(*LINE)
    c.setFillColorRGB(*LINE)
    c.setLineWidth(1.1)
    c.line(x, y0, x, y1 + 4)
    p = c.beginPath()
    p.moveTo(x, y1)
    p.lineTo(x - 3.5, y1 + 6)
    p.lineTo(x + 3.5, y1 + 6)
    p.close()
    c.drawPath(p, fill=1, stroke=0)


def neurons(c, x, y, n_show, label):
    c.setFillColorRGB(*NN)
    for i in range(n_show):
        c.circle(x + i * 7, y, 2.4, fill=1, stroke=0)
    c.setFillColorRGB(*MUTED)
    c.setFont("Helvetica", 7.5)
    c.drawString(x + n_show * 7 + 4, y - 2.5, label)


def header(c, title, sub):
    c.setFillColorRGB(*NN)
    c.rect(0, H - 16 * mm, W, 16 * mm, fill=1, stroke=0)
    c.setFillColorRGB(*WHITE)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(12 * mm, H - 7.5 * mm, title)
    c.setFont("Helvetica", 8)
    c.drawString(12 * mm, H - 12.5 * mm, sub)


def footer(c, n, tot):
    c.setFillColorRGB(*MUTED)
    c.setFont("Helvetica", 7.5)
    c.drawString(12 * mm, 7 * mm, "dual_head_tf.py · build_features.py · frame_to_tf_inputs")
    c.drawRightString(W - 12 * mm, 7 * mm, f"{n}/{tot}")


def page1(c):
    header(c, "Dal dato grezzo all’ingresso della rete", "Prima dei neuroni: aggregazioni causali fino ad as_of")
    m = 11 * mm
    uw = W - 2 * m

    c.setFillColorRGB(*RAW)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(m, H - 24 * mm, "1 · FONTI GREZZE")

    cols = [
        ("scontrini", ["pezzi, prezzo,", "sconto · data < as_of"]),
        ("products", ["modello, tessuto,", "colore, categoria"]),
        ("pdv_master", ["mq, country,", "city, apertura"]),
        ("meteo", ["temp, pioggia", "finestra 14g"]),
        ("calendar", ["festivi, saldi,", "mese, weekend"]),
        ("giacenze", ["stock a as_of", "listino"]),
    ]
    cw = (uw - 5 * 2.5 * mm) / 6
    y0 = H - 48 * mm
    for i, (t, lines) in enumerate(cols):
        box(c, m + i * (cw + 2.5 * mm), y0, cw, 20 * mm, t, lines, fill=RAWF, stroke=RAW, title_sz=8)

    arrow_v(c, W / 2, y0, y0 - 7 * mm)

    c.setFillColorRGB(*NN)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(m, y0 - 12 * mm, "2 · FEATURE ENGINEERING  →  1 riga PDV × SKU × as_of")
    box(
        c,
        m,
        y0 - 40 * mm,
        uw,
        24 * mm,
        "build_training_frames",
        [
            "run_rate 7/14/28 · trend · volatilità · is_new_sku · sconto · mq bande",
            "profilo meteo SKU · calendario · sku_prior = run_rate_14d × 15",
            "niente vendite future: solo storico ≤ as_of (no leakage)",
        ],
        fill=(0.91, 0.93, 0.96),
        stroke=(0.24, 0.35, 0.50),
    )

    arrow_v(c, W / 2, y0 - 40 * mm, y0 - 48 * mm)

    c.setFillColorRGB(*NN)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(m, y0 - 54 * mm, "3 · TENSORI PER LA RETE  (frame_to_tf_inputs)")

    tw = (uw - 8 * mm) / 3
    ty = y0 - 86 * mm
    box(
        c,
        m,
        ty,
        tw,
        28 * mm,
        "13+3 Embedding input",
        ["ciascuno shape (1,) int32", "stringa → indice encoder", "UNK = 0 se sconosciuto"],
        fill=NNF,
        stroke=NN,
        lw=1.5,
    )
    box(
        c,
        m + tw + 4 * mm,
        ty,
        tw,
        28 * mm,
        "numeric",
        ["shape (57,) float32", "(x − μ) / σ del train", "poi BatchNorm in rete"],
        fill=NNF,
        stroke=NN,
        lw=1.5,
    )
    box(
        c,
        m + 2 * (tw + 4 * mm),
        ty,
        tw,
        28 * mm,
        "sku_prior",
        ["shape (1,) float32", "RR_14d × 15", "salta il tronco → Multiply"],
        fill=OUTF,
        stroke=OUTC,
        lw=1.8,
    )

    # shape strip
    c.setFillColorRGB(*INK)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(m, 52 * mm, "Dimensioni embedding (neuroni virtuali dopo lookup)")
    c.setFont("Helvetica", 7.5)
    c.setFillColorRGB(*MUTED)
    rows = [
        "sku→12  pdv→8  country→6  cat→4  commerciale→4  modello→6  tessuto→4  colore→6",
        "campagna→4  classe→4  sclasse→4  ssclasse→4  tipologia→3  mq×2→8  weather→4",
        "SOMMA embedding = 81   +   numeric BatchNorm = 57   →   CONCAT = 138 attivazioni",
    ]
    yy = 44 * mm
    for r in rows:
        c.drawString(m, yy, r)
        yy -= 9

    c.setFillColorRGB(*OUTF)
    c.setStrokeColorRGB(*OUTC)
    c.roundRect(m, 16 * mm, uw, 12 * mm, 3, fill=1, stroke=1)
    c.setFillColorRGB(*OUTC)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(W / 2, 22.5 * mm, "Entra in rete un vettore di 138 numeri (+ prior a parte) per ogni forecast")
    footer(c, 1, 2)


def page2(c):
    header(c, "Layer, neuroni e passaggio fino a y_sku", "Shape: 138 → 256 → 256 → 128 → 64 → 1  (+ prior)")
    m = 11 * mm
    uw = W - 2 * m

    # layers stacked
    layers = [
        ("L1  Concatenate", ["81 emb ∥ 57 num"], "138", NNF, NN),
        (
            "L2  Residual #1 · Dense 256",
            ["Dense→BN→ReLU→Drop(0.15)→Dense→BN→Add→ReLU", "ogni neurone: somma pesata + bias"],
            "256",
            NNF,
            NN,
        ),
        ("L3  Residual #2 · Dense 256", ["stessa struttura · raffina x"], "256", NNF, NN),
        (
            "L4  Residual #3 · Dense 128",
            ["compressione 256→128 · shortcut proiettata"],
            "128",
            NNF,
            NN,
        ),
    ]

    y = H - 28 * mm
    for title, lines, dim, fill, stroke in layers:
        h = 18 * mm if len(lines) > 1 else 14 * mm
        box(c, m + 8 * mm, y - h, uw - 28 * mm, h, title, lines, fill=fill, stroke=stroke, lw=1.4)
        c.setFillColorRGB(*NN)
        c.setFont("Helvetica-Bold", 14)
        c.drawRightString(W - m, y - h / 2 - 4, dim)
        neurons(c, m + 14 * mm, y - h + 5, 12 if dim != "138" else 10, f"×{dim}")
        arrow_v(c, W / 2 - 10 * mm, y - h, y - h - 4 * mm)
        y = y - h - 5 * mm

    # fork heads
    c.setStrokeColorRGB(*LINE)
    c.line(m + 30 * mm, y + 1 * mm, W - m - 30 * mm, y + 1 * mm)
    hw = (uw - 6 * mm) / 2
    hh = 42 * mm
    hy = y - hh - 2 * mm

    box(
        c,
        m,
        hy,
        hw,
        hh,
        "L5a  Testa SKU  (usata)",
        [
            "Dense(64) ReLU + Drop(0.10)",
            "→ sku_scale Softplus (1 neurone, init≈1)",
            "→ sku_delta lineare (1 neurone, init=0)",
            "y_sku = ReLU( prior × scale + delta )",
        ],
        fill=NNF,
        stroke=NN,
        lw=1.8,
    )
    neurons(c, m + 10 * mm, hy + 6, 10, "×64 → 1+1")

    box(
        c,
        m + hw + 6 * mm,
        hy,
        hw,
        hh,
        "L5b  Testa categoria  (solo train)",
        [
            "stop_gradient(x) — tronco congelato",
            "Dense(64) ReLU + Drop",
            "→ Softplus(1) = y_cat",
            "loss weight 0.15 · non usato in inferenza",
        ],
        fill=(0.93, 0.94, 0.96),
        stroke=LINE,
        lw=1.2,
    )

    # example strip
    c.setFillColorRGB(*OUTC)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(m, hy - 8 * mm, "Esempio numerico")
    box(
        c,
        m,
        16 * mm,
        uw,
        hy - 12 * mm - 16 * mm,
        "RR_14d=0.8  →  prior=12   ·   rete stima scale=1.15  delta=+1.2",
        [
            "y_sku = ReLU(12 × 1.15 + 1.2) = 15.0 pezzi sui prossimi 15 giorni",
            "Lettura: parte dal ritmo recente (12), alza +15% e aggiunge +1.2 pezzi di contesto",
            "All’init della rete: scale≈1 e delta=0 → previsione = naive (prior)",
        ],
        fill=OUTF,
        stroke=OUTC,
        lw=1.6,
        title_sz=8.5,
    )
    footer(c, 2, 2)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=A4)
    page1(c)
    c.showPage()
    page2(c)
    c.save()
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
