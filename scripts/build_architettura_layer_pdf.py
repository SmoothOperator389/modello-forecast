#!/usr/bin/env python3
"""PDF didattico: struttura layer Dual-head TF + LightGBM (fedele al codice)."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parents[1] / "reports" / "CDM_architettura_layer_modelli.pdf"
W, H = A4

# palette
INK = (0.12, 0.14, 0.18)
MUTED = (0.35, 0.40, 0.48)
LINE = (0.55, 0.60, 0.68)
ACCENT = (0.15, 0.40, 0.65)
ACCENT2 = (0.10, 0.45, 0.40)
FILL = (0.93, 0.95, 0.98)
FILL_HI = (0.86, 0.92, 0.98)
FILL_GR = (0.86, 0.94, 0.91)
WHITE = (1, 1, 1)


def _rgb(c, r, g, b):
    c.setFillColorRGB(r, g, b)
    c.setStrokeColorRGB(r, g, b)


def box(c, x, y, w, h, title, lines=None, fill=FILL, stroke=LINE, title_color=INK, lw=1.2):
    c.setStrokeColorRGB(*stroke)
    c.setFillColorRGB(*fill)
    c.setLineWidth(lw)
    c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
    c.setFillColorRGB(*title_color)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(x + w / 2, y + h - 13, title)
    if lines:
        c.setFillColorRGB(*MUTED)
        c.setFont("Helvetica", 7.5)
        ty = y + h - 26
        for ln in lines:
            c.drawCentredString(x + w / 2, ty, ln)
            ty -= 10


def arrow_down(c, x, y0, y1):
    c.setStrokeColorRGB(*LINE)
    c.setFillColorRGB(*LINE)
    c.setLineWidth(1.2)
    c.line(x, y0, x, y1 + 5)
    path = c.beginPath()
    path.moveTo(x, y1)
    path.lineTo(x - 4, y1 + 7)
    path.lineTo(x + 4, y1 + 7)
    path.close()
    c.drawPath(path, fill=1, stroke=0)


def arrow_h(c, x0, x1, y):
    c.setStrokeColorRGB(*LINE)
    c.setFillColorRGB(*LINE)
    c.setLineWidth(1.2)
    c.line(x0, y, x1 - 5 if x1 > x0 else x1 + 5, y)
    path = c.beginPath()
    if x1 > x0:
        path.moveTo(x1, y)
        path.lineTo(x1 - 7, y + 4)
        path.lineTo(x1 - 7, y - 4)
    else:
        path.moveTo(x1, y)
        path.lineTo(x1 + 7, y + 4)
        path.lineTo(x1 + 7, y - 4)
    path.close()
    c.drawPath(path, fill=1, stroke=0)


def header(c, title, subtitle):
    c.setFillColorRGB(*ACCENT)
    c.rect(0, H - 18 * mm, W, 18 * mm, fill=1, stroke=0)
    c.setFillColorRGB(*WHITE)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(14 * mm, H - 9 * mm, title)
    c.setFont("Helvetica", 8.5)
    c.drawString(14 * mm, H - 14.5 * mm, subtitle)


def footer(c, page, total):
    c.setFillColorRGB(*MUTED)
    c.setFont("Helvetica", 8)
    c.drawString(14 * mm, 8 * mm, "fonte: testing/model/dual_head_tf.py · production/run_train_horizon_native_real.py")
    c.drawRightString(W - 14 * mm, 8 * mm, f"{page}/{total}")


def page_dual_head(c):
    header(
        c,
        "Dual-head TensorFlow — struttura dei layer",
        "Ogni box = layer Keras reale · flusso dall’alto verso il basso",
    )
    margin = 12 * mm
    usable_w = W - 2 * margin

    # --- INPUTS ---
    c.setFillColorRGB(*ACCENT)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin, H - 26 * mm, "1 · INGRESSI")

    iw = (usable_w - 9 * mm) / 4
    y_in = H - 52 * mm
    ih = 22 * mm
    boxes_in = [
        ("Embedding categorici", ["13 Input int32", "Embedding(n, dim)→Flatten", "sku:12 · pdv:8 · …"]),
        ("Bande / regimi", ["mq_espo · mq_stock", "weather_regime", "Embedding band"]),
        ("Numeriche", [f"~{57} float + z-score", "BatchNorm  num_bn", "RR, meteo, sconto…"]),
        ("sku_prior", ["run_rate_14d × H", "salta il tronco", "→ Multiply in testa"]),
    ]
    fills = [FILL, FILL, FILL, FILL_HI]
    strokes = [LINE, LINE, LINE, ACCENT]
    xs = []
    for i, ((t, lines), fill, st) in enumerate(zip(boxes_in, fills, strokes)):
        x = margin + i * (iw + 3 * mm)
        xs.append(x + iw / 2)
        box(c, x, y_in, iw, ih, t, lines, fill=fill, stroke=st, lw=1.8 if i == 3 else 1.2)

    # prior dashed side
    c.setStrokeColorRGB(*ACCENT)
    c.setDash(3, 2)
    c.setLineWidth(1.2)
    prior_x = xs[3]
    c.line(prior_x, y_in, prior_x, 78 * mm)
    c.setDash()
    c.setFillColorRGB(*ACCENT)
    c.setFont("Helvetica", 7)
    c.saveState()
    c.translate(prior_x + 4, 130 * mm)
    c.rotate(90)
    c.drawString(0, 0, "prior (bypass tronco)")
    c.restoreState()

    # concat
    y_cat = y_in - 18 * mm
    for x in xs[:3]:
        arrow_down(c, x, y_in, y_cat + 14 * mm)
    box(
        c,
        margin + usable_w * 0.18,
        y_cat,
        usable_w * 0.55,
        14 * mm,
        "Concatenate",
        ["embeddings ∥ numeric_bn  →  vettore condiviso"],
        fill=FILL,
    )
    arrow_down(c, margin + usable_w * 0.455, y_cat, y_cat - 6 * mm)

    # --- TRUNK ---
    c.setFillColorRGB(*ACCENT)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin, y_cat - 10 * mm, "2 · TRONCO RESIDUALE CONDIVISO")

    y_r = y_cat - 36 * mm
    rh = 20 * mm
    rw = usable_w * 0.62
    rx = margin + (usable_w - rw) / 2 - 8 * mm
    residuals = [
        ("Residual block  #1  ·  256", ["Dense(256)→BN→ReLU→Drop(0.15)", "Dense(256)→BN → Add(skip) → ReLU"]),
        ("Residual block  #2  ·  256", ["Dense(256)→BN→ReLU→Drop(0.15)", "Dense(256)→BN → Add(skip) → ReLU"]),
        ("Residual block  #3  ·  128", ["Dense(128)→BN→ReLU→Drop(0.10)", "Dense(128)→BN → Add(skip*) → ReLU"]),
    ]
    for i, (t, lines) in enumerate(residuals):
        box(c, rx, y_r, rw, rh, t, lines, fill=FILL_HI, stroke=ACCENT, lw=1.5)
        if i < 2:
            arrow_down(c, rx + rw / 2, y_r, y_r - 5 * mm)
            y_r -= 25 * mm
        else:
            y_r_last = y_r

    # zoom residual on the right
    zx = rx + rw + 4 * mm
    zw = W - margin - zx
    zy = y_cat - 36 * mm - 2 * 25 * mm
    box(
        c,
        zx,
        zy,
        zw,
        52 * mm,
        "Zoom: un residual block",
        [
            "x ──┐",
            "    Dense → BN → ReLU → Drop",
            "    Dense → BN",
            "x ──┴── Add → ReLU → out",
            "",
            "* se dim cambia: Dense",
            "  sulla shortcut (no bias)",
        ],
        fill=WHITE,
        stroke=ACCENT,
        lw=1.5,
    )

    # fork
    fork_y = y_r_last - 8 * mm
    arrow_down(c, rx + rw / 2, y_r_last, fork_y + 2 * mm)
    c.setStrokeColorRGB(*LINE)
    c.line(margin + 28 * mm, fork_y, margin + usable_w - 28 * mm, fork_y)

    c.setFillColorRGB(*ACCENT)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin, fork_y - 5 * mm, "3 · DUE TESTE")

    # SKU head
    hw = usable_w * 0.46
    hy = fork_y - 58 * mm
    hh = 48 * mm
    arrow_down(c, margin + hw / 2, fork_y, hy + hh)
    box(
        c,
        margin,
        hy,
        hw,
        hh,
        "TESTA SKU  (output operativo)",
        [
            "Dense(64) ReLU → Drop(0.10)",
            "├─ sku_scale  Softplus  (init ≈ 1)",
            "└─ sku_delta  lineare   (init = 0)",
            "",
            "y_sku = ReLU( prior × scale + delta )",
            "Loss Huber δ=5 · peso 1.0",
        ],
        fill=FILL_HI,
        stroke=ACCENT,
        lw=1.8,
    )

    # CAT head
    arrow_down(c, margin + usable_w - hw / 2, fork_y, hy + hh)
    box(
        c,
        margin + usable_w - hw,
        hy,
        hw,
        hh,
        "TESTA CATEGORIA  (solo train)",
        [
            "Lambda stop_gradient(x)",
            "  → non aggiorna il tronco",
            "Dense(64) ReLU → Drop(0.10)",
            "y_cat = Softplus( Dense(1) )",
            "",
            "target: log1p(totale cat)",
            "Loss Huber δ=1 · peso 0.15",
        ],
        fill=FILL,
        stroke=LINE,
        lw=1.2,
    )

    # formula bar
    c.setFillColorRGB(*FILL_HI)
    c.setStrokeColorRGB(*ACCENT)
    c.roundRect(margin, 16 * mm, usable_w, 12 * mm, 3, fill=1, stroke=1)
    c.setFillColorRGB(*INK)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(W / 2, 22.5 * mm, "Formula SKU:   pezzi = ReLU(  sku_prior × softplus(scala)  +  delta  )")

    footer(c, 1, 2)


def page_lightgbm(c):
    header(
        c,
        "LightGBM — struttura dell’ensemble ad alberi",
        "Non è una rete a layer: è una foresta di alberi di decisione addestrata in boosting",
    )
    margin = 12 * mm
    usable_w = W - 2 * margin

    # intro
    c.setFillColorRGB(*INK)
    c.setFont("Helvetica", 8.5)
    c.drawString(margin, H - 26 * mm, "Ogni albero corregge l’errore dei precedenti. In deploy ci sono DUE booster + un mix.")

    # INPUT
    c.setFillColorRGB(*ACCENT2)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin, H - 34 * mm, "1 · MATRICE FEATURE  (stesso schema del TF, ma tabulari)")

    y0 = H - 58 * mm
    box(
        c,
        margin,
        y0,
        usable_w,
        18 * mm,
        "X = NUMERIC_COLS (~57)  +  sku_prior  +  CAT_COLS (categorie + bande mq/meteo)",
        [
            "categorie passate come categorical_feature (split su livelli, non one-hot)",
            "target y = pezzi sull’orizzonte H (15 / 30 / 60 / 90 giorni)",
        ],
        fill=FILL_GR,
        stroke=ACCENT2,
        lw=1.5,
    )
    arrow_down(c, W / 2, y0, y0 - 6 * mm)

    # TWO MODELS
    c.setFillColorRGB(*ACCENT2)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin, y0 - 12 * mm, "2 · DUE MODELLI LGBM  (joblib)")

    mw = (usable_w - 6 * mm) / 2
    my = y0 - 68 * mm
    mh = 50 * mm
    arrow_down(c, margin + mw / 2, y0 - 6 * mm, my + mh)
    arrow_down(c, margin + usable_w - mw / 2, y0 - 6 * mm, my + mh)

    box(
        c,
        margin,
        my,
        mw,
        mh,
        "A · Tweedie  (globale)",
        [
            "objective = tweedie  (p=1.3)",
            "n_estimators ≤ 2500  (early stop)",
            "num_leaves = 95",
            "lr = 0.04 · λ = 1.2",
            "subsample / colsample = 0.85",
            "",
            "pred_tw = max(predict(X), 0)",
            "copre tutta la distribuzione",
            "(molti zeri + coda lunga)",
        ],
        fill=FILL_GR,
        stroke=ACCENT2,
        lw=1.8,
    )
    box(
        c,
        margin + mw + 6 * mm,
        my,
        mw,
        mh,
        "B · High-volume  (specialista)",
        [
            "objective = regression  (su log1p)",
            "train solo se prior ≥ soglia",
            "n_estimators ≤ 1500 · leaves=63",
            "lr = 0.05",
            "",
            "pred_hi = expm1(predict(X))",
            "affina i volumi alti",
            "(dove Tweedie sottostima)",
        ],
        fill=FILL_GR,
        stroke=ACCENT2,
        lw=1.8,
    )

    # TREE anatomy
    ty = my - 48 * mm
    arrow_down(c, W / 2, my, ty + 38 * mm)
    box(
        c,
        margin,
        ty,
        usable_w,
        38 * mm,
        "Anatomia di UN albero LightGBM  (leaf-wise)",
        [
            "radice  →  split su feature (es. run_rate_14d < 0.8  |  sku_id ∈ {…}  |  temp_media ≥ 22)",
            "         →  foglie con valore numerico (contributo additivo alla previsione)",
            "",
            "predizione finale di un booster = somma dei contributi di tutti gli alberi",
            "F(x) = F₀ + Σₖ  η · treeₖ(x)     η = learning_rate",
            "early stopping: si ferma quando la MAE di validazione non migliora (100 / 80 round)",
        ],
        fill=WHITE,
        stroke=ACCENT2,
        lw=1.5,
    )

    # mini tree drawing
    # draw a small schematic tree inside/under? Better as separate visual on left of anatomy - already full width.
    # Add a small illustrative tree to the right of page near anatomy - actually space is tight.
    # Draw tiny tree icons above the two models instead.

    # draw small tree glyphs next to section title area
    def tiny_tree(cx, cy, color):
        c.setStrokeColorRGB(*color)
        c.setFillColorRGB(*color)
        c.setLineWidth(1)
        c.circle(cx, cy + 18, 3, fill=1, stroke=0)
        c.line(cx, cy + 15, cx - 10, cy + 5)
        c.line(cx, cy + 15, cx + 10, cy + 5)
        c.circle(cx - 10, cy + 5, 2.5, fill=1, stroke=0)
        c.circle(cx + 10, cy + 5, 2.5, fill=1, stroke=0)
        c.line(cx - 10, cy + 2.5, cx - 16, cy - 6)
        c.line(cx - 10, cy + 2.5, cx - 4, cy - 6)
        c.line(cx + 10, cy + 2.5, cx + 4, cy - 6)
        c.line(cx + 10, cy + 2.5, cx + 16, cy - 6)
        for lx in (cx - 16, cx - 4, cx + 4, cx + 16):
            c.rect(lx - 3, cy - 12, 6, 5, fill=1, stroke=0)

    tiny_tree(margin + mw / 2, my + mh + 2, ACCENT2)
    tiny_tree(margin + mw + 6 * mm + mw / 2, my + mh + 2, ACCENT2)

    # BLEND
    c.setFillColorRGB(*ACCENT2)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin, ty - 8 * mm, "3 · BLEND DI DEPLOY  (nativi h30/h60/h90 e ramo LGBM del 15g)")

    by = 22 * mm
    arrow_down(c, W / 2, ty, by + 28 * mm)
    box(
        c,
        margin,
        by,
        usable_w,
        28 * mm,
        "Mix volume-aware",
        [
            "prior = run_rate_14d × H",
            "se prior basso → solo Tweedie",
            "se prior alto  →  0.55·Tweedie + 0.45·High   e   floor ≥ 0.35·naive",
            "Sul 15g l’ensemble completo aggiunge anche Dual-head + routing calibrato per fascia di prior",
        ],
        fill=FILL_GR,
        stroke=ACCENT2,
        lw=1.8,
    )

    footer(c, 2, 2)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=A4)
    page_dual_head(c)
    c.showPage()
    page_lightgbm(c)
    c.save()
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
