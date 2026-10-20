#!/usr/bin/env python3
"""Video didattico: dato grezzo → Dual-head TF → y_sku.

Usa l'esempio numerico reale (prior×scale+delta) già validato sul modello.
"""

from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "CDM_video_flusso_rete_neurale.mp4"
FPS = 12
W, H = 1280, 720

# Esempio coerente col forward pass reale (reports precedenti)
EX = {
    "as_of": "2025-09-18",
    "pdv": "BASEK",
    "sku": "010OKPY040002",
    "vendite_14": 12,
    "giorni": 9,
    "rr14": 0.8571,
    "temp": 21.5,
    "giacenza": 12,
    "prior": 12.86,
    "scale": 0.181,
    "delta": 0.742,
    "y_sku": 3.06,
}


def _fig():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="#0f1419")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_axis_off()
    ax.set_facecolor("#0f1419")
    return fig, ax


def _frame(fig) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    rgb = buf[:, :, :3].copy()
    plt.close(fig)
    return rgb


def _title(ax, text, y=8.4, size=22, color="#e8eef6"):
    ax.text(8, y, text, ha="center", va="center", fontsize=size, fontweight="bold", color=color, fontfamily="sans-serif")


def _sub(ax, text, y=7.85, size=12, color="#9aabbf"):
    ax.text(8, y, text, ha="center", va="center", fontsize=size, color=color, fontfamily="sans-serif")


def _box(ax, x, y, w, h, face, edge, lw=1.5):
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=face, edgecolor=edge, linewidth=lw,
        )
    )


def _progress(ax, t, total):
    """Barra avanzamento in basso."""
    p = t / max(total - 1, 1)
    ax.add_patch(mpatches.Rectangle((0.5, 0.25), 15, 0.12, facecolor="#243044", edgecolor="none"))
    ax.add_patch(mpatches.Rectangle((0.5, 0.25), 15 * p, 0.12, facecolor="#5b9fd4", edgecolor="none"))


def scene_title(t, n):
    fig, ax = _fig()
    _title(ax, "Come nasce una previsione", y=5.4, size=28)
    _sub(ax, "Dal dato grezzo alla Dual-head TensorFlow → pezzi previsti", y=4.7, size=14)
    _sub(ax, f"Esempio: PDV {EX['pdv']}  ·  SKU {EX['sku']}  ·  as_of {EX['as_of']}", y=3.9, size=12, color="#5b9fd4")
    # fade-in via alpha simulated by brightness of subtitle
    _progress(ax, t, n)
    return _frame(fig)


def scene_raw(t, n):
    fig, ax = _fig()
    _title(ax, "1 · Entra il dato grezzo", y=8.35)
    _sub(ax, "Solo storico fino ad as_of (niente futuro)", y=7.85)

    items = [
        (1.0, 5.2, "Scontrini", f"{EX['vendite_14']} pezzi\nin {EX['giorni']} giorni"),
        (4.2, 5.2, "Negozio", f"PDV {EX['pdv']}\nITA"),
        (7.4, 5.2, "Prodotto", f"SKU\n{EX['sku'][:10]}…"),
        (10.6, 5.2, "Meteo", f"{EX['temp']} °C\nmedia 14g"),
        (13.0, 5.2, "Stock", f"giacenza\n{EX['giacenza']}"),
    ]
    # reveal one by one
    reveal = min(len(items), 1 + int(t / max(n / (len(items) + 1), 1)))
    for i, (x, y, title, body) in enumerate(items):
        if i >= reveal:
            continue
        _box(ax, x, y, 2.6, 2.0, "#1a2332", "#4a5d78")
        ax.text(x + 1.3, y + 1.45, title, ha="center", fontsize=11, fontweight="bold", color="#5b9fd4")
        ax.text(x + 1.3, y + 0.7, body, ha="center", fontsize=10, color="#e8eef6")

    if reveal >= len(items):
        ax.annotate(
            "",
            xy=(8, 3.6),
            xytext=(8, 4.9),
            arrowprops=dict(arrowstyle="->", color="#5b9fd4", lw=2),
        )
        ax.text(8, 3.2, "aggregazione causale → 1 riga PDV × SKU × as_of", ha="center", color="#9aabbf", fontsize=12)

    _progress(ax, t, n)
    return _frame(fig)


def scene_features(t, n):
    fig, ax = _fig()
    _title(ax, "2 · Diventa feature numeriche e ID", y=8.35)
    _sub(ax, "La rete non legge lo scontrino: legge numeri e indici", y=7.85)

    _box(ax, 0.8, 3.8, 4.5, 3.4, "#1a2332", "#8a5a2b")
    ax.text(3.05, 6.8, "Dal grezzo", ha="center", fontsize=12, fontweight="bold", color="#e8b86d")
    ax.text(3.05, 6.1, f"{EX['vendite_14']} pezzi / 14 giorni", ha="center", color="#e8eef6", fontsize=11)
    ax.text(3.05, 5.4, f"→ run_rate_14d = {EX['rr14']}", ha="center", color="#5b9fd4", fontsize=12, fontweight="bold")
    ax.text(3.05, 4.7, f"→ sku_prior = RR × 15 = {EX['prior']}", ha="center", color="#5b9fd4", fontsize=12, fontweight="bold")
    ax.text(3.05, 4.2, "(ancora = previsione naive)", ha="center", color="#9aabbf", fontsize=10)

    ax.annotate("", xy=(7.2, 5.5), xytext=(5.5, 5.5), arrowprops=dict(arrowstyle="->", color="#5b9fd4", lw=2.5))

    _box(ax, 7.4, 3.8, 7.8, 3.4, "#1a2332", "#2563a8")
    ax.text(11.3, 6.8, "Tre tipi di ingresso alla rete", ha="center", fontsize=12, fontweight="bold", color="#5b9fd4")
    rows = [
        ("Embedding", "sku, pdv, colore… → vettori (es. 12 dim)"),
        ("Numeriche", "57 valori z-score + BatchNorm"),
        ("sku_prior", f"{EX['prior']}  (salta il tronco)"),
    ]
    for i, (a, b) in enumerate(rows):
        y = 6.0 - i * 0.7
        if t > i * (n / 4):
            ax.text(8.0, y, a, fontsize=11, fontweight="bold", color="#e8eef6")
            ax.text(10.2, y, b, fontsize=11, color="#9aabbf")

    _progress(ax, t, n)
    return _frame(fig)


def _neuron_row(ax, x0, y, n_show, activation, label, dim, active=True):
    ax.text(0.6, y + 0.15, label, fontsize=11, fontweight="bold", color="#5b9fd4" if active else "#4a5d78")
    ax.text(0.6, y - 0.25, f"dim {dim}", fontsize=9, color="#9aabbf")
    rng = np.random.default_rng(abs(hash(label)) % (2**31))
    # stable pseudo activations scaled by progress
    base = rng.random(n_show)
    for i in range(n_show):
        a = float(np.clip(base[i] * activation, 0, 1))
        if not active:
            a *= 0.15
        color = plt.cm.Blues(0.2 + 0.75 * a)
        ax.add_patch(plt.Circle((x0 + i * 0.18, y), 0.07, color=color, ec="#2a3a4e", lw=0.4))
    if dim > n_show:
        ax.text(x0 + n_show * 0.18 + 0.2, y, f"…×{dim}", va="center", fontsize=10, color="#9aabbf")


def scene_trunk(t, n, stage: int):
    """stage 0=concat, 1=res256a, 2=res256b, 3=res128"""
    fig, ax = _fig()
    titles = [
        "3 · Concatenate — un solo vettore di 138",
        "4 · Residual #1 — 256 neuroni si attivano",
        "5 · Residual #2 — raffina la rappresentazione",
        "6 · Residual #3 — comprime a 128 neuroni",
    ]
    expl = [
        "Embedding (81) ∥ numeriche BN (57) → 138 attivazioni in ingresso al tronco",
        "Ogni neurone: somma pesata + bias → ReLU (se <0 si spegne). Skip connection.",
        "Stessa logica: Dense→BN→ReLU→Drop→Dense→BN→Add(skip)→ReLU",
        "256→128: il segnale utile si concentra. Uscita = vettore condiviso x",
    ]
    _title(ax, titles[stage], y=8.35, size=18)
    _sub(ax, expl[stage], y=7.8, size=11)

    # progress of activation within scene
    pulse = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(2 * math.pi * t / max(n, 1)))
    act_levels = [0.4, 0.7, 0.85, 1.0]
    labels = [
        ("Concat 138", 138, 40),
        ("Residual 256", 256, 48),
        ("Residual 256", 256, 48),
        ("Residual 128", 128, 40),
    ]

    for i, (lab, dim, nshow) in enumerate(labels):
        y = 6.3 - i * 1.35
        active = i <= stage
        act = act_levels[i] * (pulse if i == stage else (0.9 if active else 0.2))
        _neuron_row(ax, 4.2, y, nshow, act, lab, dim, active=active)
        if i < stage:
            ax.annotate(
                "",
                xy=(8, y - 0.55),
                xytext=(8, y - 0.35),
                arrowprops=dict(arrowstyle="->", color="#4a5d78", lw=1.2),
            )

    if stage == 0:
        ax.text(8, 1.2, "Il prior NON passa da qui → viaggia a destra fino alla Multiply", ha="center", color="#5b9fd4", fontsize=11)
    elif stage >= 1:
        ax.text(
            8,
            1.2,
            "ReLU: neurone acceso solo se z>0  ·  Dropout (solo in training) spegne a caso",
            ha="center",
            color="#9aabbf",
            fontsize=11,
        )

    _progress(ax, t, n)
    return _frame(fig)


def scene_head(t, n):
    fig, ax = _fig()
    _title(ax, "7 · Testa SKU — due neuroni decisivi", y=8.35)
    _sub(ax, "Dal vettore x (128) → Dense(64) → scale e delta", y=7.85)

    # trunk summary
    _box(ax, 0.6, 5.5, 3.2, 1.8, "#1a2332", "#4a5d78")
    ax.text(2.2, 6.8, "tronco x", ha="center", fontweight="bold", color="#e8eef6", fontsize=12)
    ax.text(2.2, 6.2, "128 neuroni", ha="center", color="#9aabbf", fontsize=11)
    ax.text(2.2, 5.8, "condiviso", ha="center", color="#9aabbf", fontsize=10)

    ax.annotate("", xy=(5.0, 6.4), xytext=(4.0, 6.4), arrowprops=dict(arrowstyle="->", color="#5b9fd4", lw=2))

    _box(ax, 5.1, 5.5, 3.4, 1.8, "#1a2332", "#2563a8")
    ax.text(6.8, 6.8, "Dense 64 + ReLU", ha="center", fontweight="bold", color="#5b9fd4", fontsize=12)
    ax.text(6.8, 6.15, "rappresentazione\nper la testa", ha="center", color="#9aabbf", fontsize=10)

    # reveal scale/delta
    phase = t / max(n - 1, 1)
    ax.annotate("", xy=(9.6, 7.0), xytext=(8.7, 6.6), arrowprops=dict(arrowstyle="->", color="#5b9fd4", lw=1.5))
    ax.annotate("", xy=(9.6, 5.0), xytext=(8.7, 6.2), arrowprops=dict(arrowstyle="->", color="#5b9fd4", lw=1.5))

    # scale
    face_s = "#1e3a5f" if phase > 0.2 else "#1a2332"
    _box(ax, 9.7, 6.4, 5.5, 1.3, face_s, "#5b9fd4", lw=2)
    ax.text(12.45, 7.3, "sku_scale  ·  Softplus", ha="center", fontweight="bold", color="#e8eef6", fontsize=12)
    if phase > 0.25:
        ax.text(12.45, 6.75, f"= {EX['scale']:.3f}   (sempre > 0)", ha="center", color="#5b9fd4", fontsize=13, fontweight="bold")

    # delta
    face_d = "#1e3a5f" if phase > 0.45 else "#1a2332"
    _box(ax, 9.7, 4.6, 5.5, 1.3, face_d, "#5b9fd4", lw=2)
    ax.text(12.45, 5.5, "sku_delta  ·  lineare", ha="center", fontweight="bold", color="#e8eef6", fontsize=12)
    if phase > 0.5:
        ax.text(12.45, 4.95, f"= {EX['delta']:+.3f}   pezzi di correzione", ha="center", color="#5b9fd4", fontsize=13, fontweight="bold")

    # prior bypass
    _box(ax, 0.6, 3.2, 5.0, 1.4, "#1e3a5f", "#5b9fd4", lw=1.5)
    ax.text(3.1, 4.1, "sku_prior (bypass)", ha="center", fontweight="bold", color="#5b9fd4", fontsize=12)
    ax.text(3.1, 3.55, f"RR×15 = {EX['prior']}  → Multipply", ha="center", color="#e8eef6", fontsize=11)

    if phase > 0.7:
        ax.text(
            8,
            2.2,
            "Logica: non inventare da zero — correggi il ritmo recente (prior)",
            ha="center",
            color="#e8b86d",
            fontsize=12,
            fontweight="bold",
        )

    _progress(ax, t, n)
    return _frame(fig)


def scene_formula(t, n):
    fig, ax = _fig()
    _title(ax, "8 · Formula finale", y=8.35)
    _sub(ax, "y_sku = ReLU( prior × scale + delta )", y=7.8, size=14, color="#5b9fd4")

    phase = t / max(n - 1, 1)

    # animated computation
    steps = [
        (f"prior = {EX['prior']}", 6.6),
        (f"× scale = {EX['scale']}", 5.7),
        (f"= {EX['prior'] * EX['scale']:.2f}", 4.8),
        (f"+ delta = {EX['delta']:+.3f}", 3.9),
        (f"= {EX['prior'] * EX['scale'] + EX['delta']:.2f}", 3.0),
        ("ReLU → pezzi ≥ 0", 2.1),
    ]
    shown = int(phase * (len(steps) + 0.5))
    for i, (txt, y) in enumerate(steps):
        if i < shown:
            col = "#0b6e4f" if i == len(steps) - 1 and shown >= len(steps) else "#e8eef6"
            ax.text(8, y, txt, ha="center", fontsize=18 if i >= 4 else 15, fontweight="bold", color=col)

    if shown >= len(steps):
        _box(ax, 4.5, 0.7, 7, 1.0, "#0b6e4f", "#7dcea0", lw=2)
        ax.text(8, 1.2, f"y_sku = {EX['y_sku']:.2f}  pezzi / 15 giorni", ha="center", fontsize=16, fontweight="bold", color="#fff")

    _progress(ax, t, n)
    return _frame(fig)


def scene_outro(t, n):
    fig, ax = _fig()
    _title(ax, "Sintesi del percorso", y=8.0, size=24)
    lines = [
        "Grezzo (scontrini, PDV, SKU, meteo, stock)",
        "→ feature + prior (ritmo recente × 15)",
        "→ Embedding + numeriche → vettore 138",
        "→ Residual 256 → 256 → 128 (neuroni ReLU)",
        "→ scale (Softplus) e delta (lineare)",
        f"→ ReLU(prior×scale+delta) = {EX['y_sku']:.2f} pezzi / 15g",
    ]
    reveal = 1 + int(t / max(n / len(lines), 1))
    for i, ln in enumerate(lines):
        if i < reveal:
            ax.text(8, 6.6 - i * 0.7, ln, ha="center", fontsize=13, color="#e8eef6" if i < 5 else "#7dcea0", fontweight="bold" if i == 5 else "normal")

    if reveal >= len(lines):
        ax.text(8, 1.6, "In BI può ancora mescolarsi con LightGBM (routing per volume)", ha="center", color="#9aabbf", fontsize=11)

    _progress(ax, t, n)
    return _frame(fig)


def build_timeline():
    """Lista (scene_fn, n_frames) oppure (scene_fn_with_stage, n, stage)."""
    return [
        (scene_title, 2 * FPS),
        (scene_raw, 5 * FPS),
        (scene_features, 5 * FPS),
        (lambda t, n, s=0: scene_trunk(t, n, s), 4 * FPS),
        (lambda t, n, s=1: scene_trunk(t, n, s), 4 * FPS),
        (lambda t, n, s=2: scene_trunk(t, n, s), 3 * FPS),
        (lambda t, n, s=3: scene_trunk(t, n, s), 4 * FPS),
        (scene_head, 6 * FPS),
        (scene_formula, 6 * FPS),
        (scene_outro, 5 * FPS),
    ]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    timeline = build_timeline()
    total = sum(n for _, n in timeline)
    print(f"rendering {total} frames (~{total / FPS:.0f}s) → {OUT}")

    writer = imageio.get_writer(
        OUT,
        fps=FPS,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
    )

    frame_i = 0
    try:
        for fn, n in timeline:
            for t in range(n):
                frame = fn(t, n)
                # ensure even dims for yuv420p
                h, w = frame.shape[:2]
                if h % 2 or w % 2:
                    frame = frame[: h - h % 2, : w - w % 2]
                writer.append_data(frame)
                frame_i += 1
                if frame_i % 24 == 0:
                    print(f"  {frame_i}/{total}")
    finally:
        writer.close()

    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")

    # tiny HTML player
    html = OUT.with_suffix(".html")
    html.write_text(
        f"""<!DOCTYPE html>
<html lang="it"><head><meta charset="utf-8"/>
<title>Video flusso rete neurale</title>
<style>
 body{{margin:0;background:#0f1419;color:#e8eef6;font-family:IBM Plex Sans,Segoe UI,sans-serif}}
 .wrap{{max-width:1100px;margin:0 auto;padding:24px}}
 h1{{font-size:1.3rem;margin:0 0 8px}}
 p{{color:#9aabbf}}
 video{{width:100%;border-radius:12px;border:1px solid #243044;background:#000}}
</style></head><body>
<div class="wrap">
<h1>Flusso dato grezzo → Dual-head → previsione</h1>
<p>Esempio PDV {EX['pdv']} · SKU {EX['sku']} · output {EX['y_sku']} pezzi / 15 giorni</p>
<video controls autoplay muted playsinline src="{OUT.name}"></video>
</div></body></html>
""",
        encoding="utf-8",
    )
    print(f"wrote {html}")


if __name__ == "__main__":
    main()
