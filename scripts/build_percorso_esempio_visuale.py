#!/usr/bin/env python3
"""Visualizza un esempio grezzo che attraversa un albero LGBM reale + i layer Dual-head."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "production" / "artifacts" / "real"
OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from testing.features.build_features import NUMERIC_COLS  # noqa: E402
from testing.features.schema import BAND_INPUT_COLS, CATEGORICAL_COLS, EMB_DIMS  # noqa: E402

CAT_COLS = list(CATEGORICAL_COLS) + list(BAND_INPUT_COLS.keys())


# ---------------------------------------------------------------------------
# Esempio didattico (coerente con feature reali h30)
# ---------------------------------------------------------------------------

def build_example(maps: dict):
    """Ritorna raw_story, cats_raw, X (1 riga), feats, bundle."""
    bundle = joblib.load(ART / "h30" / "lgbm_bundle.joblib")
    feats = bundle["features"]

    # SKU reale presente nelle mappe se possibile
    sku = "010OKPY040002" if "010OKPY040002" in maps.get("sku_id", {}) else next(iter(maps["sku_id"]))
    pdv = "BASEK" if "BASEK" in maps.get("pdv_id", {}) else next(iter(maps["pdv_id"]))

    raw = {
        "as_of": "2025-09-18",
        "orizzonte_giorni": 30,
        "sku_id": sku,
        "pdv_id": pdv,
        "country": "ITA",
        "vendite_ultimi_14_giorni": 12,
        "giorni_con_vendita_su_14": 9,
        "max_pezzi_in_un_giorno": 3,
        "temp_media_città_14g": 21.5,
        "giorni_di_pioggia": 2,
        "giacenza": 12,
        "prezzo": 129.0,
        "in_sconto": False,
        "mese": 9,
    }
    rr14 = raw["vendite_ultimi_14_giorni"] / 14.0
    rr7 = 7 / 7.0
    rr28 = 20 / 28.0

    row = {f: 0.0 for f in feats}
    num = {
        "run_rate_14d": rr14,
        "run_rate_7d": rr7,
        "run_rate_28d": rr28,
        "trend_7v14": rr7 / max(rr14, 1e-6),
        "trend_14v28": rr14 / max(rr28, 1e-6),
        "vendite_volatilita_14d": 0.45,
        "max_giornaliero_14d": float(raw["max_pezzi_in_un_giorno"]),
        "giorni_vendita_su_14": float(raw["giorni_con_vendita_su_14"]),
        "has_history": 1.0,
        "giorni_da_lancio": 40.0,
        "log_giorni_da_lancio": float(np.log1p(40)),
        "is_new_sku": 0.0,
        "giacenza": float(raw["giacenza"]),
        "giacenza_su_run_rate": float(raw["giacenza"]) / max(rr14, 1e-6),
        "prezzo_listino": float(raw["prezzo"]),
        "prezzo_pagato": float(raw["prezzo"]),
        "rapporto_prezzo_pagato": 1.0,
        "is_in_sconto": 0.0,
        "sconto_pct": 0.0,
        "temp_media": float(raw["temp_media_città_14g"]),
        "giorni_pioggia": float(raw["giorni_di_pioggia"]),
        "mese": 9.0,
        "day_of_week": 3.0,
        "is_weekend": 0.0,
        "sin_mese": float(np.sin(2 * np.pi * 9 / 12)),
        "cos_mese": float(np.cos(2 * np.pi * 9 / 12)),
        "sin_anno": float(np.sin(2 * np.pi * 261 / 365)),
        "cos_anno": float(np.cos(2 * np.pi * 261 / 365)),
    }
    for k, v in num.items():
        if k in row:
            row[k] = float(v)
    row["sku_prior"] = rr14 * 30.0

    def pick(col: str, fallback: str = "NA", idx: int = 1) -> str:
        keys = list(maps.get(col, {fallback: 0}).keys())
        return keys[min(idx, len(keys) - 1)]

    cats_raw = {
        "country": "ITA",
        "pdv_id": pdv,
        "sku_id": sku,
        "categoria_id": pick("categoria_id", idx=1),
        "commerciale_riferimento": pick("commerciale_riferimento", idx=0),
        "modello": pick("modello", idx=5),
        "tessuto": pick("tessuto", idx=2),
        "colore": pick("colore", idx=3),
        "campagna": pick("campagna", idx=1),
        "classe": pick("classe", idx=1),
        "sclasse": pick("sclasse", idx=1),
        "ssclasse": pick("ssclasse", idx=1),
        "tipologia": pick("tipologia", idx=1),
        "mq_espositivi_band": 2,
        "mq_stock_band": 2,
        "weather_regime": 1,
    }
    for col in CAT_COLS:
        if col in maps:
            row[col] = maps[col].get(str(cats_raw.get(col, "NA")), -1)
        elif col in cats_raw:
            row[col] = int(cats_raw[col])

    X = pd.DataFrame([row])[feats].copy()
    for col in CAT_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")

    raw["run_rate_14d"] = round(rr14, 4)
    raw["sku_prior_30g"] = round(rr14 * 30, 2)
    raw["naive_30g"] = round(rr14 * 30, 2)
    return raw, cats_raw, X, feats, bundle


def walk_tree(node: dict, fname: list[str], xrow: pd.Series, path: list) -> list:
    if "split_feature" not in node:
        path.append(
            {
                "type": "leaf",
                "value": node["leaf_value"],
                "index": node.get("leaf_index"),
                "count": node.get("leaf_count"),
            }
        )
        return path
    fi = node["split_feature"]
    fn = fname[fi]
    thr = node["threshold"]
    dtype = node.get("decision_type", "<=")
    val = xrow[fn]
    val_f = float(val)
    if dtype in ("<=", "<"):
        go_left = val_f <= float(thr)
        rule = f"{fn} = {val_f:.4g}  {'≤' if go_left else '>'}  {float(thr):.4g}"
    else:
        cats = str(thr).split("||")
        go_left = str(int(val_f)) in cats
        shown = ",".join(cats[:3]) + ("…" if len(cats) > 3 else "")
        rule = f"{fn} = {int(val_f)}  {'∈' if go_left else '∉'}  {{{shown}}}"
    path.append(
        {
            "type": "split",
            "feature": fn,
            "threshold": thr,
            "value": val_f,
            "go": "L" if go_left else "R",
            "rule": rule,
            "gain": node.get("split_gain"),
            "node": node,
        }
    )
    child = node["left_child"] if go_left else node["right_child"]
    return walk_tree(child, fname, xrow, path)


# ---------------------------------------------------------------------------
# Layout albero (stile “decision tree plot”)
# ---------------------------------------------------------------------------

def _collect(node, fname, nodes, edges, depth=0, path_ids=None, on_path=False):
    nid = len(nodes)
    if "split_feature" not in node:
        nodes.append(
            {
                "id": nid,
                "depth": depth,
                "leaf": True,
                "value": node["leaf_value"],
                "label": f"{node['leaf_value']:.2f}",
                "on_path": on_path,
            }
        )
        return nid
    fn = fname[node["split_feature"]]
    thr = node["threshold"]
    dtype = node.get("decision_type", "<=")
    if dtype in ("<=", "<"):
        lab = f"{fn}\n≤ {float(thr):.3g}"
    else:
        lab = f"{fn}\n∈ cat"
    nodes.append(
        {
            "id": nid,
            "depth": depth,
            "leaf": False,
            "value": node.get("internal_value", 0.0),
            "label": lab,
            "on_path": on_path,
            "feature": fn,
        }
    )
    # which child is on path?
    left_on = right_on = False
    if on_path and path_ids is not None and depth + 1 < len(path_ids):
        # path_ids[depth] is current; path_ids[depth+1] tells go
        pass
    # We'll mark on_path by comparing object identity via path list of nodes later
    lid = _collect(node["left_child"], fname, nodes, edges, depth + 1, path_ids, False)
    rid = _collect(node["right_child"], fname, nodes, edges, depth + 1, path_ids, False)
    edges.append((nid, lid, "L"))
    edges.append((nid, rid, "R"))
    nodes[nid]["left"] = lid
    nodes[nid]["right"] = rid
    return nid


def _assign_x(nodes, edges):
    """Inorder x for leaves, parents = mid children."""
    children = {n["id"]: [] for n in nodes}
    for a, b, _ in edges:
        children[a].append(b)
    leaves = [n for n in nodes if n["leaf"]]
    for i, n in enumerate(leaves):
        n["x"] = float(i)
    # bottom-up
    by_id = {n["id"]: n for n in nodes}
    for n in sorted(nodes, key=lambda z: -z["depth"]):
        if not n["leaf"]:
            xs = [by_id[c]["x"] for c in children[n["id"]]]
            n["x"] = sum(xs) / len(xs)
    max_d = max(n["depth"] for n in nodes)
    for n in nodes:
        n["y"] = max_d - n["depth"]


def _mark_path(nodes, edges, path_rules_gos: list[str]):
    """Mark path from root following L/R sequence from walk (excluding leaf)."""
    by_id = {n["id"]: n for n in nodes}
    children = {}
    for a, b, side in edges:
        children.setdefault(a, {})[side] = b
    cur = 0
    by_id[0]["on_path"] = True
    for go in path_rules_gos:
        nxt = children.get(cur, {}).get(go)
        if nxt is None:
            break
        # mark edge
        for e in edges:
            if e[0] == cur and e[1] == nxt:
                # store on edge via parallel list later
                pass
        by_id[nxt]["on_path"] = True
        cur = nxt
    return {(a, b) for a, b, side in edges if by_id[a]["on_path"] and by_id[b]["on_path"]}


def draw_lgbm_tree(ax, booster, tree_index: int, xrow: pd.Series, path: list, title: str):
    dump = booster.dump_model()
    fname = booster.feature_name()
    root = dump["tree_info"][tree_index]["tree_structure"]
    nodes, edges = [], []
    _collect(root, fname, nodes, edges)
    _assign_x(nodes, edges)
    gos = [p["go"] for p in path if p["type"] == "split"]
    path_edges = _mark_path(nodes, edges, gos)

    # color by value
    vals = np.array([n["value"] for n in nodes], dtype=float)
    vmin, vmax = np.percentile(vals, 5), np.percentile(vals, 95)
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1
    cmap = plt.cm.RdYlBu_r

    by_id = {n["id"]: n for n in nodes}
    # edges
    segs, cols, lws = [], [], []
    for a, b, _side in edges:
        na, nb = by_id[a], by_id[b]
        segs.append([(na["x"], na["y"]), (nb["x"], nb["y"])])
        on = (a, b) in path_edges
        cols.append("#111111" if on else "#b0b0b0")
        lws.append(2.8 if on else 0.6)
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, zorder=1))

    for n in nodes:
        t = (n["value"] - vmin) / (vmax - vmin)
        t = float(np.clip(t, 0, 1))
        color = cmap(t)
        if n["on_path"]:
            ec, lw = "#111111", 2.2
        else:
            ec, lw = "#666666", 0.5
        if n["leaf"]:
            w, h = 0.55, 0.38
        else:
            w, h = 0.85, 0.55
        rect = mpatches.FancyBboxPatch(
            (n["x"] - w / 2, n["y"] - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color,
            edgecolor=ec,
            linewidth=lw,
            zorder=2,
        )
        ax.add_patch(rect)
        fontsize = 3.2 if not n["on_path"] else 5.0
        weight = "bold" if n["on_path"] else "normal"
        txt_color = "white" if t > 0.55 or t < 0.2 else "black"
        ax.text(
            n["x"],
            n["y"],
            n["label"] if n["on_path"] or n["leaf"] else "",
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=weight,
            color=txt_color,
            zorder=3,
            clip_on=True,
        )
        # always show leaf values small
        if n["leaf"] and not n["on_path"]:
            ax.text(n["x"], n["y"], f"{n['value']:.1f}", ha="center", va="center", fontsize=2.8, color=txt_color, zorder=3)

    xs = [n["x"] for n in nodes]
    ys = [n["y"] for n in nodes]
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=8)


def draw_path_panel(ax, raw: dict, path: list, pred: float, leaf_sum_raw: float, bias: float):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    lines = [
        "ESEMPIO DI DATO GREZZO",
        f"as_of {raw['as_of']}  ·  PDV {raw['pdv_id']}  ·  SKU {raw['sku_id']}",
        f"Vendite 14g = {raw['vendite_ultimi_14_giorni']} pezzi in {raw['giorni_con_vendita_su_14']} giorni",
        f"→ run_rate_14d = {raw['run_rate_14d']}  →  prior/naive 30g = {raw['naive_30g']}",
        f"Temp {raw['temp_media_città_14g']}°C · giacenza {raw['giacenza']} · no sconto · mese {raw['mese']}",
        "",
        "PERCORSO NELL’ALBERO #0  (ogni riga = uno split)",
    ]
    for i, p in enumerate(path):
        if p["type"] == "split":
            lines.append(f"  {i+1}. {p['rule']}   → vai a {'sinistra' if p['go']=='L' else 'destra'}")
        else:
            lines.append(f"  ★ FOGLIA #{p['index']}: contributo grezzo = {p['value']:.4f}")
    lines += [
        "",
        "DA UN ALBERO A TUTTI GLI ALBERI",
        f"  score grezzo (link log) ≈ bias {bias:.3f} + Σ contributi feature",
        f"  predizione Tweedie = exp(score) = {pred:.2f} pezzi / 30 giorni",
        f"  (il modello ha ~{380} alberi: ognuno aggiunge un pezzetto come questa foglia)",
    ]
    y = 0.97
    for i, ln in enumerate(lines):
        if i == 0 or ln.startswith("PERCORSO") or ln.startswith("DA UN"):
            ax.text(0.02, y, ln, fontsize=9, fontweight="bold", va="top", family="sans-serif")
        else:
            ax.text(0.02, y, ln, fontsize=7.5, va="top", family="monospace")
        y -= 0.038 if ln else 0.02


# ---------------------------------------------------------------------------
# Dual-head activation flow
# ---------------------------------------------------------------------------

def dual_head_activations(raw: dict):
    """Forward pass reale Dual-head su CPU, con attivazioni intermedie."""
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf
    from tensorflow import keras

    tf.config.set_visible_devices([], "GPU")

    from testing.model.dual_head_tf import frame_to_tf_inputs, load_dual_head_model

    enc = np.load(ART / "encoders.npz", allow_pickle=True)
    maps = enc["maps"].item()
    model = load_dual_head_model(ART / "dual_head.keras")

    rr14 = float(raw["run_rate_14d"])
    row = {c: float(enc["num_mean"][i]) for i, c in enumerate(NUMERIC_COLS)}
    row.update(
        {
            "run_rate_14d": rr14,
            "run_rate_7d": 1.0,
            "run_rate_28d": 20 / 28,
            "trend_7v14": 1.0 / max(rr14, 1e-6),
            "trend_14v28": rr14 / max(20 / 28, 1e-6),
            "vendite_volatilita_14d": 0.45,
            "max_giornaliero_14d": 3.0,
            "giorni_vendita_su_14": 9.0,
            "has_history": 1.0,
            "giorni_da_lancio": 40.0,
            "log_giorni_da_lancio": float(np.log1p(40)),
            "giacenza": 12.0,
            "giacenza_su_run_rate": 12.0 / max(rr14, 1e-6),
            "prezzo_listino": 129.0,
            "prezzo_pagato": 129.0,
            "rapporto_prezzo_pagato": 1.0,
            "temp_media": 21.5,
            "giorni_pioggia": 2.0,
            "mese": 9.0,
            "day_of_week": 3.0,
        }
    )
    # categorie: usa chiavi presenti negli encoder TF (non le mappe LGBM)
    for col in maps:
        keys = list(maps[col].keys())
        row[col] = keys[min(len(keys) // 3, len(keys) - 1)]
    # allinea story se possibile
    if raw["pdv_id"] in maps.get("pdv_id", {}):
        row["pdv_id"] = raw["pdv_id"]
    if raw["sku_id"] in maps.get("sku_id", {}):
        row["sku_id"] = raw["sku_id"]
    if "ITA" in maps.get("country", {}):
        row["country"] = "ITA"
    row["mq_espositivi_band"] = 2
    row["mq_stock_band"] = 2
    row["weather_regime"] = 1

    df = pd.DataFrame([row])
    inputs = frame_to_tf_inputs(df, maps, enc["num_mean"], enc["num_std"])

    with tf.device("/CPU:0"):
        probe = keras.Model(
            inputs=model.inputs,
            outputs={
                "concat": model.get_layer("concatenate").output,
                "res1": model.get_layer("activation_1").output,  # 256
                "res2": model.get_layer("activation_3").output,  # 256
                "res3": model.get_layer("activation_5").output,  # 128
                "head64": model.get_layer("dense_8").output,  # 64
                "sku_scale": model.get_layer("sku_scale").output,
                "sku_delta": model.get_layer("sku_delta").output,
                "y_sku": model.get_layer("y_sku").output,
            },
        )
        got = probe.predict(inputs, verbose=0)

    prior = float(np.asarray(inputs["sku_prior"]).reshape(-1)[0])
    scale = float(np.asarray(got["sku_scale"]).reshape(-1)[0])
    delta = float(np.asarray(got["sku_delta"]).reshape(-1)[0])
    y_sku = float(np.asarray(got["y_sku"]).reshape(-1)[0])
    stages = [
        ("L1 · Concatenate  (138)", np.asarray(got["concat"]).reshape(-1)),
        ("L2 · Residual → 256 neuroni", np.asarray(got["res1"]).reshape(-1)),
        ("L3 · Residual → 256 neuroni", np.asarray(got["res2"]).reshape(-1)),
        ("L4 · Residual → 128 neuroni", np.asarray(got["res3"]).reshape(-1)),
        ("L5 · Head Dense 64 neuroni", np.asarray(got["head64"]).reshape(-1)),
    ]
    return {
        "prior": prior,
        "scale": scale,
        "delta": delta,
        "y_sku": y_sku,
        "stages": stages,
        "sku_used": row["sku_id"],
        "pdv_used": row["pdv_id"],
        "formula": f"ReLU({prior:.2f} × {scale:.3f} + ({delta:+.3f})) = {y_sku:.2f}",
    }


def draw_nn_flow(ax, raw, nn):
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)

    ax.text(5, 11.6, "STESSO DATO GREZZO → Dual-head TF (15 giorni)", ha="center", fontsize=12, fontweight="bold")
    ax.text(
        5,
        11.1,
        f"PDV {raw['pdv_id']} · SKU {raw['sku_id']} · RR14={raw['run_rate_14d']} · prior15={nn['prior']:.2f}",
        ha="center",
        fontsize=8,
        color="#444",
    )

    # raw box
    ax.add_patch(mpatches.FancyBboxPatch((0.3, 9.5), 9.4, 1.3, boxstyle="round,pad=0.05", facecolor="#f7efe4", edgecolor="#8a5a2b"))
    ax.text(5, 10.45, "DATO GREZZO", ha="center", fontsize=8, fontweight="bold", color="#8a5a2b")
    ax.text(
        5,
        10.0,
        f"{raw['vendite_ultimi_14_giorni']} pezzi/14g · temp {raw['temp_media_città_14g']}°C · giacenza {raw['giacenza']} · no sconto",
        ha="center",
        fontsize=7.5,
    )
    ax.text(5, 9.7, "→ encoder ID + z-score numeriche + sku_prior", ha="center", fontsize=7, color="#555")

    # stages as neuron bars
    y = 8.6
    for name, vec in nn["stages"]:
        # show mean |act|, and a strip of neuron magnitudes
        ax.text(0.4, y + 0.35, name, fontsize=8, fontweight="bold", color="#1d4e89")
        ax.text(0.4, y + 0.05, f"dim={len(vec)}  mean|a|={np.mean(np.abs(vec)):.3f}  max={np.max(np.abs(vec)):.2f}", fontsize=6.5, color="#555")
        show = vec[: min(64, len(vec))]
        vmax = np.percentile(np.abs(vec), 95) + 1e-6
        for i, a in enumerate(show):
            t = float(np.clip(abs(a) / vmax, 0, 1))
            color = plt.cm.Blues(0.25 + 0.75 * t)
            ax.add_patch(plt.Circle((2.2 + i * 0.11, y + 0.2), 0.045, color=color, ec="#333", lw=0.2))
        if len(vec) > 64:
            ax.text(2.2 + 64 * 0.11 + 0.15, y + 0.2, f"…×{len(vec)}", fontsize=7, va="center", color="#555")
        y -= 1.15

    # head
    ax.add_patch(mpatches.FancyBboxPatch((0.3, 1.6), 4.4, 1.8, boxstyle="round,pad=0.05", facecolor="#dceaf8", edgecolor="#1d4e89", linewidth=1.5))
    ax.text(2.5, 3.1, "TESTA SKU", ha="center", fontsize=9, fontweight="bold", color="#1d4e89")
    ax.text(2.5, 2.7, f"sku_scale (1 neurone Softplus) = {nn['scale']:.3f}", ha="center", fontsize=7.5)
    ax.text(2.5, 2.35, f"sku_delta (1 neurone) = {nn['delta']:+.3f}", ha="center", fontsize=7.5)
    ax.text(2.5, 1.95, f"prior = {nn['prior']:.2f}", ha="center", fontsize=7.5)

    ax.add_patch(mpatches.FancyBboxPatch((5.3, 1.6), 4.4, 1.8, boxstyle="round,pad=0.05", facecolor="#dff3eb", edgecolor="#0b6e4f", linewidth=2))
    ax.text(7.5, 3.1, "OUTPUT FINALE", ha="center", fontsize=9, fontweight="bold", color="#0b6e4f")
    ax.text(7.5, 2.6, nn["formula"], ha="center", fontsize=8, family="monospace")
    ax.text(7.5, 2.15, f"y_sku = {nn['y_sku']:.2f} pezzi / 15 giorni", ha="center", fontsize=10, fontweight="bold", color="#0b6e4f")
    ax.text(7.5, 1.8, "(poi in BI può mescolarsi con LightGBM via routing)", ha="center", fontsize=6.5, color="#555")

    ax.text(
        5,
        0.7,
        "I cerchi sono neuroni: più scuro = attivazione più forte su QUESTO esempio.",
        ha="center",
        fontsize=7.5,
        color="#444",
    )
    ax.text(
        5,
        0.3,
        "Il prior salta il tronco e arriva solo alla Multiply finale (ancora naive).",
        ha="center",
        fontsize=7.5,
        color="#444",
    )


def main():
    maps = {
        k: {str(a): int(b) for a, b in v.items()}
        for k, v in json.loads((ART / "h30" / "lgbm_maps.json").read_text()).items()
    }
    raw, cats_raw, X, feats, bundle = build_example(maps)
    m = bundle["tweedie"]
    pred = float(np.maximum(m.predict(X), 0.0)[0])
    contrib = m.predict(X, pred_contrib=True)[0]
    bias = float(contrib[-1])
    fname = m.booster_.feature_name()
    tree0 = m.booster_.dump_model()["tree_info"][0]["tree_structure"]
    path = walk_tree(tree0, fname, X.iloc[0], [])

    # --- Figure 1: tree like user image ---
    fig1 = plt.figure(figsize=(16, 11), facecolor="white")
    gs = fig1.add_gridspec(2, 1, height_ratios=[2.6, 1.1], hspace=0.08)
    ax_t = fig1.add_subplot(gs[0])
    ax_p = fig1.add_subplot(gs[1])
    n_trees = m.booster_.num_trees()
    draw_lgbm_tree(
        ax_t,
        m.booster_,
        0,
        X.iloc[0],
        path,
        title=f"LightGBM Tweedie h30 — albero #0 di {n_trees}  ·  percorso dell’esempio in nero spesso  ·  colore = valore nodo",
    )
    draw_path_panel(ax_p, raw, path, pred, path[-1]["value"], bias)
    fig1.suptitle("Percorso di UN dato grezzo dentro LightGBM", fontsize=14, fontweight="bold", y=0.98)
    png_tree = OUT_DIR / "CDM_percorso_esempio_lgbm_albero.png"
    fig1.savefig(png_tree, dpi=160, bbox_inches="tight", facecolor="white")
    print("wrote", png_tree)

    # --- Figure 2: NN ---
    nn = dual_head_activations(raw)
    # allinea etichette story al PDV/SKU effettivamente usati dalla rete
    raw_nn = dict(raw)
    raw_nn["pdv_id"] = nn.get("pdv_used", raw["pdv_id"])
    raw_nn["sku_id"] = nn.get("sku_used", raw["sku_id"])
    fig2 = plt.figure(figsize=(12, 10), facecolor="white")
    ax2 = fig2.add_subplot(111)
    draw_nn_flow(ax2, raw_nn, nn)
    png_nn = OUT_DIR / "CDM_percorso_esempio_dualhead_neuroni.png"
    fig2.savefig(png_nn, dpi=160, bbox_inches="tight", facecolor="white")
    print("wrote", png_nn)

    # --- PDF ---
    pdf_path = OUT_DIR / "CDM_percorso_esempio_dato.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig1, bbox_inches="tight")
        pdf.savefig(fig2, bbox_inches="tight")
    print("wrote", pdf_path)

    # --- HTML ---
    html = OUT_DIR / "CDM_percorso_esempio_dato.html"
    path_html = "".join(
        f"<li><code>{p['rule']}</code> → <b>{'sinistra' if p['go']=='L' else 'destra'}</b></li>"
        if p["type"] == "split"
        else f"<li><b>FOGLIA #{p['index']}</b> contributo = <code>{p['value']:.4f}</code></li>"
        for p in path
    )
    html.write_text(
        f"""<!DOCTYPE html>
<html lang="it"><head><meta charset="utf-8"/>
<title>Percorso esempio — dato grezzo → output</title>
<style>
 body{{font-family:IBM Plex Sans,Segoe UI,sans-serif;margin:0;background:#eef1f5;color:#15202b}}
 header{{background:linear-gradient(115deg,#1a334d,#0b6e4f);color:#fff;padding:20px}}
 .wrap{{max-width:1100px;margin:0 auto;padding:16px 16px 48px}}
 section{{background:#fff;border:1px solid #d3dce6;border-radius:12px;padding:16px;margin:14px 0}}
 img{{max-width:100%;height:auto;border:1px solid #d3dce6;border-radius:8px}}
 code{{background:#f0f4f8;padding:1px 5px;border-radius:4px}}
 .out{{font-size:1.2rem;font-weight:700;color:#0b6e4f}}
 ol{{line-height:1.55}}
</style></head><body>
<header>
<h1>Un dato grezzo, tutto il percorso</h1>
<p>Esempio PDV <b>{raw['pdv_id']}</b> · SKU <b>{raw['sku_id']}</b> · as_of {raw['as_of']}</p>
</header>
<div class="wrap">
<section>
<h2>1 · Dato grezzo</h2>
<ul>
<li>Vendite ultimi 14 giorni: <b>{raw['vendite_ultimi_14_giorni']}</b> pezzi in {raw['giorni_con_vendita_su_14']} giorni → RR14 = <b>{raw['run_rate_14d']}</b></li>
<li>Temp media {raw['temp_media_città_14g']}°C · giacenza {raw['giacenza']} · prezzo {raw['prezzo']} · no sconto</li>
<li>Naive 30g = RR × 30 = <b>{raw['naive_30g']}</b></li>
</ul>
</section>
<section>
<h2>2 · Dentro un albero LightGBM (come il diagramma che mi hai mostrato)</h2>
<p>Il percorso dell’esempio è disegnato in <b>nero spesso</b>. Gli altri nodi restano sullo sfondo.</p>
<img src="{png_tree.name}" alt="Albero LightGBM con percorso"/>
<ol>{path_html}</ol>
<p class="out">Output Tweedie h30 su questo esempio: {pred:.2f} pezzi / 30 giorni</p>
<p>(somma di ~{n_trees} alberi sul link log, poi exp)</p>
</section>
<section>
<h2>3 · Stesso dato nella Dual-head (layer e neuroni)</h2>
<img src="{png_nn.name}" alt="Flusso neuroni Dual-head"/>
<p class="out">Output Dual-head: {nn['y_sku']:.2f} pezzi / 15 giorni · <code>{nn['formula']}</code></p>
</section>
<p><a href="{pdf_path.name}">Scarica PDF</a></p>
</div></body></html>
""",
        encoding="utf-8",
    )
    print("wrote", html)
    plt.close("all")


if __name__ == "__main__":
    main()
