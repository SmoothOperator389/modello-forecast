"""Genera dashboard HTML interattiva (mini-BI) per CDM."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from testing.config import DATA_DIR, PRODUCTION_N_PDV, PRODUCTION_N_SKU
from testing.features.sku_weather import write_sku_season_lookup
from testing.report_builder import REPORT_DIR, load_report_inputs

MAX_PRIORITY_ROWS = 2500  # righe dettaglio embed (qty > 0, top per qty)
# Vista «Per SKU»: tutti i negozi, anche qty = 0 (monitora).


def _load_season_map(data_dir: Path, priorities: pd.DataFrame) -> dict[str, str]:
    """Stagione stimata dalle vendite×temperatura, non tag catalogo."""
    path = data_dir / "sku_meteo_profilo.csv"
    as_of = None
    if not priorities.empty and "as_of" in priorities.columns:
        raw = str(priorities["as_of"].iloc[0])[:10]
        try:
            as_of = datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError:
            as_of = None
    if not path.exists() and as_of is not None:
        write_sku_season_lookup(as_of, data_dir=data_dir)
    if not path.exists():
        return {}
    prof = pd.read_csv(path)
    if "sku_id" not in prof.columns or "stagione_stimata" not in prof.columns:
        return {}
    return (
        prof.drop_duplicates("sku_id")
        .set_index("sku_id")["stagione_stimata"]
        .astype(str)
        .to_dict()
    )


def _season_of(sku_id: str, season_map: dict[str, str], fallback: str = "") -> str:
    sid = str(sku_id or "")
    return str(season_map.get(sid) or fallback or "poco storico")


def _prepare_priorities(
    priorities: pd.DataFrame,
    pdv: pd.DataFrame,
    products: pd.DataFrame,
    season_map: dict[str, str],
) -> list[dict]:
    if priorities.empty:
        return []
    df = priorities.copy()
    if "qty_suggerita" in df.columns:
        df = df[df["qty_suggerita"] > 0].copy()
    if df.empty:
        return []
    df = df.sort_values("qty_suggerita", ascending=False).head(MAX_PRIORITY_ROWS)

    pdv_l = pdv[["pdv_id", "country", "city", "commerciale_riferimento"]].copy()
    df = df.merge(pdv_l, on="pdv_id", how="left")
    if not products.empty and "sku_id" in products.columns:
        prod_cols = [
            c
            for c in [
                "sku_id",
                "nome",
                "modello",
                "tessuto",
                "colore",
                "categoria_id",
                "campagna",
                "ssclasse",
                "sclasse",
            ]
            if c in products.columns
        ]
        df = df.merge(products[prod_cols].drop_duplicates("sku_id"), on="sku_id", how="left")

    records = []
    for r in df.to_dict(orient="records"):
        records.append(
            {
                "as_of": str(r.get("as_of", ""))[:10],
                "pdv_id": str(r.get("pdv_id", "")),
                "sku_id": str(r.get("sku_id", "")),
                "nome": str(r.get("nome", r.get("modello", ""))),
                "country": str(r.get("country", "")),
                "city": str(r.get("city", r.get("pdv_id", ""))),
                "commerciale": str(r.get("commerciale_riferimento", "")),
                "modello": str(r.get("modello", "")),
                "categoria": str(r.get("categoria_id", "")),
                "stagione": _season_of(
                    r.get("sku_id", ""),
                    season_map,
                    str(r.get("stagione_stimata", "") or ""),
                ),
                "ssclasse": str(r.get("ssclasse", "")),
                "pred": float(r.get("pred_tf", r.get("vendite_previste_15gg", 0)) or 0),
                "pred_cdm": float(r.get("pred_cdm", 0) or 0),
                "giacenza": float(r.get("giacenza", 0) or 0),
                "qty": float(r.get("qty_suggerita", 0) or 0),
                "azione": str(r.get("azione", "")),
                "confidence": float(r.get("confidence_score", 0) or 0),
                "is_new_pdv": int(r.get("is_new_pdv", 0) or 0),
            }
        )
    return records


def _prepare_focus(
    focus: pd.DataFrame,
    pdv: pd.DataFrame,
    products: pd.DataFrame | None = None,
    season_map: dict[str, str] | None = None,
) -> list[dict]:
    if focus.empty:
        return []
    df = focus.copy()
    if products is not None and not products.empty and "sku_id" in products.columns:
        prod_cols = [c for c in ["sku_id", "nome", "ssclasse", "modello"] if c in products.columns]
        df = df.merge(products[prod_cols].drop_duplicates("sku_id"), on="sku_id", how="left")
    city_map = pdv.set_index("pdv_id")["city"].to_dict() if "city" in pdv.columns else {}
    out = []
    for r in df.to_dict(orient="records"):
        drivers = []
        for i in range(1, 5):
            lab = r.get(f"driver_{i}")
            if lab is None or (isinstance(lab, float) and pd.isna(lab)):
                continue
            drivers.append(
                {
                    "label": str(lab),
                    "delta": float(r.get(f"driver_{i}_delta", 0) or 0),
                    "dir": str(r.get(f"driver_{i}_dir", "")),
                }
            )
        out.append(
            {
                "as_of": str(r.get("as_of", ""))[:10],
                "pdv_id": str(r.get("pdv_id", "")),
                "sku_id": str(r.get("sku_id", "")),
                "nome": str(r.get("nome", r.get("modello", ""))),
                "stagione": _season_of(r.get("sku_id", ""), season_map or {}),
                "ssclasse": str(r.get("ssclasse", "")),
                "city": str(city_map.get(r.get("pdv_id"), r.get("pdv_id", ""))),
                "pred": float(r.get("pred_tf", r.get("vendite_previste_15gg", 0)) or 0),
                "pred_model": float(r.get("pred_model", 0) or 0),
                "pred_analogue": float(r.get("pred_analogue", 0) or 0),
                "pred_cdm": float(r.get("pred_cdm", 0) or 0),
                "giacenza": float(r.get("giacenza", 0) or 0),
                "qty": float(r.get("qty_suggerita", 0) or 0),
                "azione": str(r.get("azione", "")),
                "confidence": float(r.get("confidence_score", 0) or 0),
                "run_rate_14d": float(r.get("run_rate_14d", 0) or 0),
                "is_new_pdv": int(r.get("is_new_pdv", 0) or 0),
                "spiegazione": str(r.get("spiegazione_testuale", "")).replace("<strong>", "").replace("</strong>", ""),
                "drivers": drivers,
            }
        )
    return out


def _prepare_sku_network(
    priorities: pd.DataFrame,
    pdv: pd.DataFrame,
    products: pd.DataFrame,
    season_map: dict[str, str],
) -> dict:
    """Previsione 15 gg di ogni SKU in tutti i negozi (non solo top qty)."""
    if priorities.empty:
        return {"as_of": "", "skus": [], "by_sku": {}}

    df = priorities.copy()
    pdv_l = pdv[["pdv_id", "country", "city"]].copy() if not pdv.empty else pd.DataFrame()
    if not pdv_l.empty:
        df = df.merge(pdv_l, on="pdv_id", how="left")
    nome_map: dict[str, str] = {}
    stag_map: dict[str, str] = {}
    mod_map: dict[str, str] = {}
    if not products.empty and "sku_id" in products.columns:
        for r in products.drop_duplicates("sku_id").to_dict(orient="records"):
            sid = str(r.get("sku_id", ""))
            nome_map[sid] = str(r.get("nome", r.get("modello", "")) or "")
            stag_map[sid] = _season_of(sid, season_map)
            mod_map[sid] = str(r.get("modello", "") or "")

    as_of = ""
    if "as_of" in df.columns and len(df):
        as_of = str(df["as_of"].iloc[0])[:10]

    by_sku: dict[str, dict] = {}
    for r in df.to_dict(orient="records"):
        sid = str(r.get("sku_id", ""))
        if not sid:
            continue
        if sid not in by_sku:
            by_sku[sid] = {
                "nome": nome_map.get(sid, str(r.get("nome", r.get("modello", "")) or "")),
                "stagione": stag_map.get(sid) or _season_of(sid, season_map),
                "modello": mod_map.get(sid, str(r.get("modello", "") or "")),
                "rows": [],
            }
        azione = str(r.get("azione", "monitora") or "monitora")
        by_sku[sid]["rows"].append(
            [
                str(r.get("pdv_id", "")),
                str(r.get("city", r.get("pdv_id", ""))),
                str(r.get("country", "")),
                round(float(r.get("pred_tf", r.get("vendite_previste_15gg", 0)) or 0), 1),
                round(float(r.get("pred_cdm", 0) or 0), 1),
                round(float(r.get("giacenza", 0) or 0), 1),
                round(float(r.get("qty_suggerita", 0) or 0), 0),
                round(float(r.get("confidence_score", 0) or 0), 3),
                1 if azione == "valuta_riassortimento" else 0,
                int(r.get("is_new_pdv", 0) or 0),
            ]
        )

    skus = []
    for sid, meta in sorted(by_sku.items()):
        nome = meta["nome"] or sid
        skus.append(
            {
                "id": sid,
                "label": f"{sid} — {nome}",
                "nome": nome,
                "stagione": meta["stagione"],
            }
        )
    return {"as_of": as_of, "skus": skus, "by_sku": by_sku}


def _aggregate_by_country(rows: list[dict]) -> list[dict]:
    agg: dict[str, float] = {}
    for r in rows:
        c = r.get("country") or "?"
        agg[c] = agg.get(c, 0.0) + float(r.get("qty", 0))
    return [{"country": k, "qty": round(v, 1)} for k, v in sorted(agg.items(), key=lambda x: -x[1])]


def _aggregate_by_city(rows: list[dict], top: int = 12) -> list[dict]:
    agg: dict[str, float] = {}
    for r in rows:
        key = f"{r.get('city', '?')} ({r.get('country', '')})"
        agg[key] = agg.get(key, 0.0) + float(r.get("qty", 0))
    items = sorted(agg.items(), key=lambda x: -x[1])[:top]
    return [{"label": k, "qty": round(v, 1)} for k, v in items]


def build_bi_payload(data_dir: Path = DATA_DIR) -> dict:
    inputs = load_report_inputs(data_dir)
    products_path = data_dir / "products.csv"
    products = pd.read_csv(products_path) if products_path.exists() else pd.DataFrame()
    season_map = _load_season_map(data_dir, inputs["priorities"])

    meta_d = dict(zip(inputs["meta"]["metric"], inputs["meta"]["value"]))
    priorities = _prepare_priorities(inputs["priorities"], inputs["pdv"], products, season_map)
    focus = _prepare_focus(inputs["focus"], inputs["pdv"], products, season_map)
    sku_network = _prepare_sku_network(inputs["priorities"], inputs["pdv"], products, season_map)

    hist = inputs["history"]
    train_mae = hist["val_y_sku_mae"].tolist() if not hist.empty and "val_y_sku_mae" in hist.columns else []

    metrics = inputs["metrics"]
    return {
        "generated_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "meta": {
            "n_pdv": int(meta_d.get("n_pdv", 0)),
            "n_sku": int(meta_d.get("n_sku", 0)),
            "production_n_pdv": PRODUCTION_N_PDV,
            "production_n_sku": PRODUCTION_N_SKU,
            "date_min": str(meta_d.get("date_min", "")),
            "date_max": str(meta_d.get("date_max", "")),
        },
        "metrics": metrics,
        "priorities": priorities,
        "sku_network": sku_network,
        "focus": focus,
        "by_country": _aggregate_by_country(priorities),
        "by_city": _aggregate_by_city(priorities),
        "train_mae": train_mae,
        "filters": {
            "countries": sorted({r["country"] for r in priorities if r.get("country")}),
            "cities": sorted({r["city"] for r in priorities if r.get("city")}),
            "categorie": sorted({r["categoria"] for r in priorities if r.get("categoria")}),
            "stagioni": sorted({v for v in season_map.values() if v})
            or sorted({r["stagione"] for r in priorities if r.get("stagione")}),
            "ssclassi": sorted({r["ssclasse"] for r in priorities if r.get("ssclasse")}),
            "skus": sku_network.get("skus")
            or sorted(
                [
                    {
                        "id": r["sku_id"],
                        "label": f"{r['sku_id']} — {r.get('nome') or r.get('modello') or r['sku_id']}",
                    }
                    for r in {p["sku_id"]: p for p in priorities}.values()
                ],
                key=lambda x: x["id"],
            ),
        },
    }


def build_bi_html(payload: dict) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    m = payload.get("metrics", {})
    meta = payload.get("meta", {})
    mae = float(m.get("mae_tf", 0))
    r2 = float(m.get("r2", 0))
    r2_pct = int(round(r2 * 100))

    return f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CDM Forecast — Dashboard interattiva</title>
  <style>
    :root {{
      --bg: #f1f5f9; --surface: #fff; --text: #0f172a; --muted: #64748b;
      --border: #e2e8f0; --accent: #2563eb; --accent-soft: #eff6ff;
      --ok: #16a34a; --warn: #d97706;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg); color: var(--text); }}
    .app {{ display: grid; grid-template-columns: 280px 1fr; min-height: 100vh; }}
    .sidebar {{ background: var(--surface); border-right: 1px solid var(--border); padding: 20px 16px;
                position: sticky; top: 0; height: 100vh; overflow-y: auto; }}
    .main {{ padding: 20px 24px 40px; overflow-x: hidden; }}
    h1 {{ font-size: 1.35rem; margin: 0 0 4px; }}
    .sub {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 18px; }}
    label {{ display: block; font-size: 0.75rem; font-weight: 600; color: var(--muted);
             text-transform: uppercase; letter-spacing: 0.04em; margin: 14px 0 6px; }}
    input, select {{ width: 100%; padding: 8px 10px; border: 1px solid var(--border);
                     border-radius: 8px; font-size: 0.9rem; background: #fff; }}
    .tabs {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }}
    .tab {{ padding: 8px 14px; border: 1px solid var(--border); border-radius: 999px;
            background: var(--surface); cursor: pointer; font-size: 0.88rem; }}
    .tab.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
    .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .kpi {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }}
    .kpi .lbl {{ font-size: 0.72rem; color: var(--muted); text-transform: uppercase; }}
    .kpi .val {{ font-size: 1.45rem; font-weight: 700; margin-top: 4px; }}
    .kpi .hint {{ font-size: 0.78rem; color: var(--muted); margin-top: 2px; }}
    .kpi.primary {{ border-color: #93c5fd; background: var(--accent-soft); }}
    .panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
              padding: 16px; margin-bottom: 16px; }}
    .panel h2 {{ font-size: 1rem; margin: 0 0 12px; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; }}
    th, td {{ border-bottom: 1px solid var(--border); padding: 8px 6px; text-align: left; }}
    th {{ background: #f8fafc; cursor: pointer; user-select: none; position: sticky; top: 0; }}
    th:hover {{ background: #eef2ff; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    tr.row-click {{ cursor: pointer; }}
    tr.row-click:hover {{ background: #f8fafc; }}
    tr.selected {{ background: #eff6ff; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.72rem; font-weight: 600; }}
    .badge-ok {{ background: #dcfce7; color: #166534; }}
    .badge-warn {{ background: #fef3c7; color: #92400e; }}
    .detail {{ border-left: 3px solid var(--accent); padding-left: 14px; }}
    .detail h3 {{ margin: 0 0 8px; font-size: 1rem; }}
    .drivers {{ list-style: none; padding: 0; margin: 10px 0 0; }}
    .drivers li {{ padding: 6px 0; border-bottom: 1px dashed var(--border); font-size: 0.88rem; }}
    .drivers .up {{ color: var(--ok); font-weight: 700; }}
    .drivers .down {{ color: #dc2626; font-weight: 700; }}
    .bar-row {{ display: flex; align-items: center; gap: 10px; margin: 6px 0; font-size: 0.85rem; }}
    .bar-track {{ flex: 1; height: 10px; background: #e2e8f0; border-radius: 4px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: var(--accent); border-radius: 4px; }}
    .bar-lbl {{ width: 120px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .bar-val {{ width: 48px; text-align: right; font-variant-numeric: tabular-nums; }}
    .hidden {{ display: none !important; }}
    .toolbar {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }}
    .toolbar span {{ color: var(--muted); font-size: 0.85rem; }}
    .btn {{ padding: 7px 12px; border: 1px solid var(--border); border-radius: 8px; background: #fff;
            cursor: pointer; font-size: 0.85rem; }}
    .btn:hover {{ background: #f8fafc; }}
    .btn-why {{ padding: 5px 10px; font-size: 0.78rem; white-space: nowrap; color: var(--accent); border-color: #bfdbfe; }}
    .btn-why:hover {{ background: var(--accent-soft); }}
    .btn-why.active {{ background: var(--accent-soft); border-color: var(--accent); font-weight: 600; }}
    tr.why-expand td {{
      background: #f8fafc;
      padding: 12px 16px 14px;
      border-bottom: 2px solid #bfdbfe;
    }}
    tr.why-expand .detail {{ border-left: none; padding: 0; margin: 0; }}
    .table-wrap {{ max-height: 520px; overflow: auto; border: 1px solid var(--border); border-radius: 8px; }}
    .sku-pick {{ display: flex; gap: 10px; align-items: flex-end; flex-wrap: wrap; margin-bottom: 12px; }}
    .sku-pick .field {{ flex: 1; min-width: 240px; }}
    .sku-pick label {{ margin-top: 0; }}
    @media (max-width: 960px) {{
      .app {{ grid-template-columns: 1fr; }}
      .sidebar {{ height: auto; position: static; }}
      .grid2 {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <h1>CDM Forecast</h1>
    <p class="sub">Mini-BI · agg. {payload.get("generated_at", "")}<br>
    {meta.get("n_pdv", "—")} PDV · {meta.get("n_sku", "—")} SKU</p>

    <label for="f-search">Cerca PDV / SKU / città</label>
    <input id="f-search" type="search" placeholder="es. MIL_01, crop, collana, Milan…">

    <label for="f-sku">SKU</label>
    <input id="f-sku" list="sku-list" placeholder="Cerca codice o nome…">
    <datalist id="sku-list"></datalist>

    <label for="f-country">Paese</label>
    <select id="f-country"><option value="">Tutti</option></select>

    <label for="f-city">Città</label>
    <select id="f-city"><option value="">Tutte</option></select>

    <label for="f-cat">Categoria</label>
    <select id="f-cat"><option value="">Tutte</option></select>

    <label for="f-stagione">Stagione stimata (dalle vendite)</label>
    <select id="f-stagione"><option value="">Tutte</option></select>

    <label for="f-tipo">Tipo prodotto</label>
    <select id="f-tipo"><option value="">Tutti</option></select>

    <label for="f-minqty">Qty minima suggerita</label>
    <input id="f-minqty" type="number" min="0" step="1" value="1">

    <label for="f-conf">Confidence minima</label>
    <input id="f-conf" type="range" min="0" max="100" value="0">
    <div class="sub" id="conf-lbl">0%</div>

    <button class="btn" id="btn-reset" style="margin-top:16px;width:100%">Reset filtri</button>
    <button class="btn" id="btn-export" style="margin-top:8px;width:100%">Esporta CSV filtrato</button>
  </aside>

  <main class="main">
    <div class="tabs" id="tabs">
      <button class="tab active" data-tab="overview">Panoramica</button>
      <button class="tab" data-tab="sku">Per SKU</button>
      <button class="tab" data-tab="priorities">Priorità riassortimento</button>
      <button class="tab" data-tab="model">Accuratezza</button>
    </div>

    <div class="kpis" id="kpis"></div>

    <section id="tab-overview" class="tab-panel">
      <div class="grid2">
        <div class="panel">
          <h2>Qty suggerita per paese</h2>
          <div id="chart-country"></div>
        </div>
        <div class="panel">
          <h2>Top negozi per qty suggerita</h2>
          <div id="chart-city"></div>
        </div>
      </div>
      <div class="panel">
        <h2>Qty per stagione stimata</h2>
        <p class="sub">Dedotta dalla rete: temperatura a cui quello SKU ha venduto (niente tag catalogo). «Tende al caldo» / «tende al freddo» / «tutto l'anno».</p>
        <div id="chart-stagione"></div>
      </div>
      <div class="panel">
        <h2>Dettaglio selezionato</h2>
        <div id="detail-overview" class="detail"><em>Seleziona una riga in Priorità per il dettaglio.</em></div>
      </div>
    </section>

    <section id="tab-sku" class="tab-panel hidden">
      <div class="panel">
        <h2>Previsione 15 giorni per SKU, negozio per negozio</h2>
        <p class="sub" id="sku-horizon">Scegli un articolo: domanda attesa, giacenza e qty suggerita in ogni PDV. Clicca <strong>Perché</strong> su un negozio per aprire la spiegazione sotto la riga.</p>
        <div class="sku-pick">
          <div class="field">
            <label for="sku-pick">SKU</label>
            <input id="sku-pick" list="sku-list" placeholder="es. SKU_A, crop, collana…">
          </div>
          <button class="btn" id="btn-sku-export">Esporta CSV di questo SKU</button>
        </div>
        <div class="kpis" id="sku-kpis"></div>
        <div class="grid2">
          <div>
            <h2 style="font-size:1rem;margin:0 0 12px">Top negozi per domanda attesa</h2>
            <div id="chart-sku-pdv"></div>
          </div>
          <div>
            <h2 style="font-size:1rem;margin:0 0 12px">Qty suggerita per paese</h2>
            <div id="chart-sku-country"></div>
          </div>
        </div>
        <div class="toolbar" style="margin-top:14px">
          <span id="sku-count">—</span>
        </div>
        <div class="table-wrap">
          <table id="sku-table">
            <thead>
              <tr>
                <th data-sku-key="city">Negozio</th>
                <th data-sku-key="country">Paese</th>
                <th data-sku-key="pred" class="num">Prev. 15gg</th>
                <th data-sku-key="giacenza" class="num">Giac.</th>
                <th data-sku-key="qty" class="num">Qty</th>
                <th data-sku-key="confidence" class="num">Conf.</th>
                <th data-sku-key="azione">Azione</th>
                <th></th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </section>

    <section id="tab-priorities" class="tab-panel hidden">
      <div class="panel">
        <div class="toolbar">
          <span id="prio-count">—</span>
          <button class="btn" id="sort-qty">Ordina per qty ↓</button>
        </div>
        <div class="table-wrap">
          <table id="prio-table">
            <thead>
              <tr>
                <th data-key="city">Negozio</th>
                <th data-key="sku_id">SKU</th>
                <th data-key="stagione">Stagione stimata</th>
                <th data-key="categoria">Cat</th>
                <th data-key="pred" class="num">Prev. 15gg</th>
                <th data-key="giacenza" class="num">Giac.</th>
                <th data-key="qty" class="num">Qty</th>
                <th data-key="confidence" class="num">Conf.</th>
                <th></th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </section>

    <section id="tab-model" class="tab-panel hidden">
      <div class="grid2">
        <div class="panel">
          <h2>Accuratezza operativa</h2>
          <ul id="model-metrics"></ul>
        </div>
        <div class="panel">
          <h2>Apprendimento (MAE SKU validazione)</h2>
          <div id="chart-train"></div>
        </div>
      </div>
      <p class="sub">Confronto ufficiale: rete vs <strong>modello CDM originale</strong>
      (stesso test, tutti gli SKU). Lo spezzamento sotto usa etichette fake del dataset
      testing solo come controllo: la rete non le vede.</p>
    </section>
  </main>
</div>

<script>
const DATA = {data_json};

let filtered = [...DATA.priorities];
let sortKey = "qty";
let sortAsc = false;
let selectedKey = null;
let expandedPrioKey = null;
let expandedSkuPdv = null;

function fmt(n, d=1) {{ return Number(n).toLocaleString("it-IT", {{maximumFractionDigits: d}}); }}
function badge(azione) {{
  if (azione === "valuta_riassortimento") return '<span class="badge badge-warn">Valuta riassortimento</span>';
  return '<span class="badge badge-ok">Monitora</span>';
}}

function rowKey(r) {{ return r.pdv_id + "|" + r.sku_id + "|" + r.as_of; }}

function applyFilters() {{
  const q = document.getElementById("f-search").value.trim().toLowerCase();
  const skuQ = document.getElementById("f-sku").value.trim().toLowerCase();
  const country = document.getElementById("f-country").value;
  const city = document.getElementById("f-city").value;
  const cat = document.getElementById("f-cat").value;
  const stagione = document.getElementById("f-stagione").value;
  const tipo = document.getElementById("f-tipo").value;
  const minQty = Number(document.getElementById("f-minqty").value || 0);
  const minConf = Number(document.getElementById("f-conf").value || 0) / 100;

  filtered = DATA.priorities.filter(r => {{
    if (country && r.country !== country) return false;
    if (city && r.city !== city) return false;
    if (cat && r.categoria !== cat) return false;
    if (stagione && r.stagione !== stagione) return false;
    if (tipo && r.ssclasse !== tipo) return false;
    if (r.qty < minQty) return false;
    if (r.confidence < minConf) return false;
    if (skuQ) {{
      const skuBlob = [r.sku_id, r.nome, r.modello, r.ssclasse].join(" ").toLowerCase();
      if (!skuBlob.includes(skuQ)) return false;
    }}
    if (q) {{
      const blob = [r.pdv_id, r.sku_id, r.city, r.country, r.modello, r.nome, r.ssclasse, r.stagione, r.commerciale].join(" ").toLowerCase();
      if (!blob.includes(q)) return false;
    }}
    return true;
  }});
  sortRows();
  renderKpis();
  renderTable();
  renderChartsFiltered();
  document.getElementById("prio-count").textContent = filtered.length + " righe (max embed " + DATA.priorities.length + ")";
}}

function sortRows() {{
  filtered.sort((a,b) => {{
    const va = a[sortKey], vb = b[sortKey];
    if (typeof va === "string") return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    return sortAsc ? va - vb : vb - va;
  }});
}}

function renderKpis() {{
  const totQty = filtered.reduce((s,r) => s + r.qty, 0);
  const nShop = new Set(filtered.map(r => r.pdv_id)).size;
  const avgConf = filtered.length ? filtered.reduce((s,r)=>s+r.confidence,0)/filtered.length : 0;
  document.getElementById("kpis").innerHTML = `
    <div class="kpi primary"><div class="lbl">Qty totale filtrata</div><div class="val">${{fmt(totQty,0)}}</div><div class="hint">pezzi suggeriti</div></div>
    <div class="kpi"><div class="lbl">Righe / negozi</div><div class="val">${{filtered.length}} / ${{nShop}}</div></div>
    <div class="kpi"><div class="lbl">Errore medio</div><div class="val">${{fmt({mae})}}</div><div class="hint">pezzi / 15 gg · tutti gli SKU</div></div>
    <div class="kpi"><div class="lbl">Domanda spiegata</div><div class="val">{r2_pct}%</div><div class="hint">R² sul periodo di test</div></div>
    <div class="kpi"><div class="lbl">Conf. media</div><div class="val">${{Math.round(avgConf*100)}}%</div></div>`;
}}

function renderBars(elId, items, labelKey) {{
  const el = document.getElementById(elId);
  if (!items.length) {{ el.innerHTML = "<em>Nessun dato</em>"; return; }}
  const max = Math.max(...items.map(x => x.qty || x.value || 0), 1);
  el.innerHTML = items.map(it => {{
    const lbl = it[labelKey] || it.label || it.country;
    const val = it.qty || it.value || 0;
    const w = Math.max(4, (val/max)*100);
    return `<div class="bar-row"><div class="bar-lbl" title="${{lbl}}">${{lbl}}</div><div class="bar-track"><div class="bar-fill" style="width:${{w}}%"></div></div><div class="bar-val">${{fmt(val,0)}}</div></div>`;
  }}).join("");
}}

function renderChartsFiltered() {{
  const byC = {{}};
  const byCity = {{}};
  const byStag = {{}};
  filtered.forEach(r => {{
    byC[r.country] = (byC[r.country]||0) + r.qty;
    const ck = r.city + " (" + r.country + ")";
    byCity[ck] = (byCity[ck]||0) + r.qty;
    const st = r.stagione || "—";
    byStag[st] = (byStag[st]||0) + r.qty;
  }});
  const cItems = Object.entries(byC).map(([country,qty]) => ({{country, qty}})).sort((a,b)=>b.qty-a.qty);
  const cityItems = Object.entries(byCity).map(([label,qty]) => ({{label, qty}})).sort((a,b)=>b.qty-a.qty).slice(0,12);
  const order = ["tende al caldo","tende al freddo","tutto l'anno","poco storico"];
  const stagItems = Object.entries(byStag).map(([label,qty]) => ({{label, qty}}))
    .sort((a,b) => (order.indexOf(a.label)+99) - (order.indexOf(b.label)+99));
  renderBars("chart-country", cItems, "country");
  renderBars("chart-city", cityItems, "label");
  renderBars("chart-stagione", stagItems, "label");
}}

function simpleNarrative(r) {{
  const pred = Number(r.pred) || 0;
  const giac = Number(r.giacenza) || 0;
  const qty = Number(r.qty) || 0;
  let t = `Nei prossimi 15 giorni il modello stima che questo negozio venderà circa <strong>${{fmt(pred)}} pezzi</strong> di questo SKU. `;
  t += `Oggi in magazzino ce ne sono <strong>${{fmt(giac,0)}}</strong>. `;
  if (qty > 0) {{
    if (giac >= pred) {{
      t += `Quelli in negozio coprono già le vendite attese dei 15 giorni. La qty suggerita (<strong>${{fmt(qty,0)}}</strong>) serve a coprire anche i giorni di transito: mentre arriva la merce si continua a vendere. Azione: valuta riassortimento. `;
    }} else {{
      t += `Non bastano per le vendite attese, quindi qty suggerita <strong>${{fmt(qty,0)}}</strong> (valuta riassortimento). `;
    }}
  }} else {{
    t += `Con questa giacenza copri domanda e transito: di solito si monitora, senza riassortire. `;
  }}
  if (r.is_new_pdv) t += "Negozio nuovo: la previsione è più prudente (poco storico). ";
  if (r.confidence) t += `Quanto ci fidiamo dello storico di questo SKU/negozio: ${{Math.round(r.confidence*100)}}%.`;
  return t;
}}

function findFocus(pdvId, skuId) {{
  return (DATA.focus || []).find(x => x.pdv_id === pdvId && x.sku_id === skuId) || null;
}}

function mergeWhyRow(r) {{
  const focus = findFocus(r.pdv_id, r.sku_id);
  const merged = Object.assign({{}}, r, focus || {{}});
  if (!merged.sku_id) merged.sku_id = currentSkuId || r.sku_id;
  if (!merged.nome && DATA.sku_network && DATA.sku_network.by_sku && merged.sku_id) {{
    const meta = DATA.sku_network.by_sku[merged.sku_id];
    if (meta) {{ merged.nome = meta.nome; merged.stagione = merged.stagione || meta.stagione; }}
  }}
  return {{ merged, isFocus: !!focus }};
}}

function openWhy(r) {{
  const {{ merged, isFocus }} = mergeWhyRow(r);
  const html = renderDetail(merged, isFocus);
  const ov = document.getElementById("detail-overview");
  if (ov) ov.innerHTML = html;
}}

function renderDetail(r, isFocus=false) {{
  if (!r) return "<em>Nessuna selezione</em>";
  const drivers = (r.drivers || []).map(d =>
    `<li><span class="${{d.dir==='alza'?'up':'down'}}">${{d.dir==='alza'?'↑':'↓'}}</span> ${{d.label}} (${{d.delta>=0?'+':''}}${{fmt(d.delta)}} pezzi)</li>`
  ).join("");
  const narr = r.spiegazione || simpleNarrative(r);
  return `<h3>${{r.city || r.pdv_id}} · ${{r.nome || r.sku_id}} <small>(${{r.sku_id || currentSkuId || ""}}${{r.stagione ? " · "+r.stagione : ""}})</small></h3>
    <p>${{narr}}</p>
    <p>${{badge(r.azione || 'monitora')}} · Prev. 15gg: ${{fmt(r.pred)}} · Giac: ${{fmt(r.giacenza,0)}} · Qty: ${{fmt(r.qty,0)}} · Conf. ${{Math.round((r.confidence||0)*100)}}%</p>
    ${{drivers ? '<div class="chart-title">Driver principali</div><ul class="drivers">'+drivers+'</ul>' : ''}}`;
}}

function renderTable() {{
  const tb = document.querySelector("#prio-table tbody");
  const rowsHtml = [];
  filtered.slice(0, 500).forEach(r => {{
    const k = rowKey(r);
    const sel = selectedKey === k ? "selected" : "";
    const whyOpen = expandedPrioKey === k;
    rowsHtml.push(`<tr class="row-click ${{sel}}" data-key="${{k}}">
      <td>${{r.city}}<br><small>${{r.pdv_id}}</small></td>
      <td>${{r.sku_id}}<br><small>${{r.nome||r.modello||''}}</small></td>
      <td>${{r.stagione||'—'}}</td>
      <td>${{r.categoria||'—'}}</td>
      <td class="num">${{fmt(r.pred)}}</td>
      <td class="num">${{fmt(r.giacenza,0)}}</td>
      <td class="num"><strong>${{fmt(r.qty,0)}}</strong></td>
      <td class="num">${{Math.round(r.confidence*100)}}%</td>
      <td><button class="btn btn-why ${{whyOpen ? 'active' : ''}}" data-why="${{k}}">${{whyOpen ? 'Chiudi' : 'Perché'}}</button></td>
    </tr>`);
    if (whyOpen) {{
      const {{ merged, isFocus }} = mergeWhyRow(r);
      rowsHtml.push(`<tr class="why-expand"><td colspan="9"><div class="detail">${{renderDetail(merged, isFocus)}}</div></td></tr>`);
    }}
  }});
  tb.innerHTML = rowsHtml.join("");
  tb.querySelectorAll("tr.row-click").forEach(tr => {{
    tr.addEventListener("click", (ev) => {{
      if (ev.target.closest && ev.target.closest(".btn-why")) return;
      selectedKey = tr.dataset.key;
      const r = filtered.find(x => rowKey(x) === selectedKey);
      if (r) openWhy(r);
      renderTable();
    }});
  }});
  tb.querySelectorAll(".btn-why").forEach(btn => {{
    btn.addEventListener("click", (ev) => {{
      ev.stopPropagation();
      const k = btn.dataset.why;
      expandedPrioKey = expandedPrioKey === k ? null : k;
      selectedKey = k;
      renderTable();
    }});
  }});
}}

function renderExplainCards() {{}}

function renderModelTab() {{
  const m = DATA.metrics || {{}};
  const r2pct = Math.round((m.r2 || 0) * 100);
  const cold = m.mae_cold_sku_confidence;
  const coldLi = (cold || cold === 0)
    ? `<li><strong>SKU nuovi (&lt; 30 gg):</strong> errore ${{fmt(cold)}} pezzi (blend analoghi)</li>`
    : "";
  function segLi(label, s, hint) {{
    if (!s) return "";
    const lift = Number(s.improvement_pct || 0);
    const liftTxt = (lift >= 0 ? lift.toFixed(0) + "% di errore in meno vs modello CDM" : Math.abs(lift).toFixed(0) + "% di errore in più vs modello CDM");
    return `<li><strong>${{label}}:</strong> errore ${{fmt(s.mae)}} pezzi (${{liftTxt}})${{hint ? " — "+hint : ""}}</li>`;
  }}
  document.getElementById("model-metrics").innerHTML = `
    <li><strong>Rete neurale (tutti gli SKU):</strong> ${{fmt(m.mae_tf||0)}} pezzi / 15 gg · R² ${{r2pct}}%</li>
    <li><strong>Modello CDM originale:</strong> ${{fmt(m.mae_cdm || m.mae_naive || 0)}} pezzi
      · lift rete ${{(Number(m.improvement_pct||0)>=0?'+':'')}}${{Number(m.improvement_pct||0).toFixed(0)}}%</li>
    <li><strong>Naive run-rate×15:</strong> ${{fmt(m.mae_naive||0)}} pezzi (secondo controllo)</li>
    ${{coldLi}}
    <li style="margin-top:8px"><em>Controllo su etichette fake testing (non usate dalla rete):</em></li>
    ${{segLi("Prodotti ESTATE", m.seg_estate)}}
    ${{segLi("Prodotti INVERNO", m.seg_inverno)}}
    ${{segLi("Prodotti CONTINUATIVO", m.seg_continuativo)}}
    <li><strong>Orizzonte:</strong> ${{m.horizon_sku_days||15}} giorni · apprendimento ultimi ${{m.train_window_days||'—'}} giorni</li>`;
  const pts = DATA.train_mae || [];
  if (!pts.length) {{ document.getElementById("chart-train").innerHTML = "<em>Storico training non disponibile</em>"; return; }}
  const max = Math.max(...pts), min = Math.min(...pts);
  const span = Math.max(max-min, 0.01);
  const w=480,h=140,pad=30;
  const poly = pts.map((v,i)=>{{
    const x = pad + (i/Math.max(pts.length-1,1))*(w-pad-10);
    const y = pad + (1-(v-min)/span)*(h-pad-20);
    return x+','+y;
  }}).join(' ');
  document.getElementById("chart-train").innerHTML = `
    <svg viewBox="0 0 ${{w}} ${{h}}" width="100%" height="${{h}}">
      <polyline points="${{poly}}" fill="none" stroke="#2563eb" stroke-width="2"/>
    </svg>
    <div class="sub">Epoche 1–${{pts.length}} · MAE SKU validazione · Source: training_history.csv</div>`;
}}

function fillSelect(id, values) {{
  const sel = document.getElementById(id);
  values.forEach(v => {{
    const o = document.createElement("option");
    o.value = v; o.textContent = v;
    sel.appendChild(o);
  }});
}}

function exportCsv() {{
  const cols = ["as_of","country","city","pdv_id","sku_id","nome","stagione","ssclasse","categoria","modello","pred","giacenza","qty","confidence","azione"];
  const lines = [cols.join(",")].concat(filtered.map(r => cols.map(c => JSON.stringify(r[c] ?? "")).join(",")));
  const blob = new Blob([lines.join("\\n")], {{type:"text/csv;charset=utf-8"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "priorita_filtrata.csv";
  a.click();
}}

function unpackSkuRow(arr) {{
  return {{
    pdv_id: arr[0],
    city: arr[1],
    country: arr[2],
    pred: arr[3],
    pred_cdm: arr[4],
    giacenza: arr[5],
    qty: arr[6],
    confidence: arr[7],
    azione: arr[8] ? "valuta_riassortimento" : "monitora",
    is_new_pdv: arr[9] || 0,
  }};
}}

function resolveSkuId(raw) {{
  const q = (raw || "").trim();
  if (!q) return "";
  const net = DATA.sku_network || {{}};
  if (net.by_sku && net.by_sku[q]) return q;
  const lower = q.toLowerCase();
  const hit = (net.skus || DATA.filters.skus || []).find(s =>
    s.id.toLowerCase() === lower ||
    (s.label || "").toLowerCase().includes(lower) ||
    (s.nome || "").toLowerCase().includes(lower)
  );
  return hit ? hit.id : "";
}}

let skuRows = [];
let skuSortKey = "pred";
let skuSortAsc = false;
let currentSkuId = "";

function renderSkuTab() {{
  const net = DATA.sku_network || {{}};
  const raw = document.getElementById("sku-pick").value;
  const sid = resolveSkuId(raw);
  const horizonEl = document.getElementById("sku-horizon");
  const asOf = net.as_of || "";
  if (!sid || !net.by_sku || !net.by_sku[sid]) {{
    currentSkuId = "";
    expandedSkuPdv = null;
    skuRows = [];
    document.getElementById("sku-kpis").innerHTML = "";
    document.getElementById("chart-sku-pdv").innerHTML = "<em>Seleziona uno SKU dal menu.</em>";
    document.getElementById("chart-sku-country").innerHTML = "";
    document.getElementById("sku-count").textContent = (net.skus || []).length + " SKU in catalogo";
    document.querySelector("#sku-table tbody").innerHTML = "";
    if (horizonEl) horizonEl.textContent = asOf
      ? ("Chiusura " + asOf + " · domanda attesa nei 15 giorni successivi, per ogni negozio.")
      : "Scegli un articolo: domanda attesa, giacenza e qty suggerita in ogni PDV.";
    return;
  }}
  currentSkuId = sid;
  expandedSkuPdv = null;
  const meta = net.by_sku[sid];
  skuRows = (meta.rows || []).map(unpackSkuRow);
  const totPred = skuRows.reduce((s,r) => s + r.pred, 0);
  const totQty = skuRows.reduce((s,r) => s + r.qty, 0);
  const nValuta = skuRows.filter(r => r.azione === "valuta_riassortimento").length;
  const avgConf = skuRows.length ? skuRows.reduce((s,r)=>s+r.confidence,0)/skuRows.length : 0;
  if (horizonEl) {{
    horizonEl.textContent = (meta.nome || sid) + (meta.stagione ? " · " + meta.stagione : "")
      + (asOf ? " · chiusura " + asOf : "") + " · prossimi 15 giorni";
  }}
  document.getElementById("sku-kpis").innerHTML = `
    <div class="kpi primary"><div class="lbl">Domanda rete 15 gg</div><div class="val">${{fmt(totPred,0)}}</div><div class="hint">somma su ${{skuRows.length}} negozi</div></div>
    <div class="kpi"><div class="lbl">Qty suggerita</div><div class="val">${{fmt(totQty,0)}}</div><div class="hint">pezzi da valutare</div></div>
    <div class="kpi"><div class="lbl">Negozi da riassortire</div><div class="val">${{nValuta}}</div><div class="hint">su ${{skuRows.length}} PDV</div></div>
    <div class="kpi"><div class="lbl">Conf. media</div><div class="val">${{Math.round(avgConf*100)}}%</div></div>`;
  document.getElementById("sku-count").textContent = skuRows.length + " negozi · " + sid;
  const topPdv = [...skuRows].sort((a,b)=>b.pred-a.pred).slice(0,12).map(r => ({{label: r.city + " (" + r.pdv_id + ")", qty: r.pred}}));
  const byC = {{}};
  skuRows.forEach(r => {{ byC[r.country] = (byC[r.country]||0) + r.qty; }});
  const cItems = Object.entries(byC).map(([country,qty]) => ({{country, qty}})).sort((a,b)=>b.qty-a.qty);
  renderBars("chart-sku-pdv", topPdv, "label");
  renderBars("chart-sku-country", cItems, "country");
  renderSkuTable();
}}

function renderSkuTable() {{
  const tb = document.querySelector("#sku-table tbody");
  const sorted = [...skuRows].sort((a,b) => {{
    const va = a[skuSortKey], vb = b[skuSortKey];
    if (typeof va === "string") return skuSortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    return skuSortAsc ? va - vb : vb - va;
  }});
  const rowsHtml = [];
  sorted.forEach(r => {{
    const whyOpen = expandedSkuPdv === r.pdv_id;
    const sel = whyOpen ? "selected" : "";
    rowsHtml.push(`<tr class="${{sel}}" data-pdv="${{r.pdv_id}}">
      <td>${{r.city}}<br><small>${{r.pdv_id}}</small></td>
      <td>${{r.country}}</td>
      <td class="num"><strong>${{fmt(r.pred)}}</strong></td>
      <td class="num">${{fmt(r.giacenza,0)}}</td>
      <td class="num">${{fmt(r.qty,0)}}</td>
      <td class="num">${{Math.round(r.confidence*100)}}%</td>
      <td>${{badge(r.azione)}}</td>
      <td><button class="btn btn-why ${{whyOpen ? 'active' : ''}}" data-pdv="${{r.pdv_id}}">${{whyOpen ? 'Chiudi' : 'Perché'}}</button></td>
    </tr>`);
    if (whyOpen) {{
      const row = Object.assign({{}}, r, {{ sku_id: currentSkuId }});
      const {{ merged, isFocus }} = mergeWhyRow(row);
      rowsHtml.push(`<tr class="why-expand"><td colspan="8"><div class="detail">${{renderDetail(merged, isFocus)}}</div></td></tr>`);
    }}
  }});
  tb.innerHTML = rowsHtml.join("");
  tb.querySelectorAll(".btn-why").forEach(btn => {{
    btn.addEventListener("click", (ev) => {{
      ev.stopPropagation();
      const pdv = btn.dataset.pdv;
      expandedSkuPdv = expandedSkuPdv === pdv ? null : pdv;
      renderSkuTable();
    }});
  }});
}}

function exportSkuCsv() {{
  if (!currentSkuId || !skuRows.length) return;
  const cols = ["sku_id","pdv_id","city","country","pred_15gg","giacenza","qty_suggerita","confidence","azione"];
  const lines = [cols.join(",")].concat(skuRows.map(r =>
    [currentSkuId, r.pdv_id, r.city, r.country, r.pred, r.giacenza, r.qty, r.confidence, r.azione]
      .map(v => JSON.stringify(v ?? "")).join(",")
  ));
  const blob = new Blob([lines.join("\\n")], {{type:"text/csv;charset=utf-8"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "sku_" + currentSkuId + "_negozi.csv";
  a.click();
}}

document.querySelectorAll(".tab").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.add("hidden"));
    document.getElementById("tab-" + btn.dataset.tab).classList.remove("hidden");
    if (btn.dataset.tab === "sku") renderSkuTab();
  }});
}});

document.querySelectorAll("#prio-table th").forEach(th => {{
  th.addEventListener("click", () => {{
    const k = th.dataset.key;
    if (sortKey === k) sortAsc = !sortAsc; else {{ sortKey = k; sortAsc = false; }}
    sortRows(); renderTable();
  }});
}});

["f-search","f-sku","f-country","f-city","f-cat","f-stagione","f-tipo","f-minqty"].forEach(id => {{
  document.getElementById(id).addEventListener("input", applyFilters);
  document.getElementById(id).addEventListener("change", applyFilters);
}});
document.getElementById("f-conf").addEventListener("input", e => {{
  document.getElementById("conf-lbl").textContent = e.target.value + "%";
  applyFilters();
}});
document.getElementById("btn-reset").addEventListener("click", () => {{
  document.getElementById("f-search").value = "";
  document.getElementById("f-sku").value = "";
  document.getElementById("f-country").value = "";
  document.getElementById("f-city").value = "";
  document.getElementById("f-cat").value = "";
  document.getElementById("f-stagione").value = "";
  document.getElementById("f-tipo").value = "";
  document.getElementById("f-minqty").value = "1";
  document.getElementById("f-conf").value = "0";
  document.getElementById("conf-lbl").textContent = "0%";
  applyFilters();
}});
document.getElementById("btn-export").addEventListener("click", exportCsv);
document.getElementById("sort-qty").addEventListener("click", () => {{ sortKey="qty"; sortAsc=false; sortRows(); renderTable(); }});
document.getElementById("sku-pick").addEventListener("input", renderSkuTab);
document.getElementById("sku-pick").addEventListener("change", renderSkuTab);
document.getElementById("btn-sku-export").addEventListener("click", exportSkuCsv);
document.querySelectorAll("#sku-table th").forEach(th => {{
  th.addEventListener("click", () => {{
    const k = th.dataset.skuKey;
    if (!k) return;
    if (skuSortKey === k) skuSortAsc = !skuSortAsc; else {{ skuSortKey = k; skuSortAsc = false; }}
    renderSkuTable();
  }});
}});

fillSelect("f-country", DATA.filters.countries || []);
fillSelect("f-city", DATA.filters.cities || []);
fillSelect("f-cat", DATA.filters.categorie || []);
fillSelect("f-stagione", DATA.filters.stagioni || []);
fillSelect("f-tipo", DATA.filters.ssclassi || []);
(function fillSkuList() {{
  const dl = document.getElementById("sku-list");
  const list = (DATA.sku_network && DATA.sku_network.skus) || DATA.filters.skus || [];
  list.forEach(s => {{
    const o = document.createElement("option");
    o.value = s.id;
    o.label = s.label;
    dl.appendChild(o);
  }});
}})();
renderExplainCards();
renderModelTab();
applyFilters();
renderSkuTab();
</script>
</body>
</html>"""


def write_bi_dashboard(output_path: Path | None = None, data_dir: Path = DATA_DIR) -> Path:
    payload = build_bi_payload(data_dir)
    if not payload.get("metrics"):
        raise FileNotFoundError(f"Mancano metriche in {data_dir}. Esegui run_train.")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or REPORT_DIR / "cdm_dashboard.html"
    out.write_text(build_bi_html(payload), encoding="utf-8")
    return out
