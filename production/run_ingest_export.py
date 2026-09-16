"""
Ingest export Valido ufficiale (export24kk.csv) → fact_daily + stub pdv/sku.

Formato: nessuna intestazione, separatore ';', 10 campi fissi.
Regole tipodocumento:
  100 = vendita (domanda) — Qta → abs
  47, 51 = carico (stock)
  20 = azzeramento (stock)

Uso:
  OMP_NUM_THREADS=1 .venv2/bin/python -m production.run_ingest_export \\
    --input /Users/mirkobergantino/Downloads/export24kk.csv

  # smoke test veloce:
  ... --max-rows 200000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from production.connectors.valido import (
    VALIDO_EXPORT_COLUMNS,
    VALIDO_TIPODOC_CARICO_WOMEN,
    VALIDO_TIPODOC_GIACENZA,
    VALIDO_TIPODOC_VENDITA,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH_FILE = ROOT / "data" / "raw" / "export24kk.path"
DEFAULT_OUT = ROOT / "data" / "real"

SCONTO_SOGLIA = 5.0


def resolve_input_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    if DEFAULT_PATH_FILE.exists():
        p = Path(DEFAULT_PATH_FILE.read_text().strip())
        if p.exists():
            return p
    raise FileNotFoundError(
        "Nessun CSV trovato. Passa --input oppure aggiorna data/raw/export24kk.path"
    )


def _parse_float(x: str) -> float:
    try:
        return float(x) if x != "" else 0.0
    except ValueError:
        return 0.0


@dataclass
class AggSales:
    qty: float = 0.0
    listino_w: float = 0.0
    pagato_w: float = 0.0
    w: float = 0.0
    giac: float | None = None


def ingest_export(
    input_path: Path,
    out_dir: Path,
    chunksize: int = 500_000,
    max_rows: int | None = None,
) -> dict[str, Path]:
    """
    Due passate logiche in un solo scan:
    - vendite 100 → qty + prezzi medi ponderati + ultima giacenza del giorno
    - movimenti stock (100/47/51/20) → ultima giacenza del giorno (anche senza vendita)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # key: (data_str, country, pdv, sku)
    sales: dict[tuple[str, str, str, str], AggSales] = {}
    stock_giac: dict[tuple[str, str, str, str], float] = {}
    first_sale_sku: dict[str, str] = {}
    first_seen_pdv: dict[str, tuple[str, str]] = {}  # pdv -> (country, data)

    n_read = 0
    n_bad = 0
    n_sale_rows = 0
    n_stock_rows = 0

    reader = pd.read_csv(
        input_path,
        sep=";",
        header=None,
        names=list(VALIDO_EXPORT_COLUMNS),
        dtype=str,
        chunksize=chunksize,
        engine="c",
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace",
    )

    for chunk in reader:
        if max_rows is not None and n_read >= max_rows:
            break
        if max_rows is not None:
            remain = max_rows - n_read
            if len(chunk) > remain:
                chunk = chunk.iloc[:remain]

        n_read += len(chunk)
        chunk = chunk.dropna(how="all")
        if chunk.empty:
            continue

        td = chunk["tipodocumento"].astype(str).str.strip()
        mask_sale = td == VALIDO_TIPODOC_VENDITA
        mask_stock = td.isin(VALIDO_TIPODOC_GIACENZA)

        # --- vendite ---
        if mask_sale.any():
            s = chunk.loc[mask_sale].copy()
            n_sale_rows += len(s)
            qta = s["qta"].map(_parse_float).abs()
            listino = s["prezzo_pubblico"].map(_parse_float)
            pagato = s["pagato_effettivo"].map(_parse_float)
            giac = s["giacenza"].map(_parse_float)
            keys = list(
                zip(
                    s["data"].astype(str).str[:10],
                    s["country"].astype(str),
                    s["pdv_id"].astype(str),
                    s["sku_id"].astype(str),
                )
            )
            qta_a = qta.to_numpy()
            listino_a = listino.to_numpy()
            pagato_a = pagato.to_numpy()
            giac_a = giac.to_numpy()
            for k, q, pl, pp, g in zip(keys, qta_a, listino_a, pagato_a, giac_a):
                if not k[0] or not k[2] or not k[3]:
                    n_bad += 1
                    continue
                agg = sales.get(k)
                if agg is None:
                    agg = AggSales()
                    sales[k] = agg
                agg.qty += float(q)
                if pl > 0 or pp > 0:
                    w = float(q) if q > 0 else 1.0
                    agg.listino_w += pl * w
                    agg.pagato_w += pp * w
                    agg.w += w
                agg.giac = float(g)

                d, country, pdv, sku = k
                if sku not in first_sale_sku or d < first_sale_sku[sku]:
                    first_sale_sku[sku] = d
                prev = first_seen_pdv.get(pdv)
                if prev is None or d < prev[1]:
                    first_seen_pdv[pdv] = (country, d)

        # --- stock giacenza (ultima del giorno, ordine file ≈ cronologico per chiave) ---
        if mask_stock.any():
            st = chunk.loc[mask_stock]
            n_stock_rows += len(st)
            keys = list(
                zip(
                    st["data"].astype(str).str[:10],
                    st["country"].astype(str),
                    st["pdv_id"].astype(str),
                    st["sku_id"].astype(str),
                )
            )
            giac_a = st["giacenza"].map(_parse_float).to_numpy()
            for k, g in zip(keys, giac_a):
                if not k[0] or not k[2] or not k[3]:
                    continue
                stock_giac[k] = float(g)
                d, country, pdv, _sku = k
                prev = first_seen_pdv.get(pdv)
                if prev is None or d < prev[1]:
                    first_seen_pdv[pdv] = (country, d)

        if n_read % 2_000_000 < chunksize:
            print(f"  … {n_read:,} righe lette | sales-keys={len(sales):,} | stock-keys={len(stock_giac):,}", flush=True)

    print(
        f"Scan fine: rows={n_read:,} sale_rows={n_sale_rows:,} stock_rows={n_stock_rows:,} "
        f"bad={n_bad:,}",
        flush=True,
    )

    # Unione chiavi: giorni con vendita + giorni solo stock (vendite_qty=0)
    all_keys = set(sales) | set(stock_giac)
    print(f"Chiavi giornaliere PDV×SKU: {len(all_keys):,}", flush=True)

    import csv
    import json

    fact_cols = [
        "data",
        "country",
        "pdv_id",
        "sku_id",
        "categoria_id",
        "vendite_qty",
        "giacenza_qty",
        "ora",
        "prezzo_listino",
        "prezzo_pagato",
        "sconto_pct",
        "is_in_sconto",
        "modello",
        "tessuto",
        "colore",
        "campagna",
        "classe",
        "sclasse",
        "ssclasse",
        "tipologia",
        "nome",
        "data_apertura",
        "data_lancio_sku",
    ]
    fact_path = out_dir / "fact_daily.csv"
    n_fact = 0
    dmin = dmax = None
    print(f"Scrivo {fact_path} in streaming…", flush=True)
    with fact_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fact_cols)
        w.writeheader()
        # iter ordinato per data/pdv/sku senza materializzare tutta la lista di dict
        for k in sorted(all_keys):
            d, country, pdv, sku = k
            if dmin is None or d < dmin:
                dmin = d
            if dmax is None or d > dmax:
                dmax = d
            agg = sales.get(k)
            if agg is not None and agg.w > 0:
                listino = agg.listino_w / agg.w
                pagato = agg.pagato_w / agg.w
            else:
                listino = pagato = 0.0
            qty = agg.qty if agg is not None else 0.0
            giac = stock_giac.get(k)
            if giac is None and agg is not None and agg.giac is not None:
                giac = agg.giac
            if giac is None:
                giac = 0.0
            if listino > 0.01:
                sconto = float(np.clip((1.0 - pagato / listino) * 100.0, 0.0, 90.0))
            else:
                sconto = 0.0
            lancio = first_sale_sku.get(sku, d)
            apertura = first_seen_pdv.get(pdv, (country, d))[1]
            w.writerow(
                {
                    "data": d,
                    "country": country,
                    "pdv_id": pdv,
                    "sku_id": sku,
                    "categoria_id": "NA",
                    "vendite_qty": f"{qty:.4f}",
                    "giacenza_qty": f"{float(giac):.4f}",
                    "ora": "",
                    "prezzo_listino": f"{listino:.4f}",
                    "prezzo_pagato": f"{pagato:.4f}",
                    "sconto_pct": f"{sconto:.4f}",
                    "is_in_sconto": int(sconto >= SCONTO_SOGLIA),
                    "modello": "NA",
                    "tessuto": "NA",
                    "colore": "NA",
                    "campagna": "NA",
                    "classe": "NA",
                    "sclasse": "NA",
                    "ssclasse": "NA",
                    "tipologia": "NA",
                    "nome": sku,
                    "data_apertura": apertura,
                    "data_lancio_sku": lancio,
                }
            )
            n_fact += 1
            if n_fact % 2_000_000 == 0:
                print(f"  … scritte {n_fact:,} / {len(all_keys):,}", flush=True)

    print(f"Scritto {fact_path} ({n_fact:,} righe)", flush=True)

    # pdv_master stub (da completare col team)
    pdv_rows = []
    for pdv, (country, d0) in sorted(first_seen_pdv.items()):
        pdv_rows.append(
            {
                "pdv_id": pdv,
                "country": country,
                "city": "NA",
                "commerciale_riferimento": "NA",
                "mq_espositivi": 100.0,
                "mq_stock": 40.0,
                "mq_totali": 140.0,
                "numero_fatturazioni": 100.0,
                "transit_time_days": 5.0,
                "data_apertura": d0,
                "pdv_strategico": 0,
            }
        )
    pdv = pd.DataFrame(pdv_rows)
    pdv_path = out_dir / "pdv_master.csv"
    pdv.to_csv(pdv_path, index=False)
    print(f"Scritto {pdv_path} ({len(pdv):,} PDV stub)", flush=True)

    # products stub da prima vendita + SKU solo-stock
    seen_sku: set[str] = set()
    prod_rows: list[dict] = []
    for sku, d0 in sorted(first_sale_sku.items()):
        seen_sku.add(sku)
        prod_rows.append(
            {
                "sku_id": sku,
                "nome": sku,
                "data_lancio_sku": d0,
                "categoria_id": "NA",
                "modello": "NA",
                "tessuto": "NA",
                "colore": "NA",
                "campagna": "NA",
                "classe": "NA",
                "sclasse": "NA",
                "ssclasse": "NA",
                "tipologia": "NA",
            }
        )
    for k in stock_giac:
        sku = k[3]
        if sku in seen_sku:
            continue
        seen_sku.add(sku)
        prod_rows.append(
            {
                "sku_id": sku,
                "nome": sku,
                "data_lancio_sku": k[0],
                "categoria_id": "NA",
                "modello": "NA",
                "tessuto": "NA",
                "colore": "NA",
                "campagna": "NA",
                "classe": "NA",
                "sclasse": "NA",
                "ssclasse": "NA",
                "tipologia": "NA",
            }
        )
    prod = pd.DataFrame(prod_rows)
    prod_path = out_dir / "products.csv"
    prod.to_csv(prod_path, index=False)
    print(f"Scritto {prod_path} ({len(prod):,} SKU stub)", flush=True)

    # calendar minimo
    assert dmin and dmax
    cal = pd.DataFrame({"data": pd.date_range(dmin, dmax, freq="D")})
    cal["is_festivo"] = 0
    cal["is_saldi"] = 0
    cal_path = out_dir / "calendar.csv"
    cal.to_csv(cal_path, index=False)
    print(f"Scritto {cal_path} ({len(cal):,} giorni)", flush=True)

    meta = {
        "input": str(input_path),
        "n_rows_read": n_read,
        "n_sale_rows": n_sale_rows,
        "n_stock_rows": n_stock_rows,
        "n_fact_rows": n_fact,
        "n_pdv": len(pdv),
        "n_sku": int(len(prod)),
        "date_min": dmin,
        "date_max": dmax,
        "tipodoc_vendita": VALIDO_TIPODOC_VENDITA,
        "tipodoc_giacenza": sorted(VALIDO_TIPODOC_GIACENZA),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = out_dir / "ingest_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Scritto {meta_path}", flush=True)

    return {
        "fact_daily": fact_path,
        "pdv_master": pdv_path,
        "products": prod_path,
        "calendar": cal_path,
        "meta": meta_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest export24kk Valido → data/real/")
    ap.add_argument("--input", type=Path, default=None, help="Path CSV export (default: export24kk.path)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Directory output")
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--max-rows", type=int, default=None, help="Limite righe (smoke test)")
    args = ap.parse_args()

    src = resolve_input_path(args.input)
    print(f"Input: {src}")
    print(f"Output: {args.out}")
    print(
        f"Filtri: vendita={VALIDO_TIPODOC_VENDITA} "
        f"giacenza={sorted(VALIDO_TIPODOC_GIACENZA)} "
        f"(incluso carico women {VALIDO_TIPODOC_CARICO_WOMEN})"
    )
    ingest_export(src, args.out, chunksize=args.chunksize, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
