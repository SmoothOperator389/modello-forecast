#!/usr/bin/env python3
"""Report pandas delle previsioni CDM (metriche + priorità + stagione stimata).

Uso:
  cd ~/Desktop/modello-forecast
  .venv2/bin/python -m testing.run_pandas_report
  .venv2/bin/python -m testing.run_pandas_report --top 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from testing.blas_threads import limit_native_threads

limit_native_threads()

from testing.config import DATA_DIR


def _load() -> dict[str, pd.DataFrame | dict]:
    prio = pd.read_csv(DATA_DIR / "priorita_riassortimento.csv")
    products = pd.read_csv(DATA_DIR / "products.csv") if (DATA_DIR / "products.csv").exists() else pd.DataFrame()
    pdv = pd.read_csv(DATA_DIR / "pdv_master.csv") if (DATA_DIR / "pdv_master.csv").exists() else pd.DataFrame()
    meteo = (
        pd.read_csv(DATA_DIR / "sku_meteo_profilo.csv")
        if (DATA_DIR / "sku_meteo_profilo.csv").exists()
        else pd.DataFrame()
    )
    metrics_path = DATA_DIR / "report_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    if not products.empty:
        keep = [c for c in ["sku_id", "nome", "modello", "categoria_id", "ssclasse"] if c in products.columns]
        prio = prio.merge(products[keep].drop_duplicates("sku_id"), on="sku_id", how="left")
    if not pdv.empty:
        keep = [c for c in ["pdv_id", "country", "city"] if c in pdv.columns]
        prio = prio.merge(pdv[keep].drop_duplicates("pdv_id"), on="pdv_id", how="left")
    if not meteo.empty and "sku_id" in meteo.columns:
        keep = [c for c in ["sku_id", "stagione_stimata", "sku_share_hot", "sku_share_cold", "sku_temp_vendita"] if c in meteo.columns]
        prio = prio.merge(meteo[keep].drop_duplicates("sku_id"), on="sku_id", how="left")

    return {"prio": prio, "meteo": meteo, "metrics": metrics}


def _metrics_frame(metrics: dict) -> pd.DataFrame:
    rows = [
        ("MAE rete (pezzi / 15gg)", metrics.get("mae_tf")),
        ("MAE CDM originale", metrics.get("mae_cdm")),
        ("MAE naive run-rate×15", metrics.get("mae_naive")),
        ("Miglioramento vs CDM %", metrics.get("improvement_pct")),
        ("R² rete", metrics.get("r2")),
        ("MAPE rete %", metrics.get("mape_tf")),
        ("MAE SKU nuovi (<30g)", metrics.get("mae_cold_sku_confidence")),
        ("n train", metrics.get("n_train")),
        ("n test", metrics.get("n_test")),
        ("epoche", metrics.get("epochs_run")),
    ]
    return pd.DataFrame(rows, columns=["indicatore", "valore"])


def _by_azione(prio: pd.DataFrame) -> pd.DataFrame:
    g = (
        prio.groupby("azione", dropna=False)
        .agg(
            n_righe=("sku_id", "count"),
            qty_tot=("qty_suggerita", "sum"),
            pred_media=("vendite_previste_15gg", "mean"),
            conf_media=("confidence_score", "mean"),
        )
        .reset_index()
        .sort_values("qty_tot", ascending=False)
    )
    g["pred_media"] = g["pred_media"].round(2)
    g["conf_media"] = (g["conf_media"] * 100).round(1)
    g["qty_tot"] = g["qty_tot"].round(0)
    return g


def _by_stagione(prio: pd.DataFrame) -> pd.DataFrame:
    if "stagione_stimata" not in prio.columns:
        return pd.DataFrame()
    g = (
        prio.groupby("stagione_stimata", dropna=False)
        .agg(
            n_sku=("sku_id", "nunique"),
            n_righe=("sku_id", "count"),
            qty_tot=("qty_suggerita", "sum"),
            pred_media=("vendite_previste_15gg", "mean"),
        )
        .reset_index()
        .sort_values("qty_tot", ascending=False)
    )
    g["pred_media"] = g["pred_media"].round(2)
    g["qty_tot"] = g["qty_tot"].round(0)
    return g


def _by_country(prio: pd.DataFrame) -> pd.DataFrame:
    if "country" not in prio.columns:
        return pd.DataFrame()
    g = (
        prio.groupby("country", dropna=False)
        .agg(
            n_pdv=("pdv_id", "nunique"),
            qty_tot=("qty_suggerita", "sum"),
            pred_tot=("vendite_previste_15gg", "sum"),
        )
        .reset_index()
        .sort_values("qty_tot", ascending=False)
    )
    g["qty_tot"] = g["qty_tot"].round(0)
    g["pred_tot"] = g["pred_tot"].round(0)
    return g


def _top_qty(prio: pd.DataFrame, n: int) -> pd.DataFrame:
    cols = [
        c
        for c in [
            "as_of",
            "country",
            "city",
            "pdv_id",
            "sku_id",
            "nome",
            "stagione_stimata",
            "vendite_previste_15gg",
            "giacenza",
            "qty_suggerita",
            "confidence_score",
            "azione",
        ]
        if c in prio.columns
    ]
    out = prio.sort_values("qty_suggerita", ascending=False).head(n)[cols].copy()
    if "confidence_score" in out.columns:
        out["confidence_score"] = (out["confidence_score"] * 100).round(1)
    for c in ("vendite_previste_15gg", "giacenza", "qty_suggerita"):
        if c in out.columns:
            out[c] = out[c].round(1)
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report pandas previsioni CDM")
    parser.add_argument("--top", type=int, default=20, help="Righe top qty da mostrare (default 20)")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Salva anche Excel in testing/data/output/report_pandas.xlsx",
    )
    args = parser.parse_args()

    data = _load()
    prio: pd.DataFrame = data["prio"]  # type: ignore[assignment]
    metrics: dict = data["metrics"]  # type: ignore[assignment]

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", 28)

    print("\n=== 1) Metriche modello ===")
    mdf = _metrics_frame(metrics)
    print(mdf.to_string(index=False))

    print("\n=== 2) Priorità per azione ===")
    print(_by_azione(prio).to_string(index=False))

    stag = _by_stagione(prio)
    if not stag.empty:
        print("\n=== 3) Qty per stagione stimata (dalle vendite) ===")
        print(stag.to_string(index=False))

    by_c = _by_country(prio)
    if not by_c.empty:
        print("\n=== 4) Qty per paese ===")
        print(by_c.to_string(index=False))

    print(f"\n=== 5) Top {args.top} qty suggerita ===")
    top = _top_qty(prio, args.top)
    print(top.to_string(index=False))

    print(f"\nRighe priorità totali: {len(prio):,}")
    print(f"Fonte: {DATA_DIR / 'priorita_riassortimento.csv'}")

    if args.save:
        out = DATA_DIR / "report_pandas.xlsx"
        with pd.ExcelWriter(out, engine="openpyxl") as xw:
            mdf.to_excel(xw, sheet_name="metriche", index=False)
            _by_azione(prio).to_excel(xw, sheet_name="azioni", index=False)
            if not stag.empty:
                stag.to_excel(xw, sheet_name="stagione_stimata", index=False)
            if not by_c.empty:
                by_c.to_excel(xw, sheet_name="paesi", index=False)
            top.to_excel(xw, sheet_name="top_qty", index=False)
            prio.head(5000).to_excel(xw, sheet_name="priorita_sample", index=False)
        print(f"Salvato: {out}")


if __name__ == "__main__":
    main()
