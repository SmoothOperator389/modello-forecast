#!/usr/bin/env python3
"""
Genera (o rigenera) il report HTML per il reparto produzione.

Uso:
  cd ~/Desktop/modello-forecast
  .venv/bin/python -m testing.run_report_html

Richiede gli artefatti prodotti da run_train (report_metrics.json, ecc.).
"""

from __future__ import annotations

from testing.report_builder import write_report


def main() -> None:
    path = write_report()
    print(f"Report generato: {path}")
    print(f"Apri con: open {path}")


if __name__ == "__main__":
    main()
