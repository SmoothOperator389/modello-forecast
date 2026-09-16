#!/usr/bin/env python3
"""Genera dashboard interattiva CDM (mini-BI HTML).

Uso:
  cd ~/Desktop/modello-forecast
  .venv2/bin/python -m testing.run_bi_dashboard
  open reports/cdm_dashboard.html
"""

from __future__ import annotations

from testing.blas_threads import limit_native_threads

limit_native_threads()

from testing.bi_dashboard import write_bi_dashboard


def main() -> None:
    path = write_bi_dashboard()
    print(f"Dashboard CDM: {path}")
    print(f"Apri: open {path}")


if __name__ == "__main__":
    main()
