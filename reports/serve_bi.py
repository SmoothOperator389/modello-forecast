#!/usr/bin/env python3
"""Avvia server locale per la BI CDM consultabile.

  .venv2/bin/python reports/serve_bi.py
  poi apri http://127.0.0.1:8765/bi_cdm_consultazione.html

Modalità:
  - Operativo: data/bi_predictions.json (forward)
  - Storico: scegli dal/al (anche 2025) → API genera confronto prev vs reale
  - Analisi SKU: reale in forbice + forecast 15g / 1m / 2m / 3m
"""

from __future__ import annotations

import json
import os
import sys
import traceback
import urllib.parse
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
PORT = int(os.environ.get("BI_PORT", "8765"))

# import production helpers
sys.path.insert(0, str(PROJECT))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/holdout/meta":
            try:
                from production.bi_holdout_query import ensure_models, list_sample_as_ofs, _state, _refresh_index

                ensure_models()
                _refresh_index()
                as_ofs = [str(a) for a in list_sample_as_ofs()]
                self._json(
                    200,
                    {
                        "ok": True,
                        "fact_max": str(_state["max_d"]),
                        "sample_as_ofs": as_ofs,
                        "precomputed_as_ofs": as_ofs,
                        "daily_as_of": True,
                        "min_date": str(_state.get("min_as_of") or (as_ofs[0] if as_ofs else None)),
                        "max_date": str(_state["max_d"]),
                        "min_as_of": str(_state.get("min_as_of") or (as_ofs[0] if as_ofs else None)),
                        "max_as_of": str(_state["max_d"]),
                        "note": (
                            "as_of giornaliero: puoi scegliere qualsiasi giorno. "
                            "Le date biweekly precomputate sono più veloci; le altre si calcolano on-demand."
                        ),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._json(500, {"ok": False, "error": str(exc)})
            return

        if parsed.path == "/api/holdout/range":
            qs = urllib.parse.parse_qs(parsed.query)
            d_from = (qs.get("from") or [None])[0]
            d_to = (qs.get("to") or [None])[0]
            as_of = (qs.get("as_of") or [None])[0]
            if not d_from or not d_to:
                self._json(400, {"ok": False, "error": "Parametri richiesti: from, to (YYYY-MM-DD)"})
                return
            try:
                from production.bi_holdout_query import _parse_date, build_range_payload

                payload = build_range_payload(
                    _parse_date(d_from),
                    _parse_date(d_to),
                    as_of=_parse_date(as_of) if as_of else None,
                )
                self._json(200, {"ok": True, "data": payload})
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._json(500, {"ok": False, "error": str(exc)})
            return

        if parsed.path == "/api/holdout/as_of":
            qs = urllib.parse.parse_qs(parsed.query)
            as_of = (qs.get("as_of") or [None])[0]
            if not as_of:
                self._json(400, {"ok": False, "error": "Parametro richiesto: as_of=YYYY-MM-DD"})
                return
            try:
                from production.bi_holdout_query import _parse_date, clamp_as_of, generate_as_of_snapshot

                target = _parse_date(as_of)
                used = clamp_as_of(target)
                payload = generate_as_of_snapshot(used, cache=True)
                payload["meta"]["requested_as_of"] = str(target)
                payload["meta"]["as_of_snapped"] = used != target
                self._json(200, {"ok": True, "data": payload})
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._json(500, {"ok": False, "error": str(exc)})
            return

        if parsed.path == "/api/sku/window":
            qs = urllib.parse.parse_qs(parsed.query)
            q = (qs.get("q") or [None])[0]
            d_from = (qs.get("from") or [None])[0]
            d_to = (qs.get("to") or [None])[0]
            as_of = (qs.get("as_of") or [None])[0]
            if not q or not d_from or not d_to:
                self._json(400, {"ok": False, "error": "Parametri richiesti: q, from, to"})
                return
            try:
                from production.bi_holdout_query import _parse_date, build_sku_window_forecast

                payload = build_sku_window_forecast(
                    q,
                    _parse_date(d_from),
                    _parse_date(d_to),
                    as_of=_parse_date(as_of) if as_of else None,
                )
                self._json(200, {"ok": True, "data": payload})
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._json(500, {"ok": False, "error": str(exc)})
            return

        return super().do_GET()


def main() -> None:
    pred = ROOT / "data" / "bi_predictions.json"
    if not pred.exists():
        print("Manca reports/data/bi_predictions.json")
        print("Esegui forward: FORECAST_PILOT=0 python -m production.run_infer_forward_real")
        raise SystemExit(1)
    # evita "Address already in use" sticky
    ThreadingHTTPServer.allow_reuse_address = True
    with ThreadingHTTPServer(("127.0.0.1", PORT), Handler) as httpd:
        url = f"http://127.0.0.1:{PORT}/bi_cdm_consultazione.html"
        print(f"BI CDM → {url}")
        print(
            "API: /api/holdout/meta | /api/holdout/range?from=&to= | "
            "/api/holdout/as_of?as_of= | /api/sku/window?q=&from=&to="
        )
        print("Ctrl+C per chiudere")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        httpd.serve_forever()


if __name__ == "__main__":
    main()
