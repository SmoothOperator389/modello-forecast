"""Aggancio tip_budget al pipeline weekly/monthly (refresh, non one-shot).

Su VM tip_budget può vivere come modulo/script già presente sotto
``/opt/modello-forecast``. Questo wrapper lo scopre e lo invoca in modo
idempotente senza forzare un full retrain (a meno che tip_budget stesso
non lo richieda a basso costo).
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _root() -> Path:
    return Path(os.environ.get("MODELLO_ROOT", "/opt/modello-forecast"))


def discover_tip_budget() -> dict[str, Any]:
    root = _root()
    candidates = [
        root / "production" / "tip_budget.py",
        root / "production" / "ops" / "tip_budget.py",
        root / "scripts" / "tip_budget_refresh.py",
        root / "scripts" / "run_tip_budget.py",
        root / "bi" / "tip_budget.py",
    ]
    found = [str(p) for p in candidates if p.exists()]
    # glob leggero
    for pat in ("**/tip_budget*.py", "**/run_tip_budget*.sh"):
        found.extend(str(p) for p in root.glob(pat) if p.is_file())
    # dedupe preserve order
    seen = set()
    uniq = []
    for f in found:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return {"root": str(root), "candidates": uniq}


def refresh_tip_budget(*, mode: str = "refresh", dry_run: bool = False) -> dict[str, Any]:
    """Esegue il refresh tip_budget se trovato.

    ``mode``: refresh | skip | retrain_if_cheap
    Non forza full retrain salvo env ``TIP_BUDGET_ALLOW_RETRAIN=1``.
    """
    info = discover_tip_budget()
    result: dict[str, Any] = {
        "ok": False,
        "mode": mode,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "discovery": info,
    }
    if mode == "skip":
        result["ok"] = True
        result["skipped"] = True
        return result
    if not info["candidates"]:
        result["error"] = "tip_budget non trovato sotto MODELLO_ROOT"
        return result

    target = Path(info["candidates"][0])
    allow_retrain = os.environ.get("TIP_BUDGET_ALLOW_RETRAIN", "0").strip() in {
        "1",
        "true",
        "yes",
    }
    if dry_run:
        result["ok"] = True
        result["dry_run"] = True
        result["would_run"] = str(target)
        result["allow_retrain"] = allow_retrain
        return result

    env = os.environ.copy()
    env["TIP_BUDGET_MODE"] = "refresh" if not allow_retrain else mode
    env.setdefault("TIP_BUDGET_NO_FULL_RETRAIN", "0" if allow_retrain else "1")

    if target.suffix == ".py":
        # prefer import as module if under production/
        rel = None
        try:
            rel = target.relative_to(_root())
        except ValueError:
            rel = None
        if rel and rel.parts[0] == "production" and rel.suffix == ".py":
            mod_name = ".".join(rel.with_suffix("").parts)
            try:
                mod = importlib.import_module(mod_name)
                fn = getattr(mod, "refresh", None) or getattr(mod, "main", None)
                if callable(fn):
                    out = fn()
                    result["ok"] = True
                    result["via"] = f"import:{mod_name}"
                    result["output"] = out if isinstance(out, (dict, str, int, float, list)) else str(out)
                    return result
            except Exception as exc:  # noqa: BLE001 — report and fall through to subprocess
                result["import_error"] = str(exc)

        cmd = [sys.executable, str(target)]
    else:
        cmd = ["bash", str(target)]

    proc = subprocess.run(
        cmd,
        cwd=str(_root()),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    result["cmd"] = cmd
    result["returncode"] = proc.returncode
    result["stdout_tail"] = (proc.stdout or "")[-4000:]
    result["stderr_tail"] = (proc.stderr or "")[-2000:]
    result["ok"] = proc.returncode == 0
    return result


def wire_into_stack_scripts() -> dict[str, Any]:
    """Assicura che weekly/monthly richiamino tip_budget refresh.

    Cerca script esistenti e, se manca il hook, appende un blocco marcato.
    """
    root = _root()
    marker = "# >>> tip_budget_pipeline hook"
    hook = (
        f"\n{marker}\n"
        "python -m production.ops.tip_budget_pipeline --refresh || "
        "echo '[tip_budget] refresh failed (non-fatal)' >&2\n"
        f"{marker.replace('>>>', '<<<')}\n"
    )
    targets = []
    for name in (
        "scripts/run_weekly.sh",
        "scripts/run_monthly.sh",
        "scripts/pipeline_weekly.sh",
        "scripts/pipeline_monthly.sh",
        "production/run_weekly.sh",
        "production/run_monthly.sh",
    ):
        p = root / name
        if p.exists():
            targets.append(p)

    changed = []
    for p in targets:
        text = p.read_text(encoding="utf-8")
        if marker in text:
            continue
        p.write_text(text.rstrip() + "\n" + hook, encoding="utf-8")
        changed.append(str(p))

    # se non esistono weekly/monthly, crea stub minimi
    created = []
    if not targets:
        scripts = root / "scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        for name in ("run_weekly.sh", "run_monthly.sh"):
            p = scripts / name
            if p.exists():
                continue
            p.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                f"cd \"$(dirname \"$0\")/..\"\n{hook}",
                encoding="utf-8",
            )
            p.chmod(0o755)
            created.append(str(p))

    return {"changed": changed, "created": created, "targets_seen": [str(t) for t in targets]}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--wire", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    out: dict[str, Any] = {}
    if args.discover or not (args.refresh or args.wire):
        out["discover"] = discover_tip_budget()
    if args.wire:
        out["wire"] = wire_into_stack_scripts()
    if args.refresh:
        out["refresh"] = refresh_tip_budget(dry_run=args.dry_run)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.refresh and not out.get("refresh", {}).get("ok"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
