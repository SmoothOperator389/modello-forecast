"""Lock di training BI — acquisizione/rilascio sicuri (niente orphan).

Uso tipico su VM::

    from production.ops.bi_training_lock import bi_training_lock
    with bi_training_lock(reason="h30_retrain"):
        ...  # train pesante

Override path: env ``BI_TRAINING_LOCK_PATH`` (default ``/opt/modello-forecast/logs/bi_training.lock``).
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def _default_lock_path() -> Path:
    raw = os.environ.get("BI_TRAINING_LOCK_PATH", "").strip()
    if raw:
        return Path(raw)
    root = Path(os.environ.get("MODELLO_ROOT", "/opt/modello-forecast"))
    return root / "logs" / "bi_training.lock"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def lock_info(path: Path | None = None) -> dict | None:
    p = path or _default_lock_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"raw": p.read_text(encoding="utf-8", errors="replace"), "path": str(p)}


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(
    *,
    reason: str,
    owner: str | None = None,
    stale_seconds: int = 36 * 3600,
    path: Path | None = None,
) -> Path:
    """Crea il lockfile. Se esiste un lock con PID morto o troppo vecchio, lo sostituisce."""
    p = path or _default_lock_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    info = lock_info(p)
    if info and isinstance(info, dict):
        pid = int(info.get("pid") or 0)
        ts = str(info.get("acquired_at") or "")
        age_ok = True
        if ts.endswith("Z"):
            try:
                acquired = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                age_ok = (datetime.now(timezone.utc) - acquired).total_seconds() < stale_seconds
            except ValueError:
                age_ok = True
        if _pid_alive(pid) and age_ok:
            raise RuntimeError(
                f"bi_training_lock già attivo: pid={pid} reason={info.get('reason')} "
                f"acquired_at={ts} path={p}"
            )
        # stale / dead → rimuovi
        try:
            p.unlink(missing_ok=True)
        except TypeError:
            if p.exists():
                p.unlink()

    payload = {
        "pid": os.getpid(),
        "reason": reason,
        "owner": owner or os.environ.get("USER") or "unknown",
        "acquired_at": _utc_now(),
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
    }
    tmp = p.with_suffix(p.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return p


def release_lock(path: Path | None = None, *, force: bool = False) -> bool:
    p = path or _default_lock_path()
    if not p.exists():
        return False
    info = lock_info(p) or {}
    pid = int(info.get("pid") or 0) if isinstance(info, dict) else 0
    if not force and pid and pid != os.getpid() and _pid_alive(pid):
        raise RuntimeError(f"refuse release: lock owned by live pid={pid}")
    try:
        p.unlink()
    except FileNotFoundError:
        return False
    return True


@contextmanager
def bi_training_lock(
    reason: str,
    *,
    owner: str | None = None,
    path: Path | None = None,
) -> Iterator[Path]:
    """Context manager: acquisisce, registra cleanup su exit/signal, rilascia sempre."""
    lock_path = acquire_lock(reason=reason, owner=owner, path=path)
    released = {"done": False}

    def _cleanup(*_args) -> None:
        if released["done"]:
            return
        released["done"] = True
        try:
            release_lock(lock_path, force=True)
        except Exception:
            pass

    atexit.register(_cleanup)
    prev_sigterm = signal.getsignal(signal.SIGTERM)
    prev_intl = signal.getsignal(signal.SIGINT)

    def _on_signal(signum, frame):
        _cleanup()
        if callable(prev_sigterm) and signum == signal.SIGTERM:
            prev_sigterm(signum, frame)
        elif callable(prev_intl) and signum == signal.SIGINT:
            prev_intl(signum, frame)
        else:
            raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    try:
        yield lock_path
    finally:
        _cleanup()
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_intl)
        try:
            atexit.unregister(_cleanup)
        except Exception:
            pass


def heartbeat(path: Path | None = None, **extra) -> None:
    """Aggiorna il lockfile con heartbeat (non cambia pid)."""
    p = path or _default_lock_path()
    info = lock_info(p) or {}
    if not isinstance(info, dict):
        return
    if int(info.get("pid") or 0) != os.getpid():
        return
    info["heartbeat_at"] = _utc_now()
    info.update({k: v for k, v in extra.items() if v is not None})
    p.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Gestione bi_training_lock")
    ap.add_argument("cmd", choices=["status", "acquire", "release", "force-release"])
    ap.add_argument("--reason", default="manual")
    args = ap.parse_args()
    if args.cmd == "status":
        print(json.dumps(lock_info() or {"status": "free"}, indent=2))
    elif args.cmd == "acquire":
        print(acquire_lock(reason=args.reason))
        # tieni vivo finché non SIGTERM (per test)
        print("hold… Ctrl+C per release")
        while True:
            heartbeat()
            time.sleep(30)
    elif args.cmd == "release":
        print({"released": release_lock()})
    else:
        print({"released": release_lock(force=True)})
