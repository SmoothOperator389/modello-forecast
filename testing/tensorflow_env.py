"""Safe TensorFlow import with Mac / CLT Python troubleshooting and GPU setup."""

from __future__ import annotations

import os
import sys
from typing import Any


def _is_clt_python() -> bool:
    """True when the venv/interpreter is built on Apple Command Line Tools Python."""
    if "CommandLineTools" in sys.executable:
        return True
    if "CommandLineTools" in getattr(sys, "base_prefix", ""):
        return True
    return sys.platform == "darwin" and sys.version_info < (3, 10)


def _tensorflow_failure_message(exc: BaseException | None = None) -> str:
    py = sys.version.split()[0]
    exe = sys.executable
    is_clt_python = _is_clt_python()

    lines = [
        "ERRORE: impossibile importare TensorFlow.",
        f"  Python: {py} ({exe})",
    ]
    if exc is not None:
        lines.append(f"  Dettaglio: {exc}")
    lines.append("")

    if is_clt_python:
        lines.extend(
            [
                "Causa probabile: Python 3.9 da Xcode Command Line Tools su macOS.",
                "TensorFlow non è compatibile con questo interprete su macOS recenti",
                "(errori tipo «mutex lock failed» o «Lock blocking» durante import/train).",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "TensorFlow potrebbe non essere installato o non essere compatibile",
                "con questa versione di Python / sistema operativo.",
                "",
            ]
        )

    lines.extend(
        [
            "Soluzioni (scegline una):",
            "",
            "  1) Docker (consigliato su Mac finché il venv locale non funziona):",
            "       ./scripts/run_train_docker.sh",
            "",
            "  2) Venv con Python 3.11+ (python.org o Homebrew, NON CLT python3.9):",
            "       ./scripts/setup_venv_mac.sh",
            "",
            "  3) Workaround immediato senza TensorFlow (dati + baseline naive):",
            "       .venv/bin/python -m testing.run_train_lite",
        ]
    )
    return "\n".join(lines)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def configure_tensorflow_devices(tf: Any) -> dict[str, Any]:
    """Configure TF devices.

    Su Mac Apple Silicon, TensorFlow-Metal gira ma congela la testa SKU di questo
    dual-head (MAE SKU costante). Default = CPU. Per forzare GPU: FORECAST_USE_GPU=1.
    """
    force_cpu = _truthy_env("FORECAST_FORCE_CPU")
    use_gpu = _truthy_env("FORECAST_USE_GPU")
    on_apple_silicon = sys.platform == "darwin" and os.uname().machine == "arm64"
    gpus = tf.config.list_physical_devices("GPU")

    # Default Mac: CPU. Metal OK per matmul, non affidabile per questo training.
    if on_apple_silicon and not use_gpu:
        force_cpu = True
        if gpus:
            print(
                "Mac Apple Silicon: training su CPU (default).\n"
                "  Motivo: TensorFlow-Metal congela la testa SKU (MAE fissa).\n"
                "  Per forzare GPU (sperimentale): FORECAST_USE_GPU=1"
            )

    if force_cpu and gpus:
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        gpus = []

    try:
        keras = tf.keras
        keras.mixed_precision.set_global_policy("float32")
    except Exception:
        pass

    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        with tf.device("/GPU:0"):
            _ = tf.matmul(tf.random.normal((64, 64)), tf.random.normal((64, 64)))
        print(f"TensorFlow {tf.__version__} — compute su GPU: {gpus[0].name}")
        return {"backend": "gpu", "devices": [d.name for d in gpus]}

    cpus = tf.config.list_physical_devices("CPU")
    print(f"TensorFlow {tf.__version__} — compute su CPU")
    return {"backend": "cpu", "devices": [d.name for d in cpus]}


def import_tensorflow(*, configure_devices: bool = True) -> Any:
    """Import tensorflow; on failure print guidance and exit with code 1."""
    if _is_clt_python():
        print(_tensorflow_failure_message(), file=sys.stderr)
        sys.exit(1)
    try:
        import tensorflow as tf  # noqa: PLC0415 — lazy import by design
    except Exception as exc:  # noqa: BLE001 — catch hang-prone TF init failures
        print(_tensorflow_failure_message(exc), file=sys.stderr)
        sys.exit(1)
    if configure_devices:
        configure_tensorflow_devices(tf)
    return tf
