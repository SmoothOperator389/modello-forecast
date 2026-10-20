"""Routing post-modello per fascia di volume (prior causale).

prior = run_rate_14d × horizon (stesso segnale del naive).

Zone:
  prior <= low_thr              → LightGBM (volumi bassi)
  low_thr < prior <= high_thr   → mid (TF / specialista high / mix)
  prior > high_thr              → coda alta (naive / specialista / mix)
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _blend(a: np.ndarray, b: np.ndarray, w_a: float) -> np.ndarray:
    w = float(w_a)
    return w * a + (1.0 - w) * b


def route_forecast(
    prior: np.ndarray,
    pred_tf: np.ndarray,
    pred_lgbm: np.ndarray,
    pred_naive: np.ndarray,
    *,
    pred_high: np.ndarray | None = None,
    low_thr: float = 3.0,
    high_thr: float = 15.0,
    mid_mode: str = "tf",
    high_mode: str = "mix_tf_naive",
    mid_spec_weight: float = 0.55,
    high_tf_weight: float = 0.35,
    high_spec_weight: float = 0.55,
) -> np.ndarray:
    """Combina predizioni in base al prior di volume.

    mid_mode:  tf | high | mix_high_tf | mix_high_lgbm | lgbm
    high_mode: mix_tf_naive | naive | high | mix_high_naive | mix_high_tf
    """
    prior = np.asarray(prior, dtype=float)
    tf_p = np.asarray(pred_tf, dtype=float)
    lg_p = np.asarray(pred_lgbm, dtype=float)
    nv_p = np.asarray(pred_naive, dtype=float)
    hi_p = np.asarray(pred_high, dtype=float) if pred_high is not None else nv_p

    out = tf_p.copy()
    low = prior <= low_thr
    high = prior > high_thr
    mid = (~low) & (~high)

    out[low] = lg_p[low]

    mm = mid_mode.strip().lower()
    if mm == "high":
        out[mid] = hi_p[mid]
    elif mm == "lgbm":
        out[mid] = lg_p[mid]
    elif mm == "mix_high_tf":
        out[mid] = _blend(hi_p[mid], tf_p[mid], mid_spec_weight)
    elif mm == "mix_high_lgbm":
        out[mid] = _blend(hi_p[mid], lg_p[mid], mid_spec_weight)
    else:
        out[mid] = tf_p[mid]

    hm = high_mode.strip().lower()
    if hm == "naive":
        out[high] = nv_p[high]
    elif hm == "high":
        out[high] = hi_p[high]
    elif hm == "mix_high_naive":
        out[high] = _blend(hi_p[high], nv_p[high], high_spec_weight)
    elif hm == "mix_high_tf":
        out[high] = _blend(hi_p[high], tf_p[high], high_spec_weight)
    else:  # mix_tf_naive
        out[high] = _blend(tf_p[high], nv_p[high], high_tf_weight)

    return np.maximum(out, 0.0)


def floor_toward_naive(
    pred: np.ndarray,
    naive: np.ndarray,
    prior: np.ndarray,
    *,
    prior_thr: float = 15.0,
    floor_ratio: float = 0.85,
) -> np.ndarray:
    """Sulla coda alta evita sottostima: pred >= floor_ratio × naive."""
    out = np.asarray(pred, dtype=float).copy()
    nv = np.asarray(naive, dtype=float)
    mask = np.asarray(prior, dtype=float) > prior_thr
    out[mask] = np.maximum(out[mask], floor_ratio * nv[mask])
    return np.maximum(out, 0.0)


def _objective_mask(y: np.ndarray, objective: str) -> np.ndarray:
    if objective == "ops":
        return y >= 3
    if objective == "high":
        return y > 15
    if objective == "composite":
        return np.ones(len(y), dtype=bool)
    return np.ones(len(y), dtype=bool)


def _score(pred: np.ndarray, y: np.ndarray, objective: str) -> float:
    """Score da minimizzare."""
    if objective != "composite":
        m = _objective_mask(y, objective)
        if m.sum() < 50:
            m = np.ones(len(y), dtype=bool)
        return float(np.abs(pred[m] - y[m]).mean())
    # bilanciato: globale + ops + coda alta
    mae_g = float(np.abs(pred - y).mean())
    ops = y >= 3
    hi = y > 15
    mae_ops = float(np.abs(pred[ops] - y[ops]).mean()) if ops.any() else mae_g
    mae_hi = float(np.abs(pred[hi] - y[hi]).mean()) if hi.any() else mae_g
    return 0.35 * mae_g + 0.40 * mae_ops + 0.25 * (mae_hi / 4.0)


def calibrate_route_params(
    prior: np.ndarray,
    y: np.ndarray,
    pred_tf: np.ndarray,
    pred_lgbm: np.ndarray,
    pred_naive: np.ndarray,
    pred_high: np.ndarray | None = None,
    *,
    low_thrs: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0),
    high_thrs: tuple[float, ...] = (8.0, 10.0, 12.0, 15.0, 18.0),
    mid_modes: tuple[str, ...] = ("tf", "high", "mix_high_tf", "mix_high_lgbm"),
    high_modes: tuple[str, ...] = (
        "mix_tf_naive",
        "naive",
        "high",
        "mix_high_naive",
        "mix_high_tf",
    ),
    mid_spec_weights: tuple[float, ...] = (0.45, 0.6, 0.75, 0.9),
    high_tf_weights: tuple[float, ...] = (0.0, 0.2, 0.35),
    high_spec_weights: tuple[float, ...] = (0.4, 0.55, 0.7, 0.85),
    floor_ratios: tuple[float, ...] = (0.0, 0.8, 0.9, 0.95),
    objective: str = "composite",
) -> dict[str, Any]:
    """Grid search su validazione."""
    y = np.asarray(y, dtype=float)
    best: dict[str, Any] = {"score": float("inf"), "mae": float("inf")}
    trials = 0

    for low_thr in low_thrs:
        for high_thr in high_thrs:
            if high_thr <= low_thr + 1:
                continue
            for mid_mode in mid_modes:
                needs_high = mid_mode in {"high", "mix_high_tf", "mix_high_lgbm"}
                mid_ws = mid_spec_weights if mid_mode.startswith("mix_high") else (0.0,)
                for mid_w in mid_ws:
                    for high_mode in high_modes:
                        if (
                            needs_high or high_mode in {"high", "mix_high_naive", "mix_high_tf"}
                        ) and pred_high is None:
                            continue
                        if high_mode == "mix_tf_naive":
                            h_ws = high_tf_weights
                        elif high_mode.startswith("mix_high"):
                            h_ws = high_spec_weights
                        else:
                            h_ws = (0.0,)
                        for h_w in h_ws:
                            kwargs: dict[str, float | str] = {
                                "low_thr": low_thr,
                                "high_thr": high_thr,
                                "mid_mode": mid_mode,
                                "high_mode": high_mode,
                                "mid_spec_weight": mid_w,
                            }
                            if high_mode == "mix_tf_naive":
                                kwargs["high_tf_weight"] = h_w
                            elif high_mode.startswith("mix_high"):
                                kwargs["high_spec_weight"] = h_w
                            pred = route_forecast(
                                prior,
                                pred_tf,
                                pred_lgbm,
                                pred_naive,
                                pred_high=pred_high,
                                **kwargs,  # type: ignore[arg-type]
                            )
                            for fr in floor_ratios:
                                cand = (
                                    floor_toward_naive(
                                        pred, pred_naive, prior, prior_thr=high_thr, floor_ratio=fr
                                    )
                                    if fr > 0
                                    else pred
                                )
                                score = _score(cand, y, objective)
                                mae_g = float(np.abs(cand - y).mean())
                                trials += 1
                                if score < best["score"]:
                                    best = {
                                        "score": round(score, 4),
                                        "mae": round(mae_g, 4),
                                        "low_thr": low_thr,
                                        "high_thr": high_thr,
                                        "mid_mode": mid_mode,
                                        "high_mode": high_mode,
                                        "mid_spec_weight": float(mid_w),
                                        "high_tf_weight": float(kwargs.get("high_tf_weight", 0.35)),
                                        "high_spec_weight": float(kwargs.get("high_spec_weight", 0.55)),
                                        "floor_ratio": fr,
                                        "objective": objective,
                                    }
    best["trials"] = trials
    return best


def apply_calibrated_route(
    prior: np.ndarray,
    pred_tf: np.ndarray,
    pred_lgbm: np.ndarray,
    pred_naive: np.ndarray,
    params: dict[str, Any],
    pred_high: np.ndarray | None = None,
) -> np.ndarray:
    pred = route_forecast(
        prior,
        pred_tf,
        pred_lgbm,
        pred_naive,
        pred_high=pred_high,
        low_thr=float(params.get("low_thr", 3.0)),
        high_thr=float(params.get("high_thr", 15.0)),
        mid_mode=str(params.get("mid_mode", "tf")),
        high_mode=str(params.get("high_mode", "mix_tf_naive")),
        mid_spec_weight=float(params.get("mid_spec_weight", 0.55)),
        high_tf_weight=float(params.get("high_tf_weight", 0.35)),
        high_spec_weight=float(params.get("high_spec_weight", 0.55)),
    )
    fr = float(params.get("floor_ratio", 0.0))
    if fr > 0:
        pred = floor_toward_naive(
            pred,
            pred_naive,
            prior,
            prior_thr=float(params.get("high_thr", 15.0)),
            floor_ratio=fr,
        )
    return pred


def soft_reconcile_frame(
    df,
    pred: np.ndarray,
    *,
    keys: tuple[str, ...] = ("pdv_id", "modello"),
    alpha: float = 0.25,
) -> np.ndarray:
    """Shrink pred verso media di gruppo (es. pdv×modello)."""
    out = df.copy()
    out["_pred"] = np.asarray(pred, dtype=float)
    cols = [k for k in keys if k in out.columns]
    if not cols:
        return np.maximum(out["_pred"].to_numpy(), 0.0)
    gmean = out.groupby(cols, observed=True)["_pred"].transform("mean")
    blended = (1.0 - alpha) * out["_pred"] + alpha * gmean
    return np.maximum(blended.to_numpy(), 0.0)
