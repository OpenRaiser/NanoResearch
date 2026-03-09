"""Automated training curve analysis.

Detects convergence speed, overfitting, stability from training logs.
"""
from __future__ import annotations

import math


def analyze_training_dynamics(training_log: list[dict]) -> dict:
    """Analyze training curve for convergence, overfitting, stability.

    Args:
        training_log: List of dicts with keys like "epoch", "train_loss",
                      "val_loss", plus optional metric keys.

    Returns:
        Dict with convergence_epoch, overfitting_detected, stability, etc.
    """
    val_losses = [e["val_loss"] for e in training_log
                  if isinstance(e.get("val_loss"), (int, float))
                  and _is_finite(e["val_loss"])]
    train_losses = [e["train_loss"] for e in training_log
                    if isinstance(e.get("train_loss"), (int, float))
                    and _is_finite(e["train_loss"])]

    result: dict = {
        "total_epochs": len(training_log),
        "has_val_loss": len(val_losses) > 0,
        "has_train_loss": len(train_losses) > 0,
    }

    if len(val_losses) < 3:
        result["analysis_skipped"] = (
            "insufficient data (need >= 3 val_loss entries)"
        )
        return result

    # 1. Convergence speed
    initial = val_losses[0]
    final_best = min(val_losses)
    target_90 = initial - 0.9 * (initial - final_best)
    convergence_epoch = len(val_losses)  # default: never
    for i, loss in enumerate(val_losses):
        if loss <= target_90:
            convergence_epoch = i
            break
    result["convergence_epoch"] = convergence_epoch
    result["convergence_ratio"] = round(
        convergence_epoch / len(val_losses), 3
    )

    # 2. Best epoch
    best_epoch = int(_argmin(val_losses))
    result["best_epoch"] = best_epoch
    result["best_val_loss"] = round(val_losses[best_epoch], 6)
    result["early_stopping_recommended"] = best_epoch < len(val_losses) * 0.7

    # 3. Overfitting detection (last 1/3 of training)
    split = max(1, len(val_losses) * 2 // 3)
    if len(val_losses[split:]) >= 2:
        val_tail = val_losses[split:]
        val_trend = _linear_slope(val_tail)
        result["val_loss_tail_trend"] = round(val_trend, 6)

        overfitting = False
        if (len(train_losses) >= len(val_losses)
                and len(train_losses[split:]) >= 2):
            train_tail = train_losses[split:]
            train_trend = _linear_slope(train_tail)
            result["train_loss_tail_trend"] = round(train_trend, 6)
            overfitting = val_trend > 0.001 and train_trend < -0.001
        else:
            overfitting = val_trend > 0.001
        result["overfitting_detected"] = overfitting

    # 4. Train-val gap (at last epoch)
    if train_losses and val_losses:
        idx = min(len(train_losses), len(val_losses)) - 1
        result["final_train_val_gap"] = round(
            train_losses[idx] - val_losses[idx], 6
        )

    # 5. Loss stability (std of epoch-to-epoch differences)
    diffs = [val_losses[i + 1] - val_losses[i]
             for i in range(len(val_losses) - 1)]
    mean_loss = sum(val_losses) / len(val_losses)
    if mean_loss > 1e-8:
        stability = _std(diffs) / mean_loss
        result["loss_stability_ratio"] = round(stability, 4)
        result["loss_stability"] = (
            "stable" if stability < 0.1
            else "noisy" if stability < 0.5
            else "unstable"
        )

    return result


# ── Helpers ──────────────────────────────────────────────────────────────

def _is_finite(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def _argmin(xs: list[float]) -> int:
    return min(range(len(xs)), key=lambda i: xs[i])


def _linear_slope(ys: list[float]) -> float:
    """Least-squares slope (no numpy needed)."""
    n = len(ys)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(ys) / n
    num = sum((i - x_mean) * (ys[i] - y_mean) for i in range(n))
    denom = sum((i - x_mean) ** 2 for i in range(n))
    return num / denom if denom > 1e-12 else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    return (sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
