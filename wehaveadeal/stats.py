"""Shared statistics helpers for the validation notebooks
(the paper's validation analysis, the paper's validation analysis)
-- one implementation of bootstrap confidence intervals so both notebooks
report numbers the same, defensible way, instead of two copies that could
quietly drift apart.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(values, n_boot: int = 5000, ci: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """(mean, ci_lo, ci_hi) via a percentile bootstrap. Deliberately not a
    normal-approximation interval: every outcome distribution seen so far in
    this project (margins, profits) is skewed, not bell-shaped."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(values.mean()), float(lo), float(hi)


def paired_wilcoxon(a_values, b_values, n_boot: int = 5000, ci: float = 0.95, seed: int = 0):
    """Same-day paired comparison: `a_values[i]`/`b_values[i]` must already
    be aligned (same day, two different agents). Returns
    (mean_diff, ci_lo, ci_hi, p_raw) -- Wilcoxon signed-rank on the
    differences (non-parametric, appropriate for skewed outcome
    distributions), bootstrap CI on the mean difference. `p_raw` still needs
    multiple-comparison correction by the caller (e.g.
    `statsmodels.stats.multitest.multipletests(..., method="holm")`) if more
    than one pair is being tested at once -- that correction is scoped
    per-experiment by design, so it stays in the calling notebook rather
    than being baked in here."""
    a = np.asarray(a_values, dtype=float)
    b = np.asarray(b_values, dtype=float)
    diff = a - b
    mean_diff, ci_lo, ci_hi = bootstrap_ci(diff, n_boot=n_boot, ci=ci, seed=seed)
    try:
        _, p_raw = stats.wilcoxon(diff)
    except ValueError:
        p_raw = 1.0
    return mean_diff, ci_lo, ci_hi, float(p_raw)
