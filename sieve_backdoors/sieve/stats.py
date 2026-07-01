# ---------------------------------------------------------------------------
# VENDORED from the SIEVE validity-audit codebase (sieve-audit).
#   source repo:   sieve-audit  (author's SIEVE / latent-horizon project)
#   source path:   src/sieve_audit/stats.py
#   source commit: f9632ec0796d4ac2beb44fcce44874d608286c20
#   vendored:      2026-07-01
# PROVENANCE: copied verbatim, then package-relative imports rewritten
#   (sieve_audit -> sieve_backdoors.sieve) so this repo is standalone.
# POLICY (build-prompt §11): WRAP, DO NOT FORK. The gate/control/verdict LOGIC
#   in this file is NOT edited; only import paths were adjusted. If a behavior
#   change is ever needed, wrap this module from sieve_backdoors, never edit here.
# ---------------------------------------------------------------------------
"""Statistical primitives used by every gate: AUROC, bootstrap CIs, agreement.

All randomness flows through an explicit ``numpy.random.Generator`` so audits
are reproducible from (bundle, config, seed) alone.

Design notes (hardened after adversarial review):

- AUROC bootstraps are *stratified* (resampled within class) so class
  imbalance cannot push resamples to a single class and drag the CI toward
  0.5.
- Steering-arm comparisons use per-prompt effect *magnitudes*, paired by
  prompt where possible. Comparing |mean| would let a control whose effects
  are large but mixed-sign cancel to ~0, making a weak probe look superior.
- The dose-response p-value comes from a within-prompt permutation test, not
  from treating per-(prompt, alpha) points as independent: every prompt
  contributes a delta at each alpha, so naive p-values are pseudo-replicated.
- Cohen's kappa is NaN (not 1.0) for constant raters: two judges that always
  output the same class provide no evidence of reliability.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as _scipy_stats
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class CI:
    """A point estimate with a two-sided bootstrap confidence interval."""

    point: float
    lo: float
    hi: float
    level: float = 0.95

    def excludes(self, value: float) -> bool:
        return self.lo > value or self.hi < value

    def to_dict(self) -> dict:
        return {"point": self.point, "lo": self.lo, "hi": self.hi, "level": self.level}


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC; 0.5 when only one class is present (no information either way)."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def _stratified_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One stratified bootstrap resample: indices drawn within each class."""
    idx_parts = []
    for cls in np.unique(labels):
        cls_idx = np.flatnonzero(labels == cls)
        idx_parts.append(rng.choice(cls_idx, len(cls_idx), replace=True))
    return np.concatenate(idx_parts)


def bootstrap_auroc(
    labels: np.ndarray,
    scores: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 2000,
    level: float = 0.95,
) -> CI:
    """Stratified bootstrap CI for AUROC."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    point = auroc(labels, scores)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        idx = _stratified_indices(labels, rng)
        reps[b] = auroc(labels[idx], scores[idx])
    lo, hi = _percentile_ci(reps, level)
    return CI(point, lo, hi, level)


def bootstrap_auroc_diff(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 2000,
    level: float = 0.95,
) -> CI:
    """Stratified bootstrap CI for AUROC(a) - AUROC(b) on the same examples."""
    labels = np.asarray(labels)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    point = auroc(labels, scores_a) - auroc(labels, scores_b)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        idx = _stratified_indices(labels, rng)
        reps[b] = auroc(labels[idx], scores_a[idx]) - auroc(labels[idx], scores_b[idx])
    lo, hi = _percentile_ci(reps, level)
    return CI(point, lo, hi, level)


def bootstrap_mean(
    values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 2000,
    level: float = 0.95,
) -> CI:
    """Bootstrap CI for the mean of a sample."""
    values = np.asarray(values, dtype=float)
    point = float(values.mean())
    n = len(values)
    idx = rng.integers(0, n, (n_boot, n))
    reps = values[idx].mean(axis=1)
    lo, hi = _percentile_ci(reps, level)
    return CI(point, lo, hi, level)


def bootstrap_magnitude_diff(
    probe_by_prompt: dict[str, float],
    control_by_prompt: dict[str, float],
    rng: np.random.Generator,
    n_boot: int = 2000,
    level: float = 0.95,
) -> CI:
    """CI for mean(|probe effect|) - mean(|control effect|), paired by prompt.

    Magnitudes are compared per prompt so a control with large but mixed-sign
    effects cannot cancel to a deceptively small aggregate. Prompts present in
    both arms are resampled jointly (paired); if the arms share no prompts,
    falls back to independent resampling with a wider, honest CI.
    """
    shared = sorted(set(probe_by_prompt) & set(control_by_prompt))
    if shared:
        d = np.array(
            [abs(probe_by_prompt[p]) - abs(control_by_prompt[p]) for p in shared]
        )
        return bootstrap_mean(d, rng, n_boot, level)
    a = np.abs(np.array(list(probe_by_prompt.values())))
    b = np.abs(np.array(list(control_by_prompt.values())))
    point = float(a.mean() - b.mean())
    reps = np.empty(n_boot)
    for i in range(n_boot):
        ra = a[rng.integers(0, len(a), len(a))]
        rb = b[rng.integers(0, len(b), len(b))]
        reps[i] = ra.mean() - rb.mean()
    lo, hi = _percentile_ci(reps, level)
    return CI(point, lo, hi, level)


def dose_response_clustered(
    deltas: dict[float, dict[str, float]],
    rng: np.random.Generator,
    n_perm: int = 1000,
) -> tuple[float, float]:
    """Spearman rho of effect vs alpha, with a within-prompt permutation p.

    ``deltas[alpha][prompt_id]`` is the behavioral delta vs alpha=0. The same
    prompt contributes one delta per alpha, so per-record points are not
    independent; the null distribution is built by permuting the alpha labels
    *within each prompt*, which preserves every prompt's own delta set.
    """
    alphas = sorted(deltas)
    if len(alphas) < 3:
        return 0.0, 1.0
    prompts = sorted(set.intersection(*(set(deltas[a]) for a in alphas)))
    if len(prompts) < 3:
        return 0.0, 1.0

    # matrix: rows = prompts, cols = alphas
    mat = np.array([[deltas[a][p] for a in alphas] for p in prompts])
    alpha_arr = np.array(alphas, dtype=float)

    def stat(m: np.ndarray) -> float:
        xs = np.tile(alpha_arr, len(prompts))
        ys = m.ravel()
        rho = _scipy_stats.spearmanr(xs, ys).statistic
        return 0.0 if np.isnan(rho) else float(rho)

    observed = stat(mat)
    count = 0
    for _ in range(n_perm):
        perm = np.array([row[rng.permutation(len(alphas))] for row in mat])
        if abs(stat(perm)) >= abs(observed):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return observed, float(p)


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two binary raters; NaN when chance agreement is 1.

    Constant raters (both always 0 or always 1) carry no information about
    reliability, so they must not score as perfect agreement.
    """
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    po = float(np.mean(a == b))
    pa = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    if pa >= 1.0:
        return float("nan")
    return float((po - pa) / (1 - pa))


def _percentile_ci(reps: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    return float(np.quantile(reps, tail)), float(np.quantile(reps, 1.0 - tail))
