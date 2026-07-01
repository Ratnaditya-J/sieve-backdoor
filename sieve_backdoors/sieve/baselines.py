# ---------------------------------------------------------------------------
# VENDORED from the SIEVE validity-audit codebase (sieve-audit).
#   source repo:   sieve-audit  (author's SIEVE / latent-horizon project)
#   source path:   src/sieve_audit/baselines.py
#   source commit: f9632ec0796d4ac2beb44fcce44874d608286c20
#   vendored:      2026-07-01
# PROVENANCE: copied verbatim, then package-relative imports rewritten
#   (sieve_audit -> sieve_backdoors.sieve) so this repo is standalone.
# POLICY (build-prompt §11): WRAP, DO NOT FORK. The gate/control/verdict LOGIC
#   in this file is NOT edited; only import paths were adjusted. If a behavior
#   change is ever needed, wrap this module from sieve_backdoors, never edit here.
# ---------------------------------------------------------------------------
"""Surface baselines: the must-beat opponents for any decodability claim.

If a logistic regression over prompt length or TF-IDF features matches the
probe, the "signal" is explainable by surface text statistics and no
activation-level claim is warranted (DESIGN.md section 4). Baselines are
trained on the training families and evaluated on held-out families, exactly
like the probe is supposed to generalize.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SURFACE_BASELINES = ("length", "tfidf")


def _length_features(texts: list[str]) -> np.ndarray:
    """Character count, word count, mean word length, digit and punctuation rates."""
    feats = []
    for t in texts:
        words = t.split()
        n_char = max(len(t), 1)
        feats.append(
            [
                len(t),
                len(words),
                (sum(len(w) for w in words) / len(words)) if words else 0.0,
                sum(c.isdigit() for c in t) / n_char,
                sum(not c.isalnum() and not c.isspace() for c in t) / n_char,
            ]
        )
    return np.asarray(feats, dtype=float)


def fit_baseline_scores(
    name: str,
    train_texts: list[str],
    train_labels: np.ndarray,
    eval_texts: list[str],
    seed: int = 0,
) -> np.ndarray:
    """Train one surface baseline on the train split, return scores on the eval split."""
    if name == "length":
        model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)
        )
        model.fit(_length_features(train_texts), train_labels)
        return model.predict_proba(_length_features(eval_texts))[:, 1]
    if name == "tfidf":
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(max_iter=1000, random_state=seed),
        )
        model.fit(train_texts, train_labels)
        return model.predict_proba(eval_texts)[:, 1]
    raise ValueError(f"unknown surface baseline: {name!r}")
