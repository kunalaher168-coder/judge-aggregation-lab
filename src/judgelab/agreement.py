"""Inter-rater agreement: Cohen's kappa, Fleiss' kappa, Krippendorff's alpha.

Implemented from the textbook definitions and cross-checked in the test
suite against scikit-learn, statsmodels and the `krippendorff` package.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from judgelab.io import RatingSet


def _weight_matrix(n_classes: int, weights: str | None) -> np.ndarray:
    idx = np.arange(n_classes)
    diff = np.abs(idx[:, None] - idx[None, :]).astype(float)
    if weights is None:
        return (diff > 0).astype(float)
    if weights == "linear":
        return diff
    if weights == "quadratic":
        return diff**2
    raise ValueError(f"unknown weights '{weights}' (None, 'linear', 'quadratic')")


def cohen_kappa(a, b, n_classes: int | None = None, weights: str | None = None) -> float:
    """Cohen's kappa between two raters' class indices on the same items.

    weights=None gives the unweighted kappa; 'linear' / 'quadratic' give the
    weighted kappa for ordinal scales. Returns nan when chance agreement is 1
    (both raters used a single identical class).
    """
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("cohen_kappa needs two equal-length, non-empty label arrays")
    K = n_classes or int(max(a.max(), b.max()) + 1)
    obs = np.zeros((K, K))
    np.add.at(obs, (a, b), 1.0)
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    w = _weight_matrix(K, weights)
    denom = (w * exp).sum()
    if denom == 0:
        return float("nan")
    return float(1.0 - (w * obs).sum() / denom)


def pairwise_cohen(rs: RatingSet, weights: str | None = None, min_shared: int = 2) -> dict:
    """Cohen's kappa for every rater pair on the items both rated.

    Returns {"pairs": {(r1, r2): {"kappa": k, "n": shared}}, "mean": m}, where
    the mean is weighted by the number of shared items (pairs with fewer than
    `min_shared` items or undefined kappa are skipped).
    """
    dense = rs.dense()
    pairs = {}
    total, weight = 0.0, 0.0
    for i, j in combinations(range(rs.n_raters), 2):
        both = (dense[:, i] >= 0) & (dense[:, j] >= 0)
        n = int(both.sum())
        if n < min_shared:
            continue
        k = cohen_kappa(dense[both, i], dense[both, j], rs.n_classes, weights)
        pairs[(rs.raters[i], rs.raters[j])] = {"kappa": k, "n": n}
        if not np.isnan(k):
            total += k * n
            weight += n
    return {"pairs": pairs, "mean": total / weight if weight else float("nan")}


def percent_agreement(rs: RatingSet) -> float:
    """Share of agreeing rater pairs, pooled over items with 2+ ratings."""
    counts = rs.vote_counts()
    n = counts.sum(axis=1)
    keep = n >= 2
    agree = (counts[keep] * (counts[keep] - 1)).sum()
    possible = (n[keep] * (n[keep] - 1)).sum()
    return float(agree / possible) if possible else float("nan")


def fleiss_kappa(rs: RatingSet) -> float:
    """Fleiss' kappa, generalised to a variable number of raters per item.

    P_i = (sum_j n_ij^2 - n_i) / (n_i (n_i - 1)) for items with n_i >= 2;
    p_j = category share over all ratings on those items. With a constant
    n_i this is exactly Fleiss (1971).
    """
    counts = rs.vote_counts()
    n = counts.sum(axis=1)
    keep = n >= 2
    counts, n = counts[keep], n[keep]
    if counts.shape[0] == 0:
        return float("nan")
    p_i = ((counts**2).sum(axis=1) - n) / (n * (n - 1))
    p_bar = p_i.mean()
    p_j = counts.sum(axis=0) / n.sum()
    p_e = (p_j**2).sum()
    if p_e == 1:
        return float("nan")
    return float((p_bar - p_e) / (1 - p_e))


def _delta(level: str, n_c: np.ndarray, values: np.ndarray) -> np.ndarray:
    K = n_c.size
    if level == "nominal":
        return 1.0 - np.eye(K)
    if level == "interval":
        return (values[:, None] - values[None, :]) ** 2
    if level == "ordinal":
        cum = np.concatenate([[0.0], np.cumsum(n_c)])
        d = np.zeros((K, K))
        for c in range(K):
            for k in range(K):
                lo, hi = min(c, k), max(c, k)
                d[c, k] = (cum[hi + 1] - cum[lo] - (n_c[c] + n_c[k]) / 2.0) ** 2
        return d
    raise ValueError(f"unknown level '{level}' (nominal, ordinal, interval)")


def krippendorff_alpha(rs: RatingSet, level: str = "nominal") -> float:
    """Krippendorff's alpha via the coincidence matrix.

    Handles missing ratings natively; only items with 2+ ratings are pairable.
    level: 'nominal', 'ordinal' (category order = sorted label order) or
    'interval' (uses numeric label values).
    """
    counts = rs.vote_counts()
    m = counts.sum(axis=1)
    keep = m >= 2
    counts, m = counts[keep], m[keep]
    if counts.shape[0] == 0:
        return float("nan")
    # o_ck = sum_u (n_uc * n_uk - [c==k] n_uc) / (m_u - 1)
    scaled = counts / (m - 1)[:, None]
    o = scaled.T @ counts - np.diag(scaled.sum(axis=0))
    n_c = o.sum(axis=1)
    n = n_c.sum()
    delta = _delta(level, n_c, rs.class_values())
    d_o = (o * delta).sum()
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n - 1)
    if d_e == 0:
        return float("nan")
    return float(1.0 - d_o / d_e)


def agreement_summary(rs: RatingSet, ordinal: bool | None = None) -> dict[str, float]:
    """All agreement statistics in one dict. Ordinal stats are included when
    labels are numeric (or ordinal=True)."""
    if ordinal is None:
        ordinal = all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in rs.classes)
    out = {
        "n_items": rs.n_items,
        "n_raters": rs.n_raters,
        "n_ratings": rs.n_obs,
        "percent_agreement": percent_agreement(rs),
        "cohen_kappa_mean": pairwise_cohen(rs)["mean"],
        "fleiss_kappa": fleiss_kappa(rs),
        "krippendorff_alpha_nominal": krippendorff_alpha(rs, "nominal"),
    }
    if ordinal and rs.n_classes > 2:
        out["cohen_kappa_quadratic_mean"] = pairwise_cohen(rs, weights="quadratic")["mean"]
        out["krippendorff_alpha_ordinal"] = krippendorff_alpha(rs, "ordinal")
        out["krippendorff_alpha_interval"] = krippendorff_alpha(rs, "interval")
    return out
