"""Label aggregation: majority vote, reliability-weighted vote and Dawid-Skene EM.

All methods take a RatingSet and return an AggregationResult holding one
inferred label per item plus a per-item posterior over classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from judgelab.io import RatingSet

EPS = 1e-12


@dataclass
class AggregationResult:
    method: str
    items: list[str]
    classes: list
    posterior: np.ndarray  # items x classes, rows sum to 1
    label_idx: np.ndarray  # items
    rater_weights: dict[str, float] = field(default_factory=dict)
    rater_accuracy: dict[str, float] = field(default_factory=dict)
    confusion: np.ndarray | None = None  # raters x true x observed (Dawid-Skene)
    priors: np.ndarray | None = None
    n_ties: int = 0
    n_iter: int = 0
    converged: bool = True
    log_likelihood: float | None = None

    @property
    def labels(self) -> list:
        return [self.classes[int(k)] for k in self.label_idx]

    def as_dict(self) -> dict[str, object]:
        return {item: self.classes[int(k)] for item, k in zip(self.items, self.label_idx)}

    def confidence(self) -> np.ndarray:
        return self.posterior[np.arange(len(self.items)), self.label_idx]


def _argmax_with_ties(
    scores: np.ndarray, rng: np.random.Generator | None
) -> tuple[np.ndarray, int]:
    """Row-wise argmax. Ties broken uniformly at random (rng) or by lowest index."""
    best = scores.max(axis=1, keepdims=True)
    is_max = np.isclose(scores, best, rtol=0.0, atol=1e-9)
    n_max = is_max.sum(axis=1)
    out = np.argmax(scores, axis=1)
    tied_rows = np.flatnonzero(n_max > 1)
    if rng is not None:
        for i in tied_rows:
            out[i] = rng.choice(np.flatnonzero(is_max[i]))
    return out, int(tied_rows.size)


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def majority_vote(rs: RatingSet, tie_break: str = "random", seed: int = 0) -> AggregationResult:
    """Plurality vote. Posterior = vote share. Ties: 'random' (seeded) or 'lowest'."""
    counts = rs.vote_counts()
    rng = np.random.default_rng(seed) if tie_break == "random" else None
    labels, n_ties = _argmax_with_ties(counts, rng)
    posterior = counts / np.maximum(counts.sum(axis=1, keepdims=True), EPS)
    return AggregationResult(
        method="majority",
        items=rs.items,
        classes=rs.classes,
        posterior=posterior,
        label_idx=labels,
        rater_weights={r: 1.0 for r in rs.raters},
        n_ties=n_ties,
    )


def _loo_accuracy(rs: RatingSet, weights: np.ndarray, prior_strength: float = 1.0) -> np.ndarray:
    """Each rater's agreement with the weighted vote of the *other* raters.

    For every observation the rater's own vote is removed from the item's
    weighted tally; agreement with a k-way tie among the others counts 1/k.
    Returns a smoothed accuracy estimate per rater (Beta(1,1)-style prior).
    """
    K = rs.n_classes
    w_obs = weights[rs.rater_idx]
    tally = np.zeros((rs.n_items, K))
    np.add.at(tally, (rs.item_idx, rs.label_idx), w_obs)
    n_votes = np.bincount(rs.item_idx, minlength=rs.n_items)

    others = tally[rs.item_idx].copy()
    others[np.arange(rs.n_obs), rs.label_idx] -= w_obs
    has_others = n_votes[rs.item_idx] > 1
    best = others.max(axis=1, keepdims=True)
    is_max = np.isclose(others, best, rtol=0.0, atol=1e-9)
    agree = is_max[np.arange(rs.n_obs), rs.label_idx] / is_max.sum(axis=1)
    agree = np.where(has_others, agree, 0.0)

    agree_sum = np.bincount(rs.rater_idx, weights=agree, minlength=rs.n_raters)
    n_scored = np.bincount(rs.rater_idx, weights=has_others.astype(float), minlength=rs.n_raters)
    return (agree_sum + prior_strength) / (n_scored + 2.0 * prior_strength)


def _gold_accuracy(
    rs: RatingSet, gold: dict[str, object], prior_strength: float = 1.0
) -> np.ndarray:
    class_pos = {c: i for i, c in enumerate(rs.classes)}
    gold_idx = np.full(rs.n_items, -1)
    for i, item in enumerate(rs.items):
        if item in gold and gold[item] in class_pos:
            gold_idx[i] = class_pos[gold[item]]
    g = gold_idx[rs.item_idx]
    scored = g >= 0
    correct = (g == rs.label_idx) & scored
    hits = np.bincount(rs.rater_idx, weights=correct.astype(float), minlength=rs.n_raters)
    n = np.bincount(rs.rater_idx, weights=scored.astype(float), minlength=rs.n_raters)
    return (hits + prior_strength) / (n + 2.0 * prior_strength)


def log_odds_weights(accuracy: np.ndarray, n_classes: int) -> np.ndarray:
    """Optimal vote weights under the symmetric 'one-coin' rater model.

    w = log((K-1) * a / (1 - a)); clipped at 0 so raters at or below chance
    get no say (a negative weight has no meaning for K > 2 classes).
    """
    a = np.clip(accuracy, 1e-6, 1 - 1e-6)
    return np.maximum(np.log((n_classes - 1) * a / (1 - a)), 0.0)


def weighted_vote(
    rs: RatingSet,
    weights: dict[str, float] | None = None,
    gold: dict[str, object] | None = None,
    n_iter: int = 3,
    seed: int = 0,
) -> AggregationResult:
    """Reliability-weighted vote.

    Weight source, in order of precedence:
      1. `weights`: explicit per-rater weights (e.g. from a calibration study);
      2. `gold`: accuracy on gold-labelled items -> log-odds weights;
      3. otherwise: accuracy against the leave-one-out weighted vote of the
         other raters, re-estimated `n_iter` times -> log-odds weights.
    """
    K = rs.n_classes
    accuracy = None
    if weights is not None:
        w = np.array([float(weights.get(r, 0.0)) for r in rs.raters])
    elif gold:
        accuracy = _gold_accuracy(rs, gold)
        w = log_odds_weights(accuracy, K)
    else:
        w = np.ones(rs.n_raters)
        for _ in range(max(1, n_iter)):
            accuracy = _loo_accuracy(rs, w)
            w = log_odds_weights(accuracy, K)
            if not np.any(w > 0):
                w = np.ones(rs.n_raters)
                break
    if not np.any(w > 0):
        w = np.ones(rs.n_raters)

    scores = np.zeros((rs.n_items, K))
    np.add.at(scores, (rs.item_idx, rs.label_idx), w[rs.rater_idx])
    labels, n_ties = _argmax_with_ties(scores, np.random.default_rng(seed))
    return AggregationResult(
        method="weighted",
        items=rs.items,
        classes=rs.classes,
        posterior=_softmax_rows(scores),
        label_idx=labels,
        rater_weights={r: float(x) for r, x in zip(rs.raters, w)},
        rater_accuracy={}
        if accuracy is None
        else {r: float(a) for r, a in zip(rs.raters, accuracy)},
        n_ties=n_ties,
    )


def bayes_posterior(
    rs: RatingSet, confusion: np.ndarray, priors: np.ndarray
) -> tuple[np.ndarray, float]:
    """Posterior over true classes given rater confusion matrices and class priors.

    This is the Dawid-Skene E-step. With the *true* confusion matrices it is
    the Bayes-optimal aggregator (used as an upper bound in the benchmark).
    Returns (posterior items x K, total log-likelihood).
    """
    K = rs.n_classes
    log_conf = np.log(np.maximum(confusion, EPS))
    log_post = np.tile(np.log(np.maximum(priors, EPS)), (rs.n_items, 1))
    contrib = log_conf[rs.rater_idx, :, rs.label_idx]  # n_obs x K
    np.add.at(log_post, rs.item_idx, contrib)
    m = log_post.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(log_post - m).sum(axis=1))
    post = np.exp(log_post - lse[:, None])
    assert post.shape == (rs.n_items, K)
    return post, float(lse.sum())


def dawid_skene(
    rs: RatingSet,
    max_iter: int = 500,
    tol: float = 1e-7,
    smoothing: float = 0.5,
    init: str = "majority",
) -> AggregationResult:
    """Dawid & Skene (1979) EM with a full confusion matrix per rater.

    Model: true class y_i ~ Cat(pi); rater r reports l with probability
    confusion[r, y_i, l], independently across raters given y_i.

    `smoothing` is a symmetric Dirichlet pseudo-count added to every
    confusion-matrix cell in the M-step (MAP estimate); it keeps sparse
    raters from getting zero-probability cells. Initialised from majority-vote
    vote shares, which also fixes the label permutation.
    """
    K, R = rs.n_classes, rs.n_raters
    if init == "majority":
        counts = rs.vote_counts()
        post = counts / np.maximum(counts.sum(axis=1, keepdims=True), EPS)
    elif init == "uniform":
        post = np.full((rs.n_items, K), 1.0 / K)
    else:
        raise ValueError(f"unknown init '{init}'")

    k_range = np.arange(K)[None, :]
    prev_ll = -np.inf
    ll = -np.inf
    converged = False
    n_iter = 0
    confusion = np.full((R, K, K), 1.0 / K)
    priors = np.full(K, 1.0 / K)
    for it in range(1, max_iter + 1):
        n_iter = it
        # M-step
        priors = (post.sum(axis=0) + 1e-3) / (rs.n_items + K * 1e-3)
        cnt = np.full((R, K, K), float(smoothing))
        np.add.at(
            cnt,
            (rs.rater_idx[:, None], k_range, rs.label_idx[:, None]),
            post[rs.item_idx],
        )
        confusion = cnt / np.maximum(cnt.sum(axis=2, keepdims=True), EPS)
        # E-step
        post, ll = bayes_posterior(rs, confusion, priors)
        if abs(ll - prev_ll) < tol * max(1.0, abs(ll)):
            converged = True
            break
        prev_ll = ll

    labels, n_ties = _argmax_with_ties(post, None)
    diag = confusion[:, k_range[0], k_range[0]]  # R x K
    accuracy = diag @ priors
    return AggregationResult(
        method="dawid_skene",
        items=rs.items,
        classes=rs.classes,
        posterior=post,
        label_idx=labels,
        rater_accuracy={r: float(a) for r, a in zip(rs.raters, accuracy)},
        confusion=confusion,
        priors=priors,
        n_ties=n_ties,
        n_iter=n_iter,
        converged=converged,
        log_likelihood=ll,
    )


METHODS = {
    "majority": majority_vote,
    "weighted": weighted_vote,
    "dawid_skene": dawid_skene,
}


def aggregate(rs: RatingSet, method: str, **kwargs) -> AggregationResult:
    aliases = {
        "mv": "majority",
        "wv": "weighted",
        "ds": "dawid_skene",
        "dawid-skene": "dawid_skene",
    }
    name = aliases.get(method, method)
    if name not in METHODS:
        raise ValueError(f"unknown method '{method}' (choose from {sorted(METHODS)})")
    return METHODS[name](rs, **kwargs)
