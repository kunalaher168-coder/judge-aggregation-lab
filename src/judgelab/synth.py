"""Synthetic rating data with known ground truth.

Each rater is defined by an explicit confusion matrix C[true, reported], so
the true reliability of every rater is known and the Bayes-optimal
aggregate can be computed as an upper bound. Rater archetypes mirror known
failure modes of human and LLM raters:

  - ordinal noise: mistakes land mostly on adjacent scores (careful humans)
  - spammer: answers independent of the item (careless or bot raters)
  - lenient / harsh: systematic +1 / -1 shift (LLM-judge leniency bias)
  - correlated group: judges that share errors (same model family / prompt)

All data is generated from seeded random numbers. No real data is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CLASSES = [1, 2, 3, 4]
PRIORS = np.array([0.15, 0.35, 0.35, 0.15])


@dataclass
class RaterSpec:
    rater_id: str
    rater_type: str
    confusion: np.ndarray
    group: str | None = None
    rho: float = 0.0  # chance of copying the group's shared draw

    def true_accuracy(self, priors: np.ndarray) -> float:
        return float(np.diag(self.confusion) @ priors)


@dataclass
class SynthData:
    scenario: str
    classes: list
    priors: np.ndarray
    specs: list[RaterSpec]
    records: list[dict]
    truth: dict[str, int]
    meta: dict = field(default_factory=dict)


def ordinal_noise(K: int, accuracy: float, decay: float = 1.5) -> np.ndarray:
    """Diagonal = accuracy; errors fall off as exp(-decay * distance)."""
    idx = np.arange(K)
    dist = np.abs(idx[:, None] - idx[None, :])
    off = np.exp(-decay * dist) * (dist > 0)
    off = off / off.sum(axis=1, keepdims=True) * (1 - accuracy)
    return off + np.eye(K) * accuracy


def spammer(K: int, preference: np.ndarray | None = None) -> np.ndarray:
    """Every row identical: the report carries no information about the truth."""
    p = np.full(K, 1.0 / K) if preference is None else np.asarray(preference, float)
    return np.tile(p / p.sum(), (K, 1))


def shifted(K: int, shift: int, bias: float, accuracy_rest: float = 0.85) -> np.ndarray:
    """With probability `bias` report true+shift (clipped); otherwise ordinal noise."""
    base = ordinal_noise(K, accuracy_rest)
    shift_m = np.zeros((K, K))
    for k in range(K):
        shift_m[k, int(np.clip(k + shift, 0, K - 1))] = 1.0
    return bias * shift_m + (1 - bias) * base


def scenario_specs(name: str, K: int = 4) -> tuple[list[RaterSpec], int | None]:
    """Rater pool and raters-per-item for each named scenario (None = all raters)."""
    H, L = "human", "llm"
    if name == "independent":
        accs = [0.80, 0.70, 0.70, 0.65, 0.65, 0.60, 0.60]
        return [RaterSpec(f"human_{i + 1}", H, ordinal_noise(K, a)) for i, a in enumerate(accs)], 5
    if name == "spammers":
        specs = [
            RaterSpec(f"human_{i + 1}", H, ordinal_noise(K, a))
            for i, a in enumerate([0.80, 0.70, 0.65, 0.60])
        ]
        specs += [RaterSpec(f"careless_{i + 1}", H, spammer(K)) for i in range(3)]
        return specs, 5
    if name == "biased_llm":
        specs = [
            RaterSpec(f"human_{i + 1}", H, ordinal_noise(K, a))
            for i, a in enumerate([0.75, 0.65, 0.60])
        ]
        specs += [
            RaterSpec("llm_strong", L, ordinal_noise(K, 0.80)),
            RaterSpec("llm_lenient_a", L, shifted(K, +1, 0.55)),
            RaterSpec("llm_lenient_b", L, shifted(K, +1, 0.50)),
            RaterSpec("llm_harsh", L, shifted(K, -1, 0.50)),
        ]
        return specs, 5
    if name == "mixed":
        specs = [
            RaterSpec(f"human_{i + 1}", H, ordinal_noise(K, a))
            for i, a in enumerate([0.80, 0.65, 0.60])
        ]
        specs += [
            RaterSpec("careless_1", H, spammer(K, [1, 2, 4, 3])),
            RaterSpec("llm_strong", L, ordinal_noise(K, 0.80)),
            RaterSpec("llm_weak", L, ordinal_noise(K, 0.50)),
            RaterSpec("llm_lenient_a", L, shifted(K, +1, 0.55)),
            RaterSpec("llm_lenient_b", L, shifted(K, +1, 0.50)),
            RaterSpec("llm_harsh", L, shifted(K, -1, 0.50)),
        ]
        return specs, 5
    if name == "correlated_llm":
        specs = [
            RaterSpec(f"human_{i + 1}", H, ordinal_noise(K, a)) for i, a in enumerate([0.75, 0.70])
        ]
        specs += [
            RaterSpec(f"llm_family_{c}", L, ordinal_noise(K, 0.60), group="family", rho=0.7)
            for c in "abc"
        ]
        return specs, None
    raise ValueError(f"unknown scenario '{name}' (choose from {SCENARIOS})")


SCENARIOS = ["independent", "spammers", "biased_llm", "mixed", "correlated_llm"]


def generate(
    scenario: str = "mixed",
    n_items: int = 400,
    raters_per_item: int | None = -1,
    seed: int = 0,
) -> SynthData:
    """Draw true labels and ratings. raters_per_item=-1 uses the scenario default."""
    rng = np.random.default_rng(seed)
    K = len(CLASSES)
    specs, default_rpi = scenario_specs(scenario, K)
    rpi = default_rpi if raters_per_item == -1 else raters_per_item
    n_raters = len(specs)
    rpi = n_raters if rpi is None else min(rpi, n_raters)

    truth_idx = rng.choice(K, size=n_items, p=PRIORS)
    groups = {s.group for s in specs if s.group}
    records, truth = [], {}
    width = len(str(n_items))
    for i in range(n_items):
        item_id = f"item_{i:0{width}d}"
        y = int(truth_idx[i])
        truth[item_id] = CLASSES[y]
        chosen = (
            rng.choice(n_raters, size=rpi, replace=False) if rpi < n_raters else np.arange(n_raters)
        )
        shared = {}
        for g in groups:
            row = next(s for s in specs if s.group == g).confusion[y]
            shared[g] = int(rng.choice(K, p=row))
        for r in sorted(chosen):
            spec = specs[r]
            if spec.group and rng.random() < spec.rho:
                lab = shared[spec.group]
            else:
                lab = int(rng.choice(K, p=spec.confusion[y]))
            records.append(
                {
                    "item_id": item_id,
                    "rater_id": spec.rater_id,
                    "rater_type": spec.rater_type,
                    "criterion": "overall",
                    "label": CLASSES[lab],
                }
            )
    meta = {
        "scenario": scenario,
        "n_items": n_items,
        "raters_per_item": rpi,
        "seed": seed,
        "raters": {
            s.rater_id: {
                "type": s.rater_type,
                "true_accuracy": round(s.true_accuracy(PRIORS), 4),
                "correlated_group": s.group,
            }
            for s in specs
        },
    }
    return SynthData(scenario, list(CLASSES), PRIORS.copy(), specs, records, truth, meta)
