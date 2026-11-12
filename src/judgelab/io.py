"""Loading ratings from JSONL into an indexed, sparse representation.

Input format: one JSON object per line, one rating per object.

    {"item_id": "q17", "rater_id": "judge_haiku", "label": 3,
     "rater_type": "llm", "criterion": "accuracy"}

Required keys: item_id, rater_id, label. Optional: rater_type (free text,
e.g. "human" / "llm"), criterion (defaults to "overall"). An item does not
need a rating from every rater; missing ratings are simply absent.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_CRITERION = "overall"


@dataclass
class RatingSet:
    """Sparse ratings for one criterion.

    Observations are parallel arrays (item_idx[n], rater_idx[n], label_idx[n]).
    Labels are mapped to class indices 0..K-1 in sorted order, so numeric
    labels keep their natural order (needed for ordinal / interval metrics).
    """

    items: list[str]
    raters: list[str]
    classes: list
    item_idx: np.ndarray
    rater_idx: np.ndarray
    label_idx: np.ndarray
    rater_types: dict[str, str] = field(default_factory=dict)
    criterion: str = DEFAULT_CRITERION
    duplicates_dropped: int = 0

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def n_raters(self) -> int:
        return len(self.raters)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def n_obs(self) -> int:
        return int(self.item_idx.size)

    def dense(self) -> np.ndarray:
        """items x raters matrix of class indices, -1 where a rating is missing."""
        mat = np.full((self.n_items, self.n_raters), -1, dtype=int)
        mat[self.item_idx, self.rater_idx] = self.label_idx
        return mat

    def vote_counts(self) -> np.ndarray:
        """items x classes matrix of raw vote counts."""
        counts = np.zeros((self.n_items, self.n_classes))
        np.add.at(counts, (self.item_idx, self.label_idx), 1.0)
        return counts

    def class_values(self) -> np.ndarray:
        """Numeric value of each class (for interval metrics); falls back to rank."""
        try:
            return np.array([float(c) for c in self.classes])
        except (TypeError, ValueError):
            return np.arange(self.n_classes, dtype=float)

    def label_of(self, class_index: int):
        return self.classes[int(class_index)]


def _sort_key(value):
    # Numbers sort numerically and before strings; strings sort lexically.
    if isinstance(value, bool):
        return (1, str(value))
    if isinstance(value, (int, float)):
        return (0, float(value))
    return (1, str(value))


def ratings_from_records(
    records: Iterable[dict],
    criterion: str | None = None,
    classes: list | None = None,
) -> RatingSet:
    """Build a RatingSet from dict records, optionally filtered to one criterion.

    If the same (item, rater) pair appears twice, the last rating wins and the
    number of dropped duplicates is recorded on the result.
    """
    latest: dict[tuple[str, str], object] = {}
    rater_types: dict[str, str] = {}
    duplicates = 0
    chosen_criterion = criterion
    for rec in records:
        for key in ("item_id", "rater_id", "label"):
            if key not in rec:
                raise ValueError(f"rating record missing '{key}': {rec}")
        rec_criterion = str(rec.get("criterion", DEFAULT_CRITERION))
        if chosen_criterion is None:
            chosen_criterion = rec_criterion
        if rec_criterion != chosen_criterion:
            continue
        label = rec["label"]
        if label is None:
            continue
        key = (str(rec["item_id"]), str(rec["rater_id"]))
        if key in latest:
            duplicates += 1
        latest[key] = label
        if "rater_type" in rec:
            rater_types[key[1]] = str(rec["rater_type"])

    if not latest:
        raise ValueError(
            "no ratings found" + (f" for criterion '{criterion}'" if criterion else "")
        )

    items = sorted({k[0] for k in latest})
    raters = sorted({k[1] for k in latest})
    if classes is None:
        classes = sorted(set(latest.values()), key=_sort_key)
    class_pos = {c: i for i, c in enumerate(classes)}
    item_pos = {x: i for i, x in enumerate(items)}
    rater_pos = {x: i for i, x in enumerate(raters)}

    n = len(latest)
    item_idx = np.empty(n, dtype=int)
    rater_idx = np.empty(n, dtype=int)
    label_idx = np.empty(n, dtype=int)
    for j, ((item, rater), label) in enumerate(sorted(latest.items())):
        if label not in class_pos:
            raise ValueError(f"label {label!r} not in declared classes {classes}")
        item_idx[j] = item_pos[item]
        rater_idx[j] = rater_pos[rater]
        label_idx[j] = class_pos[label]

    return RatingSet(
        items=items,
        raters=raters,
        classes=list(classes),
        item_idx=item_idx,
        rater_idx=rater_idx,
        label_idx=label_idx,
        rater_types=rater_types,
        criterion=chosen_criterion or DEFAULT_CRITERION,
        duplicates_dropped=duplicates,
    )


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: str | Path, text: str) -> None:
    """Write UTF-8 text with LF line endings on every platform."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def criteria_in(records: Iterable[dict]) -> list[str]:
    seen: dict[str, None] = {}
    for rec in records:
        seen.setdefault(str(rec.get("criterion", DEFAULT_CRITERION)), None)
    return list(seen)


def load_ratings(path: str | Path, criterion: str | None = None) -> RatingSet:
    return ratings_from_records(read_jsonl(path), criterion=criterion)


def load_truth(path: str | Path, criterion: str | None = None) -> dict[str, object]:
    """Load gold labels: JSONL rows with item_id, label and optional criterion."""
    truth = {}
    for rec in read_jsonl(path):
        if criterion is not None and str(rec.get("criterion", DEFAULT_CRITERION)) != criterion:
            continue
        truth[str(rec["item_id"])] = rec["label"]
    return truth
