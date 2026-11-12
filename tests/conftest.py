import numpy as np
import pytest

from judgelab.io import ratings_from_records


def records_from_matrix(mat, missing=None, criterion="overall"):
    """Rows = items, columns = raters; `missing` marks an absent rating."""
    recs = []
    for i, row in enumerate(mat):
        for r, lab in enumerate(row):
            if lab is missing or (isinstance(lab, float) and np.isnan(lab)):
                continue
            recs.append(
                {
                    "item_id": f"i{i:03d}",
                    "rater_id": f"r{r:02d}",
                    "label": lab,
                    "criterion": criterion,
                }
            )
    return recs


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def tiny_rs():
    mat = [
        [1, 1, 2],
        [2, 2, 2],
        [1, 2, None],
        [3, 3, 1],
    ]
    return ratings_from_records(records_from_matrix(mat))
