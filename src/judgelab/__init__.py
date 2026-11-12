"""judgelab: aggregate human and LLM judgements, measure agreement, run a multi-agent judge."""

from judgelab.aggregate import AggregationResult, dawid_skene, majority_vote, weighted_vote
from judgelab.agreement import cohen_kappa, fleiss_kappa, krippendorff_alpha, pairwise_cohen
from judgelab.io import RatingSet, load_ratings, ratings_from_records

__all__ = [
    "AggregationResult",
    "RatingSet",
    "cohen_kappa",
    "dawid_skene",
    "fleiss_kappa",
    "krippendorff_alpha",
    "load_ratings",
    "majority_vote",
    "pairwise_cohen",
    "ratings_from_records",
    "weighted_vote",
]

__version__ = "0.1.0"
