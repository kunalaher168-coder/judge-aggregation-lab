"""The three agent roles: planner, judge and aggregator (reconciler).

planner     reads the rubric once and writes self-contained instructions for
            each criterion (one judge agent per criterion)
judge       scores one criterion of one item: pass / fail + confidence
aggregator  combines the panel of judges; unanimous verdicts pass straight
            through, disagreements go to a reconciler model call
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from judgelab.agents.backends import Backend, BackendError, CallRecord, _parse_rubric_criteria

VERDICTS = ("pass", "fail")

PLANNER_SYSTEM = (
    "You are the planner agent in a multi-agent evaluation pipeline. Read the grading rubric and "
    "split it into independent criteria, one per judge agent. For each criterion write self-contained "
    "judge instructions: what to check, what counts as pass, what counts as fail, and which problems to "
    "ignore because another judge covers them. Keep each criterion id exactly as it appears in square "
    "brackets in the rubric."
)

PLANNER_SCHEMA = {
    "type": "object",
    "properties": {
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "judge_instructions": {"type": "string"},
                },
                "required": ["id", "name", "judge_instructions"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["criteria"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = (
    "You are one judge agent in an evaluation panel. You assess exactly one criterion of the response "
    "shown and nothing else; other judges cover the other criteria. Work only from the material "
    "provided and recompute any numbers yourself rather than trusting the response. Reply with a short "
    "rationale (at most two sentences), a verdict of 'pass' or 'fail', and your confidence that the "
    "verdict is correct as a number between 0.5 and 1.0."
)

RECONCILER_SYSTEM = (
    "You are the aggregator agent in a multi-agent evaluation panel. Several judge agents scored the "
    "same criterion of the same response and disagreed. Re-check the response against the criterion "
    "yourself, weigh each judge's rationale on its merits rather than by head count, and give the "
    "final verdict with a short rationale (at most two sentences) and your confidence between 0.5 and 1.0."
)

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "confidence": {"type": "number"},
    },
    "required": ["rationale", "verdict", "confidence"],
    "additionalProperties": False,
}

_BRACKET_ID = re.compile(r"\[([a-z0-9_]+)\]")


@dataclass
class Criterion:
    id: str
    name: str
    judge_instructions: str


@dataclass
class Verdict:
    item_id: str
    criterion: str
    judge: str  # tier name, or "reconciler"
    verdict: str | None
    confidence: float | None
    rationale: str = ""


def render_item(item: dict, fields: list[str] | None = None) -> str:
    fields = fields or [k for k in item if k != "item_id"]
    return "\n".join(f"<{k}>\n{item[k]}\n</{k}>" for k in fields)


def _clean_verdict(data: dict) -> tuple[str, float, str]:
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        raise BackendError(f"invalid verdict {data.get('verdict')!r}")
    try:
        conf = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return verdict, min(max(conf, 0.5), 1.0), str(data.get("rationale", ""))[:600]


def plan_criteria(
    backend: Backend, rubric: str, model: str, effort: str = "medium"
) -> tuple[list[Criterion], CallRecord]:
    """Planner agent. Falls back to the rubric's own wording if the planner's
    criterion ids do not match the bracketed ids in the rubric."""
    expected = _BRACKET_ID.findall(rubric)
    t0 = time.perf_counter()
    try:
        resp = backend.complete_json(
            model=model,
            system=PLANNER_SYSTEM,
            user=f"<rubric>\n{rubric}\n</rubric>",
            schema=PLANNER_SCHEMA,
            effort=effort,
            max_tokens=8000,
            meta={"agent": "planner"},
        )
    except BackendError as exc:
        record = CallRecord.failed("planner", model, None, None, time.perf_counter() - t0, str(exc))
        return [Criterion(**c) for c in _parse_rubric_criteria(rubric)], record
    record = CallRecord.from_response("planner", resp, None, None)
    criteria = [
        Criterion(c["id"].strip(), c["name"].strip(), c["judge_instructions"].strip())
        for c in resp.data.get("criteria", [])
    ]
    if expected and sorted(c.id for c in criteria) != sorted(expected):
        record.error = (
            f"planner ids {[c.id for c in criteria]} != rubric ids {expected}; used rubric text"
        )
        criteria = [Criterion(**c) for c in _parse_rubric_criteria(rubric)]
    return criteria, record


def judge_one(
    backend: Backend,
    model: str,
    tier: str,
    criterion: Criterion,
    item: dict,
    effort: str = "low",
    fields: list[str] | None = None,
) -> tuple[Verdict, CallRecord]:
    """Judge agent: one criterion, one item."""
    user = (
        f"Criterion: {criterion.name} [{criterion.id}]\n"
        f"Judge instructions: {criterion.judge_instructions}\n\n"
        f"{render_item(item, fields)}\n\n"
        "Judge only this criterion."
    )
    meta = {"agent": "judge", "item_id": item["item_id"], "criterion": criterion.id}
    t0 = time.perf_counter()
    try:
        resp = backend.complete_json(
            model=model,
            system=JUDGE_SYSTEM,
            user=user,
            schema=VERDICT_SCHEMA,
            effort=effort,
            max_tokens=4096,
            meta=meta,
        )
        verdict, conf, why = _clean_verdict(resp.data)
    except BackendError as exc:
        rec = CallRecord.failed(
            "judge", model, item["item_id"], criterion.id, time.perf_counter() - t0, str(exc)
        )
        rec.tier = tier
        return Verdict(item["item_id"], criterion.id, tier, None, None, ""), rec
    rec = CallRecord.from_response("judge", resp, item["item_id"], criterion.id)
    rec.tier = tier
    return Verdict(item["item_id"], criterion.id, tier, verdict, conf, why), rec


def reconcile(
    backend: Backend,
    model: str,
    criterion: Criterion,
    item: dict,
    panel: list[Verdict],
    effort: str = "medium",
    fields: list[str] | None = None,
) -> tuple[Verdict, CallRecord]:
    """Aggregator agent's model call for a criterion the panel disagreed on."""
    lines = [
        f"- {v.judge}: {v.verdict} (confidence {v.confidence:.2f}). Rationale: {v.rationale}"
        for v in panel
        if v.verdict is not None
    ]
    user = (
        f"Criterion: {criterion.name} [{criterion.id}]\n"
        f"Judge instructions: {criterion.judge_instructions}\n\n"
        f"{render_item(item, fields)}\n\n"
        "Judge verdicts:\n" + "\n".join(lines)
    )
    meta = {"agent": "reconciler", "item_id": item["item_id"], "criterion": criterion.id}
    t0 = time.perf_counter()
    try:
        resp = backend.complete_json(
            model=model,
            system=RECONCILER_SYSTEM,
            user=user,
            schema=VERDICT_SCHEMA,
            effort=effort,
            max_tokens=8000,
            meta=meta,
        )
        verdict, conf, why = _clean_verdict(resp.data)
    except BackendError as exc:
        rec = CallRecord.failed(
            "reconciler", model, item["item_id"], criterion.id, time.perf_counter() - t0, str(exc)
        )
        return Verdict(item["item_id"], criterion.id, "reconciler", None, None, ""), rec
    rec = CallRecord.from_response("reconciler", resp, item["item_id"], criterion.id)
    return Verdict(item["item_id"], criterion.id, "reconciler", verdict, conf, why), rec
