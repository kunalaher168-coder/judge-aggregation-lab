import json
import re
from decimal import Decimal
from types import SimpleNamespace

import pytest

from judgelab.agents.backends import AnthropicBackend, BackendError, SimulatedBackend
from judgelab.agents.dataset import CRITERIA, RUBRIC, generate_support_items
from judgelab.agents.pipeline import PipelineConfig, run_pipeline, write_outputs
from judgelab.agents.pricing import call_cost
from judgelab.agents.roles import Criterion, judge_one, plan_criteria
from judgelab.agents.routing import recommend, routing_policies, tier_table, wilson
from judgelab.io import load_ratings, read_jsonl

# ---------------------------------------------------------------- dataset


def test_support_dataset_truth_is_consistent():
    items, truth = generate_support_items(200, seed=3)
    by_key = {(t["item_id"], t["criterion"]): t for t in truth}
    for it in items:
        assert set(it) == {"item_id", "policy", "order", "customer_message", "agent_reply"}
        reply = it["agent_reply"]
        stated = Decimal(re.search(r"refund will be \$([0-9]+\.[0-9]{2})", reply).group(1))
        calc = by_key[(it["item_id"], "refund_calculation")]
        correct = Decimal(calc["correct_refund"])
        assert (stated == correct) == (calc["label"] == "pass")
        window = int(re.search(r"within (\d+) days", it["policy"]).group(1))
        stated_window = int(re.search(r"returns within (\d+) days", reply).group(1))
        if stated_window != window:
            assert by_key[(it["item_id"], "policy_fidelity")]["label"] == "fail"
        if "store credit to your account" in reply or "free gift" in reply:
            assert by_key[(it["item_id"], "no_unsupported_promises")]["label"] == "fail"
    fail_rate = sum(t["label"] == "fail" for t in truth) / len(truth)
    assert 0.25 < fail_rate < 0.45
    assert all(ord(ch) < 128 for it in items for v in it.values() for ch in v)


# ---------------------------------------------------------------- pricing


def test_call_cost():
    assert call_cost("claude-opus-5-5", 1_000_000, 0) == pytest.approx(4.0)
    assert call_cost("claude-haiku-5-5", 0, 1_000_000) == pytest.approx(0.5)
    assert call_cost("claude-sonnet-5-5", 1000, 1000) == pytest.approx(0.012)
    with pytest.raises(KeyError):
        call_cost("not-a-model", 1, 1)


# ---------------------------------------------------------------- Anthropic backend with a fake client


class FakeClient:
    def __init__(
        self,
        text='{"rationale": "ok", "verdict": "pass", "confidence": 0.9}',
        stop="end_turn",
        served=None,
    ):
        self.calls = []
        self.text, self.stop, self.served = text, stop, served
        self.messages = SimpleNamespace(create=self._make("messages"))
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._make("beta")))

    def _make(self, path):
        def create(**kwargs):
            self.calls.append((path, kwargs))
            return SimpleNamespace(
                stop_reason=self.stop,
                stop_details=SimpleNamespace(category="cyber") if self.stop == "refusal" else None,
                content=[
                    SimpleNamespace(type="thinking", thinking=""),
                    SimpleNamespace(type="text", text=self.text),
                ],
                usage=SimpleNamespace(
                    input_tokens=1000,
                    output_tokens=200,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=None,
                ),
                model=self.served or kwargs["model"],
                _request_id="req_test",
            )

        return create


def _call(backend, model):
    return backend.complete_json(
        model=model, system="s", user="u", schema={"type": "object"}, effort="low", max_tokens=100
    )


def test_anthropic_backend_request_shape_and_accounting():
    fake = FakeClient()
    be = AnthropicBackend(client=fake)
    r = _call(be, "claude-haiku-5-5")
    path, kw = fake.calls[-1]
    assert path == "messages"  # no server-side fallback for Haiku
    assert kw["output_config"] == {
        "format": {"type": "json_schema", "schema": {"type": "object"}},
        "effort": "low",
    }
    assert "thinking" not in kw and "temperature" not in kw
    assert r.data["verdict"] == "pass"
    assert r.cost_usd == pytest.approx(call_cost("claude-haiku-5-5", 1000, 200))
    assert r.request_id == "req_test"

    _call(be, "claude-opus-5-5")
    path, kw = fake.calls[-1]
    assert (
        path == "beta"
        and kw["fallbacks"] == "default"
        and kw["betas"] == ["server-side-fallback-2026-07-01"]
    )

    _call(AnthropicBackend(client=fake, use_fallbacks=False), "claude-sonnet-5-5")
    assert fake.calls[-1][0] == "messages"


def test_anthropic_backend_logs_fallback_model():
    be = AnthropicBackend(client=FakeClient(served="claude-opus-4-8"))
    r = _call(be, "claude-opus-5-5")
    assert r.served_model == "claude-opus-4-8"
    assert r.cost_usd == pytest.approx(call_cost("claude-opus-4-8", 1000, 200))


@pytest.mark.parametrize(
    "client",
    [
        FakeClient(stop="refusal"),
        FakeClient(stop="max_tokens"),
        FakeClient(text="not json"),
        FakeClient(text="[1, 2]"),
    ],
)
def test_anthropic_backend_failures_raise_and_keep_billing(client):
    with pytest.raises(BackendError) as info:
        _call(AnthropicBackend(client=client), "claude-haiku-5-5")
    # the API answered, so the call was billed: the error carries its accounting
    billed = info.value.response
    assert billed is not None and billed.input_tokens == 1000 and billed.output_tokens == 200
    assert billed.cost_usd == pytest.approx(call_cost("claude-haiku-5-5", 1000, 200))


def test_judge_records_error_instead_of_crashing():
    be = AnthropicBackend(
        client=FakeClient(text='{"rationale": "x", "verdict": "maybe", "confidence": 2}')
    )
    crit = Criterion("tone", "Tone", "be nice")
    verdict, rec = judge_one(be, "claude-haiku-5-5", "haiku", crit, {"item_id": "i1", "text": "hi"})
    assert verdict.verdict is None and rec.error and rec.tier == "haiku"
    # an unusable answer is still a billed call
    assert rec.cost_usd == pytest.approx(call_cost("claude-haiku-5-5", 1000, 200))
    assert rec.request_id == "req_test"


def test_refused_judge_call_is_logged_with_its_cost():
    be = AnthropicBackend(client=FakeClient(stop="refusal"))
    crit = Criterion("tone", "Tone", "be nice")
    verdict, rec = judge_one(be, "claude-opus-5-5", "opus", crit, {"item_id": "i1", "text": "hi"})
    assert verdict.verdict is None and rec.error.startswith("refusal")
    assert rec.output_tokens == 200 and rec.cost_usd > 0


def test_planner_falls_back_to_rubric_ids():
    be = AnthropicBackend(
        client=FakeClient(
            text=json.dumps(
                {"criteria": [{"id": "made_up", "name": "X", "judge_instructions": "y"}]}
            )
        )
    )
    criteria, rec = plan_criteria(be, RUBRIC, "claude-opus-5-5")
    assert [c.id for c in criteria] == CRITERIA
    assert rec.error and "rubric" in rec.error


# ---------------------------------------------------------------- pipeline + routing (simulated)


@pytest.fixture
def sim_run(tmp_path):
    items, truth_rows = generate_support_items(12, seed=1)
    truth = {(t["item_id"], t["criterion"]): t["label"] for t in truth_rows}
    result = run_pipeline(
        items,
        RUBRIC,
        SimulatedBackend(truth=truth, seed=2),
        PipelineConfig(max_workers=4),
        log=lambda *_: None,
    )
    write_outputs(result, tmp_path)
    return result, truth, tmp_path


def test_pipeline_call_accounting(sim_run):
    result, truth, out = sim_run
    agents = [c.agent for c in result.calls]
    assert agents.count("planner") == 1
    assert agents.count("judge") == 3 * 12 * 4
    n_reconciled = sum(1 for s in result.final_source.values() if s == "reconciled")
    assert agents.count("reconciler") == n_reconciled
    for key, src in result.final_source.items():
        votes = {v.verdict for v in result.panel[key]}
        assert (src == "unanimous") == (len(votes) == 1)
    assert all(c.cost_usd > 0 and c.latency_s > 0 for c in result.calls)
    for name in (
        "criteria.json",
        "calls.jsonl",
        "panel_ratings.jsonl",
        "verdicts.jsonl",
        "item_scores.json",
    ):
        assert (out / name).exists()
    # the panel file is in judgelab rating format and feeds straight into the aggregators
    rs = load_ratings(out / "panel_ratings.jsonl", criterion="refund_calculation")
    assert set(rs.raters) == {"haiku", "sonnet", "opus"} and rs.n_items == 12


def test_routing_replay(sim_run):
    result, truth, out = sim_run
    calls = read_jsonl(out / "calls.jsonl")
    final = {(r["item_id"], r["criterion"]): r for r in read_jsonl(out / "verdicts.jsonl")}
    tiers = tier_table(calls, truth)
    assert [t["tier"] for t in tiers] == ["haiku", "sonnet", "opus"]
    pol = {p["policy"]: p for p in routing_policies(calls, truth, final)}
    assert pol["haiku only"]["cost_usd"] == pytest.approx(tiers[0]["cost_usd"])
    assert pol["opus only"]["accuracy"] == pytest.approx(tiers[2]["accuracy"])
    casc = pol["haiku -> opus if haiku confidence < 0.9"]
    assert tiers[0]["cost_usd"] <= casc["cost_usd"] <= tiers[0]["cost_usd"] + tiers[2]["cost_usd"]
    assert "pipeline: panel + aggregator agent" in pol
    assert recommend(list(pol.values()))["policy"] in pol


def test_wilson_interval():
    lo, hi = wilson(90, 100)
    assert 0.82 < lo < 0.84 and 0.94 < hi < 0.95
