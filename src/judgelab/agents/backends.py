"""LLM backends: the real Anthropic API and an offline simulator for tests/CI.

Both expose `complete_json(...)`, which returns parsed JSON plus the
accounting fields (latency, tokens, cost) logged for every agent call.
"""

from __future__ import annotations

import json
import re
import time
import zlib
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np

from judgelab.agents.pricing import call_cost, tier_for

# Models for which server-side refusal fallbacks are requested (Haiku has none).
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class BackendError(RuntimeError):
    """A call failed (API error, refusal, truncation or unparseable output).

    `response` is set when the API did answer, and so billed the call, but the
    answer was unusable; the failed call is then logged with its real tokens
    and cost instead of $0.
    """

    def __init__(self, message: str, response: LLMResponse | None = None):
        super().__init__(message)
        self.response = response


@dataclass
class LLMResponse:
    data: dict
    model: str
    served_model: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    stop_reason: str = "end_turn"
    request_id: str | None = None


@dataclass
class CallRecord:
    """One logged agent call (one line of calls.jsonl)."""

    agent: str  # planner | judge | reconciler
    tier: str
    model: str
    served_model: str
    item_id: str | None
    criterion: str | None
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    stop_reason: str | None = None
    error: str | None = None
    output: dict = field(default_factory=dict)
    request_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def failed(
        cls,
        agent: str,
        model: str,
        item_id,
        criterion,
        latency_s: float,
        error: str,
        response: LLMResponse | None = None,
    ) -> CallRecord:
        """A failed call. With `response` (the API answered but the answer was
        unusable), the record keeps that call's tokens, cost and latency."""
        if response is not None:
            rec = cls.from_response(agent, response, item_id, criterion)
            rec.error = error
            return rec
        return cls(
            agent,
            tier_for(model),
            model,
            model,
            item_id,
            criterion,
            latency_s,
            0,
            0,
            0.0,
            error=error,
        )

    @classmethod
    def from_response(cls, agent: str, resp: LLMResponse, item_id, criterion) -> CallRecord:
        return cls(
            agent=agent,
            tier=tier_for(resp.model),
            model=resp.model,
            served_model=resp.served_model,
            item_id=item_id,
            criterion=criterion,
            latency_s=resp.latency_s,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            cost_usd=resp.cost_usd,
            cache_read_tokens=resp.cache_read_tokens,
            cache_write_tokens=resp.cache_write_tokens,
            stop_reason=resp.stop_reason,
            output=resp.data,
            request_id=resp.request_id,
        )


class Backend(Protocol):
    name: str

    def complete_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: dict,
        effort: str,
        max_tokens: int,
        meta: dict | None = None,
    ) -> LLMResponse: ...


class AnthropicBackend:
    """Calls the Claude Messages API with structured JSON output.

    - `output_config.format` constrains the reply to the given JSON schema;
      `output_config.effort` sets thinking depth (thinking is adaptive by
      default on the current Haiku / Sonnet / Opus models).
    - For Opus and Sonnet, server-side refusal fallbacks are requested
      (`fallbacks="default"`); the model that actually served the reply is
      logged as `served_model`, so a fallback can never silently contaminate
      a per-tier comparison.
    """

    name = "anthropic"

    def __init__(
        self, client=None, use_fallbacks: bool = True, max_retries: int = 4, timeout: float = 180.0
    ):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=max_retries, timeout=timeout)
        self.client = client
        self.use_fallbacks = use_fallbacks

    def _create(self, model: str, kwargs: dict):
        if self.use_fallbacks and model in FALLBACK_MODELS:
            return self.client.beta.messages.create(
                betas=[FALLBACK_BETA], fallbacks="default", **kwargs
            )
        return self.client.messages.create(**kwargs)

    def complete_json(
        self, *, model, system, user, schema, effort, max_tokens, meta=None
    ) -> LLMResponse:
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {
                "format": {"type": "json_schema", "schema": schema},
                "effort": effort,
            },
        }
        import anthropic

        t0 = time.perf_counter()
        try:
            resp = self._create(model, kwargs)
        except anthropic.RateLimitError as exc:  # 429 after the SDK's own retries
            raise BackendError(f"rate_limited: {exc.message}") from exc
        except anthropic.BadRequestError as exc:  # 400: request shape problem, not retryable
            raise BackendError(f"bad_request: {exc.message}") from exc
        except anthropic.APIStatusError as exc:  # other 4xx / 5xx after retries
            raise BackendError(f"api_error_{exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:  # network / timeout after retries
            raise BackendError(f"connection_error: {exc}") from exc
        latency = time.perf_counter() - t0

        # Accounting first: from here on the call has been billed, even if the
        # answer turns out to be unusable.
        usage = resp.usage
        served = getattr(resp, "model", None) or model
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        c_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        c_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        try:
            cost = call_cost(served, in_tok, out_tok, c_read, c_write)
        except KeyError:
            cost = call_cost(model, in_tok, out_tok, c_read, c_write)
        billed = LLMResponse(
            data={},
            model=model,
            served_model=served,
            latency_s=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read_tokens=c_read,
            cache_write_tokens=c_write,
            cost_usd=cost,
            stop_reason=resp.stop_reason,
            request_id=getattr(resp, "_request_id", None),
        )

        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise BackendError(f"refusal (category={category})", billed)
        if resp.stop_reason == "max_tokens":
            raise BackendError("max_tokens reached before the JSON answer was complete", billed)
        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
        if text is None:
            raise BackendError(
                f"no text block in response (stop_reason={resp.stop_reason})", billed
            )
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BackendError(f"unparseable JSON output: {exc}", billed) from exc
        if not isinstance(data, dict):
            raise BackendError(f"expected a JSON object, got {type(data).__name__}", billed)
        billed.data = data
        return billed


# --------------------------------------------------------------------------
# Offline simulator
# --------------------------------------------------------------------------


@dataclass
class TierProfile:
    accuracy: float  # chance a judge verdict matches the truth
    latency_median_s: float
    output_tokens: int  # mean billed output tokens (incl. thinking) per judge call


SIM_PROFILES: dict[str, TierProfile] = {
    "claude-haiku-5-5": TierProfile(0.80, 1.4, 260),
    "claude-sonnet-5-5": TierProfile(0.88, 3.2, 420),
    "claude-opus-5-5": TierProfile(0.93, 5.5, 520),
}


def _stable_seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


class SimulatedBackend:
    """Deterministic stand-in for the API, used by tests, CI and `--backend simulated`.

    It is a test double, not a model: judge verdicts are the known truth
    flipped at a per-tier error rate, latency and token counts are drawn
    from per-tier profiles, and cost uses the real price table. Its output
    exercises the pipeline and the routing analysis; it says nothing about
    real model quality.
    """

    name = "simulated"

    def __init__(
        self,
        truth: dict[tuple[str, str], str] | None = None,
        seed: int = 0,
        profiles: dict[str, TierProfile] | None = None,
    ):
        self.truth = truth or {}
        self.seed = seed
        self.profiles = profiles or SIM_PROFILES

    def complete_json(
        self, *, model, system, user, schema, effort, max_tokens, meta=None
    ) -> LLMResponse:
        meta = meta or {}
        agent = meta.get("agent", "judge")
        prof = self.profiles.get(model, TierProfile(0.85, 3.0, 400))
        rng = np.random.default_rng(
            _stable_seed(self.seed, model, agent, meta.get("item_id"), meta.get("criterion"))
        )

        if agent == "planner":
            data = {"criteria": _parse_rubric_criteria(user)}
            out_tokens = 300
        else:
            key = (meta.get("item_id"), meta.get("criterion"))
            truth = self.truth.get(key, "pass")
            correct = rng.random() < prof.accuracy
            verdict = truth if correct else ("fail" if truth == "pass" else "pass")
            conf = float(np.clip(rng.beta(9, 2) if correct else rng.beta(5, 3), 0.5, 0.99))
            data = {
                "rationale": f"simulated {agent} verdict",
                "verdict": verdict,
                "confidence": round(conf, 2),
            }
            out_tokens = int(max(40, rng.normal(prof.output_tokens, prof.output_tokens * 0.2)))

        in_tokens = (len(system) + len(user)) // 4
        latency = float(prof.latency_median_s * rng.lognormal(0.0, 0.35))
        return LLMResponse(
            data=data,
            model=model,
            served_model=model,
            latency_s=latency,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cost_usd=call_cost(model, in_tokens, out_tokens),
        )


_CRITERION_LINE = re.compile(r"^\s*\d+[.)]\s+(.+?)\s+\[([a-z0-9_]+)\]\s+-\s+(.+)$")


def _parse_rubric_criteria(text: str) -> list[dict]:
    """Deterministic rubric parser used by the simulator ('N. Name [id] - description')."""
    out = []
    for line in text.splitlines():
        m = _CRITERION_LINE.match(line)
        if m:
            desc = m.group(3).strip()
            out.append({"id": m.group(2), "name": m.group(1).strip(), "judge_instructions": desc})
    return out
