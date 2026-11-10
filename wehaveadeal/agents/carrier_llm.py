"""CarrierLLM: an LLM (served locally/remotely via Ollama) playing the
carrier role, used only for the LLM-vs-engines validation study -- NOT
registered in `wehaveadeal/simulate.py`'s `CARRIER_KINDS`, the dashboard, or
`wehaveadeal/api.py`. A real model call takes seconds, which is fine for a
batch validation script left running for hours but incompatible with the
live SSE dashboard, so this deliberately stays out of that registry
instead of half-supporting it there.

Mirrors every other engine's defense-in-depth: the hard last-turn floor is
checked unconditionally before ever calling the model (so a timed-out or
malformed response on the final turn still can't cost a legally-available
deal), and nothing the model proposes reaches the round loop without
passing `wehaveadeal.simulate._validate_move` first -- an illegal or malformed
response is retried a few times, then forfeited as a REJECT, exactly like
running out of turns. See `wehaveadeal/llm/prompt.py`'s module docstring for the
two explicit disclosure choices (full hard-band numbers, per-table history
only) baked into what this engine tells the model.
"""
from __future__ import annotations

from typing import Optional, Union

from wehaveadeal.agents.base import ACCEPT, DecisionEngine, REJECT, make_offer
from wehaveadeal.llm.ollama_client import DEFAULT_HOST, LLMResponse, call_ollama
from wehaveadeal.llm.prompt import RESPONSE_SCHEMA, build_messages
from wehaveadeal.ml.features import role_band
from wehaveadeal.models import Move, TableState

# NOT imported at module scope: wehaveadeal.simulate imports wehaveadeal.agents.base,
# which -- because wehaveadeal/agents/__init__.py eagerly imports every agent
# module including this one -- would make this a circular import at
# package-load time. Deferred to inside _parse() instead, by which point
# wehaveadeal.simulate has always finished initializing.


class CarrierLLM(DecisionEngine):
    engine_name = "CarrierLLM"
    version = "v1"

    def __init__(
        self,
        agent_id: str,
        model: str,
        frozen: bool = True,
        host: str = DEFAULT_HOST,
        timeout_s: float = 90.0,
        max_retries: int = 2,
        think: Optional[Union[bool, str]] = None,
        log_sink: Optional[list] = None,
    ) -> None:
        super().__init__(agent_id, frozen=frozen)
        self.model = model
        self.host = host
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.think = think
        # If given, every model call's prompt/response/latency/error is
        # appended here -- the qualitative log a paper appendix wants, and
        # how the paper's LLM study measures structured-output
        # compliance and latency per model. None by default so unit tests
        # that don't care about logging stay simple.
        self.log_sink = log_sink

    def _call(self, model: str, messages: list[dict]) -> LLMResponse:
        return call_ollama(model, messages, RESPONSE_SCHEMA, host=self.host, timeout_s=self.timeout_s, think=self.think)

    def next_move(self, state: TableState) -> Move:
        turn_index = state.my_turns_used + 1
        turns_per_side = state.turns_per_side
        worst, best = role_band(state)  # carrier: worst=floor (rp_t), best=ceiling (target_t)
        shipper_offer = state.opp_last_offer

        if turn_index > turns_per_side:
            return REJECT
        if shipper_offer is not None and turn_index >= turns_per_side and shipper_offer >= worst:
            return ACCEPT

        messages = build_messages(state, role="carrier", worst=worst, best=best)
        return self._decide(messages, state, worst)

    def _decide(self, messages: list[dict], state: TableState, worst: float) -> Move:
        for attempt in range(self.max_retries + 1):
            response = self._call(self.model, messages)
            self._log(state, attempt, response)
            move = self._parse(response, state, worst)
            if move is not None:
                return move
            messages = messages + [
                {"role": "assistant", "content": response.raw_text or ""},
                {"role": "user", "content": "That response was invalid or illegal. Reply again with ONLY the required JSON object, respecting the hard rules."},
            ]
        self._log(state, "forfeit", None, error="exhausted_retries")
        return REJECT

    def _parse(self, response: LLMResponse, state: TableState, worst: float) -> Optional[Move]:
        from wehaveadeal.simulate import IllegalMoveError, _validate_move  # see the module docstring's note on this

        if response.parsed is None:
            return None
        action = response.parsed.get("action")
        amount = response.parsed.get("amount")

        if action == "reject":
            return REJECT
        if action == "accept":
            shipper_offer = state.opp_last_offer
            if shipper_offer is None or shipper_offer < worst:
                return None  # nothing legal to accept -- malformed, retry
            return ACCEPT
        if action == "offer":
            if not isinstance(amount, (int, float)):
                return None
            move = make_offer(float(amount))
            try:
                _validate_move(move, state)
            except IllegalMoveError:
                return None
            return move
        return None

    def _log(self, state: TableState, attempt, response: Optional[LLMResponse], error: Optional[str] = None) -> None:
        if self.log_sink is None:
            return
        self.log_sink.append({
            "agent_id": self.agent_id, "model": self.model, "table_id": state.table_id,
            "turn": state.turn, "attempt": attempt,
            "latency_s": response.latency_s if response else None,
            "raw_text": response.raw_text if response else None,
            "error": (response.error if response else None) or error,
        })
