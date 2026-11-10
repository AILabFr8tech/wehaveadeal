"""ShipperLLM: mirrors `CarrierLLM` exactly -- see that module's docstring
for the full rationale (not registered anywhere outside the LLM validation
study, defense-in-depth around the model's output, etc.). Mirrored
specifics: worst=RP (ceiling, the shipper may never pay more), best=target
(floor, the lowest price achievable); accepting requires the carrier's
offer to be AT OR BELOW worst, not above it.
"""
from __future__ import annotations

from typing import Optional, Union

from wehaveadeal.agents.base import ACCEPT, DecisionEngine, REJECT, make_offer
from wehaveadeal.llm.ollama_client import DEFAULT_HOST, LLMResponse, call_ollama
from wehaveadeal.llm.prompt import RESPONSE_SCHEMA, build_messages
from wehaveadeal.ml.features import role_band
from wehaveadeal.models import Move, TableState
from wehaveadeal.value_functions import shipper_route_is_feasible

# wehaveadeal.simulate's IllegalMoveError/_validate_move are imported lazily
# inside _parse() instead of here -- see the equivalent comment in
# carrier_llm.py for why a module-scope import here would be circular
# (wehaveadeal/agents/__init__.py eagerly imports every agent module, including
# this one, and wehaveadeal.simulate imports wehaveadeal.agents.base).


class ShipperLLM(DecisionEngine):
    engine_name = "ShipperLLM"
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
        self.log_sink = log_sink

    def _call(self, model: str, messages: list[dict]) -> LLMResponse:
        return call_ollama(model, messages, RESPONSE_SCHEMA, host=self.host, timeout_s=self.timeout_s, think=self.think)

    def next_move(self, state: TableState) -> Move:
        # Same deliberate walk-away every hand-designed shipper engine has
        # (Strategy spec §13.0): if even the best case is unaffordable, don't
        # bother spending a model call on it at all.
        if not shipper_route_is_feasible(state):
            return REJECT

        turn_index = state.my_turns_used + 1
        turns_per_side = state.turns_per_side
        worst, best = role_band(state)  # shipper: worst=RP (ceiling), best=target (floor)
        carrier_offer = state.opp_last_offer

        if turn_index > turns_per_side:
            return REJECT
        if carrier_offer is not None and turn_index >= turns_per_side and carrier_offer <= worst:
            return ACCEPT

        messages = build_messages(state, role="shipper", worst=worst, best=best)
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
        from wehaveadeal.simulate import IllegalMoveError, _validate_move  # see carrier_llm.py's module docstring

        if response.parsed is None:
            return None
        action = response.parsed.get("action")
        amount = response.parsed.get("amount")

        if action == "reject":
            return REJECT
        if action == "accept":
            carrier_offer = state.opp_last_offer
            if carrier_offer is None or carrier_offer > worst:
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
