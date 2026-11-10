"""An LLM-backed carrier that runs on YOUR API key / YOUR model server.

Nothing in the toolkit pays for or proxies model calls: this agent calls any
OpenAI-compatible chat endpoint you point it at (a hosted provider, vLLM,
LM Studio, or Ollama at http://localhost:11434/v1), configured by environment
variables:

    LLM_BASE_URL   e.g. http://localhost:11434/v1      (required)
    LLM_MODEL      e.g. llama3.1:8b                    (required)
    LLM_API_KEY    your key, if the endpoint needs one (optional)

LLM calls are slow, so raise the per-move time limit when benchmarking:

    wehaveadeal benchmark examples/llm_carrier.py --preset quick --move-timeout 60

If the endpoint is not configured or a reply cannot be parsed, the agent falls
back to a safe rule-based move instead of forfeiting the table. The toolkit
also ships CarrierLLM/ShipperLLM (wehaveadeal.agents), the Ollama-native agents
used in the paper.
"""
import json
import os
import re
import urllib.request

from wehaveadeal import ACCEPT, REJECT, DecisionEngine, Move, TableState, legal_offer_range, offer

SYSTEM = (
    "You are a freight carrier negotiating a transport rate with a shipper. Higher rates are better for you, "
    "but no deal earns nothing. Reply with JSON only: "
    '{"action": "offer", "amount": <number>} or {"action": "accept"} or {"action": "reject"}.'
)


class LLMCarrier(DecisionEngine):
    role = "carrier"
    engine_name = "LLMCarrier"

    def __init__(self, agent_id: str, frozen: bool = True, timeout_s: float = 45.0) -> None:
        super().__init__(agent_id, frozen=frozen)
        self.base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
        self.model = os.environ.get("LLM_MODEL", "")
        self.api_key = os.environ.get("LLM_API_KEY", "")
        self.timeout_s = timeout_s

    def ask(self, prompt: str) -> str:
        """One chat-completion call. Separate method so tests can replace it."""
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "temperature": 0,
        }).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]

    def fallback(self, state: TableState, lo: float, hi: float) -> Move:
        if state.opp_last_offer is not None and state.opp_last_offer >= state.refprice_mean:
            return ACCEPT
        return REJECT if lo > hi else offer(hi - 0.15 * (hi - lo))

    def next_move(self, state: TableState) -> Move:
        lo, hi = legal_offer_range(state)
        if not (self.base_url and self.model):
            return self.fallback(state, lo, hi)
        prompt = (
            f"Reference market rate for this route: {state.refprice_mean:.2f} (volatility {state.sigma:.2f}).\n"
            f"Your legal offer range right now: [{lo:.2f}, {hi:.2f}].\n"
            f"Shipper's current offer: {state.opp_last_offer}. Your last offer: {state.my_last_offer}.\n"
            f"You have used {state.my_turns_used} of {state.turns_per_side} turns.\n"
            f"Deals already closed on your other tables: {sum(t.is_closed for t in state.my_portfolio)}."
        )
        try:
            text = self.ask(prompt)
            data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            action = str(data.get("action", "")).lower()
            if action == "accept" and state.opp_last_offer is not None:
                return ACCEPT
            if action == "reject":
                return REJECT
            if action == "offer" and lo <= hi:
                return offer(min(max(float(data["amount"]), lo), hi))
        except Exception:  # noqa: BLE001 -- network/parse problems fall back, never forfeit
            pass
        return self.fallback(state, lo, hi)
