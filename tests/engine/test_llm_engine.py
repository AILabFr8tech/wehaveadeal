"""Tests for CarrierLLM/ShipperLLM. No real Ollama server is available in
this environment (or needed): every test overrides `agent._call` with a
stub returning a canned `LLMResponse`, the same "swap in a stub" pattern
`test_ml.py`/`test_rl_engines.py` already use for their models -- these
tests are about the engine's OWN logic (legality enforcement, retry/forfeit,
hard last-turn floor, prompt content), not about any specific model's
negotiating skill.
"""
import json

import pytest

from wehaveadeal.agents import CarrierLLM, ShipperLLM
from wehaveadeal.llm.ollama_client import LLMResponse
from wehaveadeal.llm.prompt import build_messages
from wehaveadeal.market import MarketState
from wehaveadeal.models import Action, TableState


def _resp(action=None, amount=None, reasoning="ok", error=None, raw=None) -> LLMResponse:
    if raw is None:
        raw = json.dumps({"reasoning": reasoning, "action": action, "amount": amount})
    parsed = None if error else {"reasoning": reasoning, "action": action, "amount": amount}
    return LLMResponse(raw_text=raw, parsed=parsed, latency_s=0.01, error=error)


def _sample_carrier_state(seed=7, turn=4, my_last_offer=None, opp_last_offer=None, history=None):
    market = MarketState(seed=seed, n_negotiations=1)
    route = market.assign_route(market.negotiation_rng(0))
    return TableState(
        role="carrier", table_id="t", negotiation_seed=seed, turn=turn, turns_max=12,
        my_turns_used=turn // 2 - 1 if turn > 1 else 0, opp_turns_used=turn // 2,
        my_last_offer=my_last_offer, opp_last_offer=opp_last_offer, history=history or [],
        sigma=route.refprice_sigma, refprice_mean=route.refprice_mean,
    )


def _sample_shipper_state(seed=11, turn=1, my_last_offer=None, opp_last_offer=None, history=None):
    market = MarketState(seed=seed, n_negotiations=1)
    route = market.assign_route(market.negotiation_rng(0))
    return TableState(
        role="shipper", table_id="t", negotiation_seed=seed, turn=turn, turns_max=12,
        my_turns_used=turn // 2, opp_turns_used=turn // 2,
        my_last_offer=my_last_offer, opp_last_offer=opp_last_offer, history=history or [],
        sigma=route.refprice_sigma,
        sale_price=route.refprice_mean + 0.6 * route.refprice_sigma, goods_cost=0.0,
        refprice_est=route.refprice_mean,
    )


# ---------------------------------------------------------------- carrier --

def test_carrier_llm_legal_offer_is_used_as_is():
    state = _sample_carrier_state(turn=4)
    rp_t = state.refprice_mean - 0.5 * state.sigma
    target_t = state.refprice_mean + 0.5 * state.sigma
    legal_amount = (rp_t + target_t) / 2

    agent = CarrierLLM(agent_id="carrier-llm-1", model="stub")
    agent._call = lambda model, messages: _resp(action="offer", amount=legal_amount)

    move = agent.next_move(state)
    assert move.action == Action.OFFER
    assert move.amount == pytest.approx(legal_amount)


def test_carrier_llm_out_of_band_offer_forfeits_after_retries():
    state = _sample_carrier_state(turn=4)
    target_t = state.refprice_mean + 0.5 * state.sigma
    calls = []

    agent = CarrierLLM(agent_id="carrier-llm-1", model="stub", max_retries=2)

    def fake_call(model, messages):
        calls.append(messages)
        return _resp(action="offer", amount=target_t + 1000)  # always illegal, way above ceiling

    agent._call = fake_call
    move = agent.next_move(state)
    assert move.action == Action.REJECT
    assert len(calls) == 3  # initial + 2 retries


def test_carrier_llm_will_not_blindly_accept_an_illegal_offer():
    """Regression-style test mirroring CarrierML's real anchoring bug: the
    engine must not just trust the model's "accept" -- it has to check the
    offer actually on the table is legal for it first."""
    state = _sample_carrier_state(turn=4, opp_last_offer=1.0)  # nonsense, certainly below rp_t
    agent = CarrierLLM(agent_id="carrier-llm-1", model="stub", max_retries=1)
    agent._call = lambda model, messages: _resp(action="accept")

    move = agent.next_move(state)
    assert move.action == Action.REJECT  # forfeited, not a bogus accept


def test_carrier_llm_accepts_a_legal_offer():
    probe = _sample_carrier_state(turn=4)
    rp_t = probe.refprice_mean - 0.5 * probe.sigma
    state = _sample_carrier_state(turn=4, opp_last_offer=rp_t + 1.0)
    agent = CarrierLLM(agent_id="carrier-llm-1", model="stub")
    agent._call = lambda model, messages: _resp(action="accept")

    move = agent.next_move(state)
    assert move.action == Action.ACCEPT


def test_carrier_llm_hard_last_turn_floor_never_calls_the_model():
    state = _sample_carrier_state(turn=12)
    rp_t = state.refprice_mean - 0.5 * state.sigma
    state = _sample_carrier_state(turn=12, my_last_offer=rp_t + 5, opp_last_offer=rp_t + 1.0)

    agent = CarrierLLM(agent_id="carrier-llm-1", model="stub")
    agent._call = lambda model, messages: (_ for _ in ()).throw(AssertionError("should not call the model"))

    move = agent.next_move(state)
    assert move.action == Action.ACCEPT


def test_carrier_llm_invalid_json_retries_then_forfeits():
    state = _sample_carrier_state(turn=2)
    agent = CarrierLLM(agent_id="carrier-llm-1", model="stub", max_retries=1)
    agent._call = lambda model, messages: _resp(error="invalid_json", raw="not json")

    move = agent.next_move(state)
    assert move.action == Action.REJECT


def test_carrier_llm_logs_every_attempt():
    state = _sample_carrier_state(turn=2)
    log = []
    agent = CarrierLLM(agent_id="carrier-llm-1", model="stub", max_retries=1, log_sink=log)
    agent._call = lambda model, messages: _resp(error="invalid_json", raw="not json")

    agent.next_move(state)
    assert len(log) == 3  # 2 real attempts + the final forfeit record
    assert log[0]["model"] == "stub"
    assert log[-1]["error"] == "exhausted_retries"


# ---------------------------------------------------------------- shipper --

def test_shipper_llm_legal_offer_is_used_as_is():
    state = _sample_shipper_state(turn=1)
    rp = state.sale_price - state.goods_cost
    agent = ShipperLLM(agent_id="shipper-llm-1", model="stub")
    agent._call = lambda model, messages: _resp(action="offer", amount=rp - 10)

    move = agent.next_move(state)
    assert move.action == Action.OFFER
    assert move.amount <= rp + 1e-6


def test_shipper_llm_walks_away_when_route_infeasible_without_calling_model():
    market = MarketState(seed=11, n_negotiations=1)
    route = market.assign_route(market.negotiation_rng(0))
    state = TableState(
        role="shipper", table_id="t", negotiation_seed=11, turn=1, turns_max=12,
        my_turns_used=0, opp_turns_used=0,
        sigma=route.refprice_sigma,
        sale_price=route.refprice_mean - 10 * route.refprice_sigma,  # impossibly cheap -> infeasible
        goods_cost=0.0, refprice_est=route.refprice_mean,
    )
    agent = ShipperLLM(agent_id="shipper-llm-1", model="stub")
    agent._call = lambda model, messages: (_ for _ in ()).throw(AssertionError("should not call the model"))

    move = agent.next_move(state)
    assert move.action == Action.REJECT


def test_shipper_llm_out_of_band_offer_forfeits_after_retries():
    state = _sample_shipper_state(turn=1)
    rp = state.sale_price - state.goods_cost
    agent = ShipperLLM(agent_id="shipper-llm-1", model="stub", max_retries=1)
    agent._call = lambda model, messages: _resp(action="offer", amount=rp + 500)  # above the hard ceiling

    move = agent.next_move(state)
    assert move.action == Action.REJECT


# ----------------------------------------------------------------- prompt --

def test_prompt_discloses_hard_band_and_full_history():
    history = [
        {"turn": 1, "actor": "shipper", "action": "offer", "amount": 1500.0},
        {"turn": 2, "actor": "carrier", "action": "offer", "amount": 1800.0},
    ]
    state = _sample_carrier_state(turn=3, my_last_offer=1800.0, opp_last_offer=1500.0, history=history)
    worst, best = 1200.0, 1900.0
    messages = build_messages(state, role="carrier", worst=worst, best=best)
    system, user = messages[0]["content"], messages[1]["content"]

    assert f"${worst:.2f}" in system
    assert f"${best:.2f}" in system
    assert "1500.00" in user
    assert "1800.00" in user
    assert "not the only carrier" in system  # competition disclosed, without naming any rival
