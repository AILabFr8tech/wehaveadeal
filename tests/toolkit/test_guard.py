"""Buggy, slow, cheating and malicious-ish agents: the referee must contain all of them.

The rule being enforced is the protocol's own: an illegal action forfeits the
table (it becomes an impasse), the agent keeps playing its other tables, and
the benchmark always finishes.
"""
from __future__ import annotations

import time

import pytest

from wehaveadeal import ACCEPT, REJECT, DecisionEngine, Move, offer
from wehaveadeal.models import Action
from wehaveadeal.toolkit.benchmark import run_benchmark
from wehaveadeal.toolkit.guard import GuardedAgent, check_move

from .conftest import FAST


def carrier_state(**overrides):
    from wehaveadeal.models import TableState

    base = dict(
        role="carrier", table_id="shipper-r-1::carrier-x-1", negotiation_seed=1, turn=2, turns_max=12,
        my_turns_used=0, opp_turns_used=1, opp_last_offer=900.0, sigma=100.0, refprice_mean=1000.0,
    )
    base.update(overrides)
    return TableState(**base)


class Raises(DecisionEngine):
    role = "carrier"

    def next_move(self, state):
        raise ZeroDivisionError("oops")


class ReturnsGarbage(DecisionEngine):
    role = "carrier"

    def next_move(self, state):
        return {"action": "offer", "amount": 1000}


class OutOfBand(DecisionEngine):
    role = "carrier"

    def next_move(self, state):
        return offer(state.refprice_mean + 10 * state.sigma)


class RaisesPrice(DecisionEngine):
    """Concessions must be monotonic: a carrier may never raise its own ask."""
    role = "carrier"

    def next_move(self, state):
        lo = state.refprice_mean - 0.5 * state.sigma
        return offer(lo if state.my_last_offer is None else lo + 0.25 * state.sigma)


class AcceptsNothing(DecisionEngine):
    role = "carrier"

    def next_move(self, state):
        return ACCEPT


class AcceptsNothingShipper(DecisionEngine):
    """Shippers open the table (turn 1), so an immediate ACCEPT accepts nothing."""
    role = "shipper"

    def next_move(self, state):
        return ACCEPT


class Slow(DecisionEngine):
    role = "carrier"

    def next_move(self, state):
        time.sleep(0.05)
        return REJECT


class Tamperer(DecisionEngine):
    """Tries to rewrite the state it is handed (history, the opponent's offer)."""
    role = "carrier"

    def next_move(self, state):
        state.history.clear()
        state.opp_last_offer = 10**9
        return REJECT


@pytest.mark.parametrize("agent_cls,kind", [
    (Raises, "exception"),
    (ReturnsGarbage, "invalid_return"),
    (OutOfBand, "illegal_move"),
])
def test_single_move_violations_forfeit_the_table(agent_cls, kind):
    g = GuardedAgent(agent_cls(agent_id="carrier-x-1"))
    move = g.next_move(carrier_state())
    assert move.action == Action.REJECT
    assert [v.kind for v in g.violations] == [kind]
    assert g.violations[0].table_id == "shipper-r-1::carrier-x-1"


def test_carrier_always_accepting_is_legal_but_shipper_is_not():
    """Carriers move second, so a counter-offer is always on the table for them."""
    report = run_benchmark(AcceptsNothing, **FAST)
    assert report.violations.total_violations == 0
    report = run_benchmark(AcceptsNothingShipper, **FAST)
    assert set(report.violations.counts) == {"illegal_accept"}


def test_accept_with_nothing_on_the_table():
    g = GuardedAgent(AcceptsNothing(agent_id="carrier-x-1"))
    assert g.next_move(carrier_state(opp_last_offer=None)) == REJECT
    assert g.violations[0].kind == "illegal_accept"


def test_non_monotonic_concession():
    g = GuardedAgent(RaisesPrice(agent_id="carrier-x-1"))
    g.next_move(carrier_state(my_last_offer=960.0))
    assert g.violations[0].kind == "illegal_move"
    assert "non-monotonic" in g.violations[0].detail


def test_timeout_forfeits_even_a_legal_answer():
    g = GuardedAgent(Slow(agent_id="carrier-x-1"), move_timeout_s=0.01)
    g.next_move(carrier_state())
    assert g.violations[0].kind == "timeout"
    assert g.max_think_s > 0.01  # sleep(0.05) can measure a hair under 0.05 on Windows timers


def test_no_timeout_when_disabled():
    g = GuardedAgent(Slow(agent_id="carrier-x-1"), move_timeout_s=None)
    g.next_move(carrier_state())
    assert g.violations == []


def test_agent_cannot_tamper_with_the_arena_state():
    state = carrier_state(opp_last_offer=900.0)
    GuardedAgent(Tamperer(agent_id="carrier-x-1")).next_move(state)
    assert state.opp_last_offer == 900.0


def test_legal_moves_pass_through_untouched():
    class Legal(DecisionEngine):
        role = "carrier"

        def next_move(self, state):
            return offer(state.refprice_mean)

    g = GuardedAgent(Legal(agent_id="carrier-x-1"))
    move = g.next_move(carrier_state())
    assert move == Move(action=Action.OFFER, amount=1000.0)
    assert g.violations == []


def test_check_move_accepts_band_edges():
    s = carrier_state()
    assert check_move(offer(950.0), s) is None
    assert check_move(offer(1050.0), s) is None
    assert check_move(offer(1050.1), s)[0] == "illegal_move"


@pytest.mark.parametrize("agent_cls", [Raises, ReturnsGarbage, OutOfBand, AcceptsNothingShipper, Tamperer])
def test_benchmark_survives_a_broken_agent(agent_cls):
    """A broken agent still yields a complete report -- with its violations on record."""
    report = run_benchmark(agent_cls, **FAST)
    assert len(report.regimes) == 3
    assert all(r.n_market_days > 0 for r in report.regimes)
    if agent_cls is not Tamperer:
        assert report.violations.total_violations > 0
        assert report.violations.examples
    # A forfeited table is an impasse, so the broken agent never closes a deal it broke.
    if agent_cls in (Raises, ReturnsGarbage, OutOfBand, AcceptsNothingShipper):
        assert report.violations.forfeit_rate == 1.0
        assert all(r.entrant.match_rate == 0.0 for r in report.regimes)


def test_benchmark_with_slow_agent_records_timeouts():
    report = run_benchmark(Slow, move_timeout_s=0.001, regimes=("balanced",), baselines=FAST["baselines"],
                           n_subrounds=1, n_negotiations=1, preset="smoke")
    assert report.violations.counts.get("timeout", 0) == report.violations.total_moves
