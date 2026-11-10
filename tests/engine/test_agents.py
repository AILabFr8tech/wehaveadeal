import itertools

import pytest

from wehaveadeal.agents import (
    CarrierIS,
    CarrierOPT,
    CarrierOPT2,
    CarrierR,
    CarrierRV,
    ShipperIS,
    ShipperOPT,
    ShipperOPT2,
    ShipperR,
    ShipperRV,
)
from wehaveadeal.models import Action
from wehaveadeal.simulate import IllegalMoveError, run_round

ALL_CARRIER_CLASSES = [CarrierR, CarrierRV, CarrierIS, CarrierOPT, CarrierOPT2]
ALL_SHIPPER_CLASSES = [ShipperR, ShipperRV, ShipperIS, ShipperOPT, ShipperOPT2]


@pytest.mark.parametrize("carrier_cls", ALL_CARRIER_CLASSES)
def test_carrier_never_offers_outside_band_or_non_monotonic(carrier_cls):
    """Fuzz across many seeds/negotiations: DoD requires agents never offer
    outside [RP, band] or produce a non-monotonic concession. run_round
    raises IllegalMoveError internally if this is ever violated."""
    shippers = {f"shipper-{i+1}": ShipperR(agent_id=f"shipper-{i+1}") for i in range(4)}
    carriers = {f"carrier-{i+1}": carrier_cls(agent_id=f"carrier-{i+1}") for i in range(3)}
    for seed in range(10):
        run_round(round_index=0, seed=seed, shipper_engines=shippers, carrier_engines=carriers, n_negotiations=10)


def test_shipper_r_never_exceeds_reservation_price():
    shippers = {f"shipper-{i+1}": ShipperR(agent_id=f"shipper-{i+1}") for i in range(4)}
    carriers = {f"carrier-{i+1}": CarrierR(agent_id=f"carrier-{i+1}") for i in range(3)}
    result = run_round(round_index=0, seed=123, shipper_engines=shippers, carrier_engines=carriers, n_negotiations=20)
    for neg in result.negotiations:
        rp = neg.sale_price - neg.goods_cost
        for table in neg.tables:
            for rec in table.history:
                if rec.actor == "shipper" and rec.action == Action.OFFER:
                    assert rec.amount <= rp + 1e-6


def test_shipper_r_offers_are_monotonically_non_decreasing():
    shippers = {f"shipper-{i+1}": ShipperR(agent_id=f"shipper-{i+1}") for i in range(4)}
    carriers = {f"carrier-{i+1}": CarrierR(agent_id=f"carrier-{i+1}") for i in range(3)}
    result = run_round(round_index=0, seed=9, shipper_engines=shippers, carrier_engines=carriers, n_negotiations=20)
    for neg in result.negotiations:
        for table in neg.tables:
            offers = [r.amount for r in table.history if r.actor == "shipper" and r.action == Action.OFFER]
            assert offers == sorted(offers)


def test_carrier_counters_are_monotonically_non_increasing():
    shippers = {f"shipper-{i+1}": ShipperR(agent_id=f"shipper-{i+1}") for i in range(4)}
    carriers = {f"carrier-{i+1}": CarrierRV(agent_id=f"carrier-{i+1}") for i in range(3)}
    result = run_round(round_index=0, seed=11, shipper_engines=shippers, carrier_engines=carriers, n_negotiations=20)
    for neg in result.negotiations:
        for table in neg.tables:
            offers = [r.amount for r in table.history if r.actor == "carrier" and r.action == Action.OFFER]
            assert offers == sorted(offers, reverse=True)


def test_carrier_rv_and_carrier_r_are_both_pluggable_decision_engines():
    from wehaveadeal.agents.base import DecisionEngine

    for cls in ALL_CARRIER_CLASSES + ALL_SHIPPER_CLASSES:
        assert issubclass(cls, DecisionEngine)
        agent = cls(agent_id="x")
        assert agent.frozen is True
        assert hasattr(agent, "next_move")


@pytest.mark.parametrize("carrier_cls", [CarrierIS, CarrierOPT, CarrierOPT2])
def test_new_carrier_engines_offers_are_monotonically_non_increasing(carrier_cls):
    shippers = {f"shipper-{i+1}": ShipperR(agent_id=f"shipper-{i+1}") for i in range(4)}
    carriers = {f"carrier-{i+1}": carrier_cls(agent_id=f"carrier-{i+1}") for i in range(3)}
    result = run_round(round_index=0, seed=17, shipper_engines=shippers, carrier_engines=carriers, n_negotiations=15)
    for neg in result.negotiations:
        for table in neg.tables:
            offers = [r.amount for r in table.history if r.actor == "carrier" and r.action == Action.OFFER]
            assert offers == sorted(offers, reverse=True)


@pytest.mark.parametrize("carrier_cls", [CarrierIS, CarrierOPT, CarrierOPT2])
def test_new_carrier_engines_stay_within_hard_band(carrier_cls):
    shippers = {f"shipper-{i+1}": ShipperR(agent_id=f"shipper-{i+1}") for i in range(4)}
    carriers = {f"carrier-{i+1}": carrier_cls(agent_id=f"carrier-{i+1}") for i in range(3)}
    result = run_round(round_index=0, seed=23, shipper_engines=shippers, carrier_engines=carriers, n_negotiations=15)
    for neg in result.negotiations:
        for table in neg.tables:
            route = neg.routes[table.shipper_id]
            rp_t = route.refprice_mean - 0.5 * route.refprice_sigma
            target_t = route.refprice_mean + 0.5 * route.refprice_sigma
            for rec in table.history:
                if rec.actor == "carrier" and rec.action == Action.OFFER:
                    assert rp_t - 1e-6 <= rec.amount <= target_t + 1e-6


@pytest.mark.parametrize("carrier_cls", [CarrierIS, CarrierOPT, CarrierOPT2])
def test_new_carrier_engines_beat_random_walk_on_margin(carrier_cls):
    """Sanity check that each smarter engine isn't accidentally worse than
    doing nothing clever: against the fixed ShipperR sparring partner, all
    three should close most negotiations with a non-catastrophic margin."""
    shippers = {f"shipper-{i+1}": ShipperR(agent_id=f"shipper-{i+1}") for i in range(4)}
    carriers = {f"carrier-{i+1}": carrier_cls(agent_id=f"carrier-{i+1}") for i in range(3)}
    result = run_round(round_index=0, seed=99, shipper_engines=shippers, carrier_engines=carriers, n_negotiations=50)
    for carrier_id, standing in result.carrier_standings.items():
        assert standing["occupancy"] > 0.5


def test_carrier_is_particle_sampling_is_deterministic_given_seed():
    shippers = {f"shipper-{i+1}": ShipperR(agent_id=f"shipper-{i+1}") for i in range(4)}
    carriers = {f"carrier-{i+1}": CarrierIS(agent_id=f"carrier-{i+1}") for i in range(3)}
    r1 = run_round(round_index=0, seed=55, shipper_engines=shippers, carrier_engines=carriers, n_negotiations=5)
    r2 = run_round(round_index=0, seed=55, shipper_engines=shippers, carrier_engines=carriers, n_negotiations=5)
    assert r1.carrier_standings == r2.carrier_standings


def test_carrier_opt_noise_is_reproducible_given_state_and_stays_legal():
    """CarrierOPT's concession-direction jitter is seeded from
    (negotiation_seed, table_id, turn), not wall-clock entropy, so calling
    next_move twice on the identical state must give the identical offer --
    and that offer must still respect the monotonic ceiling."""
    from wehaveadeal.market import MarketState
    from wehaveadeal.models import TableState

    agent = CarrierOPT(agent_id="carrier-opt-1")
    market = MarketState(seed=7, n_negotiations=1)
    route = market.assign_route(market.negotiation_rng(0))
    state = TableState(
        role="carrier",
        table_id="t",
        negotiation_seed=7,
        turn=4,
        turns_max=12,
        my_turns_used=1,
        opp_turns_used=2,
        my_last_offer=route.refprice_mean + 0.3 * route.refprice_sigma,
        opp_last_offer=route.refprice_mean - 0.2 * route.refprice_sigma,
        sigma=route.refprice_sigma,
        refprice_mean=route.refprice_mean,
    )
    move1 = agent.next_move(state)
    move2 = agent.next_move(state)
    assert move1.amount == move2.amount
    assert move1.amount <= state.my_last_offer + 1e-6


def test_new_carrier_engines_offers_are_not_perfectly_static_across_turns():
    """Without any jitter, a pure gradient optimum can settle onto the same
    counter turn after turn once EU's optimum stabilizes -- exactly the
    "always the same price" complaint the noise was added to fix. Across
    enough tables, at least some should show more than one distinct value."""
    shippers = {f"shipper-{i+1}": ShipperR(agent_id=f"shipper-{i+1}") for i in range(4)}
    carriers = {f"carrier-{i+1}": CarrierOPT(agent_id=f"carrier-{i+1}") for i in range(3)}
    result = run_round(round_index=0, seed=17, shipper_engines=shippers, carrier_engines=carriers, n_negotiations=15)
    distinct_counts = []
    for neg in result.negotiations:
        for table in neg.tables:
            offers = [r.amount for r in table.history if r.actor == "carrier" and r.action == Action.OFFER]
            if len(offers) >= 3:
                distinct_counts.append(len({round(o, 6) for o in offers}))
    assert any(c > 1 for c in distinct_counts)


def test_shipper_opt_noise_is_reproducible_given_state_and_stays_legal():
    from wehaveadeal.market import MarketState
    from wehaveadeal.models import TableState

    agent = ShipperOPT(agent_id="shipper-opt-1")
    market = MarketState(seed=7, n_negotiations=1)
    route = market.assign_route(market.negotiation_rng(0))
    # RP kept close to the route's own scale (not an arbitrarily generous
    # sale_price/goods_cost pair) so the carrier's counter isn't trivially
    # a bargain relative to RP -- that would make ShipperOPT accept outright
    # instead of exercising the offer path this test wants to check.
    state = TableState(
        role="shipper",
        table_id="t",
        negotiation_seed=7,
        turn=3,
        turns_max=12,
        my_turns_used=1,
        opp_turns_used=1,
        my_last_offer=route.refprice_mean - 0.3 * route.refprice_sigma,
        opp_last_offer=route.refprice_mean + 0.2 * route.refprice_sigma,
        sigma=route.refprice_sigma,
        sale_price=route.refprice_mean + 0.3 * route.refprice_sigma,
        goods_cost=0.0,
        refprice_est=route.refprice_mean,
    )
    move1 = agent.next_move(state)
    move2 = agent.next_move(state)
    assert move1.amount == move2.amount
    assert move1.amount >= state.my_last_offer - 1e-6


def test_carrier_opt2_envelope_never_improves_on_its_own_optimum():
    """The anti-detection envelope only concedes further from x*, so the
    emitted offer is never more aggressive than the gradient optimum."""
    from wehaveadeal.market import MarketState

    agent = CarrierOPT2(agent_id="carrier-opt2-1")
    market = MarketState(seed=7, n_negotiations=1)
    route = market.assign_route(market.negotiation_rng(0))
    from wehaveadeal.models import TableState

    state = TableState(
        role="carrier",
        table_id="t",
        negotiation_seed=7,
        turn=2,
        turns_max=12,
        my_turns_used=0,
        opp_turns_used=1,
        my_last_offer=None,
        opp_last_offer=route.refprice_mean - 0.4 * route.refprice_sigma,
        sigma=route.refprice_sigma,
        refprice_mean=route.refprice_mean,
    )
    rp_t = route.refprice_mean - 0.5 * route.refprice_sigma
    target_t = route.refprice_mean + 0.5 * route.refprice_sigma
    threshold = agent.estimate_shipper_threshold(state)
    x_star = agent._optimize_offer(state, rp_t, target_t, target_t, threshold, 0.5)
    x = agent._anti_detection_envelope(x_star, rp_t, target_t, threshold, state, 0.5)
    assert rp_t - 1e-6 <= x <= x_star + 1e-6


# ---------------------------------------------------------------------------
# Shipper-side mirrors of the five carrier architectures (ShipperR was the
# only pre-existing shipper engine; ShipperRV/IS/OPT/OPT2 are new). Fixed
# opponent is CarrierR throughout, mirroring how the carrier tests above fix
# ShipperR as their sparring partner.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shipper_cls", ALL_SHIPPER_CLASSES)
def test_shipper_never_offers_above_rp_or_non_monotonic(shipper_cls):
    """Fuzz across many seeds/negotiations: shippers must never offer above
    their RP or produce a non-monotonic (decreasing) concession. run_round
    raises IllegalMoveError internally if this is ever violated."""
    shippers = {f"shipper-{i+1}": shipper_cls(agent_id=f"shipper-{i+1}") for i in range(4)}
    carriers = {f"carrier-{i+1}": CarrierR(agent_id=f"carrier-{i+1}") for i in range(3)}
    for seed in range(10):
        run_round(round_index=0, seed=seed, shipper_engines=shippers, carrier_engines=carriers, n_negotiations=10)


@pytest.mark.parametrize("shipper_cls", ALL_SHIPPER_CLASSES)
def test_shipper_engines_never_exceed_reservation_price(shipper_cls):
    shippers = {f"shipper-{i+1}": shipper_cls(agent_id=f"shipper-{i+1}") for i in range(4)}
    carriers = {f"carrier-{i+1}": CarrierR(agent_id=f"carrier-{i+1}") for i in range(3)}
    result = run_round(round_index=0, seed=123, shipper_engines=shippers, carrier_engines=carriers, n_negotiations=20)
    for neg in result.negotiations:
        rp = neg.sale_price - neg.goods_cost
        for table in neg.tables:
            for rec in table.history:
                if rec.actor == "shipper" and rec.action == Action.OFFER:
                    assert rec.amount <= rp + 1e-6


@pytest.mark.parametrize("shipper_cls", [ShipperRV, ShipperIS, ShipperOPT, ShipperOPT2])
def test_new_shipper_engines_offers_are_monotonically_non_decreasing(shipper_cls):
    shippers = {f"shipper-{i+1}": shipper_cls(agent_id=f"shipper-{i+1}") for i in range(4)}
    carriers = {f"carrier-{i+1}": CarrierR(agent_id=f"carrier-{i+1}") for i in range(3)}
    result = run_round(round_index=0, seed=17, shipper_engines=shippers, carrier_engines=carriers, n_negotiations=15)
    for neg in result.negotiations:
        for table in neg.tables:
            offers = [r.amount for r in table.history if r.actor == "shipper" and r.action == Action.OFFER]
            assert offers == sorted(offers)


@pytest.mark.parametrize("shipper_cls", [ShipperRV, ShipperIS, ShipperOPT, ShipperOPT2])
def test_new_shipper_engines_beat_random_walk_on_closes(shipper_cls):
    """Sanity check mirrored from the carrier-side occupancy check: against
    the fixed CarrierR sparring partner, each smarter shipper engine should
    close most negotiations, not accidentally do worse than doing nothing
    clever."""
    shippers = {f"shipper-{i+1}": shipper_cls(agent_id=f"shipper-{i+1}") for i in range(4)}
    carriers = {f"carrier-{i+1}": CarrierR(agent_id=f"carrier-{i+1}") for i in range(3)}
    n_negotiations = 50
    result = run_round(
        round_index=0, seed=99, shipper_engines=shippers, carrier_engines=carriers, n_negotiations=n_negotiations
    )
    for shipper_id, standing in result.shipper_standings.items():
        assert standing["closes"] / n_negotiations > 0.5


def test_shipper_is_particle_sampling_is_deterministic_given_seed():
    shippers = {f"shipper-{i+1}": ShipperIS(agent_id=f"shipper-{i+1}") for i in range(4)}
    carriers = {f"carrier-{i+1}": CarrierR(agent_id=f"carrier-{i+1}") for i in range(3)}
    r1 = run_round(round_index=0, seed=55, shipper_engines=shippers, carrier_engines=carriers, n_negotiations=5)
    r2 = run_round(round_index=0, seed=55, shipper_engines=shippers, carrier_engines=carriers, n_negotiations=5)
    assert r1.shipper_standings == r2.shipper_standings


def test_shipper_opt2_envelope_never_improves_on_its_own_optimum():
    """The anti-detection envelope only concedes further from x* (upward,
    for a shipper), so the emitted offer is never more aggressive than the
    gradient optimum."""
    from wehaveadeal.market import MarketState
    from wehaveadeal.models import TableState

    agent = ShipperOPT2(agent_id="shipper-opt2-1")
    market = MarketState(seed=7, n_negotiations=1)
    route = market.assign_route(market.negotiation_rng(0))

    state = TableState(
        role="shipper",
        table_id="t",
        negotiation_seed=7,
        turn=2,
        turns_max=12,
        my_turns_used=0,
        opp_turns_used=1,
        my_last_offer=None,
        opp_last_offer=route.refprice_mean + 0.4 * route.refprice_sigma,
        sigma=route.refprice_sigma,
        sale_price=5000.0,
        goods_cost=1000.0,
        refprice_est=route.refprice_mean,
    )
    rp, target = agent._band(state)
    threshold = agent.estimate_carrier_floor(state)
    x_star = agent._optimize_offer(state, rp, target, target, threshold, 0.5)
    x = agent._anti_detection_envelope(x_star, rp, target, threshold, state, 0.5)
    assert x_star - 1e-6 <= x <= rp + 1e-6
