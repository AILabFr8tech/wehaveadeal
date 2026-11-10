"""Role-agnostic feature extraction for imitation-learning engines.

Every price-like feature is expressed as a position on the same [0,1] band
scale `value_bar` already uses everywhere else in this package (0 = worst
anchor, 1 = best anchor) rather than as a raw price. That's what lets ONE
feature schema -- and, if trained that way, one model -- serve both roles:
only the anchors swap (carrier: worst=RP_t, best=target_t; shipper:
worst=RP, best=target), exactly the same mirroring principle
`wehaveadeal/value_functions.py` already relies on. A model trained on
carrier rows and fed shipper rows (or vice versa) would at least be looking
at numbers on a consistent, comparable scale, even if in practice we train
one model per role for cleaner evaluation (see `collect.py`/`train.py`).

Nothing here is specific to any one engine architecture: `role_band` uses
the same (worst, best) formula every engine of that role already computes
for its own hard/tactical band (CarrierR/RV/IS/OPT/OPT2 all derive
`rp_t/target_t` from `refprice_mean`; ShipperR/RV/IS/OPT/OPT2 all derive
`rp/target` from `sale_price/goods_cost/refprice_est`), so features
extracted here line up with what any of those engines was "thinking" at the
same decision point.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from wehaveadeal.market import SHIPPER_NOISE_FRACTION
from wehaveadeal.models import Action, TableState
from wehaveadeal.value_functions import portfolio_occupancy, value_bar

TARGET_CUSHION_K = 0.5  # matches every shipper engine's own default

FEATURE_NAMES: Tuple[str, ...] = (
    "turn_frac",
    "my_turn_frac",
    "opp_turn_frac",
    "opp_last_pos",
    "opp_last_seen",
    "my_last_pos",
    "my_last_seen",
    "opp_trend_slope",
    "opp_trend_intercept_pos",
    "opp_trend_seen",
    "n_opp_offers",
    "portfolio_promise",
)


def role_band(state: TableState) -> Tuple[float, float]:
    """(worst, best) anchors for `state`'s role -- identical to the hard
    band every carrier engine uses, and the tactical band every shipper
    engine uses (shippers have no hard floor, see
    the paper Section 6.0)."""
    if state.role == "carrier":
        rp_t = state.refprice_mean - 0.5 * state.sigma
        target_t = state.refprice_mean + 0.5 * state.sigma
        return rp_t, target_t
    rp = state.sale_price - state.goods_cost
    sigma_eps = state.sigma * SHIPPER_NOISE_FRACTION
    target = state.refprice_est - 0.5 * state.sigma + TARGET_CUSHION_K * sigma_eps
    return rp, target


def _fit_trend(positions: Sequence[float]) -> Tuple[float, float]:
    """Closed-form OLS slope/intercept of `position ~ intercept + slope*k`
    over the opponent's observed offer positions. CarrierOPT2/ShipperOPT2
    fit the same kind of line via explicit gradient-descent steps because
    there the point was co-optimizing belief and offer through one
    mechanism (see carrier_opt2.py); here we only want the number as an
    input feature, so the closed form is the right tool. n<2 -> no
    detectable trend yet."""
    n = len(positions)
    if n == 0:
        return 0.0, 0.5
    if n == 1:
        return 0.0, positions[0]
    xs = range(n)
    mean_x = (n - 1) / 2.0
    mean_y = sum(positions) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, positions))
    var = sum((x - mean_x) ** 2 for x in xs)
    slope = cov / var if var > 0 else 0.0
    intercept = mean_y - slope * mean_x
    return slope, intercept


def extract_features(state: TableState) -> Dict[str, float]:
    """TableState -> fixed-size, role-agnostic feature dict (see module
    docstring for why every price-like value is a `value_bar` position
    rather than a raw price)."""
    worst, best = role_band(state)

    def pos(price: float) -> float:
        return value_bar(price, worst, best)

    opp_role = "shipper" if state.role == "carrier" else "carrier"
    opp_offer_positions = [
        pos(r.amount) for r in state.history if r.actor == opp_role and r.action == Action.OFFER and r.amount is not None
    ]
    slope, intercept = _fit_trend(opp_offer_positions)

    turns_per_side = state.turns_max // 2
    threshold = state.my_last_offer if state.my_last_offer is not None else (worst + best) / 2
    p_promise = portfolio_occupancy(state.my_portfolio, threshold, higher_is_better=(state.role == "carrier"))

    return {
        "turn_frac": state.turn / state.turns_max,
        "my_turn_frac": state.my_turns_used / turns_per_side,
        "opp_turn_frac": state.opp_turns_used / turns_per_side,
        "opp_last_pos": pos(state.opp_last_offer) if state.opp_last_offer is not None else 0.5,
        "opp_last_seen": 1.0 if state.opp_last_offer is not None else 0.0,
        "my_last_pos": pos(state.my_last_offer) if state.my_last_offer is not None else 0.5,
        "my_last_seen": 1.0 if state.my_last_offer is not None else 0.0,
        "opp_trend_slope": slope,
        "opp_trend_intercept_pos": intercept,
        "opp_trend_seen": 1.0 if opp_offer_positions else 0.0,
        "n_opp_offers": float(len(opp_offer_positions)),
        "portfolio_promise": p_promise,
    }


def features_to_vector(features: Dict[str, float]) -> List[float]:
    return [features[name] for name in FEATURE_NAMES]
