from wehaveadeal.models import ShipperNegotiationOutcome
from wehaveadeal.scoring import apply_round_bonus, score_shippers


def _o(shipper_id, matched, profit, declined=False):
    return ShipperNegotiationOutcome(shipper_id=shipper_id, matched=matched, profit=profit, declined_to_ship=declined)


def test_points_only_to_matched_closed_shippers():
    outcomes = [
        _o("A", True, 100.0),
        _o("B", True, 80.0),
        _o("C", True, 50.0),
        _o("D", False, 0.0),  # unmatched
    ]
    score_shippers(outcomes)
    points = {o.shipper_id: o.points for o in outcomes}
    assert points == {"A": 6.0, "B": 3.0, "C": 1.0, "D": 0.0}


def test_unmatched_and_zero_close_score_zero():
    outcomes = [_o("A", True, 10.0), _o("B", False, 0.0)]
    score_shippers(outcomes)
    assert outcomes[1].points == 0.0


def test_declined_to_ship_scores_zero_even_if_matched_flag_unset():
    outcomes = [_o("A", True, 10.0), _o("B", False, 0.0, declined=True)]
    score_shippers(outcomes)
    assert outcomes[1].points == 0.0


def test_tie_shares_mean_of_positions():
    outcomes = [
        _o("A", True, 100.0),
        _o("B", True, 100.0),  # tie for 1st/2nd -> (6+3)/2 = 4.5 each
        _o("C", True, 50.0),  # 3rd -> 1
    ]
    score_shippers(outcomes)
    points = {o.shipper_id: o.points for o in outcomes}
    assert points["A"] == 4.5
    assert points["B"] == 4.5
    assert points["C"] == 1.0


def test_three_way_tie_for_first_shares_mean_of_top_three():
    outcomes = [_o("A", True, 10.0), _o("B", True, 10.0), _o("C", True, 10.0)]
    score_shippers(outcomes)
    expected = (6.0 + 3.0 + 1.0) / 3
    assert all(o.points == expected for o in outcomes)


def test_round_bonus_goes_to_highest_accumulated_profit():
    standings = {
        "A": {"points": 10.0, "profit": 500.0},
        "B": {"points": 12.0, "profit": 900.0},
    }
    apply_round_bonus(standings, bonus=200.0)
    assert standings["B"]["points"] == 212.0
    assert standings["A"]["points"] == 10.0
    assert standings["B"].get("got_bonus") is True
