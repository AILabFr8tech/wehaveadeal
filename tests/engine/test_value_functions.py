from wehaveadeal.value_functions import carrier_value, shipper_value, value_bar


def test_value_bar_is_1_at_best_and_0_at_worst():
    assert value_bar(100, worst_anchor=100, best_anchor=50) == 0.0
    assert value_bar(50, worst_anchor=100, best_anchor=50) == 1.0
    assert value_bar(75, worst_anchor=100, best_anchor=50) == 0.5


def test_shipper_value_terminal_ignores_pressure():
    # terminal branch returns the raw normalized close value, no time-pressure factor
    v = shipper_value(rp=100, target=50, terminal=True, close_price=50, turn=1, turns_max=12, weight_assignment=0.5, p_assignment=0.0)
    assert v == 1.0


def test_shipper_and_carrier_value_are_mirrored_via_shared_core():
    # shipper: lower price better. carrier: higher price better. Same shared
    # core (_value_of_position / value_bar), swapped anchors.
    v_shipper_good = shipper_value(
        rp=100, target=50, carrier_offer_on_table=50, my_last_offer=50,
        turn=1, turns_max=12, weight_assignment=0.0, p_assignment=1.0,
    )
    v_carrier_good = carrier_value(
        rp_t=50, target_t=100, shipper_offer_on_table=100, my_last_counter=100,
        turn=1, turns_max=12, weight_occupancy=0.0, p_occupancy=1.0,
    )
    assert v_shipper_good == v_carrier_good  # both are "best possible position, turn 1"


def test_value_decays_with_time_pressure():
    kwargs = dict(rp=100, target=50, carrier_offer_on_table=75, my_last_offer=75, weight_assignment=0.0, p_assignment=1.0)
    v_early = shipper_value(turn=1, turns_max=12, **kwargs)
    v_late = shipper_value(turn=11, turns_max=12, **kwargs)
    assert v_late < v_early


def test_carrier_value_zero_when_turns_exhausted():
    v = carrier_value(
        rp_t=50, target_t=100, shipper_offer_on_table=75, my_last_counter=75,
        turn=12, turns_max=12, weight_occupancy=0.5, p_occupancy=0.5,
    )
    assert v == 0.0
