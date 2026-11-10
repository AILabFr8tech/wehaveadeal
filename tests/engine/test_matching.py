from wehaveadeal.matching import gale_shapley
from wehaveadeal.models import TableResult


def _closed(shipper_id, carrier_id, rate, turns):
    return TableResult(shipper_id=shipper_id, carrier_id=carrier_id, closed=True, impasse=False, rate=rate, turns_to_close=turns)


def _impasse(shipper_id, carrier_id):
    return TableResult(shipper_id=shipper_id, carrier_id=carrier_id, closed=False, impasse=True)


def test_required_regression_worked_example():
    """A closes with T1 @ 1000; B closes with T1 @ 1300 AND T2 @ 1250.
    Correct deferred acceptance: A matched with T1 (NOT excluded), B matched
    with T2. See the paper (docs/PROTOCOL.md) "Matching" for the reasoning:
    B's cheapest close is T2 (1250 < 1300), so B never even proposes to T1
    unless rejected by T2 -- and T2 accepts B outright. So T1's only
    proposer is A, who is matched.
    """
    tables = [
        _closed("A", "T1", 1000, 4),
        _closed("B", "T1", 1300, 6),
        _closed("B", "T2", 1250, 4),
    ]
    result = gale_shapley(tables, all_shipper_ids=["A", "B"], seed=123)

    matches = {m.shipper_id: (m.carrier_id, m.rate) for m in result.matches}
    assert matches["A"] == ("T1", 1000)
    assert matches["B"] == ("T2", 1250)
    assert result.unmatched_shippers == []


def test_impasse_pairs_are_excluded_entirely():
    tables = [
        _closed("A", "T1", 1000, 4),
        _impasse("A", "T2"),
        _impasse("B", "T1"),
    ]
    result = gale_shapley(tables, all_shipper_ids=["A", "B"], seed=1)
    assert "T2" not in result.shipper_preferences.get("A", [])
    assert "B" not in result.carrier_preferences.get("T1", [])
    matches = {m.shipper_id: m.carrier_id for m in result.matches}
    assert matches["A"] == "T1"
    assert "B" in result.unmatched_shippers


def test_shipper_with_zero_closes_is_unmatched():
    tables = [_closed("A", "T1", 1000, 4)]
    result = gale_shapley(tables, all_shipper_ids=["A", "B", "C"], seed=1)
    assert set(result.unmatched_shippers) == {"B", "C"}


def test_carrier_prefers_higher_rate_when_contested():
    # C1 has two closes; it should end up with the pricier one, freeing the
    # cheaper shipper to match elsewhere if it has another option, or go
    # unmatched otherwise.
    tables = [
        _closed("A", "C1", 900, 3),
        _closed("B", "C1", 1100, 3),
    ]
    result = gale_shapley(tables, all_shipper_ids=["A", "B"], seed=7)
    matches = {m.shipper_id: m.carrier_id for m in result.matches}
    assert matches.get("B") == "C1"
    assert "A" in result.unmatched_shippers


def test_tie_break_fewer_turns_wins_then_seed_is_deterministic():
    tables = [
        _closed("A", "C1", 1000, 6),
        _closed("B", "C1", 1000, 2),  # same rate, fewer turns -> B preferred by C1
    ]
    result = gale_shapley(tables, all_shipper_ids=["A", "B"], seed=99)
    matches = {m.shipper_id: m.carrier_id for m in result.matches}
    assert matches.get("B") == "C1"

    # Determinism: same inputs + same seed -> identical result.
    result2 = gale_shapley(tables, all_shipper_ids=["A", "B"], seed=99)
    assert {m.shipper_id: m.carrier_id for m in result2.matches} == matches
