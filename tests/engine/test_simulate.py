from wehaveadeal.simulate import default_agents, run_round


def test_full_round_runs_end_to_end():
    shippers, carriers = default_agents()
    result = run_round(round_index=0, seed=42, shipper_engines=shippers, carrier_engines=carriers, n_negotiations=100)

    assert len(result.negotiations) == 100
    for neg in result.negotiations:
        assert len(neg.tables) == 12  # 4 shippers x 3 carriers
        # Structural fact of the 4-vs-3 scarcity ratio: at least one shipper
        # is always unmatched, regardless of how negotiations went.
        assert len(neg.matching.unmatched_shippers) >= 1
        assert len(neg.matching.matches) <= 3

    assert result.round_winner_shipper in shippers
    assert result.round_winner_carrier in carriers
    assert set(result.shipper_standings.keys()) == set(shippers.keys())
    assert set(result.carrier_standings.keys()) == set(carriers.keys())


def test_no_illegal_moves_across_several_seeds():
    shippers, carriers = default_agents()
    for seed in (1, 2, 3, 4, 5):
        # Would raise IllegalMoveError internally if any agent ever offered
        # outside its band or produced a non-monotonic concession.
        run_round(round_index=0, seed=seed, shipper_engines=shippers, carrier_engines=carriers, n_negotiations=15)


def test_carrier_and_shipper_standings_are_internally_consistent():
    shippers, carriers = default_agents()
    result = run_round(round_index=0, seed=7, shipper_engines=shippers, carrier_engines=carriers, n_negotiations=30)

    total_matches = sum(len(n.matching.matches) for n in result.negotiations)
    total_carrier_matches = sum(st["matches"] for st in result.carrier_standings.values())
    assert total_matches == total_carrier_matches

    for carrier_id, st in result.carrier_standings.items():
        assert 0.0 <= st["occupancy"] <= 1.0
