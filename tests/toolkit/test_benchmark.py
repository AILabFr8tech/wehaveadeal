"""The benchmark protocol itself: reproducible, fair across entrants, honest statistics."""
from __future__ import annotations

import json
import re

import numpy as np
import pytest

from wehaveadeal import DecisionEngine, REJECT
from wehaveadeal.models import RoundResult
from wehaveadeal.toolkit.benchmark import (
    BASELINE_KINDS,
    PROTOCOL_VERSION,
    BenchmarkReport,
    _holm,
    _lineup,
    format_report,
    regime_sizes,
    run_benchmark,
)
from wehaveadeal.toolkit.loader import load_agent

from .conftest import FAST, GOOD_CARRIER, GOOD_SHIPPER


@pytest.fixture
def carrier_spec(agent_dir):
    return load_agent(str(agent_dir.write("good_carrier.py", GOOD_CARRIER)))


@pytest.fixture
def shipper_spec(agent_dir):
    return load_agent(str(agent_dir.write("good_shipper.py", GOOD_SHIPPER)))


def test_same_seed_same_results(carrier_spec):
    a = run_benchmark(carrier_spec, seed=11, **FAST)
    b = run_benchmark(carrier_spec, seed=11, **FAST)
    assert a.results_fingerprint() == b.results_fingerprint()


def test_different_seed_different_markets(carrier_spec):
    a = run_benchmark(carrier_spec, seed=11, **FAST)
    b = run_benchmark(carrier_spec, seed=12, **FAST)
    assert a.results_fingerprint()["regimes"] != b.results_fingerprint()["regimes"]


def test_common_random_numbers_markets_do_not_depend_on_the_entrant(carrier_spec, tmp_path):
    """Two different entrants with the same seed face the same market days and line-ups:
    the round logs must have identical seeds, routes, prices and rosters (minus the entrant)."""

    class Passive(DecisionEngine):
        role = "carrier"
        engine_name = "Passive"

        def next_move(self, state):
            return REJECT

    run_benchmark(carrier_spec, seed=5, save_rounds_dir=tmp_path / "a", **FAST)
    run_benchmark(Passive, seed=5, save_rounds_dir=tmp_path / "b", **FAST)
    market = lambda p: re.sub(r"^bench_u_[a-z0-9]+_", "", p.name)  # noqa: E731 -- drop the entrant's name
    files_a = sorted(market(p) for p in (tmp_path / "a").glob("*.json"))
    files_b = sorted(market(p) for p in (tmp_path / "b").glob("*.json"))
    assert files_a == files_b and len(files_a) == 6  # 3 regimes x 2 sub-rounds

    for pa in sorted((tmp_path / "a").glob("*.json")):
        pb = next((tmp_path / "b").glob("*_" + market(pa)))
        ra = RoundResult.model_validate_json(pa.read_text())
        rb = RoundResult.model_validate_json(pb.read_text())
        assert ra.seed == rb.seed and ra.shipper_ids == rb.shipper_ids
        strip = lambda ids: [i for i in ids if "-u_" not in i]  # noqa: E731
        assert strip(ra.carrier_ids) == strip(rb.carrier_ids)
        for na, nb in zip(ra.negotiations, rb.negotiations):
            assert (na.sale_price, na.goods_cost) == (nb.sale_price, nb.goods_cost)
            assert na.routes == nb.routes


@pytest.mark.parametrize("role", ["carrier", "shipper"])
def test_regime_sizes_follow_the_paper(role):
    own_scarce = f"{role}_scarce"
    other_scarce = "shipper_scarce" if role == "carrier" else "carrier_scarce"
    assert regime_sizes("balanced", role, 9) == (9, 9)
    assert regime_sizes(own_scarce, role, 9) == (7, 9)
    assert regime_sizes(other_scarce, role, 9) == (9, 7)


def test_report_regimes_have_the_right_market_sizes(shipper_spec):
    report = run_benchmark(shipper_spec, **FAST)
    sizes = {r.regime: (r.own_side_size, r.other_side_size) for r in report.regimes}
    assert sizes == {"balanced": (5, 5), "shipper_scarce": (3, 5), "carrier_scarce": (5, 3)}
    for r in report.regimes:
        assert r.n_market_days == FAST["n_subrounds"] * FAST["n_negotiations"]
        assert 1 <= r.entrant_rank <= r.n_ranked
        assert {b.arch for b in r.baselines} <= {k.upper() for k in FAST["baselines"]}


def test_lineup_is_deterministic_and_never_repeats_a_baseline():
    for sub in range(20):
        own, other, seed = _lineup(3, 1, sub, BASELINE_KINDS, 7, 9)
        assert (own, other, seed) == _lineup(3, 1, sub, BASELINE_KINDS, 7, 9)
        assert len(own) == 6 and len(set(own)) == 6
        assert sorted(other) == sorted(BASELINE_KINDS)


def test_entrant_sees_only_its_own_side_of_the_information(agent_dir):
    """Information hiding is structural: a carrier never sees the shipper's private
    economics, a shipper never sees the true reference price."""
    seen = []

    class Snoop(DecisionEngine):
        role = "carrier"

        def next_move(self, state):
            seen.append((state.sale_price, state.goods_cost, state.refprice_est, state.refprice_mean))
            return REJECT

    run_benchmark(Snoop, regimes=("balanced",), baselines=FAST["baselines"], n_subrounds=1, n_negotiations=2, preset="smoke")
    assert seen and all(s[:3] == (None, None, None) and s[3] is not None for s in seen)

    seen_shipper = []

    class SnoopShipper(DecisionEngine):
        role = "shipper"

        def next_move(self, state):
            seen_shipper.append((state.refprice_mean, state.refprice_est))
            return REJECT

    run_benchmark(SnoopShipper, regimes=("balanced",), baselines=FAST["baselines"], n_subrounds=1, n_negotiations=2, preset="smoke")
    assert seen_shipper and all(m is None and e is not None for m, e in seen_shipper)


def test_stateful_agents_persist_across_negotiations(agent_dir):
    """Online-learning agents get one instance for the whole run, so they can adapt."""
    p = agent_dir.write("learner.py", '''
        from wehaveadeal import REJECT, DecisionEngine

        class Learner(DecisionEngine):
            role = "shipper"
            seen_seeds = set()

            def next_move(self, state):
                type(self).seen_seeds.add(state.negotiation_seed)
                return REJECT
    ''')
    spec = load_agent(str(p))
    run_benchmark(spec, **FAST)
    assert len(spec.cls.seen_seeds) == 3 * FAST["n_subrounds"] * FAST["n_negotiations"]


def test_report_json_round_trip_and_protocol_version(carrier_spec, tmp_path):
    report = run_benchmark(carrier_spec, **FAST)
    path = tmp_path / "r.json"
    path.write_text(report.model_dump_json(indent=2))
    again = BenchmarkReport.model_validate_json(path.read_text())
    assert again.results_fingerprint() == report.results_fingerprint()
    data = json.loads(path.read_text())
    assert data["settings"]["protocol_version"] == PROTOCOL_VERSION
    assert set(data) >= {"agent", "settings", "headline", "regimes", "violations", "timing"}
    text = format_report(report)
    assert "GoodCarrier" in text and "balanced" in text


def test_regime_subset_and_validation_errors(carrier_spec):
    report = run_benchmark(carrier_spec, regimes=("shipper_scarce",), **{k: v for k, v in FAST.items()})
    assert [r.regime for r in report.regimes] == ["shipper_scarce"]
    with pytest.raises(ValueError, match="preset"):
        run_benchmark(carrier_spec, preset="huge")
    with pytest.raises(ValueError, match="unknown baseline"):
        run_benchmark(carrier_spec, baselines=("r", "rv", "gpt"))
    with pytest.raises(ValueError, match="at least 3"):
        run_benchmark(carrier_spec, baselines=("r", "rv"))


def test_holm_correction_matches_reference_values():
    # statsmodels.stats.multitest.multipletests([0.01, 0.04, 0.03, 0.5], method="holm")
    assert np.allclose(_holm([0.01, 0.04, 0.03, 0.5]), [0.04, 0.09, 0.09, 0.5])
    assert _holm([0.9]) == [0.9]


def test_paired_comparisons_cover_every_present_baseline(carrier_spec):
    report = run_benchmark(carrier_spec, **FAST)
    for r in report.regimes:
        assert {c.baseline for c in r.vs_baselines} == {b.arch for b in r.baselines}
        for c in r.vs_baselines:
            assert c.p_holm >= c.p_raw - 1e-12
            assert c.verdict in {"better", "worse", "no significant difference", "insufficient data"}


def test_full_roster_with_learned_baselines_loads_bundled_weights(carrier_spec, monkeypatch, tmp_path):
    """The default benchmark uses all nine architectures, including ML/ML2/RL/RL2 whose
    weights ship inside the package. Run from an unrelated cwd to prove paths are absolute."""
    monkeypatch.chdir(tmp_path)
    report = run_benchmark(carrier_spec, regimes=("balanced",), n_subrounds=1, n_negotiations=1, preset="smoke")
    assert report.settings.baselines == list(BASELINE_KINDS)
    assert len(report.regimes[0].baselines) == 8  # entrant + 8 of the 9 on its side
