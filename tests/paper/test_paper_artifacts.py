"""Everything the paper reports is in paper/, internally consistent, and reproducible.

These tests are what lets a reviewer trust paper/: the figures exist, the
summary tables are exactly what the raw data says, and the public package
re-simulates the published tournament.
"""
from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

PAPER = Path(__file__).resolve().parents[2] / "paper"
VALID = PAPER / "data" / "validation"
LLM = PAPER / "data" / "llm_validation"
KINDS = ["r", "rv", "is", "opt", "opt2", "ml", "ml2", "rl", "rl2"]

# Every \includegraphics in the manuscript.
PAPER_FIGURES = [
    "ranking_by_config.png",
    *[f"win_matrix_{role}_{cfg}.png" for role in ("carrier", "shipper")
      for cfg in ("balanced", "carrier_scarce", "shipper_scarce")],
    "llm_vs_engines.png",
    "llm_by_tier_family.png",
    "llm_trajectory_clean_examples.png",
    "llm_trajectory_timeout_example.png",
]


@pytest.fixture(scope="module")
def raw():
    df = pd.read_csv(VALID / "raw_results.csv")
    df["arch"] = df["kind"].str.upper()  # same labels as validation_exhaustive.ipynb's ARCH_LABELS
    return df


@pytest.mark.parametrize("name", PAPER_FIGURES)
def test_every_paper_figure_is_published(name):
    assert (PAPER / "figures" / name).stat().st_size > 1000


def test_tournament_sample_sizes_match_the_design(raw):
    """400 balanced days; 36 exclusion pairs x 12 days = 432 days per scarce condition."""
    days = raw.groupby("config")["day_id"].nunique().to_dict()
    assert days == {"balanced": 400, "carrier_scarce": 432, "shipper_scarce": 432}
    assert raw[raw.config != "balanced"].groupby("config")["excluded"].nunique().eq(36).all()


def test_summary_table_recomputes_from_raw_data(raw):
    """Tables tab:carrier / tab:shipper and fig:ranking come from summary_stats.csv;
    its n, match rate and mean must be exactly what raw_results.csv says."""
    summary = pd.read_csv(VALID / "summary_stats.csv")
    g = raw.groupby(["config", "role", "arch"]).agg(n=("outcome", "size"), match_rate=("matched", "mean"), mean_outcome=("outcome", "mean"))
    merged = summary.set_index(["config", "role", "arch"]).join(g, rsuffix="_raw")
    assert len(merged) == 3 * 2 * 9
    assert (merged.n == merged.n_raw).all()
    assert np.allclose(merged.match_rate, merged.match_rate_raw)
    assert np.allclose(merged.mean_outcome, merged.mean_outcome_raw)
    assert (merged.ci_lo <= merged.mean_outcome).all() and (merged.mean_outcome <= merged.ci_hi).all()


def test_pairwise_tests_recompute_from_raw_data(raw):
    """The win matrices come from pairwise_tests.csv: same-day paired differences."""
    tests = pd.read_csv(VALID / "pairwise_tests.csv")
    assert len(tests) == 3 * 2 * 36
    for _, row in tests.sample(25, random_state=0).iterrows():
        sub = raw[(raw.config == row.config) & (raw.role == row.role) & raw.arch.isin([row.arch_a, row.arch_b])]
        wide = sub.pivot_table(index="day_id", columns="arch", values="outcome").dropna()
        diff = wide[row.arch_a] - wide[row.arch_b]
        assert len(diff) == row.n_days
        assert np.isclose(diff.mean(), row.mean_diff)
    assert (tests.p_holm >= tests.p_raw - 1e-12).all()
    assert (tests.significant == (tests.p_holm < 0.05)).all()


def test_ablation_uses_the_same_protocol(raw):
    abl = pd.read_csv(VALID / "ablation_rl2fixed_raw_results.csv")
    assert abl.groupby("config")["day_id"].nunique().to_dict() == raw.groupby("config")["day_id"].nunique().to_dict()
    # Same markets: the 8 unchanged architectures' line-ups and day ids match the original study.
    assert set(abl.day_id) == set(raw.day_id)


def test_llm_study_logs_are_complete():
    from wehaveadeal.llm.roster import family_of, tier_of

    outcome_files = sorted(LLM.glob("*_outcomes.jsonl"))
    calls_files = sorted(LLM.glob("*_calls.jsonl"))
    assert len(outcome_files) == len(calls_files) == 9 * 2  # nine models x two roles
    for path in outcome_files:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        llm_rows = [r for r in rows if r.get("is_llm")]
        assert llm_rows, path.name
        model = llm_rows[0]["model"]
        assert tier_of(model) == llm_rows[0]["tier"] and family_of(model) == llm_rows[0]["family"]
        assert len({r["day_index"] for r in rows}) >= 35, f"{path.name}: too few logged days"


def test_tournament_reproduces_from_the_public_package(raw):
    """Re-simulate the first carrier_scarce sub-round (12 market days, all nine
    architectures incl. the learned ones) and compare with the published rows.
    Deals must be identical; payoffs equal up to floating-point noise."""
    from wehaveadeal.simulate import build_carrier, build_shipper, run_round

    pair = list(itertools.combinations(KINDS, 2))[0]
    kept = [k for k in KINDS if k not in pair]
    carriers = {f"carrier-{k}-1": build_carrier(k, f"carrier-{k}-1") for k in kept}
    shippers = {f"shipper-{k}-1": build_shipper(k, f"shipper-{k}-1") for k in KINDS}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_round(round_index=0, seed=20_000_000, shipper_engines=shippers, carrier_engines=carriers, n_negotiations=12)

    published = raw[(raw.config == "carrier_scarce") & (raw.excluded == "+".join(pair))]
    published = {(r.day_id.rsplit("|", 1)[1], r.role, r.kind): (r.matched, r.outcome) for r in published.itertuples()}
    n = 0
    for neg in result.negotiations:
        for o in neg.carrier_outcomes:
            key = (str(neg.negotiation_index), "carrier", o.carrier_id.split("-")[1])
            assert published[key][0] == o.matched
            assert abs(published[key][1] - o.margin) < 1e-3
            n += 1
        for o in neg.shipper_outcomes:
            key = (str(neg.negotiation_index), "shipper", o.shipper_id.split("-")[1])
            assert published[key][0] == o.matched
            assert abs(published[key][1] - o.profit) < 1e-3
            n += 1
    assert n == len(published) == 12 * (7 + 9)


def test_rl2fixed_ablation_agents_load_bundled_weights():
    from wehaveadeal.agents import CarrierRL2Fixed, ShipperRL2Fixed
    from wehaveadeal.simulate import build_carrier, build_shipper, run_round

    carriers = {"carrier-rl2fixed-1": CarrierRL2Fixed(agent_id="carrier-rl2fixed-1"), "carrier-r-1": build_carrier("r", "carrier-r-1")}
    shippers = {"shipper-rl2fixed-1": ShipperRL2Fixed(agent_id="shipper-rl2fixed-1"), "shipper-r-1": build_shipper("r", "shipper-r-1")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_round(round_index=0, seed=1, shipper_engines=shippers, carrier_engines=carriers, n_negotiations=2)
    assert len(result.negotiations) == 2
