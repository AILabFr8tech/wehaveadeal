"""Shared fixtures for the community-user tests.

These tests play the part of an outside researcher: they write agent files to a
fresh folder (never inside the package), load them the way the CLI does, and
run small benchmarks. Benchmarks use only the hand-designed baselines and a
tiny budget unless a test is specifically about the learned ones, so the whole
suite stays fast.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from wehaveadeal.toolkit.benchmark import HAND_DESIGNED_KINDS

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"

# Small but non-trivial: 2 sub-rounds x 3 market days per regime.
FAST = dict(baselines=HAND_DESIGNED_KINDS, n_subrounds=2, n_negotiations=3, preset="smoke")

GOOD_CARRIER = '''
from wehaveadeal import ACCEPT, REJECT, DecisionEngine, legal_offer_range, offer

class GoodCarrier(DecisionEngine):
    role = "carrier"
    engine_name = "GoodCarrier"

    def next_move(self, state):
        lo, hi = legal_offer_range(state)
        if state.opp_last_offer is not None and state.opp_last_offer >= state.refprice_mean:
            return ACCEPT
        if lo > hi:
            return REJECT
        return offer(hi - (hi - lo) * state.my_turns_used / state.turns_per_side)
'''

GOOD_SHIPPER = '''
from wehaveadeal import ACCEPT, REJECT, DecisionEngine, legal_offer_range, offer

class GoodShipper(DecisionEngine):
    role = "shipper"
    engine_name = "GoodShipper"

    def next_move(self, state):
        lo, hi = legal_offer_range(state)
        if state.opp_last_offer is not None and state.opp_last_offer <= hi:
            return ACCEPT
        if lo > hi:
            return REJECT
        return offer(lo + (hi - lo) * 0.5)
'''


@pytest.fixture
def agent_dir(tmp_path: Path):
    """A user's own agents folder. Call the fixture's `.write(name, code)` to add a file."""
    d = tmp_path / "agents"
    d.mkdir()

    class Dir:
        path = d

        @staticmethod
        def write(filename: str, code: str) -> Path:
            p = d / filename
            p.write_text(textwrap.dedent(code), encoding="utf-8")
            return p

    return Dir
