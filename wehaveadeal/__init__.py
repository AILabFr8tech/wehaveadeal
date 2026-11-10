"""We Have a Deal -- a research toolkit for bilateral freight negotiation.

Write an agent:

    from wehaveadeal import DecisionEngine, Move, TableState, ACCEPT, REJECT, offer, legal_offer_range

    class MyCarrier(DecisionEngine):
        role = "carrier"
        engine_name = "MyCarrier"

        def next_move(self, state: TableState) -> Move:
            lo, hi = legal_offer_range(state)
            ...

Benchmark it:  `wehaveadeal benchmark path/to/my_carrier.py`
"""
__version__ = "0.1.0"

from wehaveadeal.agents.base import ACCEPT, REJECT, DecisionEngine, make_offer
from wehaveadeal.models import Action, Move, PortfolioTableSnapshot, TableState, TurnRecord
from wehaveadeal.toolkit.helpers import clamp_offer, legal_offer_range

offer = make_offer

__all__ = [
    "__version__",
    "ACCEPT",
    "REJECT",
    "Action",
    "DecisionEngine",
    "Move",
    "PortfolioTableSnapshot",
    "TableState",
    "TurnRecord",
    "clamp_offer",
    "legal_offer_range",
    "make_offer",
    "offer",
]
