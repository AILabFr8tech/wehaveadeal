"""DecisionEngine: the one interface every agent (shipper or carrier, any
architecture) must implement (WHAD spec §11.1 / Strategy spec §10.0).

  DecisionEngine.next_move(table_state) -> Offer | Accept | Reject

New engines (ISMCTS, PPO, LLM-baseline in later phases) plug in here without
touching the arena, matching, or scoring code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from wehaveadeal.models import Action, Move, TableState


class DecisionEngine(ABC):
    engine_name: str = "base"
    version: str = "v1"

    def __init__(self, agent_id: str, frozen: bool = True) -> None:
        self.agent_id = agent_id
        # Frozen/evaluation-mode flag (WHAD spec §9.3): R and RV have no trainable
        # parameters, so this is a no-op for Phase 1, but every engine carries
        # the flag for forward compatibility with future learned agents.
        self.frozen = frozen

    @abstractmethod
    def next_move(self, state: TableState) -> Move:
        ...

    def observe_negotiation_result(self, **kwargs) -> None:
        """Hook for online within-round adaptation (WHAD spec §9.1). No-op by
        default; R/RV do not override it since they have no adaptive state."""
        return None

    @property
    def identifier(self) -> str:
        return f"{self.agent_id}:{self.engine_name}:{self.version}"


def make_offer(amount: float) -> Move:
    return Move(action=Action.OFFER, amount=amount)


ACCEPT = Move(action=Action.ACCEPT)
REJECT = Move(action=Action.REJECT)
