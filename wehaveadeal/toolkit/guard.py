"""GuardedAgent: the referee between a community agent and the arena.

The arena itself treats an illegal move as fatal (`IllegalMoveError`), which is
right for the reference engines but wrong for a benchmark: one bug in one
community agent must not abort a 1,000-negotiation run. GuardedAgent applies
the protocol's own rule instead -- an illegal action forfeits the table -- and
records why, so the report can show it:

  * the agent raised an exception                      -> "exception"
  * it returned something that is not a Move           -> "invalid_return"
  * it broke the hard band / monotonic concession rule -> "illegal_move"
  * it accepted with no counter-offer on the table     -> "illegal_accept"
  * it took longer than `move_timeout_s` to answer     -> "timeout"

Forfeit = the guard plays REJECT on that table (an impasse); the agent keeps
playing every other table. The timeout is enforced after the fact (Python
cannot safely interrupt a running call), so a slow agent still costs wall-clock
time but never gains from it.

The agent also receives a deep copy of the TableState, so it cannot tamper
with the arena's own bookkeeping.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass
from typing import List, Optional

from wehaveadeal.agents.base import REJECT, DecisionEngine
from wehaveadeal.models import Action, Move, TableState
from wehaveadeal.simulate import IllegalMoveError, _validate_move

DEFAULT_MOVE_TIMEOUT_S = 5.0


@dataclass
class Violation:
    kind: str
    agent_id: str
    table_id: str
    turn: int
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


def check_move(move: object, state: TableState) -> Optional[tuple[str, str]]:
    """(violation_kind, detail) if `move` is not a legal answer to `state`, else None."""
    if not isinstance(move, Move):
        return "invalid_return", f"next_move returned {type(move).__name__}, expected a Move"
    if move.action == Action.ACCEPT and state.opp_last_offer is None:
        return "illegal_accept", "accepted with no counter-offer on the table"
    try:
        _validate_move(move, state)
    except IllegalMoveError as exc:
        return "illegal_move", str(exc)
    return None


class GuardedAgent(DecisionEngine):
    def __init__(self, inner: DecisionEngine, move_timeout_s: Optional[float] = DEFAULT_MOVE_TIMEOUT_S) -> None:
        super().__init__(inner.agent_id, frozen=inner.frozen)
        self.inner = inner
        self.engine_name = getattr(inner, "engine_name", type(inner).__name__)
        self.version = getattr(inner, "version", "v1")
        self.move_timeout_s = move_timeout_s
        self.violations: List[Violation] = []
        self.n_moves = 0
        self.total_think_s = 0.0
        self.max_think_s = 0.0

    def _forfeit(self, kind: str, state: TableState, detail: str) -> Move:
        self.violations.append(Violation(kind, self.agent_id, state.table_id, state.turn, detail))
        return REJECT

    def next_move(self, state: TableState) -> Move:
        self.n_moves += 1
        t0 = time.perf_counter()
        try:
            move = self.inner.next_move(state.model_copy(deep=True))
        except Exception as exc:  # noqa: BLE001 -- community code; any failure forfeits the table
            tb = traceback.format_exception_only(type(exc), exc)[-1].strip()
            return self._forfeit("exception", state, tb)
        finally:
            elapsed = time.perf_counter() - t0
            self.total_think_s += elapsed
            self.max_think_s = max(self.max_think_s, elapsed)

        if self.move_timeout_s is not None and elapsed > self.move_timeout_s:
            return self._forfeit("timeout", state, f"took {elapsed:.2f}s (limit {self.move_timeout_s}s)")
        problem = check_move(move, state)
        if problem is not None:
            return self._forfeit(problem[0], state, problem[1])
        return move

    def observe_negotiation_result(self, **kwargs) -> None:
        try:
            self.inner.observe_negotiation_result(**kwargs)
        except Exception:  # noqa: BLE001
            pass
