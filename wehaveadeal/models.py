"""Pydantic data models shared across the engine.

Naming follows the paper (docs/PROTOCOL.md)'s terminology mapping:
Tournament -> Round -> Negotiation -> Table -> Turn,
Shipper (buyer) / Carrier (seller), RefPrice (renamed from "API price").
"""
from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Role = Literal["shipper", "carrier"]


class Action(str, Enum):
    OFFER = "offer"
    ACCEPT = "accept"
    REJECT = "reject"  # deliberate walk-away, distinct from running out of turns


class Move(BaseModel):
    """What a DecisionEngine returns from next_move()."""

    action: Action
    amount: Optional[float] = None  # required and only meaningful when action == OFFER

    def model_post_init(self, __context) -> None:
        if self.action == Action.OFFER and self.amount is None:
            raise ValueError("OFFER moves must include an amount")


class TurnRecord(BaseModel):
    turn: int
    actor: Role
    action: Action
    amount: Optional[float] = None


class PortfolioTableSnapshot(BaseModel):
    """A snapshot of one of *my own* other concurrent tables in this negotiation.

    A carrier legitimately sees all its own tables (it is negotiating all of
    them itself); a shipper likewise sees all three of its own tables. Neither
    sees a rival's tables. See REVIEW_NOTES.md "Information asymmetry".
    """

    counterpart_id: str
    is_closed: bool
    is_impasse: bool
    closed_rate: Optional[float] = None
    latest_offer_on_table: Optional[float] = None


class TableState(BaseModel):
    """State handed to DecisionEngine.next_move() at one decision point.

    Fields are role-scoped by construction: the simulate.py loop only ever
    populates the fields a given role is allowed to observe, leaving the
    other side's private fields as None. This makes information hiding
    structural rather than a matter of agent self-discipline.
    """

    role: Role
    table_id: str
    negotiation_seed: int

    turn: int  # global turn about to be decided, 1-indexed
    turns_max: int  # global max turns across both sides (12 by default)
    my_turns_used: int
    opp_turns_used: int

    my_last_offer: Optional[float] = None
    opp_last_offer: Optional[float] = None  # what's currently on the table, if any
    history: List[TurnRecord] = Field(default_factory=list)

    sigma: float  # route sigma, known to both sides (Strategy spec §1.3)

    # shipper-only observables (None when role == "carrier")
    sale_price: Optional[float] = None
    goods_cost: Optional[float] = None
    refprice_est: Optional[float] = None  # RefPrice + epsilon

    # carrier-only observables (None when role == "shipper")
    refprice_mean: Optional[float] = None

    # own-portfolio visibility (both roles use this for occupancy/assignment proxies)
    my_portfolio: List[PortfolioTableSnapshot] = Field(default_factory=list)

    @property
    def turns_per_side(self) -> int:
        return self.turns_max // 2


class TableResult(BaseModel):
    shipper_id: str
    carrier_id: str
    closed: bool
    impasse: bool
    rate: Optional[float] = None
    turns_to_close: Optional[int] = None
    history: List[TurnRecord] = Field(default_factory=list)


class MatchEntry(BaseModel):
    shipper_id: str
    carrier_id: str
    rate: float
    turns_to_close: int


class MatchingResult(BaseModel):
    matches: List[MatchEntry] = Field(default_factory=list)
    unmatched_shippers: List[str] = Field(default_factory=list)
    # rate ranking per shipper/carrier, exposed for dashboard transparency
    shipper_preferences: dict = Field(default_factory=dict)  # shipper_id -> [carrier_id,...] asc by rate
    carrier_preferences: dict = Field(default_factory=dict)  # carrier_id -> [shipper_id,...] desc by rate


class ShipperNegotiationOutcome(BaseModel):
    shipper_id: str
    matched: bool
    carrier_id: Optional[str] = None
    rate: Optional[float] = None
    profit: float = 0.0
    declined_to_ship: bool = False
    points: float = 0.0


class CarrierNegotiationOutcome(BaseModel):
    carrier_id: str
    matched: bool
    shipper_id: Optional[str] = None
    rate: Optional[float] = None
    margin: float = 0.0


class RouteInfo(BaseModel):
    route_id: str
    refprice_mean: float
    refprice_sigma: float


class NegotiationResult(BaseModel):
    negotiation_index: int
    seed: int
    sale_price: float
    goods_cost: float
    routes: dict = Field(default_factory=dict)  # shipper_id -> RouteInfo
    tables: List[TableResult] = Field(default_factory=list)
    matching: MatchingResult
    shipper_outcomes: List[ShipperNegotiationOutcome] = Field(default_factory=list)
    carrier_outcomes: List[CarrierNegotiationOutcome] = Field(default_factory=list)


class RoundResult(BaseModel):
    round_index: int
    seed: int
    mode: Literal["frozen", "learning"] = "frozen"
    shipper_ids: List[str]
    carrier_ids: List[str]
    negotiations: List[NegotiationResult] = Field(default_factory=list)
    shipper_standings: dict = Field(default_factory=dict)  # shipper_id -> {points, profit}
    carrier_standings: dict = Field(default_factory=dict)  # carrier_id -> {margin, occupancy}
    round_winner_shipper: Optional[str] = None
    round_winner_carrier: Optional[str] = None
