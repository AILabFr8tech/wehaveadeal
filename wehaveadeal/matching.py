"""Gale-Shapley deferred acceptance, shippers propose.

See the paper (docs/PROTOCOL.md) "Matching" for the full rule set and the
required regression test's worked example. Key rules:
  - Only CLOSED tables enter preference lists; impasse pairs are excluded
    entirely (never appear anywhere, not even as a last resort).
  - Shipper preference: carriers ranked by rate ASCENDING (cheapest first).
  - Carrier preference: shippers ranked by rate DESCENDING (priciest first).
  - Tie-break (equal rate): fewer turns-to-close wins, then a pseudo-random
    order seeded by the negotiation's seed.
"""
from __future__ import annotations

import random
from collections import defaultdict, deque
from typing import Dict, List, Tuple

from wehaveadeal.models import MatchEntry, MatchingResult, TableResult


def _tiebreak(seed: int, a_id: str, b_id: str) -> float:
    return random.Random(f"{seed}:{a_id}:{b_id}").random()


def _closed_offers(tables: List[TableResult]) -> Tuple[Dict[str, list], Dict[str, list]]:
    shipper_offers: Dict[str, list] = defaultdict(list)
    carrier_offers: Dict[str, list] = defaultdict(list)
    for t in tables:
        if not t.closed or t.rate is None or t.turns_to_close is None:
            continue  # impasse: excluded entirely from both preference lists
        shipper_offers[t.shipper_id].append((t.carrier_id, t.rate, t.turns_to_close))
        carrier_offers[t.carrier_id].append((t.shipper_id, t.rate, t.turns_to_close))
    return shipper_offers, carrier_offers


def gale_shapley(
    tables: List[TableResult],
    all_shipper_ids: List[str],
    seed: int,
) -> MatchingResult:
    shipper_offers, carrier_offers = _closed_offers(tables)

    shipper_prefs: Dict[str, List[Tuple[str, float, int]]] = {
        s: sorted(
            items,
            key=lambda item: (item[1], item[2], _tiebreak(seed, s, item[0])),
        )
        for s, items in shipper_offers.items()
    }
    carrier_prefs: Dict[str, List[Tuple[str, float, int]]] = {
        c: sorted(
            items,
            key=lambda item: (-item[1], item[2], _tiebreak(seed, c, item[0])),
        )
        for c, items in carrier_offers.items()
    }
    carrier_rank: Dict[str, Dict[str, int]] = {
        c: {s_id: idx for idx, (s_id, _, _) in enumerate(prefs)} for c, prefs in carrier_prefs.items()
    }

    next_proposal_idx = {s: 0 for s in shipper_prefs}
    shipper_matched: Dict[str, str] = {}
    carrier_held: Dict[str, str] = {}
    free_shippers = deque(shipper_prefs.keys())

    while free_shippers:
        s = free_shippers.popleft()
        prefs = shipper_prefs.get(s, [])
        idx = next_proposal_idx[s]
        if idx >= len(prefs):
            continue  # exhausted its preference list: stays unmatched
        carrier_id, _, _ = prefs[idx]
        next_proposal_idx[s] += 1

        if carrier_id not in carrier_held:
            carrier_held[carrier_id] = s
            shipper_matched[s] = carrier_id
        else:
            held_shipper = carrier_held[carrier_id]
            rank = carrier_rank[carrier_id]
            if rank[s] < rank[held_shipper]:
                carrier_held[carrier_id] = s
                shipper_matched[s] = carrier_id
                del shipper_matched[held_shipper]
                free_shippers.append(held_shipper)
            else:
                free_shippers.append(s)

    matches: List[MatchEntry] = []
    for s, c in shipper_matched.items():
        rate, turns = next((r, t) for (cid, r, t) in shipper_offers[s] if cid == c)
        matches.append(MatchEntry(shipper_id=s, carrier_id=c, rate=rate, turns_to_close=turns))

    unmatched = [s for s in all_shipper_ids if s not in shipper_matched]

    return MatchingResult(
        matches=matches,
        unmatched_shippers=unmatched,
        shipper_preferences={s: [c for c, _, _ in prefs] for s, prefs in shipper_prefs.items()},
        carrier_preferences={c: [s for s, _, _ in prefs] for c, prefs in carrier_prefs.items()},
    )
