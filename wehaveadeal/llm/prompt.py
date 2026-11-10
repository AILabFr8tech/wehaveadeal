"""Prompt construction for the LLM-backed engines (`wehaveadeal/agents/carrier_llm.py`,
`wehaveadeal/agents/shipper_llm.py`).

Two explicit design choices, both made directly by the user rather than
guessed at, worth recording here since they shape every prompt this module
builds:

1. **Full disclosure of the hard band** ("topes y pisos"): the LLM is told
   its exact legal floor/ceiling in dollars, the same numbers `OPT2` and
   every other hand-designed engine compute internally. This is not the
   "natural/human-like" framing that was on the table earlier in the
   project's design discussion -- it was explicitly resolved in favor of
   full disclosure, so there's no ambiguity left to ablate over.
2. **Per-table memory only**: the prompt includes this table's own full
   turn history (every offer either side has made), matching exactly what
   `TableState.history` already gives every other engine -- nothing more
   (no memory across different negotiations, no visibility into any rival's
   offers, no portfolio view). Information asymmetry stays structural, the
   same way it does for every hand-designed/ML/RL engine in this package.

One thing the raw `TableState` fields do NOT convey on their own, and which
a human or a hand-designed engine gets "for free" from how this whole
package is built (Gale-Shapley, `wehaveadeal/matching.py`): that closing a deal
does not guarantee winning it -- the counterpart only keeps the best offer
among everyone who reaches a deal with them. Since the LLM cannot infer
this from one turn's state and cannot observe a rival's offer either (that
would be actual information leakage, unlike this being told the rule),
the system prompt states the RULE in the abstract (competition exists, be
competitive) without ever naming or quantifying any specific rival -- the
same distinction `wehaveadeal/rl/league_env.py`'s reward already encodes
implicitly for RL2, made explicit in words here because an LLM has no
training loop to learn it from repeated play.
"""
from __future__ import annotations

from wehaveadeal.models import Action, TableState

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "One or two sentences of reasoning, plain text."},
        "action": {"type": "string", "enum": ["offer", "accept", "reject"]},
        "amount": {
            "type": ["number", "null"],
            "description": "Required and only meaningful when action is 'offer'. Must be null otherwise.",
        },
    },
    "required": ["action", "amount"],
    # Tried making the amount/action link structural via if/then/else (a flat schema
    # lets a model emit action=offer with amount=null, since amount is nullable
    # unconditionally, and small models do exactly that): Ollama's structured-output
    # engine accepts if/then/else without error but does not actually enforce it
    # (confirmed both in isolation and via a real pilot re-run -- forfeit rate was
    # unchanged/noise, null-amount offers unchanged). Not worth the added complexity.
}

_CARRIER_RULES = """\
You are an AI agent playing the CARRIER (transportation provider) in a bilateral price \
negotiation against one SHIPPER (customer) over the price of a single shipment.

HARD RULES (breaking these forfeits this table automatically):
- Every offer you make must be a number between ${worst:.2f} (your absolute floor -- you \
may never go below this) and ${best:.2f} (your absolute ceiling -- the best price you \
could possibly get).
- Each new offer you make must never be HIGHER than your own previous offer. You may only \
hold your price or lower it, turn after turn -- never raise it back up.
- You may ACCEPT the shipper's current offer only if it is at or above ${worst:.2f}.
- You have {turns_per_side} turns total. If this is your last turn and the shipper's \
current offer is legal for you (at or above ${worst:.2f}), you must accept it -- running \
out of turns without a deal is the worst outcome for you.

COMPETITION (important, and NOT visible to you directly): you are not the only carrier \
this shipper may be talking to. If you and this shipper reach a deal, the shipper still \
keeps only the CHEAPEST deal among every carrier who reached one with them -- you can lose \
the job even after agreeing on a price, if some other carrier (whose offers you cannot see) \
was cheaper. Holding out for your absolute best price every single time is not necessarily \
the winning strategy if it costs you the deal entirely.

Your goal: close as high a price as you legally can, while staying competitive enough to \
actually be the one the shipper picks.
"""

_SHIPPER_RULES = """\
You are an AI agent playing the SHIPPER (customer) in a bilateral price negotiation \
against one CARRIER (transportation provider) over the price of a single shipment.

HARD RULES (breaking these forfeits this table automatically):
- Every offer you make must be a number between ${worst:.2f} (your absolute ceiling -- you \
may never pay more than this) and ${best:.2f} (your absolute floor -- the best, lowest \
price you could possibly get).
- Each new offer you make must never be LOWER than your own previous offer. You may only \
hold your price or raise it, turn after turn -- never lower it back down.
- You may ACCEPT the carrier's current offer only if it is at or below ${worst:.2f}.
- You have {turns_per_side} turns total. If this is your last turn and the carrier's \
current offer is legal for you (at or below ${worst:.2f}), you must accept it -- running \
out of turns without a deal is the worst outcome for you.

COMPETITION (important, and NOT visible to you directly): you are not the only shipper \
this carrier may be talking to. If you and this carrier reach a deal, the carrier still \
keeps only the MOST EXPENSIVE deal among every shipper who reached one with them -- you \
can lose the carrier even after agreeing on a price, if some other shipper (whose offers \
you cannot see) offered more. Holding out for your absolute best price every single time \
is not necessarily the winning strategy if it costs you the deal entirely.

Your goal: close as low a price as you legally can, while staying competitive enough to \
actually be the one the carrier picks.
"""

_OUTPUT_INSTRUCTIONS = """\
Respond with ONLY a JSON object matching this exact shape:
  {{"reasoning": "<1-2 sentences>", "action": "offer" | "accept" | "reject", "amount": <number or null>}}
"amount" is required (a number) when action is "offer", and must be null for "accept" or \
"reject". Do not include anything outside the JSON object.
"""


def _format_history(state: TableState) -> str:
    if not state.history:
        return "(no offers have been made yet -- this is the opening move of the table)"
    me = state.role
    lines = []
    for rec in state.history:
        who = "you" if rec.actor == me else "the counterpart"
        if rec.action == Action.OFFER:
            lines.append(f"Turn {rec.turn}: {who} offered ${rec.amount:.2f}")
        elif rec.action == Action.ACCEPT:
            lines.append(f"Turn {rec.turn}: {who} accepted")
        else:
            lines.append(f"Turn {rec.turn}: {who} walked away")
    return "\n".join(lines)


def build_messages(state: TableState, role: str, worst: float, best: float) -> list[dict]:
    """Chat messages (system + user) for one decision point. `worst`/`best`
    are this role's band anchors from `wehaveadeal.ml.features.role_band` --
    passed in rather than recomputed here so there is exactly one place in
    the codebase that owns the (worst, best) formula per role."""
    rules_template = _CARRIER_RULES if role == "carrier" else _SHIPPER_RULES
    system = rules_template.format(worst=worst, best=best, turns_per_side=state.turns_per_side) + "\n" + _OUTPUT_INSTRUCTIONS

    turn_index = state.my_turns_used + 1
    current_offer = state.opp_last_offer
    current_offer_line = (
        f"The counterpart's current offer on the table is ${current_offer:.2f}."
        if current_offer is not None
        else "There is no offer currently on the table from the counterpart."
    )
    user = f"""\
This is turn {state.turn} overall (your decision #{turn_index} of {state.turns_per_side}).

Full history of this table so far:
{_format_history(state)}

{current_offer_line}

What do you do?
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
