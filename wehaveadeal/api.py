"""Live-negotiation streaming API.

A small FastAPI service whose only job is to run ONE fresh negotiation
through `wehaveadeal.simulate.stream_negotiation` and forward each turn event to
the dashboard over Server-Sent Events (SSE), with a short pause between
turns so a human can actually watch it unfold -- the engine itself computes
a negotiation in microseconds, so the pacing here is purely presentational.

Community agents: every agent found in the folder named by the
`WHAD_AGENTS_DIR` environment variable (default `./agents`) is offered next to
the nine reference architectures, under its own kind (`u_<name>`). The folder
is rescanned on every request, so dropping in or editing a file needs no
restart. Community agents always run behind `GuardedAgent`: an illegal move,
an exception or a timeout forfeits that table instead of killing the stream.

This is a LOCAL tool: it executes the Python files in the agents folder.
No auth, `allow_origins=["*"]` -- never expose it on a network with
untrusted clients, and never point it at agent files you have not read.

Run with:
    wehaveadeal serve            (or: uvicorn wehaveadeal.api:app --port 8000)
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from wehaveadeal.agents.carrier_ml import DEFAULT_MODEL_PATH as CARRIER_ML_MODEL_PATH
from wehaveadeal.agents.carrier_ml2 import DEFAULT_ACCEPT_MODEL_PATH as CARRIER_ML2_ACCEPT_MODEL_PATH
from wehaveadeal.agents.carrier_ml2 import DEFAULT_OFFER_MODEL_PATH as CARRIER_ML2_OFFER_MODEL_PATH
from wehaveadeal.agents.carrier_rl import DEFAULT_MODEL_PATH as CARRIER_RL_MODEL_PATH
from wehaveadeal.agents.carrier_rl2 import DEFAULT_MODEL_PATH as CARRIER_RL2_MODEL_PATH
from wehaveadeal.agents.shipper_ml import DEFAULT_MODEL_PATH as SHIPPER_ML_MODEL_PATH
from wehaveadeal.agents.shipper_ml2 import DEFAULT_ACCEPT_MODEL_PATH as SHIPPER_ML2_ACCEPT_MODEL_PATH
from wehaveadeal.agents.shipper_ml2 import DEFAULT_OFFER_MODEL_PATH as SHIPPER_ML2_OFFER_MODEL_PATH
from wehaveadeal.agents.shipper_rl import DEFAULT_MODEL_PATH as SHIPPER_RL_MODEL_PATH
from wehaveadeal.agents.shipper_rl2 import DEFAULT_MODEL_PATH as SHIPPER_RL2_MODEL_PATH
from wehaveadeal.market import MarketState
from wehaveadeal.models import RouteInfo
from wehaveadeal.simulate import CARRIER_KINDS, SHIPPER_KINDS, build_carrier, build_shipper, stream_negotiation
from wehaveadeal.toolkit.guard import GuardedAgent
from wehaveadeal.toolkit.loader import DiscoveryResult, discover_agents

# Live view pacing makes per-move latency visible anyway; LLM agents need headroom.
LIVE_MOVE_TIMEOUT_S = float(os.environ.get("WHAD_MOVE_TIMEOUT_S", "60"))


def _agents_dir() -> Path:
    return Path(os.environ.get("WHAD_AGENTS_DIR", "agents"))


def _community() -> DiscoveryResult:
    return discover_agents(_agents_dir())


def _community_kinds(community: DiscoveryResult, role: str) -> dict:
    return {a.kind: a for a in community.for_role(role)}

app = FastAPI(title="We Have a Deal -- live negotiation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


MAX_SHIPPERS = 8
MAX_CARRIERS = 8


def _parse_kinds(value: Optional[str], role: str, valid_kinds: tuple, default: tuple, max_count: int) -> tuple:
    if value is None:
        return default
    kinds = tuple(k.strip().lower() for k in value.split(","))
    if not (1 <= len(kinds) <= max_count):
        raise HTTPException(
            status_code=400, detail=f"expected between 1 and {max_count} {role} kinds, got {len(kinds)}: {kinds}"
        )
    for kind in kinds:
        if kind not in valid_kinds:
            raise HTTPException(status_code=400, detail=f"unknown {role} kind {kind!r}; expected one of {valid_kinds}")
    return kinds


def _parse_carrier_kinds(carriers: Optional[str], extra: tuple = ()) -> tuple:
    return _parse_kinds(carriers, "carrier", CARRIER_KINDS + tuple(extra), ("r", "r", "rv"), MAX_CARRIERS)


def _parse_shipper_kinds(shippers: Optional[str], extra: tuple = ()) -> tuple:
    return _parse_kinds(shippers, "shipper", SHIPPER_KINDS + tuple(extra), ("r", "r", "r", "r"), MAX_SHIPPERS)


def _build_side(kinds: tuple, role: str, community: dict) -> Dict:
    """Same id convention as simulate.default_agents ("<role>-<kind>-<n>"), plus community kinds."""
    builder = build_carrier if role == "carrier" else build_shipper
    counts: Dict[str, int] = {}
    engines = {}
    for kind in kinds:
        counts[kind] = counts.get(kind, 0) + 1
        agent_id = f"{role}-{kind}-{counts[kind]}"
        if kind in community:
            engines[agent_id] = GuardedAgent(community[kind].build(agent_id), move_timeout_s=LIVE_MOVE_TIMEOUT_S)
        else:
            engines[agent_id] = builder(kind, agent_id)
    return engines


TRAINED_MODEL_PATHS = {
    "carrier": {
        "ml": (CARRIER_ML_MODEL_PATH,),
        "ml2": (CARRIER_ML2_OFFER_MODEL_PATH, CARRIER_ML2_ACCEPT_MODEL_PATH),
        "rl": (CARRIER_RL_MODEL_PATH,),
        "rl2": (CARRIER_RL2_MODEL_PATH,),
    },
    "shipper": {
        "ml": (SHIPPER_ML_MODEL_PATH,),
        "ml2": (SHIPPER_ML2_OFFER_MODEL_PATH, SHIPPER_ML2_ACCEPT_MODEL_PATH),
        "rl": (SHIPPER_RL_MODEL_PATH,),
        "rl2": (SHIPPER_RL2_MODEL_PATH,),
    },
}
TRAIN_HINTS = {
    "ml": "wehaveadeal/ml/collect.py then python -m wehaveadeal.ml.train",
    "ml2": "wehaveadeal/ml/collect.py then python -m wehaveadeal.ml.train and python -m wehaveadeal.ml.train_accept",
    "rl": "python -m wehaveadeal.rl.train",
    "rl2": "python -m wehaveadeal.rl.train_league",
}


def _engine_status() -> dict:
    """A kind counts as available only if EVERY file it needs exists --
    "ml2" needs both the offer regressor and the accept classifier (see
    `wehaveadeal/agents/carrier_ml2.py`), so a half-trained ML2 (only one of the
    two) still reports as unavailable rather than crashing mid-negotiation."""
    return {
        role: {kind: all(Path(p).exists() for p in paths) for kind, paths in kinds.items()}
        for role, kinds in TRAINED_MODEL_PATHS.items()
    }


@app.get("/api/engine-status")
def engine_status() -> dict:
    """Whether a trained model exists for each kind that needs one ("ml",
    "ml2", "rl", "rl2") on each role, so the dashboard can disable those
    engine options instead of letting a user pick one and only find out
    mid-negotiation that there's nothing to load (see wehaveadeal/ml/ and
    wehaveadeal/rl/ to produce one)."""
    status = _engine_status()
    community = _community()
    status["community"] = {
        role: [{"kind": a.kind, "label": a.name, "source": a.source} for a in community.for_role(role)]
        for role in ("carrier", "shipper")
    }
    status["community_errors"] = community.errors
    return status


def _check_trained_kinds_available(kinds: tuple, role: str, status: dict) -> None:
    for kind in ("ml", "ml2", "rl", "rl2"):
        if kind in kinds and not status[role][kind]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{kind!r} was requested for {role} but no trained model exists yet. "
                    f"Run `{TRAIN_HINTS[kind]} --role {role} ...` first."
                ),
            )


async def _live_negotiation_events(
    seed: int, delay_ms: int, carrier_kinds: tuple, shipper_kinds: tuple, community: Optional[DiscoveryResult] = None
):
    community = community or DiscoveryResult()
    shipper_engines = _build_side(shipper_kinds, "shipper", _community_kinds(community, "shipper"))
    carrier_engines = _build_side(carrier_kinds, "carrier", _community_kinds(community, "carrier"))
    shipper_ids = list(shipper_engines.keys())
    carrier_ids = list(carrier_engines.keys())

    market = MarketState(seed, 1)
    neg_rng = market.negotiation_rng(0)
    routes: Dict[str, RouteInfo] = {}
    refprice_est: Dict[str, float] = {}
    for s in shipper_ids:
        route = market.assign_route(neg_rng)
        routes[s] = route
        eps = market.sample_shipper_epsilon(route.refprice_sigma, neg_rng)
        refprice_est[s] = route.refprice_mean + eps

    setup_event = {
        "kind": "setup",
        "seed": seed,
        "sale_price": market.sale_prices[0],
        "goods_cost": market.goods_costs[0],
        "shipper_ids": shipper_ids,
        "carrier_ids": carrier_ids,
        "routes": {s: routes[s].model_dump() for s in shipper_ids},
    }
    yield _sse(setup_event)
    await asyncio.sleep(delay_ms / 1000)

    for event in stream_negotiation(
        negotiation_index=0,
        seed=market.negotiation_seed_int(0),
        sale_price=market.sale_prices[0],
        goods_cost=market.goods_costs[0],
        shipper_engines=shipper_engines,
        carrier_engines=carrier_engines,
        routes=routes,
        refprice_est=refprice_est,
    ):
        if event["kind"] == "result":
            yield _sse({"kind": "result", "negotiation": event["negotiation"].model_dump(mode="json")})
        else:
            yield _sse(event)
            await asyncio.sleep(delay_ms / 1000)


@app.get("/api/live/negotiation")
async def live_negotiation(
    seed: Optional[int] = Query(default=None),
    delay_ms: int = Query(default=350, ge=0, le=3000),
    carriers: Optional[str] = Query(
        default=None,
        description=f"comma-separated list of 1-{MAX_CARRIERS} carrier kinds from {CARRIER_KINDS} or community kinds u_<name> (default: r,r,rv)",
    ),
    shippers: Optional[str] = Query(
        default=None,
        description=f"comma-separated list of 1-{MAX_SHIPPERS} shipper kinds from {SHIPPER_KINDS} or community kinds u_<name> (default: r,r,r,r)",
    ),
):
    """SSE stream of one fresh negotiation, turn by turn.

    Event shapes:
      {"kind": "setup", seed, sale_price, goods_cost, shipper_ids, carrier_ids, routes}
      {"kind": "turn", table_id, shipper_id, carrier_id, turn, actor, action, amount, closed, impasse}
      {"kind": "result", "negotiation": <NegotiationResult>}
    """
    community = _community()
    carrier_kinds = _parse_carrier_kinds(carriers, tuple(_community_kinds(community, "carrier")))
    shipper_kinds = _parse_shipper_kinds(shippers, tuple(_community_kinds(community, "shipper")))
    status = _engine_status()
    _check_trained_kinds_available(carrier_kinds, "carrier", status)
    _check_trained_kinds_available(shipper_kinds, "shipper", status)
    resolved_seed = seed if seed is not None else secrets.randbelow(2**31)
    return StreamingResponse(
        _live_negotiation_events(resolved_seed, delay_ms, carrier_kinds, shipper_kinds, community),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
