"""Community-facing layer: load agents, referee them, benchmark them."""
from wehaveadeal.toolkit.benchmark import (
    BASELINE_KINDS,
    HAND_DESIGNED_KINDS,
    PRESETS,
    PROTOCOL_VERSION,
    REGIMES,
    BenchmarkReport,
    format_report,
    run_benchmark,
)
from wehaveadeal.toolkit.guard import GuardedAgent, Violation, check_move
from wehaveadeal.toolkit.helpers import clamp_offer, legal_offer_range
from wehaveadeal.toolkit.loader import AgentLoadError, AgentSpec, DiscoveryResult, discover_agents, load_agent

__all__ = [
    "BASELINE_KINDS",
    "HAND_DESIGNED_KINDS",
    "PRESETS",
    "PROTOCOL_VERSION",
    "REGIMES",
    "AgentLoadError",
    "AgentSpec",
    "BenchmarkReport",
    "DiscoveryResult",
    "GuardedAgent",
    "Violation",
    "check_move",
    "clamp_offer",
    "discover_agents",
    "format_report",
    "legal_offer_range",
    "load_agent",
    "run_benchmark",
]
