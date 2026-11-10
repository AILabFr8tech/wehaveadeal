"""The community benchmark: one entrant agent vs the reference architectures.

Protocol (version PROTOCOL_VERSION; see docs/PROTOCOL.md for the rationale)
--------------------------------------------------------------------------
The entrant joins the market on its own side, competing against the
reference architectures playing the same role, while the reference
architectures also fill the opposite side. Three market regimes, mirroring
the paper's controlled tournament (B = number of baseline kinds, 9 by default):

  regime            entrant's side size      opposite side size
  balanced          B   (entrant + B-1)      B
  <own>_scarce      B-2 (entrant + B-3)      B
  <other>_scarce    B   (entrant + B-1)      B-2

("carrier_scarce" / "shipper_scarce" name the role that is scarce, as in
the paper -- so for a carrier entrant, carrier_scarce is the own-scarce one.)

Each regime runs `n_subrounds` sub-rounds of `n_negotiations` market days.
Which baselines sit out each sub-round is drawn from a seeded RNG that
depends ONLY on (seed, regime, sub-round) -- never on the entrant -- so
every entrant ever benchmarked with the same seed and preset faces exactly
the same markets and the same rival line-ups (common random numbers). That
is what makes leaderboard entries comparable.

Statistics: the entrant's per-day outcome (carrier margin or shipper
profit) is compared with each same-side baseline on the days both were in
the market (paired differences), with a bootstrap CI on the mean
difference and a Wilcoxon signed-rank test, Holm-corrected across the
baselines of each regime.
"""
from __future__ import annotations

import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Type, Union

import numpy as np
from pydantic import BaseModel, Field

from wehaveadeal import __version__
from wehaveadeal.agents.base import DecisionEngine
from wehaveadeal.simulate import CARRIER_KINDS, build_carrier, build_shipper, run_round
from wehaveadeal.stats import bootstrap_ci, paired_wilcoxon
from wehaveadeal.toolkit.guard import DEFAULT_MOVE_TIMEOUT_S, GuardedAgent
from wehaveadeal.toolkit.loader import AgentSpec, _to_spec

PROTOCOL_VERSION = "1.0"
BASELINE_KINDS: tuple[str, ...] = tuple(CARRIER_KINDS)  # the paper's nine: r rv is opt opt2 ml ml2 rl rl2
HAND_DESIGNED_KINDS: tuple[str, ...] = ("r", "rv", "is", "opt", "opt2")  # no trained weights -- fastest
REGIMES: tuple[str, ...] = ("balanced", "carrier_scarce", "shipper_scarce")
ARCH_LABELS = {k: k.upper() for k in BASELINE_KINDS}

# (n_subrounds, n_negotiations per sub-round). "standard" matches the paper's
# scarce-regime budget (36 sub-rounds x 12 days = 432 market days per regime).
PRESETS: Dict[str, tuple[int, int]] = {
    "smoke": (1, 2),
    "quick": (6, 5),
    "standard": (36, 12),
}
ALPHA = 0.05


# --------------------------------------------------------------------------- report schema


class AgentInfo(BaseModel):
    name: str
    role: str
    source: str
    source_sha256: Optional[str] = None


class Settings(BaseModel):
    protocol_version: str = PROTOCOL_VERSION
    preset: str
    seed: int
    n_subrounds: int
    n_negotiations: int
    baselines: List[str]
    regimes: List[str]
    move_timeout_s: Optional[float]


class ArchStats(BaseModel):
    arch: str
    n_days: int
    match_rate: float
    mean_outcome: float
    ci_lo: float
    ci_hi: float


class PairedComparison(BaseModel):
    baseline: str
    n_days: int
    mean_diff: float  # entrant minus baseline, same days
    ci_lo: float
    ci_hi: float
    p_raw: float
    p_holm: float
    verdict: str  # "better" | "worse" | "no significant difference" | "insufficient data"


class RegimeResult(BaseModel):
    regime: str
    own_side_size: int
    other_side_size: int
    n_market_days: int
    entrant: ArchStats
    entrant_rank: int  # 1 = best mean outcome among entrant + same-side baselines
    n_ranked: int
    baselines: List[ArchStats]
    vs_baselines: List[PairedComparison]


class ViolationSummary(BaseModel):
    total_moves: int
    total_violations: int
    forfeit_rate: float
    counts: Dict[str, int] = Field(default_factory=dict)
    examples: List[dict] = Field(default_factory=list)


class Timing(BaseModel):
    wall_s: float
    mean_think_ms: float
    max_think_ms: float


class Headline(BaseModel):
    mean_outcome: float  # average over regimes of the entrant's mean outcome
    mean_rank: float  # average over regimes of the entrant's rank
    wins: int  # significant "better" verdicts, all regimes
    losses: int  # significant "worse" verdicts, all regimes


class BenchmarkReport(BaseModel):
    toolkit_version: str
    agent: AgentInfo
    settings: Settings
    headline: Headline
    regimes: List[RegimeResult]
    violations: ViolationSummary
    timing: Timing
    environment: Dict[str, str]
    generated_at: str

    def results_fingerprint(self) -> dict:
        """The deterministic part of the report (everything but timing/timestamps)."""
        return self.model_dump(include={"agent", "settings", "headline", "regimes", "violations"})


# --------------------------------------------------------------------------- protocol


def regime_sizes(regime: str, role: str, n_baselines: int) -> tuple[int, int]:
    """(own_side_size including the entrant, other_side_size)."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")
    full, scarce = n_baselines, max(2, n_baselines - 2)
    if regime == "balanced":
        return full, full
    scarce_role = regime.split("_")[0]
    return (scarce, full) if scarce_role == role else (full, scarce)


def _lineup(seed: int, regime_idx: int, subround: int, baselines: Sequence[str], own_n: int, other_n: int):
    """Which baselines play this sub-round. Depends only on (seed, regime, sub-round)."""
    rng = np.random.default_rng([seed, regime_idx, subround])
    idx = np.arange(len(baselines))
    own = sorted(rng.choice(idx, size=own_n - 1, replace=False).tolist())
    other = sorted(rng.choice(idx, size=other_n, replace=False).tolist())
    market_seed = int(np.random.SeedSequence([seed, regime_idx, subround]).generate_state(1)[0])
    return [baselines[i] for i in own], [baselines[i] for i in other], market_seed


def _holm(p_values: List[float]) -> List[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted.tolist()


def _stats(arch: str, rows: List[dict]) -> ArchStats:
    outcomes = np.array([r["outcome"] for r in rows], dtype=float)
    mean, lo, hi = bootstrap_ci(outcomes)
    return ArchStats(
        arch=arch,
        n_days=len(rows),
        match_rate=float(np.mean([r["matched"] for r in rows])) if rows else float("nan"),
        mean_outcome=mean,
        ci_lo=lo,
        ci_hi=hi,
    )


def as_spec(agent: Union[AgentSpec, Type[DecisionEngine]]) -> AgentSpec:
    if isinstance(agent, AgentSpec):
        return agent
    if isinstance(agent, type) and issubclass(agent, DecisionEngine):
        return _to_spec(agent, source=f"{agent.__module__}:{agent.__qualname__}", sha=None)
    raise TypeError(f"expected an AgentSpec or a DecisionEngine subclass, got {agent!r}")


ProgressFn = Callable[[str, int, int], None]


def run_benchmark(
    agent: Union[AgentSpec, Type[DecisionEngine]],
    preset: str = "quick",
    seed: int = 2026,
    regimes: Sequence[str] = REGIMES,
    baselines: Sequence[str] = BASELINE_KINDS,
    n_subrounds: Optional[int] = None,
    n_negotiations: Optional[int] = None,
    move_timeout_s: Optional[float] = DEFAULT_MOVE_TIMEOUT_S,
    save_rounds_dir: Optional[Path] = None,
    progress: Optional[ProgressFn] = None,
) -> BenchmarkReport:
    """Benchmark one entrant. `preset` sets (n_subrounds, n_negotiations); either can be
    overridden. Leaderboard submissions must use preset="standard" with the default
    seed, regimes and baselines -- see docs/LEADERBOARD.md."""
    spec = as_spec(agent)
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; expected one of {tuple(PRESETS)}")
    p_sub, p_neg = PRESETS[preset]
    n_subrounds = n_subrounds or p_sub
    n_negotiations = n_negotiations or p_neg
    baselines = list(baselines)
    unknown = [k for k in baselines if k not in BASELINE_KINDS]
    if unknown:
        raise ValueError(f"unknown baseline kinds {unknown}; expected a subset of {BASELINE_KINDS}")
    if len(baselines) < 3:
        raise ValueError("at least 3 baseline kinds are needed to form the market regimes")

    role, other_role = spec.role, ("shipper" if spec.role == "carrier" else "carrier")
    own_builder = build_carrier if role == "carrier" else build_shipper
    other_builder = build_shipper if role == "carrier" else build_carrier
    # Baselines are stateless between decisions (their randomness is derived from the
    # negotiation seed, table and turn), so one instance per kind is reused throughout --
    # trained weights load once per run instead of once per sub-round.
    own_pool = {k: own_builder(k, f"{role}-{k}-1") for k in baselines}
    other_pool = {k: other_builder(k, f"{other_role}-{k}-1") for k in baselines}
    entrant_id = f"{role}-{spec.kind}-1"
    entrant = GuardedAgent(spec.build(entrant_id), move_timeout_s=move_timeout_s)

    if save_rounds_dir is not None:
        save_rounds_dir = Path(save_rounds_dir)
        save_rounds_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    regime_results: List[RegimeResult] = []
    for regime_idx, regime in enumerate(REGIMES):
        if regime not in regimes:
            continue
        own_n, other_n = regime_sizes(regime, role, len(baselines))
        rows: List[dict] = []  # one per (market day, same-side agent)
        for sub in range(n_subrounds):
            if progress:
                progress(regime, sub, n_subrounds)
            own_kinds, other_kinds, market_seed = _lineup(seed, regime_idx, sub, baselines, own_n, other_n)
            own_side: Dict[str, DecisionEngine] = {entrant_id: entrant}
            own_side.update({own_pool[k].agent_id: own_pool[k] for k in own_kinds})
            other_side = {other_pool[k].agent_id: other_pool[k] for k in other_kinds}
            id_to_arch = {entrant_id: "ENTRANT", **{own_pool[k].agent_id: ARCH_LABELS[k] for k in own_kinds}}

            shippers, carriers = (other_side, own_side) if role == "carrier" else (own_side, other_side)
            result = run_round(
                round_index=sub,
                seed=market_seed,
                shipper_engines=shippers,
                carrier_engines=carriers,
                n_negotiations=n_negotiations,
            )
            if save_rounds_dir is not None:
                out = save_rounds_dir / f"bench_{spec.kind}_{regime}_{sub:03d}_seed_{market_seed}.json"
                out.write_text(result.model_dump_json(indent=2))

            for neg in result.negotiations:
                day = f"{regime}|{sub}|{neg.negotiation_index}"
                outcomes = neg.carrier_outcomes if role == "carrier" else neg.shipper_outcomes
                for o in outcomes:
                    aid = o.carrier_id if role == "carrier" else o.shipper_id
                    value = o.margin if role == "carrier" else o.profit
                    rows.append({"day": day, "arch": id_to_arch[aid], "matched": o.matched, "outcome": value})

        regime_results.append(_summarize_regime(regime, own_n, other_n, rows))

    wall = time.perf_counter() - t_start
    return BenchmarkReport(
        toolkit_version=__version__,
        agent=AgentInfo(name=spec.name, role=spec.role, source=spec.source, source_sha256=spec.source_sha256),
        settings=Settings(
            preset=preset,
            seed=seed,
            n_subrounds=n_subrounds,
            n_negotiations=n_negotiations,
            baselines=baselines,
            regimes=[r for r in REGIMES if r in regimes],
            move_timeout_s=move_timeout_s,
        ),
        headline=_headline(regime_results),
        regimes=regime_results,
        violations=_violation_summary(entrant),
        timing=Timing(
            wall_s=wall,
            mean_think_ms=1000 * entrant.total_think_s / max(1, entrant.n_moves),
            max_think_ms=1000 * entrant.max_think_s,
        ),
        environment={"python": platform.python_version(), "platform": platform.platform()},
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _summarize_regime(regime: str, own_n: int, other_n: int, rows: List[dict]) -> RegimeResult:
    by_arch: Dict[str, List[dict]] = {}
    for r in rows:
        by_arch.setdefault(r["arch"], []).append(r)
    entrant_rows = by_arch.pop("ENTRANT", [])
    entrant_stats = _stats("ENTRANT", entrant_rows)
    baseline_stats = [_stats(a, rs) for a, rs in sorted(by_arch.items())]

    entrant_by_day = {r["day"]: r["outcome"] for r in entrant_rows}
    raw = []
    for arch, arch_rows in sorted(by_arch.items()):
        pairs = [(entrant_by_day[r["day"]], r["outcome"]) for r in arch_rows if r["day"] in entrant_by_day]
        if len(pairs) < 2:
            raw.append((arch, len(pairs), float("nan"), float("nan"), float("nan"), 1.0))
            continue
        a, b = zip(*pairs)
        with warnings.catch_warnings():  # scipy warns on small samples; the verdict already guards n < 10
            warnings.simplefilter("ignore")
            mean_diff, lo, hi, p = paired_wilcoxon(a, b)
        raw.append((arch, len(pairs), mean_diff, lo, hi, 1.0 if np.isnan(p) else p))

    adjusted = _holm([r[5] for r in raw]) if raw else []
    comparisons = []
    for (arch, n, mean_diff, lo, hi, p), p_adj in zip(raw, adjusted):
        if n < 10:
            verdict = "insufficient data"
        elif p_adj < ALPHA:
            verdict = "better" if mean_diff > 0 else "worse"
        else:
            verdict = "no significant difference"
        comparisons.append(
            PairedComparison(baseline=arch, n_days=n, mean_diff=mean_diff, ci_lo=lo, ci_hi=hi, p_raw=p, p_holm=p_adj, verdict=verdict)
        )

    ranked = sorted([entrant_stats, *baseline_stats], key=lambda s: -s.mean_outcome)
    rank = next(i for i, s in enumerate(ranked, start=1) if s.arch == "ENTRANT")
    return RegimeResult(
        regime=regime,
        own_side_size=own_n,
        other_side_size=other_n,
        n_market_days=len(entrant_rows),
        entrant=entrant_stats,
        entrant_rank=rank,
        n_ranked=len(ranked),
        baselines=baseline_stats,
        vs_baselines=comparisons,
    )


def _headline(regimes: List[RegimeResult]) -> Headline:
    if not regimes:
        return Headline(mean_outcome=float("nan"), mean_rank=float("nan"), wins=0, losses=0)
    return Headline(
        mean_outcome=float(np.mean([r.entrant.mean_outcome for r in regimes])),
        mean_rank=float(np.mean([r.entrant_rank for r in regimes])),
        wins=sum(c.verdict == "better" for r in regimes for c in r.vs_baselines),
        losses=sum(c.verdict == "worse" for r in regimes for c in r.vs_baselines),
    )


def _violation_summary(entrant: GuardedAgent) -> ViolationSummary:
    counts: Dict[str, int] = {}
    for v in entrant.violations:
        counts[v.kind] = counts.get(v.kind, 0) + 1
    return ViolationSummary(
        total_moves=entrant.n_moves,
        total_violations=len(entrant.violations),
        forfeit_rate=len(entrant.violations) / max(1, entrant.n_moves),
        counts=counts,
        examples=[v.to_dict() for v in entrant.violations[:20]],
    )


def format_report(report: BenchmarkReport) -> str:
    """Human-readable summary for the terminal."""
    lines = [
        f"We Have a Deal benchmark -- {report.agent.name} ({report.agent.role}), "
        f"preset={report.settings.preset}, seed={report.settings.seed}, protocol v{report.settings.protocol_version}",
        "",
    ]
    for r in report.regimes:
        e = r.entrant
        lines.append(
            f"[{r.regime}]  {r.own_side_size} {report.agent.role}s vs {r.other_side_size}  |  "
            f"rank {r.entrant_rank}/{r.n_ranked}  |  mean outcome {e.mean_outcome:,.1f} "
            f"[{e.ci_lo:,.1f}, {e.ci_hi:,.1f}]  |  match rate {e.match_rate:.0%}  |  {e.n_days} days"
        )
        for c in r.vs_baselines:
            lines.append(f"    vs {c.baseline:<5} {c.mean_diff:>+10,.1f}  (n={c.n_days:>4}, p_holm={c.p_holm:.3f})  {c.verdict}")
        lines.append("")
    h, v = report.headline, report.violations
    lines.append(f"Headline: mean outcome {h.mean_outcome:,.1f}, mean rank {h.mean_rank:.2f}, {h.wins} wins / {h.losses} losses")
    lines.append(
        f"Violations: {v.total_violations} of {v.total_moves} moves forfeited ({v.forfeit_rate:.1%})"
        + (f" -- {v.counts}" if v.counts else "")
    )
    lines.append(f"Time: {report.timing.wall_s:.1f}s wall, {report.timing.mean_think_ms:.2f} ms/move mean, {report.timing.max_think_ms:.1f} ms max")
    return "\n".join(lines)
