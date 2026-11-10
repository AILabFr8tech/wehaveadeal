"""`wehaveadeal` command-line interface.

    wehaveadeal new-agent MyCarrier --role carrier     scaffold agents/my_carrier.py
    wehaveadeal list-agents                            what is in agents/ (and what failed to load)
    wehaveadeal validate agents/my_carrier.py          60-second smoke test: does it play legally?
    wehaveadeal benchmark agents/my_carrier.py         full benchmark vs the 9 reference architectures
    wehaveadeal serve                                  backend for the local dashboard
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from string import Template
from typing import List, Optional

from wehaveadeal import __version__

DEFAULT_AGENTS_DIR = "agents"
TEMPLATE_PATH = Path(__file__).resolve().parent / "toolkit" / "templates" / "agent.py.tmpl"


def _snake(name: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


def cmd_new_agent(args) -> int:
    name = args.name
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        print(f"error: {name!r} is not a valid Python class name (letters, digits, underscores).", file=sys.stderr)
        return 2
    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_snake(name)}.py"
    if path.exists() and not args.force:
        print(f"error: {path} already exists (use --force to overwrite).", file=sys.stderr)
        return 2
    code = Template(TEMPLATE_PATH.read_text(encoding="utf-8")).substitute(
        class_name=name, role=args.role, file_name=path.as_posix()
    )
    path.write_text(code, encoding="utf-8")
    print(f"Created {path}  ({args.role} agent {name})")
    print(f"Next:  wehaveadeal validate {path.as_posix()}")
    return 0


def cmd_list_agents(args) -> int:
    from wehaveadeal.toolkit.benchmark import BASELINE_KINDS
    from wehaveadeal.toolkit.loader import discover_agents

    found = discover_agents(args.dir)
    print(f"Reference architectures (both roles): {', '.join(k.upper() for k in BASELINE_KINDS)}")
    print(f"Community agents in {args.dir}/:")
    if not found.agents:
        print("  (none)")
    for a in found.agents:
        print(f"  {a.role:<8} {a.name:<24} kind={a.kind:<24} {a.source}")
    if found.errors:
        print("Failed to load:")
        for e in found.errors:
            print(f"  {e['file']}: {e['error']}")
    return 1 if found.errors else 0


def _load_or_exit(spec: str):
    from wehaveadeal.toolkit.loader import AgentLoadError, load_agent

    try:
        return load_agent(spec)
    except AgentLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


def cmd_validate(args) -> int:
    """Short run against the fast hand-designed baselines only. Exit code 0 = clean."""
    from wehaveadeal.toolkit.benchmark import HAND_DESIGNED_KINDS, run_benchmark

    spec = _load_or_exit(args.agent)
    print(f"Validating {spec.name} ({spec.role}) from {spec.source} ...")
    report = run_benchmark(
        spec,
        preset="smoke",
        seed=args.seed,
        baselines=HAND_DESIGNED_KINDS,
        n_subrounds=2,
        n_negotiations=3,
        move_timeout_s=args.move_timeout,
    )
    v = report.violations
    print(f"  {v.total_moves} moves played in {report.timing.wall_s:.1f}s "
          f"({report.timing.mean_think_ms:.2f} ms/move mean, {report.timing.max_think_ms:.1f} ms max)")
    deals = sum(round(r.entrant.match_rate * r.entrant.n_days) for r in report.regimes)
    days = sum(r.entrant.n_days for r in report.regimes)
    print(f"  matched on {deals} of {days} market days")
    if v.total_violations:
        print(f"  FAILED: {v.total_violations} forfeited moves {v.counts}")
        for ex in v.examples[:5]:
            print(f"    turn {ex['turn']} {ex['table_id']}: [{ex['kind']}] {ex['detail']}")
        return 1
    print("  OK: no illegal moves, exceptions or timeouts.")
    print(f"Next:  wehaveadeal benchmark {args.agent}")
    return 0


def cmd_benchmark(args) -> int:
    from wehaveadeal.toolkit.benchmark import BASELINE_KINDS, REGIMES, format_report, run_benchmark

    spec = _load_or_exit(args.agent)
    regimes = tuple(args.regimes.split(",")) if args.regimes else REGIMES
    baselines = tuple(args.baselines.split(",")) if args.baselines else BASELINE_KINDS

    def progress(regime: str, sub: int, total: int) -> None:
        if not args.quiet:
            print(f"\r  {regime:<15} sub-round {sub + 1}/{total}", end="", file=sys.stderr, flush=True)

    try:
        report = run_benchmark(
            spec,
            preset=args.preset,
            seed=args.seed,
            regimes=regimes,
            baselines=baselines,
            n_subrounds=args.subrounds,
            n_negotiations=args.negotiations,
            move_timeout_s=args.move_timeout,
            save_rounds_dir=Path(args.save_rounds) if args.save_rounds else None,
            progress=progress,
        )
    except ValueError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2
    if not args.quiet:
        print("", file=sys.stderr)
    print(format_report(report))
    out = Path(args.out) if args.out else Path("results") / f"{spec.kind}_{spec.role}_{args.preset}_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nReport written to {out}")
    if args.save_rounds:
        print(f"Round logs written to {args.save_rounds}/ -- open them in the dashboard's history view.")
    return 0


def cmd_serve(args) -> int:
    import os

    try:
        import uvicorn
    except ImportError:
        print("error: the dashboard backend needs `pip install wehaveadeal[server]`.", file=sys.stderr)
        return 2
    os.environ["WHAD_AGENTS_DIR"] = str(Path(args.agents_dir).resolve())
    print(f"Serving on http://{args.host}:{args.port}  (community agents from {os.environ['WHAD_AGENTS_DIR']})")
    uvicorn.run("wehaveadeal.api:app", host=args.host, port=args.port, reload=False)
    return 0


def cmd_schema(args) -> int:
    from wehaveadeal.toolkit.benchmark import BenchmarkReport

    print(json.dumps(BenchmarkReport.model_json_schema(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wehaveadeal", description="We Have a Deal research toolkit")
    p.add_argument("--version", action="version", version=f"wehaveadeal {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("new-agent", help="scaffold a new agent from the template")
    s.add_argument("name", help="class name, e.g. MyCarrier")
    s.add_argument("--role", choices=["carrier", "shipper"], required=True)
    s.add_argument("--dir", default=DEFAULT_AGENTS_DIR)
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_new_agent)

    s = sub.add_parser("list-agents", help="list community agents found in a folder")
    s.add_argument("--dir", default=DEFAULT_AGENTS_DIR)
    s.set_defaults(func=cmd_list_agents)

    timeout_help = "seconds an agent may take per move before it forfeits that table (default 5; raise it for LLM agents)"

    s = sub.add_parser("validate", help="quick legality smoke test for an agent")
    s.add_argument("agent", help="path/to/agent.py[:ClassName] or module:ClassName")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--move-timeout", type=float, default=5.0, help=timeout_help)
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("benchmark", help="benchmark an agent against the reference architectures")
    s.add_argument("agent", help="path/to/agent.py[:ClassName] or module:ClassName")
    s.add_argument("--preset", choices=["smoke", "quick", "standard"], default="quick",
                   help="quick for iterating; standard is the leaderboard protocol")
    s.add_argument("--seed", type=int, default=2026)
    s.add_argument("--regimes", help="comma-separated subset of balanced,carrier_scarce,shipper_scarce")
    s.add_argument("--baselines", help="comma-separated subset of r,rv,is,opt,opt2,ml,ml2,rl,rl2")
    s.add_argument("--subrounds", type=int, help="override the preset's sub-rounds per regime")
    s.add_argument("--negotiations", type=int, help="override the preset's market days per sub-round")
    s.add_argument("--move-timeout", type=float, default=5.0, help=timeout_help)
    s.add_argument("--out", help="report path (default results/<agent>_<role>_<preset>_seed<seed>.json)")
    s.add_argument("--save-rounds", metavar="DIR", help="also write every round log (e.g. to data/ for the dashboard)")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_benchmark)

    s = sub.add_parser("serve", help="run the live-negotiation API for the local dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--agents-dir", default=DEFAULT_AGENTS_DIR)
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("schema", help="print the JSON schema of benchmark reports")
    s.set_defaults(func=cmd_schema)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
