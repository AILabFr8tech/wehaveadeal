"""End-to-end: what a newcomer does on day one, through the real CLI.

    pip install wehaveadeal
    wehaveadeal new-agent MyCarrier --role carrier
    wehaveadeal validate agents/my_carrier.py
    wehaveadeal benchmark agents/my_carrier.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from wehaveadeal.cli import main
from wehaveadeal.toolkit.benchmark import BenchmarkReport
from wehaveadeal.toolkit.loader import load_agent

from .conftest import REPO_ROOT


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("role,name", [("carrier", "MyCarrier"), ("shipper", "MyShipper")])
def test_scaffold_validate_benchmark(in_tmp: Path, role, name, capsys):
    assert main(["new-agent", name, "--role", role]) == 0
    path = in_tmp / "agents" / f"my_{role}.py"
    assert path.exists()

    spec = load_agent(str(path))
    assert spec.name == name and spec.role == role

    assert main(["validate", str(path)]) == 0
    assert "OK: no illegal moves" in capsys.readouterr().out

    out = in_tmp / "report.json"
    rc = main([
        "benchmark", str(path), "--preset", "smoke", "--baselines", "r,rv,is,opt,opt2",
        "--subrounds", "1", "--negotiations", "2", "--out", str(out), "--quiet",
    ])
    assert rc == 0
    report = BenchmarkReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert report.agent.name == name
    assert [r.regime for r in report.regimes] == ["balanced", "carrier_scarce", "shipper_scarce"]
    assert report.violations.total_violations == 0
    assert report.agent.source_sha256 and len(report.agent.source_sha256) == 64


def test_new_agent_refuses_to_overwrite_and_bad_names(in_tmp: Path):
    assert main(["new-agent", "Foo", "--role", "carrier"]) == 0
    assert main(["new-agent", "Foo", "--role", "carrier"]) == 2
    assert main(["new-agent", "Foo", "--role", "carrier", "--force"]) == 0
    assert main(["new-agent", "not-a-class", "--role", "carrier"]) == 2


def test_list_agents_reports_found_and_broken(in_tmp: Path, capsys):
    main(["new-agent", "Alpha", "--role", "carrier"])
    (in_tmp / "agents" / "broken.py").write_text("this is not python\n", encoding="utf-8")
    rc = main(["list-agents"])
    out = capsys.readouterr().out
    assert "Alpha" in out and "broken.py" in out
    assert rc == 1  # non-zero so CI catches a broken agent file


def test_validate_fails_loudly_for_an_illegal_agent(in_tmp: Path, capsys):
    path = in_tmp / "greedy.py"
    path.write_text(
        "from wehaveadeal import DecisionEngine, offer\n"
        "class Greedy(DecisionEngine):\n"
        "    role = 'carrier'\n"
        "    def next_move(self, state):\n"
        "        return offer(state.refprice_mean * 10)  # far above the hard band\n",
        encoding="utf-8",
    )
    assert main(["validate", str(path)]) == 1
    out = capsys.readouterr().out
    assert "FAILED" in out and "illegal_move" in out


def test_benchmark_rejects_unknown_baseline(in_tmp: Path, capsys):
    main(["new-agent", "Alpha", "--role", "carrier"])
    rc = main(["benchmark", "agents/alpha.py", "--baselines", "r,rv,nope", "--quiet"])
    assert rc == 2
    assert "unknown baseline" in capsys.readouterr().err


def test_unloadable_agent_exits_with_clear_message(in_tmp: Path, capsys):
    (in_tmp / "norole.py").write_text(
        "from wehaveadeal import DecisionEngine, REJECT\n"
        "class NoRole(DecisionEngine):\n"
        "    def next_move(self, state):\n"
        "        return REJECT\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        main(["validate", "norole.py"])
    assert exc.value.code == 2
    assert 'role = "carrier"' in capsys.readouterr().err


def test_cli_runs_as_a_module_from_any_directory(tmp_path: Path):
    """`python -m wehaveadeal` from a folder that is not the repo: the bundled
    baseline weights must still be found."""
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    scaffold = subprocess.run(
        [sys.executable, "-m", "wehaveadeal", "new-agent", "Cli", "--role", "shipper"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )
    assert scaffold.returncode == 0, scaffold.stderr
    bench = subprocess.run(
        [sys.executable, "-m", "wehaveadeal", "benchmark", "agents/cli.py", "--preset", "smoke",
         "--baselines", "r,ml2,rl2", "--subrounds", "1", "--negotiations", "1", "--quiet",
         "--out", "r.json"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600,
    )
    assert bench.returncode == 0, bench.stderr
    data = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert data["settings"]["baselines"] == ["r", "ml2", "rl2"]


def test_schema_command_prints_report_schema(capsys):
    assert main(["schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["title"] == "BenchmarkReport"
    assert "headline" in schema["properties"]
