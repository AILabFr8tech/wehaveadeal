"""The local dashboard backend with community agents dropped into its agents/ folder."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from wehaveadeal import api  # noqa: E402

from .conftest import GOOD_CARRIER, GOOD_SHIPPER  # noqa: E402


@pytest.fixture
def client(agent_dir, monkeypatch):
    monkeypatch.setenv("WHAD_AGENTS_DIR", str(agent_dir.path))
    return TestClient(api.app)


def _events(client, **params):
    with client.stream("GET", "/api/live/negotiation", params={"delay_ms": 0, **params}) as resp:
        assert resp.status_code == 200, resp.read()
        body = "".join(resp.iter_text())
    return [json.loads(line[len("data: "):]) for line in body.splitlines() if line.startswith("data: ")]


def test_engine_status_lists_community_agents_and_errors(client, agent_dir):
    agent_dir.write("good_carrier.py", GOOD_CARRIER)
    agent_dir.write("good_shipper.py", GOOD_SHIPPER)
    agent_dir.write("broken.py", "import nope_not_installed\n")
    data = client.get("/api/engine-status").json()
    assert data["community"]["carrier"] == [
        {"kind": "u_goodcarrier", "label": "GoodCarrier", "source": str(agent_dir.path / "good_carrier.py")}
    ]
    assert [a["kind"] for a in data["community"]["shipper"]] == ["u_goodshipper"]
    assert len(data["community_errors"]) == 1
    # Bundled weights -> every learned baseline is available out of the box.
    assert all(data["carrier"].values()) and all(data["shipper"].values())


def test_new_agent_file_appears_without_restart(client, agent_dir):
    assert client.get("/api/engine-status").json()["community"]["carrier"] == []
    agent_dir.write("good_carrier.py", GOOD_CARRIER)
    assert client.get("/api/engine-status").json()["community"]["carrier"][0]["kind"] == "u_goodcarrier"


def test_live_negotiation_with_community_agents_on_both_sides(client, agent_dir):
    agent_dir.write("good_carrier.py", GOOD_CARRIER)
    agent_dir.write("good_shipper.py", GOOD_SHIPPER)
    events = _events(client, seed=3, carriers="u_goodcarrier,opt2", shippers="u_goodshipper,r")
    assert events[0]["kind"] == "setup"
    assert events[0]["carrier_ids"] == ["carrier-u_goodcarrier-1", "carrier-opt2-1"]
    assert events[0]["shipper_ids"] == ["shipper-u_goodshipper-1", "shipper-r-1"]
    assert events[-1]["kind"] == "result"
    assert any(e["kind"] == "turn" and e["carrier_id"] == "carrier-u_goodcarrier-1" for e in events)


def test_crashing_community_agent_does_not_kill_the_stream(client, agent_dir):
    agent_dir.write("crash.py", '''
        from wehaveadeal import DecisionEngine

        class Crash(DecisionEngine):
            role = "carrier"
            def next_move(self, state):
                raise RuntimeError("bug in user code")
    ''')
    events = _events(client, seed=1, carriers="u_crash,r", shippers="r,r")
    assert events[-1]["kind"] == "result"
    crash_turns = [
        e for e in events
        if e["kind"] == "turn" and e["carrier_id"] == "carrier-u_crash-1" and e["actor"] == "carrier"
    ]
    assert crash_turns and all(e["action"] == "reject" for e in crash_turns)


def test_unknown_or_wrong_role_kinds_are_rejected(client, agent_dir):
    agent_dir.write("good_carrier.py", GOOD_CARRIER)
    r = client.get("/api/live/negotiation", params={"carriers": "u_nobody", "shippers": "r"})
    assert r.status_code == 400
    # A carrier agent cannot be seated on the shipper side.
    r = client.get("/api/live/negotiation", params={"carriers": "r", "shippers": "u_goodcarrier"})
    assert r.status_code == 400


def test_baseline_only_negotiation_still_works(client):
    events = _events(client, seed=9, carriers="r,rv,ml2", shippers="r,rl2")
    assert events[-1]["kind"] == "result"
