"""Every way a user can point the toolkit at their agent -- and every way they can get it wrong."""
from __future__ import annotations

import sys

import pytest

from wehaveadeal.toolkit.loader import AgentLoadError, discover_agents, kind_for, load_agent

from .conftest import GOOD_CARRIER, GOOD_SHIPPER


def test_load_single_agent_file(agent_dir):
    p = agent_dir.write("good_carrier.py", GOOD_CARRIER)
    spec = load_agent(str(p))
    assert (spec.name, spec.role, spec.kind) == ("GoodCarrier", "carrier", "u_goodcarrier")
    assert spec.build("carrier-u_goodcarrier-1").agent_id == "carrier-u_goodcarrier-1"


def test_load_file_with_class_name_including_windows_style_absolute_path(agent_dir):
    p = agent_dir.write("both.py", GOOD_CARRIER + GOOD_SHIPPER.split("\n", 2)[2])
    spec = load_agent(f"{p.resolve()}:GoodShipper")
    assert spec.role == "shipper"


def test_file_with_several_agents_needs_a_class_name(agent_dir):
    p = agent_dir.write("both.py", GOOD_CARRIER + GOOD_SHIPPER.split("\n", 2)[2])
    with pytest.raises(AgentLoadError, match="several agents"):
        load_agent(str(p))
    with pytest.raises(AgentLoadError, match="No DecisionEngine subclass named 'Nope'"):
        load_agent(f"{p}:Nope")


def test_load_from_an_installed_module(agent_dir, monkeypatch):
    agent_dir.write("my_pkg_agent.py", GOOD_SHIPPER)
    monkeypatch.syspath_prepend(str(agent_dir.path))
    spec = load_agent("my_pkg_agent:GoodShipper")
    assert spec.name == "GoodShipper"
    sys.modules.pop("my_pkg_agent", None)


def test_missing_role_is_explained(agent_dir):
    p = agent_dir.write("norole.py", GOOD_CARRIER.replace('role = "carrier"', ""))
    with pytest.raises(AgentLoadError, match='role = "carrier" or role = "shipper"'):
        load_agent(str(p))


def test_wrong_role_value_is_explained(agent_dir):
    p = agent_dir.write("badrole.py", GOOD_CARRIER.replace('role = "carrier"', 'role = "trucker"'))
    with pytest.raises(AgentLoadError, match="trucker"):
        load_agent(str(p))


def test_syntax_and_import_errors_are_reported_not_raised_raw(agent_dir):
    p = agent_dir.write("syntax.py", "def broken(:\n")
    with pytest.raises(AgentLoadError, match="SyntaxError"):
        load_agent(str(p))
    p = agent_dir.write("missing_dep.py", "import a_package_nobody_has\n")
    with pytest.raises(AgentLoadError, match="ModuleNotFoundError"):
        load_agent(str(p))


def test_constructor_with_required_args_is_explained(agent_dir):
    p = agent_dir.write("ctor.py", GOOD_CARRIER + '''
    def __init__(self, agent_id, api_key):
        super().__init__(agent_id)
''')
    spec = load_agent(str(p))
    with pytest.raises(AgentLoadError, match="default"):
        spec.build("carrier-x-1")


def test_missing_file_and_wrong_extension(agent_dir):
    with pytest.raises(AgentLoadError, match="not found"):
        load_agent(str(agent_dir.path / "ghost.py"))
    with pytest.raises(AgentLoadError, match="Could not import module"):
        load_agent("agent.txt")


def test_imported_baselines_are_not_mistaken_for_the_users_agent(agent_dir):
    """A user subclassing or importing a reference engine must not get it picked up as theirs."""
    p = agent_dir.write("wrapper.py", '''
        from wehaveadeal.agents import CarrierOPT2, ShipperR

        class TunedOPT2(CarrierOPT2):
            role = "carrier"
            engine_name = "TunedOPT2"
    ''')
    spec = load_agent(str(p))
    assert spec.name == "TunedOPT2"


def test_agent_can_import_its_own_helper_module(agent_dir):
    agent_dir.write("_helpers.py", "def opening(state):\n    return state.refprice_mean + 0.5 * state.sigma\n")
    p = agent_dir.write("uses_helper.py", '''
        from _helpers import opening
        from wehaveadeal import DecisionEngine, offer, legal_offer_range, REJECT

        class HelperCarrier(DecisionEngine):
            role = "carrier"
            def next_move(self, state):
                lo, hi = legal_offer_range(state)
                return offer(min(opening(state), hi)) if lo <= hi else REJECT
    ''')
    assert load_agent(str(p)).name == "HelperCarrier"


def test_discovery_finds_good_agents_despite_broken_neighbours(agent_dir):
    agent_dir.write("a_carrier.py", GOOD_CARRIER)
    agent_dir.write("b_shipper.py", GOOD_SHIPPER)
    agent_dir.write("c_broken.py", "raise RuntimeError('boom at import')\n")
    agent_dir.write("d_norole.py", GOOD_CARRIER.replace('role = "carrier"', "").replace("GoodCarrier", "Other"))
    agent_dir.write("_private.py", "raise RuntimeError('must be skipped')\n")
    (agent_dir.path / "notes.md").write_text("# not python", encoding="utf-8")

    found = discover_agents(agent_dir.path)
    assert sorted(a.name for a in found.agents) == ["GoodCarrier", "GoodShipper"]
    assert [a.name for a in found.for_role("carrier")] == ["GoodCarrier"]
    failed = sorted(e["file"].replace("\\", "/").rsplit("/", 1)[1] for e in found.errors)
    assert failed == ["c_broken.py", "d_norole.py"]


def test_discovery_flags_name_clashes(agent_dir):
    agent_dir.write("one.py", GOOD_CARRIER)
    agent_dir.write("two.py", GOOD_CARRIER)
    found = discover_agents(agent_dir.path)
    assert len(found.agents) == 1
    assert "same name" in found.errors[0]["error"]


def test_discovery_of_missing_folder_is_empty_not_an_error(tmp_path):
    found = discover_agents(tmp_path / "does-not-exist")
    assert found.agents == [] and found.errors == []


@pytest.mark.parametrize("name,kind", [
    ("MyCarrier", "u_mycarrier"),
    ("My Carrier v2!", "u_my_carrier_v2"),
    ("R", "u_r"),  # can never collide with the baseline kind "r"
    ("---", "u_agent"),
])
def test_kind_is_id_safe(name, kind):
    assert kind_for(name) == kind
    assert "-" not in kind
