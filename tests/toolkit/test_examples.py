"""The shipped examples and template must work as documented -- they are the first
code a newcomer copies. Also checks the bring-your-own-compute promise for LLM agents."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

from wehaveadeal.toolkit.benchmark import run_benchmark
from wehaveadeal.toolkit.helpers import clamp_offer, legal_offer_range
from wehaveadeal.toolkit.loader import load_agent

from .conftest import EXAMPLES_DIR, FAST
from .test_guard import carrier_state

EXAMPLES = sorted(p.name for p in EXAMPLES_DIR.glob("*.py"))


def test_examples_exist():
    assert {"random_shipper.py", "tit_for_tat_carrier.py", "llm_carrier.py",
            "noisy_mean_carrier.py", "linreg_carrier.py"} <= set(EXAMPLES)
    assert (EXAMPLES_DIR / "linreg_carrier_model" / "model.joblib").is_file(), "the trained example model must ship"


def test_linreg_training_script_produces_a_working_model(tmp_path):
    """The model-file walkthrough end to end: train -> save -> the agent loads and plays."""
    import importlib.util

    import joblib

    path = EXAMPLES_DIR / "linreg_carrier_model" / "train.py"
    spec = importlib.util.spec_from_file_location("linreg_train", path)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)

    out = tmp_path / "model.joblib"
    train.main(["--rounds", "2", "--negotiations", "1", "--out", str(out)])
    model = joblib.load(out)
    assert model.coef_.shape == (len(train.FEATURE_NAMES),)

    agent = load_agent(str(EXAMPLES_DIR / "linreg_carrier.py")).build("carrier-u_linregcarrier-1")
    move = agent.next_move(carrier_state())
    assert move.action.value in {"offer", "accept", "reject"}


@pytest.mark.parametrize("filename", EXAMPLES)
def test_every_example_plays_legally(filename, monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    spec = load_agent(str(EXAMPLES_DIR / filename))
    report = run_benchmark(spec, **FAST)
    assert report.violations.total_violations == 0, report.violations.examples[:3]
    assert any(r.entrant.match_rate > 0 for r in report.regimes), "an example should close at least one deal"


def test_llm_example_makes_no_network_calls_unless_configured(monkeypatch):
    """No key / endpoint configured -> it must not phone anywhere (and never to us)."""
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    def forbidden(*a, **k):
        raise AssertionError("network call attempted")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    spec = load_agent(str(EXAMPLES_DIR / "llm_carrier.py"))
    report = run_benchmark(spec, regimes=("balanced",), baselines=FAST["baselines"], n_subrounds=1, n_negotiations=2, preset="smoke")
    assert report.violations.total_violations == 0


def test_llm_example_uses_the_users_own_endpoint_and_key(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example-user.org/v1")
    monkeypatch.setenv("LLM_MODEL", "their-model")
    monkeypatch.setenv("LLM_API_KEY", "sk-user-owned")
    calls = []

    class FakeResponse:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return self.body

    def fake_urlopen(req, timeout):
        calls.append((req.full_url, req.get_header("Authorization")))
        return FakeResponse(b'{"choices": [{"message": {"content": "{\\"action\\": \\"offer\\", \\"amount\\": 99999}"}}]}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    agent = load_agent(str(EXAMPLES_DIR / "llm_carrier.py")).build("carrier-u_llmcarrier-1")
    move = agent.next_move(carrier_state())
    assert calls == [("https://llm.example-user.org/v1/chat/completions", "Bearer sk-user-owned")]
    assert move.amount == 1050.0  # an out-of-band LLM answer is clamped into the legal range


@pytest.mark.parametrize("reply", ["no json here", '{"action": "dance"}', '{"action": "offer"}'])
def test_llm_example_falls_back_on_unusable_replies(reply, monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    agent = load_agent(str(EXAMPLES_DIR / "llm_carrier.py")).build("carrier-u_llmcarrier-1")
    monkeypatch.setattr(agent, "ask", lambda prompt: reply)
    move = agent.next_move(carrier_state())
    assert move.action.value in {"offer", "accept", "reject"}


def test_legal_offer_range_helper():
    s = carrier_state(my_last_offer=1020.0)
    assert legal_offer_range(s) == (950.0, 1020.0)
    assert clamp_offer(s, 2000) == 1020.0 and clamp_offer(s, 0) == 950.0

    from wehaveadeal.models import TableState

    ship = TableState(role="shipper", table_id="t", negotiation_seed=1, turn=3, turns_max=12, my_turns_used=1,
                      opp_turns_used=1, my_last_offer=800.0, sigma=100.0, sale_price=5000.0, goods_cost=4000.0,
                      refprice_est=950.0)
    assert legal_offer_range(ship) == (800.0, 1000.0)


def test_template_is_packaged_with_placeholders():
    tmpl = Path(__file__).resolve().parents[2] / "wehaveadeal" / "toolkit" / "templates" / "agent.py.tmpl"
    text = tmpl.read_text(encoding="utf-8")
    for placeholder in ("$class_name", "$role", "$file_name"):
        assert placeholder in text
