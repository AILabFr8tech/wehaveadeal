# Writing an agent

## The contract

```python
class MyAgent(DecisionEngine):
    role = "carrier" | "shipper"     # required
    engine_name = "MyAgent"           # display name (defaults to the class name)

    def __init__(self, agent_id: str, frozen: bool = True, **your_params_with_defaults): ...
    def next_move(self, state: TableState) -> Move: ...
```

Return one of three moves:
- `offer(amount)`
- `ACCEPT`, which takes the counter-offer currently on the table
- `REJECT`, which walks away from this table for good

Your agent must be constructible as `MyAgent(agent_id=...)`, so every other
parameter needs a default.

One instance plays the whole benchmark. It can keep state and learn online
across negotiations, but it must not rely on global randomness: seed any RNG
from the state, for example `random.Random(f"{state.negotiation_seed}:{state.table_id}:{state.turn}")`.
That way results are reproducible.

## How a negotiation works

- A **negotiation** is one market day. Every shipper negotiates with every
  carrier at the same time, so S×C bilateral **tables** run in lockstep.
- A table has up to 12 global turns. Shippers act on odd turns and carriers on
  even turns, so each side gets 6. The shipper opens.
- After the turns, closed tables enter **Gale–Shapley matching**. Each shipper
  gets at most one carrier and vice versa. Closing a table does not guarantee
  you the load.
- **Outcomes:**
  - A carrier earns `rate - refprice_mean` if matched, otherwise 0.
  - A shipper earns `sale_price - goods_cost - rate` if matched, otherwise 0. A
    matched shipper with negative profit declines to ship and gets 0.

## What you can see (`TableState`)

| field | shipper | carrier |
|---|---|---|
| `turn`, `turns_max`, `my_turns_used`, `opp_turns_used` | ✓ | ✓ |
| `my_last_offer`, `opp_last_offer`, `history` (this table) | ✓ | ✓ |
| `sigma`: route volatility | ✓ | ✓ |
| `sale_price`, `goods_cost`: your economics | ✓ | `None` |
| `refprice_est`: a *noisy* estimate of the route's reference price | ✓ | `None` |
| `refprice_mean`: the true reference price | `None` | ✓ |
| `my_portfolio`: your other tables right now | ✓ | ✓ |

Information hiding is structural: the fields you may not see are `None`, so
there is nothing to "peek" at. You also receive a copy of the state, so
modifying it changes nothing in the arena.

## The rules (enforced)

`legal_offer_range(state)` returns the `(lo, hi)` you may offer. If `lo > hi`,
no legal offer remains.

- **Carrier:** offers stay inside `[refprice_mean - sigma/2, refprice_mean + sigma/2]`
  and never go above your own previous offer.
- **Shipper:** offers never exceed `sale_price - goods_cost` and never go below
  your own previous offer.
- **ACCEPT** needs a counter-offer on the table. A shipper cannot accept on
  turn 1.

Breaking a rule, raising an exception, returning something that is not a
`Move`, or taking longer than the move timeout **forfeits that table**: it
becomes an impasse. You keep playing your other tables, and the report lists
every violation. Use `wehaveadeal validate` to catch violations early.

## Learning-based agents

- **Reinforcement learning:** `wehaveadeal.rl.env.NegotiationEnv` is a
  Gymnasium environment for one table against the reference opponents.
  `wehaveadeal.rl.league_env` is the competitive multi-table version used to
  train RL2. Their training scripts (`python -m wehaveadeal.rl.train --help`)
  show the full pipeline.
- **Imitation learning:** `python -m wehaveadeal.ml.collect` records
  self-play data and `python -m wehaveadeal.ml.train` fits a model.
  `wehaveadeal.ml.features.extract_features` is the role-agnostic feature
  vector the learned baselines use.

## Agents with a trained model file

Any model format works (`.joblib`/`.pkl`, `.h5`/`.keras`, `.pt`, `.zip`
from Stable-Baselines3, `.onnx`, ...). The toolkit only ever calls
`next_move`, so your agent file loads the weights itself.

A complete worked example: [examples/linreg_carrier.py](../examples/linreg_carrier.py)
with its training script in
[examples/linreg_carrier_model/train.py](../examples/linreg_carrier_model/train.py).
The training script lets the reference agents play, records what each carrier
saw at every decision, labels each record with the rate that table closed at,
and fits a scikit-learn `LinearRegression`:

```bash
python examples/linreg_carrier_model/train.py     # ~1 min, writes model.joblib
wehaveadeal benchmark examples/linreg_carrier.py
```

**Layout.** Put the weights in a folder next to the agent. Only top-level
`.py` files in `agents/` are treated as agents, so subfolders are free for
weights, training code and helpers:

```
agents/
  my_carrier.py            the agent
  my_carrier_model/
    model.joblib           weights
    train.py               how you made them (optional)
```

**Loading.** Four rules, all followed by the example:

1. Locate the file relative to the agent: `Path(__file__).parent / "my_carrier_model" / "model.joblib"`.
   Absolute paths break on other machines and in Docker.
2. Load in `__init__`, not in `next_move`. The per-move time limit applies to
   `next_move` only, so a slow first-move load (TensorFlow can take several
   seconds) forfeits tables. The benchmark builds your agent once per run.
3. Cache the model at module level. The dashboard creates a fresh agent for
   every negotiation it shows, and the cache makes it load only once.
4. Share the feature code between training and playing (the example's
   `features()` function), so the model sees exactly what it was trained on.

**Libraries.** scikit-learn, torch and Stable-Baselines3 are already
installed. Install anything else yourself (`pip install tensorflow`,
`pip install onnxruntime`), and for the Docker dashboard list it in
`agents/requirements.txt`.

**Train on different seeds than you test on.** The example trains on seeds
from 10,000 up. The benchmark's default seed is 2026, so the model is never
scored on market days it was trained on.

**Sharing.** A leaderboard entry needs only the JSON report. For others to
re-run your entry, the weights must be downloadable: put large files in a
GitHub Release or on Hugging Face rather than in git. Pickle (`.pkl`,
`.joblib`) and `.h5` files can execute code when loaded, so treat someone
else's weights like their code. ONNX and `.safetensors` are safe to share.

## LLM agents: bring your own compute

[examples/llm_carrier.py](../examples/llm_carrier.py) calls any
OpenAI-compatible endpoint configured through `LLM_BASE_URL`, `LLM_MODEL` and
`LLM_API_KEY`: a hosted provider, vLLM, or Ollama. The Ollama-native agents used
in the paper are `wehaveadeal.agents.CarrierLLM` and `ShipperLLM`.

LLMs are slow, so raise the per-move limit:
`wehaveadeal benchmark my_llm_agent.py --move-timeout 60`.

Budget the calls. A standard run is about 1,300 market days with up to 6 moves
per table, and your agent sits at up to 9 tables per day. Develop with
`--preset quick` or `--regimes balanced --subrounds 2`.

## Extra dependencies

Your agent may import anything installed in your environment. For the Docker
dashboard, list the extra packages in `agents/requirements.txt`; they are
installed when the container starts.
