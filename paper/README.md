# Reproducing the paper

> **Before AI Agents Negotiate for Us: A Competitive Tournament of Rule-Based,
> Search, Optimization, Learned, and Language-Model Negotiators Under Scarcity**

**Archived version used in the paper:** release v0.1.0,
[doi:10.5281/zenodo.24122038](https://doi.org/10.5281/zenodo.24122038).

Everything the paper reports is in this folder. You can **re-check** every
number from the saved data in a few minutes. You can **regenerate** the
tournament data from scratch, which takes tens of minutes on a laptop. The LLM
study needs your own Ollama server, so we publish its complete call logs
instead of asking you to re-run it.

```bash
pip install -e ".[paper]"          # from the repo root
pytest tests/paper                 # integrity + reproduction checks (about 30 s)
```

## Where every result comes from

| In the paper | Data | Produced by |
|---|---|---|
| Table `tab:hyperparams` | the code: `wehaveadeal/agents/*.py` | n/a |
| Tables `tab:carrier`, `tab:shipper` (match rate, mean outcome, 95% CI) | `data/validation/summary_stats.csv`, from `raw_results.csv` | `notebooks/validation_exhaustive.ipynb` |
| "% of sale" / margin-% columns | `data/validation/pct_outcomes.csv` | `scripts/_compute_pct_outcomes.py` |
| Fig. `fig:ranking` | `figures/ranking_by_config.png` | `notebooks/validation_exhaustive.ipynb` |
| Figs. `fig:winmatrix-*` (six win matrices) | `data/validation/pairwise_tests.csv` | `notebooks/validation_exhaustive.ipynb`, `scripts/_regen_english_figures.py` |
| Tables `tab:llm-carrier`, `tab:llm-shipper` | `data/llm_validation/summary_stats.csv`, `tier_family_summary.csv`, `tier_summary.csv`, `pairwise_vs_rivals.csv` | `notebooks/analyze_llm_validation.ipynb` |
| Table `tab:llm-compliance` (structured-output compliance, latency) | `data/llm_validation/compliance_by_role.csv`, from `*_calls.jsonl` | `notebooks/paper_results.ipynb` |
| Figs. `fig:llm-vs-engines`, `fig:llm-tier-family` | `figures/llm_vs_engines.png`, `figures/llm_by_tier_family.png` | `notebooks/analyze_llm_validation.ipynb` |
| Figs. `fig:llm-trajectory-*`, Table `tab:llm-trajectory-struggle` | `data/llm_validation/*_calls.jsonl` | `scripts/_build_trajectory_figures.py`, `notebooks/paper_results.ipynb` |
| All numbers in one place | n/a | `notebooks/paper_results.ipynb` (reads only saved data) |

The pipeline, taxonomy and design figures are diagrams with no data behind
them.

## Supplementary material (not reported in the paper)

**RL2Fixed** is RL2 retrained with real portfolio visibility during training,
the corrected RL2 that the paper's discussion names as a next step rather than
a reported result. We include its trained weights, the ablation script and its
raw results for transparency, but the paper does not analyze them and no
confidence intervals or significance tests have been computed on them.

| Material | Data | Produced by |
|---|---|---|
| RL2 vs RL2Fixed ablation (same seeds and protocol) | `data/validation/ablation_rl2fixed_raw_results.csv` | `scripts/run_rl2fixed_ablation.py` |
| RL2Fixed weights | `wehaveadeal/baselines_data/rl/*rl2fixed*` | `scripts/train_rl2fixed_resilient.sh` |

## The data

- **`data/validation/raw_results.csv`:** one row per agent per market day for
  the three market conditions (400 + 432 + 432 days, 9 shippers and 9 carriers,
  or 7 on the scarce side). Columns:
  - `config`: the market condition
  - `excluded`: the exclusion pair in scarce conditions
  - `day_id`, `role`, `kind`, `matched`
  - `outcome`: carrier margin in USD over the reference price, or shipper profit in USD
- **`data/validation/ablation_rl2fixed_raw_results.csv`** (supplementary, not in
  the paper): the same protocol and seeds, with RL2 replaced by RL2Fixed.
- **`data/llm_validation/<model>_<role>_outcomes.jsonl`:** one row per agent
  per day in the LLM study, for the LLM and its three same-side rivals.
- **`data/llm_validation/<model>_<role>_calls.jsonl`:** every model call: the
  raw response text, latency, retry attempt and parse error. These are the
  logs behind the compliance and latency results.
- **Seeds:** balanced uses base 10,000,000; carrier-scarce uses 20,000,000 + pair
  index; shipper-scarce uses 30,000,000 + pair index. Each LLM run uses
  90,000,000 + crc32(role:model) mod 10⁶; see `scripts/run_llm_validation.py`.

## Regenerating from scratch

Run all commands from the repo root.

```bash
# The 9-architecture tournament (~20-40 min on a laptop CPU): open the notebook,
# set FORCE_RERUN = True in its second code cell, run all cells.
jupyter lab paper/notebooks/validation_exhaustive.ipynb

# Supplementary, not in the paper: the RL2Fixed ablation (same seeds and protocol)
python paper/scripts/run_rl2fixed_ablation.py

# Retrain RL2Fixed (PPO, 300k timesteps per role; writes to paper/models/)
bash paper/scripts/train_rl2fixed_resilient.sh carrier
bash paper/scripts/train_rl2fixed_resilient.sh shipper

# Percent-of-sale / margin-% table columns (from raw_results.csv)
python paper/scripts/_compute_pct_outcomes.py

# The LLM study needs an Ollama server with the models pulled (see
# wehaveadeal/llm/roster.py for the nine model tags); writes data/llm_validation/
python paper/scripts/run_llm_validation.py --days 40 --host http://localhost:11434
```

The bundled weights of the learned architectures (ML, ML2, RL, RL2) are the
exact ones used in the paper; the RL2Fixed weights are supplementary. They are in
`wehaveadeal/baselines_data/`, and their training scripts are in
`wehaveadeal/ml/` and `wehaveadeal/rl/`.

## How exact is "reproducible"?

Before release, we re-simulated the balanced condition (all 400 days) with
this public package and compared it with `raw_results.csv`:
- **Deal outcomes:** whether each of the 7,200 agent-days ended in a deal is
  identical.
- **Payoffs:** 91% of payoffs are bit-identical. The rest differ by at most
  0.0003 USD, which is floating-point noise between library versions (torch /
  scikit-learn inference).

`tests/paper/test_paper_artifacts.py` repeats a slice of this check, and
re-derives the published summary and pairwise tables from the raw data, on
every CI run.

**Verified environments.** The full test suite, including the paper
reproduction tests, also passes on a fresh `pip install` with current
libraries: Python 3.11, numpy 2.4.6, scikit-learn 1.8.0, torch 2.15.0 (CPU),
stable-baselines3 2.9.2. scikit-learn is capped below 1.9 because 1.9 can no
longer load the bundled ML/ML2 models.

The environments the paper's numbers were checked in:
- **Simulation and scripts:** Python 3.11, numpy 2.4.6, scikit-learn 1.5.2,
  torch 2.14.1 (CPU), stable-baselines3 2.9.0, gymnasium 1.3.0, pydantic 2.13.
  See `requirements-verified.txt`.
- **Analysis notebooks:** pandas, matplotlib 3.10, seaborn 0.13, statsmodels 0.14.
- **numpy 2 is required.** The RL2Fixed weights were serialized under numpy 2
  and do not load under numpy 1.x.

## Notes

- **Notebooks:** `validation_exhaustive.ipynb` and `analyze_llm_validation.ipynb`
  are the working notebooks that produced the results; `paper_results.ipynb` is the
  consolidated view of every number in the paper.
- **Renamed figure:** the LLM-vs-engines figure is `llm_vs_engines.png` (label
  `fig:llm-vs-engines`); earlier manuscript drafts called it `llm_vs_motores.png`.
- **LLM seeding:** as the paper explains, an early version of the LLM runner
  seeded runs non-reproducibly across restarts. The fix (`zlib.crc32`) is in
  the published script. Same-day comparisons within a run are unaffected.
