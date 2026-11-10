# Benchmark protocol, version 1.0

This is the protocol implemented by `wehaveadeal.toolkit.benchmark.run_benchmark`.
Any change to it bumps `PROTOCOL_VERSION`. Leaderboard entries are only
compared within the same protocol version.

## Market

The market follows the arena used in the paper:
- **Prices:** sale price and goods cost follow Ornstein–Uhlenbeck processes.
- **Routes:** each shipper is assigned one of 6 routes, with reference prices in
  [800, 3000] and σ = 15% of the reference price.
- **Shipper information:** shippers see the reference price plus noise with
  σ_ε = 0.3σ.
- **Matching:** each market day ends with Gale–Shapley matching over the closed
  tables.

## Entrant placement

The entrant plays one role. Let B be the number of reference architectures
(9: R, RV, IS, OPT, OPT2, ML, ML2, RL, RL2).

| regime | entrant's side | opposite side |
|---|---|---|
| `balanced` | entrant + B−1 references | B references |
| `<entrant role>_scarce` | entrant + B−3 references | B references |
| `<other role>_scarce` | entrant + B−1 references | B−2 references |

With B = 9 this reproduces the paper's 9-vs-9 and 7-vs-9 markets.

## Sampling

Each regime runs `n_subrounds` sub-rounds of `n_negotiations` market days.

| preset | sub-rounds × days | market days per regime |
|---|---|---|
| `smoke` | 1 × 2 | 2 |
| `quick` | 6 × 5 | 30 |
| `standard` (leaderboard) | 36 × 12 | 432, the paper's scarce-regime budget |

For sub-round *i* of regime *r*, the line-up of references on each side and
the market seed are drawn from `numpy.random.default_rng([seed, r, i])` and
`SeedSequence([seed, r, i])`. They depend only on the seed, never on the
entrant. These are **common random numbers**: two entrants benchmarked with the
same seed face identical market days and identical rivals, which makes their
scores directly comparable.

The reference architectures are deterministic given the state. Their
randomness is derived from (negotiation seed, table, turn), so a report is
exactly reproducible from its settings, provided the entrant is also
deterministic.

## Outcome and statistics

- **Per-day outcome** of an agent:
  - carrier: `rate − refprice_mean` if matched, else 0
  - shipper: `sale_price − goods_cost − rate` if matched, else 0
- **Mean outcome:** the mean over the market days the agent was in, with a 95%
  percentile-bootstrap CI (5,000 resamples).
- **Rank:** the entrant's position by mean outcome among all agents on its side.
- **Paired comparison** against each reference: the differences on the days
  both were present, with a bootstrap CI on the mean difference and a Wilcoxon
  signed-rank test. P-values are Holm-corrected across the references within
  each regime.
- **Verdict:** `better` / `worse` at α = 0.05, `no significant difference`
  otherwise, and `insufficient data` below 10 paired days.
- **Headline:**
  - the entrant's mean outcome, averaged over regimes
  - its mean rank
  - its total wins and losses
  - **Primary leaderboard key:** mean rank. Mean outcome breaks ties.

## Fair play

Every entrant runs behind `GuardedAgent`. These events forfeit the table: the
guard plays REJECT for the entrant, and the event is recorded in
`report.violations`:
- an exception
- a non-`Move` return
- an illegal offer (out of band, or a non-monotonic concession)
- an ACCEPT with nothing on the table
- exceeding `move_timeout_s` (default 5 s)

The time limit is measured, not preempted. A slow agent costs wall-clock time
but cannot gain from it.

## What the protocol does not do

- **It does not sandbox code.** Benchmarks run on your own machine, so run only
  agents whose code you trust.
- **It does not include other entrants.** The roster is the references plus
  one entrant. Head-to-head play between community agents would be a separate,
  future protocol version.
