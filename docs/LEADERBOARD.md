# Submitting to the leaderboard

The leaderboard is a folder of JSON reports in this repository. Nobody runs your
agent for you, and submitting costs nothing. Instead, every entry is
reproducible: anyone can re-run it and must get the same numbers.

## Steps

1. Run the official protocol with default seed, regimes and baselines:

   ```bash
   wehaveadeal benchmark path/to/your_agent.py --preset standard \
       --out leaderboard/submissions/<your-team>__<AgentName>__<role>.json
   ```

   For LLM agents, add `--move-timeout <seconds>`. The value is recorded in the
   report.

2. Open a pull request that adds:
   - the report JSON
   - a link to your agent's source code. The report records the SHA-256 of the
     agent file, so the linked code must match it.
   - for LLM agents: the model name and version, and the settings used
     (temperature and so on)
   - if your agent uses trained weights: a link to them

3. A maintainer checks the following, then merges:
   - `settings.preset == "standard"`, the default seed and all nine baselines
   - `settings.protocol_version` is current
   - `violations.forfeit_rate` is below 1%
   - for deterministic agents, a spot re-run reproduces the report

## Ranking

Entries are grouped by role and by protocol version, and ranked by:
1. **Mean rank** across the three regimes (lower is better)
2. **Mean outcome**, to break ties

The report also lists wins and losses against each reference architecture with
Holm-corrected tests. These show *where* an agent is strong, for example only
when carriers are scarce.

## Report schema

Run `wehaveadeal schema` to print the JSON schema of reports.
