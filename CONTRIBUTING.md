# Contributing

- **New agents:** you don't need to contribute them here. Publish your agent
  anywhere and submit its benchmark report to the leaderboard
  ([docs/LEADERBOARD.md](docs/LEADERBOARD.md)).
- **Bugs and arena questions:** open an issue with the agent spec, seed and
  command that reproduce the problem.
- **Changes to the arena or protocol:** anything that changes outcomes for a
  fixed seed must bump `PROTOCOL_VERSION` in `wehaveadeal/toolkit/benchmark.py`
  and be discussed in an issue first, because it invalidates leaderboard
  comparisons.
- **Before a pull request:** run `pip install -e ".[dev]" && pytest`.
- **License of contributions:** by opening a pull request you agree that your
  contribution is licensed under the [Apache License 2.0](LICENSE), the same
  license as the rest of the project (section 5 of the license).
