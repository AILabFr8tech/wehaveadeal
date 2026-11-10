"""One-off helper: finds good illustrative (file, table_id, episode) examples
for negotiation-trajectory plots, splitting calls.jsonl's per-table rows back
into individual day-episodes (no day_index is logged in calls.jsonl, so this
infers episode boundaries from turn/attempt sequence resets)."""
import json
from pathlib import Path
import pandas as pd


def split_episodes(rows):
    episodes, cur = [], []
    prev_turn, prev_attempt = None, None
    for r in rows:
        turn, attempt = r["turn"], r["attempt"]
        is_retry = prev_turn == turn and isinstance(attempt, int) and isinstance(prev_attempt, int) and attempt == prev_attempt + 1
        is_post_forfeit_same_turn = prev_turn == turn and attempt == "forfeit"
        new_episode = cur and (
            (prev_turn is not None and turn < prev_turn)
            or (turn == prev_turn and not is_retry and not is_post_forfeit_same_turn)
        )
        if new_episode:
            episodes.append(cur)
            cur = []
        cur.append(r)
        prev_turn, prev_attempt = turn, attempt
    if cur:
        episodes.append(cur)
    return episodes


LLM_DIR = Path("data/llm_validation")
best = []  # (n_distinct_turns, fname, table_id, episode_index, episode)
for fname in LLM_DIR.glob("*_calls.jsonl"):
    rows = [json.loads(l) for l in open(fname, encoding="utf-8")]
    if not rows:
        continue
    df = pd.DataFrame(rows)
    for table_id, sub in df.groupby("table_id"):
        episodes = split_episodes(sub.to_dict("records"))
        for i, ep in enumerate(episodes):
            n_distinct = len(set(r["turn"] for r in ep))
            best.append((n_distinct, fname.name, table_id, i, ep))

best.sort(key=lambda x: x[0], reverse=True)
print("Top 15 episodes by number of distinct turns:")
for n_distinct, fname, table_id, i, ep in best[:15]:
    turns = [(r["turn"], r["attempt"]) for r in ep]
    print(f"{n_distinct} turns | {fname:40s} | {table_id:35s} | ep#{i} | {turns}")
