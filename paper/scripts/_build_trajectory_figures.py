"""One-off builder: negotiation-trajectory figures from already-saved LLM call
logs (data/llm_validation/*_calls.jsonl) -- no re-simulation, no new Ollama
calls. calls.jsonl logs only the LLM's own responses (never the counterpart's
offers, which were never persisted -- only visible inside the model's own
prompt at the time), so these are the LLM's own half of each table: what it
offered/decided, turn by turn, across one illustrative historical day.

calls.jsonl has no day_index field, so a single table_id's rows blend all ~40
days of that (model, role) combo; split_episodes() infers day boundaries from
the turn/attempt sequence (turns only increase within one real day; a drop or
an out-of-sequence attempt means a new day started).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLM_DIR = PROJECT_ROOT / "data" / "llm_validation"
FIG_DIR = PROJECT_ROOT / "figures"


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


def get_episode(fname, table_id, episode_index):
    rows = [json.loads(l) for l in open(LLM_DIR / fname, encoding="utf-8")]
    df = pd.DataFrame(rows)
    sub = df[df.table_id == table_id].to_dict("records")
    return split_episodes(sub)[episode_index]


def parsed_amount(raw_text):
    if not raw_text:
        return None, None
    try:
        obj = json.loads(raw_text)
        return obj.get("action"), obj.get("amount")
    except (json.JSONDecodeError, TypeError):
        return None, None


def final_per_turn(episode):
    """One point per distinct turn: the LAST attempt logged for that turn is
    the one that was actually used (the retry loop returns as soon as an
    attempt validates; a forfeit row is logged only after every retry failed,
    never a value that was used). accept/reject carry amount=None by schema
    design (only "offer" has a price); those are still returned so the caller
    can annotate them, just excluded from the plotted price line."""
    by_turn = {}
    for r in episode:
        if r["attempt"] == "forfeit":
            continue
        by_turn[r["turn"]] = r
    turns = sorted(by_turn)
    actions, amounts = [], []
    for t in turns:
        a, amt = parsed_amount(by_turn[t]["raw_text"])
        actions.append(a)
        amounts.append(amt)
    return turns, actions, amounts


def plot_trajectory(ax, turns, actions, amounts, color):
    """Plots only the numeric (offer) points as a line; accept/reject points
    (amount=None by schema design) are annotated as text at the last numeric
    point instead of being given a (possibly-None) coordinate of their own."""
    num_t = [t for t, amt in zip(turns, amounts) if amt is not None]
    num_a = [amt for amt in amounts if amt is not None]
    ax.plot(num_t, num_a, "o-", color=color)
    for t, a, amt in zip(turns, actions, amounts):
        if amt is not None:
            ax.annotate(a, (t, amt), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
        elif num_t:
            ax.annotate(f"{a} (turn {t})", (num_t[-1], num_a[-1]), textcoords="offset points", xytext=(0, -16), fontsize=8, ha="center", color="#CF222E")


def main():
    # ---- Figure: two clean/illustrative price trajectories ------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ep = get_episode("llama3.1-8b_carrier_calls.jsonl", "shipper-is-1::carrier-llm-1", 4)
    turns, actions, amounts = final_per_turn(ep)
    plot_trajectory(axes[0], turns, actions, amounts, "#4a3aa7")
    axes[0].set_title("llama3.1:8b (carrier)\nvs IS, one historical day")
    axes[0].set_xlabel("turn")
    axes[0].set_ylabel("own offer ($)")
    axes[0].grid(alpha=0.3)

    ep = get_episode("qwen2.5-72b_shipper_calls.jsonl", "shipper-llm-1::carrier-opt2-1", 5)
    turns, actions, amounts = final_per_turn(ep)
    plot_trajectory(axes[1], turns, actions, amounts, "#eb6834")
    axes[1].set_title("qwen2.5:72b (shipper)\nvs OPT2, one historical day")
    axes[1].set_xlabel("turn")
    axes[1].set_ylabel("own offer ($)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "llm_trajectory_clean_examples.png", dpi=150)
    plt.close()
    print("wrote llm_trajectory_clean_examples.png")

    # (The "struggling" example -- deepseek-r1:7b repeatedly attempting an
    # illegal accept, forfeiting at turn 9 -- is presented as a table directly in
    # the notebook instead of a chart: its amount values aren't a meaningful
    # price axis, since the schema makes "amount" irrelevant for "accept".)

    # ---- Figure: deepseek-r1:70b latency wall (no trajectory possible -- no
    # attempt produces a parseable response: timeouts plus one server error) ---
    rows = [json.loads(l) for l in open(LLM_DIR / "deepseek-r1-70b_carrier_calls.jsonl", encoding="utf-8")]
    df = pd.DataFrame(rows)
    sub = df[(df.table_id == "shipper-r-1::carrier-llm-1") & (df.turn == 2)].head(12)
    # The 12 rows are three days of (attempt 0, 1, 2, forfeit). A forfeit row has
    # no latency, so it is drawn as a marker instead of a zero-height bar.
    fig, ax = plt.subplots(figsize=(8, 5))
    xs, labels, day = [], [], 0
    for i, (_, row) in enumerate(sub.iterrows()):
        x = i + day * 0.6  # small gap between days
        if row["attempt"] == "forfeit":
            ax.scatter([x], [4], marker="X", s=110, color="#CF222E", zorder=3)
            labels.append("forfeit")
            day += 1
        else:
            timed_out = "HTTPConnectionPool" in str(row["error"])
            ax.bar([x], [row["latency_s"]], color="#999999" if timed_out else "#eb6834")
            labels.append(f"attempt {int(row['attempt']) + 1}")
        xs.append(x)
    ax.axhline(90, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    for d in range(day):
        ax.text(xs[4 * d + 1], 101, f"day {d + 1}", ha="center", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("latency (s)")
    ax.set_title("deepseek-r1:70b (carrier) vs R, first decision (turn 2) on one table, three days", fontsize=10)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="#999999", label="timed out"),
        plt.Rectangle((0, 0), 1, 1, color="#eb6834", label="server error"),
        plt.Line2D([], [], marker="X", color="#CF222E", linestyle="", markersize=9, label="table forfeited"),
        plt.Line2D([], [], color="black", linestyle="--", label="90 s timeout"),
    ], fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "llm_trajectory_timeout_example.png", dpi=150)
    plt.close()
    print("wrote llm_trajectory_timeout_example.png")


if __name__ == "__main__":
    main()
