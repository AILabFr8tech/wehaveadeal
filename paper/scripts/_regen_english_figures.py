"""One-off: regenerates the handful of figures that had Spanish-language text baked
into them (plot titles/legends), after fixing the source cells in
notebooks/validation_exhaustive.ipynb and notebooks/analyze_llm_validation.ipynb.

Reads only already-saved summary CSVs (no re-simulation, no new LLM calls) and writes
each figure to its original data directory and to figures/, matching how those
PNGs were produced and copied originally.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLM_DIR = PROJECT_ROOT / "data" / "llm_validation"
VALID_DIR = PROJECT_ROOT / "data" / "validation"
FIG_DIR = PROJECT_ROOT / "figures"

sns.set_theme(style="whitegrid")

TIER_ORDER = ["tiny", "small", "medium"]
FAMILY_ORDER = ["llama", "qwen", "deepseek-r1"]
ARCH_LABELS = {"r": "R", "rv": "RV", "is": "IS", "opt": "OPT", "opt2": "OPT2",
               "ml": "ML", "ml2": "ML2", "rl": "RL", "rl2": "RL2"}


def save_both(fig, filename):
    fig.savefig(filename[0] / filename[1], dpi=150)
    fig.savefig(FIG_DIR / filename[1], dpi=150)


def llm_vs_engines():
    summary = pd.read_csv(LLM_DIR / "summary_stats.csv")
    roles = summary["role"].unique()
    fig, axes = plt.subplots(1, len(roles), figsize=(7 * len(roles), 5), squeeze=False)
    for i, role in enumerate(roles):
        ax = axes[0][i]
        sub = summary[summary.role == role].sort_values("mean_outcome", ascending=False)
        colors = ["#4a3aa7" if lbl in ("r", "opt2", "ml2") else "#eb6834" for lbl in sub["label"]]
        ax.barh(sub["label"], sub["mean_outcome"], color=colors,
                xerr=[sub["mean_outcome"] - sub["ci_lo"], sub["ci_hi"] - sub["mean_outcome"]], capsize=3)
        ax.set_title(f"{role}  (purple = our engine, orange = LLM)")
        ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    save_both(fig, (LLM_DIR, "llm_vs_engines.png"))
    plt.close(fig)
    print("wrote llm_vs_engines.png")


def llm_by_tier_family():
    tier_family_summary = pd.read_csv(LLM_DIR / "tier_family_summary.csv")
    tier_family_summary["tier"] = pd.Categorical(tier_family_summary["tier"], categories=TIER_ORDER, ordered=True)
    tier_family_summary["family"] = pd.Categorical(tier_family_summary["family"], categories=FAMILY_ORDER, ordered=True)
    roles = tier_family_summary["role"].unique()
    fig, axes = plt.subplots(1, len(roles), figsize=(6.5 * len(roles), 5), squeeze=False, sharey=True)
    family_colors = {"llama": "#4a3aa7", "qwen": "#eb6834", "deepseek-r1": "#1b9e91"}
    bar_width = 0.25
    x_base = np.arange(len(TIER_ORDER))
    for i, role in enumerate(roles):
        ax = axes[0][i]
        sub = tier_family_summary[tier_family_summary.role == role]
        for j, family in enumerate(FAMILY_ORDER):
            fam_sub = sub[sub.family == family].set_index("tier").reindex(TIER_ORDER)
            x = x_base + (j - 1) * bar_width
            yerr = [fam_sub["mean_outcome"] - fam_sub["ci_lo"], fam_sub["ci_hi"] - fam_sub["mean_outcome"]]
            ax.bar(x, fam_sub["mean_outcome"], width=bar_width, yerr=yerr, capsize=3,
                   color=family_colors[family], label=family)
        ax.set_xticks(x_base)
        ax.set_xticklabels(TIER_ORDER)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(role)
        ax.legend(title="family")
    plt.tight_layout()
    save_both(fig, (LLM_DIR, "llm_by_tier_family.png"))
    plt.close(fig)
    print("wrote llm_by_tier_family.png")


def win_matrices():
    raw = pd.read_csv(VALID_DIR / "raw_results.csv")
    raw["arch"] = raw["kind"].map(ARCH_LABELS)
    pairwise = pd.read_csv(VALID_DIR / "pairwise_tests.csv")
    for role in ["carrier", "shipper"]:
        for config in ["balanced", "carrier_scarce", "shipper_scarce"]:
            kinds_here = sorted(raw[(raw.config == config) & (raw.role == role)]["kind"].unique())
            archs = [ARCH_LABELS[k] for k in kinds_here]
            mat = pd.DataFrame(0, index=archs, columns=archs, dtype=float)
            sub = pairwise[(pairwise.config == config) & (pairwise.role == role)]
            for _, row in sub.iterrows():
                if not row["significant"]:
                    continue
                if row["mean_diff"] > 0:
                    mat.loc[row["arch_a"], row["arch_b"]] = 1
                else:
                    mat.loc[row["arch_b"], row["arch_a"]] = 1
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(mat, annot=False, cmap="Greens", cbar=False, linewidths=0.5, linecolor="white", ax=ax)
            ax.set_title(f"{role} - {config}\n(row beats column, after Holm-Bonferroni)")
            plt.tight_layout()
            fname = f"win_matrix_{role}_{config}.png"
            save_both(fig, (VALID_DIR, fname))
            plt.close(fig)
            print("wrote", fname)


def main():
    llm_vs_engines()
    llm_by_tier_family()
    win_matrices()


if __name__ == "__main__":
    main()
