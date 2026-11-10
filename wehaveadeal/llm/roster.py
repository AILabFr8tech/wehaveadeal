"""Single source of truth for the LLM validation roster: which Ollama tags
are in it, and which size tier / model family each belongs to. Imported by
`scripts/run_llm_validation.py` (to stamp every logged row) and
`notebooks/analyze_llm_validation.ipynb` (to group/order results), so the
two never drift apart on what a given tag's tier or family is.

Tiers are by parameter count: tiny <3B, small 4-14B, medium 15-70B. Three
families (Llama, Qwen, DeepSeek R1 distills) each contribute one model per
tier, so a tier-level conclusion never rests on a single architecture, and
the two same-family pairs per tier let the analysis also isolate a pure
scale effect within one architecture.
"""
from __future__ import annotations

TIER_ORDER = ["tiny", "small", "medium"]
FAMILY_ORDER = ["llama", "qwen", "deepseek-r1"]

# model tag -> (family, tier)
MODEL_ROSTER: dict[str, tuple[str, str]] = {
    "llama3.2:1b": ("llama", "tiny"),
    "qwen2.5:1.5b": ("qwen", "tiny"),
    "deepseek-r1:1.5b": ("deepseek-r1", "tiny"),
    "llama3.1:8b": ("llama", "small"),
    "qwen2.5:7b": ("qwen", "small"),
    "deepseek-r1:7b": ("deepseek-r1", "small"),
    "llama3.3:70b": ("llama", "medium"),
    "qwen2.5:72b": ("qwen", "medium"),
    "deepseek-r1:70b": ("deepseek-r1", "medium"),
}

DEFAULT_MODELS = list(MODEL_ROSTER)


def family_of(model: str) -> str:
    return MODEL_ROSTER.get(model, (None, None))[0] or "unknown"


def tier_of(model: str) -> str:
    return MODEL_ROSTER.get(model, (None, None))[1] or "unknown"
