#!/usr/bin/env python
"""
Judge responses from a generation CSV using swap-augmented win-rate and
per-model quality metrics (clarity, groundedness, relevancy, persona_adherence).

Win-Rate Formulae
-----------------
Default mode — swap-augmented (AlpacaEval / MT-Bench style, 2 judge calls per sample):
    s1 = judge(trained, ref)   → 1 if trained wins, 0.5 if tie, 0 if ref wins
    s2 = judge(ref, trained)   → 1 if ref wins,     0.5 if tie, 0 if trained wins
    w_i = (s1 + (1 - s2)) / 2
    WR  = mean(w_i)

--simple-winrate mode — single call with random position swap (1 judge call per sample):
    For each sample i, swap ∼ Bernoulli(0.5).
    If not swapped: judge sees (trained, ref); g_i ∈ {1, 0.5, 0} (1 = first position wins).
    If swapped:     judge sees (ref, trained); same g_i scale.
    w_i = g_i        if not swapped   (order: trained, ref)
    w_i = 1 − g_i   if swapped       (order: ref, trained)
    WR  = mean(w_i)

Two comparisons:
  - trained vs base      (base_response column)
  - trained vs base_fs   (base_fs_response column)

Four quality metrics evaluated per model (trained / base / base_fs),
based on the answer text only (reasoning excluded):
  clarity, groundedness, relevancy, persona_adherence

Usage
-----
  python scripts/judge_from_csv.py \\
      --csv hypotheses/final_3/clean/tmp/eval_results_generate_dpo.csv \\
      --provider gemini \\
      --gemini-project-id my-gcp-project \\
      --personas-path personas/personas_all.json \\
      --wandb-project personalization-dpo \\
      --wandb-run-name judge-run-1 \\
      --concurrency 5

  python scripts/judge_from_csv.py \\
      --csv eval_results/eval_results_run-name.csv \\
      --provider openai \\
      --model gpt-4o \\
      --env-file .env \\
      --personas-path personas/ \\
      --wandb-project personalization-dpo

  python scripts/judge_from_csv.py \\
      --csv eval_results/eval_results_run-name.csv \\
      --provider deepseek \\
      --model deepseek-chat \\
      --env-file .env \\
      --wandb-project personalization-dpo
"""

import csv
import json
import os
import random
import re
import sys
import threading
import time
import argparse
import concurrent.futures
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, select_autoescape
from tqdm import tqdm

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR       = Path(__file__).resolve().parent
_REPO_ROOT        = _SCRIPT_DIR.parent
_EVAL3_TEMPLATES  = _REPO_ROOT / "prompt_templates" / "eval_3"
_JUDGE_TEMPLATES  = _EVAL3_TEMPLATES
_METRIC_TEMPLATES = _EVAL3_TEMPLATES
_DATA_GEN_TPL     = _REPO_ROOT / "prompt_templates" / "data_generation"
_SRC              = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from api_client.gemini_client import GeminiClient, ThinkingLevel          # noqa: E402
from api_client.gemini_parser import GeminiResponseParser                  # noqa: E402

try:
    from openai import OpenAI as _OpenAI
except ImportError:
    _OpenAI = None  # type: ignore[assignment,misc]


# ── Constants ─────────────────────────────────────────────────────────────────

METRICS = ["groundedness", "relevancy", "persona_adherence"]
METRIC_MAX_SCORES = {
    "groundedness":      3,
    "relevancy":         3,
    "persona_adherence": 3,
}
MODELS = ["trained", "trained_fs", "base", "base_fs"]

_PROVIDER_DEFAULTS = {
    "openai":   "gpt-4o",
    "deepseek": "deepseek-chat",
    "gemini":   "gemini-2.5-pro",
}
_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


# ── Output columns ────────────────────────────────────────────────────────────

def _build_output_cols() -> list[str]:
    cols = ["user_id", "question", "context"]
    for comp in ["vs_base", "vs_base_fs", "fs_vs_base", "fs_vs_base_fs"]:
        cols += [
            f"wr_{comp}_verdict_normal",   # judge(A, ref)
            f"wr_{comp}_verdict_swapped",  # judge(ref, A)
            f"wr_{comp}_wi",               # per-row swap-augmented score
        ]
    for model in MODELS:
        for metric in METRICS:
            cols += [f"{model}_{metric}_score", f"{model}_{metric}_reasoning"]
    return cols


OUTPUT_COLS = _build_output_cols()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Judge generation CSV with swap-augmented win-rate "
            "and per-model quality metrics, then log to W&B."
        )
    )
    p.add_argument("--csv", required=True,
                   help="Path to generation CSV. Required columns: user_id, question, "
                        "context, trained_response, base_response, base_fs_response.")
    p.add_argument("--output", default=None,
                   help="Output CSV path. Defaults to <input>_judged.csv.")
    p.add_argument("--provider", default="openai",
                   choices=["openai", "deepseek", "gemini"],
                   help="LLM provider (default: openai).")
    p.add_argument("--model", default=None,
                   help="Model name override. Provider defaults: "
                        "openai→gpt-4o, deepseek→deepseek-chat, gemini→gemini-2.5-pro.")
    p.add_argument("--env-file", default=".env",
                   help="Path to .env file with API keys (default: .env).")
    p.add_argument("--gemini-project-id", default=None,
                   help="GCP project ID. Required when --provider=gemini.")
    p.add_argument("--gemini-location", default="global",
                   help="Vertex AI location (default: global).")
    p.add_argument("--thinking-level", default="high",
                   choices=["low", "medium", "high"],
                   help="Thinking/reasoning effort for metric calls (default: high).")
    p.add_argument("--personas-path", default=None,
                   help="Path to a personas JSON file or directory. When omitted, "
                        "falls back to 'persona_description' column in the CSV.")
    p.add_argument("--judge-templates-dir", default=str(_JUDGE_TEMPLATES),
                   help="Directory containing judge_pairwise.j2 template (default: eval_3/).")
    p.add_argument("--metric-templates-dir", default=str(_METRIC_TEMPLATES),
                   help="Directory with per-metric templates (default: eval_3/).")
    p.add_argument("--max-retries", type=int, default=3,
                   help="Max retries per LLM call (default: 3).")
    p.add_argument("--concurrency", type=int, default=5,
                   help="Number of concurrent row evaluations (default: 5).")
    p.add_argument("--simple-winrate", action="store_true",
                   help="Single-call win-rate with random position swap instead of "
                        "swap-augmented two-call evaluation. Halves judge API calls.")
    # W&B
    p.add_argument("--wandb-project", default=None,
                   help="W&B project name. Skips W&B if not provided.")
    p.add_argument("--wandb-entity", default=None,
                   help="W&B entity (team or username).")
    p.add_argument("--wandb-run-name", default=None,
                   help="W&B run name. Auto-generated if omitted.")
    p.add_argument("--wandb-tags", nargs="*", default=None,
                   help="Optional W&B tags (space-separated).")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_api_key(provider: str, env_file: str) -> str | None:
    load_dotenv(env_file)
    key_map = {"openai": "OPENAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}
    if provider in key_map:
        key = os.getenv(key_map[provider])
        if not key:
            print(f"ERROR: {key_map[provider]} not found in {env_file}")
            sys.exit(1)
        return key
    return None  # gemini: uses project_id instead


def _load_personas_raw(path: str) -> dict[str, dict]:
    path = Path(path)
    if path.is_dir():
        persona_map: dict = {}
        for json_file in sorted(path.glob("*.json")):
            with open(json_file) as f:
                data = json.load(f)
            if isinstance(data, list):
                for entry in data:
                    persona_map[entry["persona_id"]] = entry
            elif "persona_id" in data:
                persona_map[data["persona_id"]] = data
    else:
        with open(path) as f:
            data = json.load(f)
        persona_map = (
            {p["persona_id"]: p for p in data}
            if isinstance(data, list)
            else data
        )
    return persona_map


def _build_persona_text_from_dict(persona: dict, template) -> str:
    return template.render(
        display_name=persona.get("display_name", ""),
        core_philosophy=persona.get("core_philosophy", ""),
        areas_of_expertise=", ".join(persona.get("areas_of_expertise", [])),
        communication_style=persona.get("communication_style", ""),
        what_i_look_for=", ".join(persona.get("what_i_look_for", [])),
        what_i_reject=", ".join(persona.get("what_i_reject", [])),
        hypothesis_signature=", ".join(persona.get("hypothesis_signature", [])),
        vocabulary_markers=", ".join(persona.get("vocabulary_markers", [])),
    )


# ── Resume support ────────────────────────────────────────────────────────────

def _load_processed_keys(output_path: Path) -> set[tuple[str, str]]:
    """Return set of (user_id, question) already written to the output CSV."""
    if not output_path.exists():
        return set()
    processed: set[tuple[str, str]] = set()
    with output_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add((row.get("user_id", ""), row.get("question", "")))
    return processed


# ── Win-rate helpers ──────────────────────────────────────────────────────────

def _verdict_to_s(verdict: str | None) -> float:
    """Map a judge verdict to a score for model A: A→1, tie→0.5, B/None→0."""
    if verdict == "A":
        return 1.0
    if verdict == "tie":
        return 0.5
    return 0.0


def _compute_wi(s1: float, s2: float) -> float:
    """
    Swap-augmented per-sample win score.

    s1 = judge(trained, ref)  → 1 if trained wins, 0.5 if tie, 0 if ref wins
    s2 = judge(ref, trained)  → 1 if ref wins,     0.5 if tie, 0 if trained wins
    w_i = (s1 + (1 - s2)) / 2
    """
    return (s1 + (1.0 - s2)) / 2.0


# ── Output-format parsers ─────────────────────────────────────────────────────

def _parse_verdict_from_text(text: str) -> str | None:
    """Return 'A', 'B', or 'tie' from a plain-text judge response, or None."""
    clean = text.strip().strip('"\' ').rstrip('.').strip()
    if clean == "A":
        return "A"
    if clean == "B":
        return "B"
    if clean.lower() == "tie":
        return "tie"
    return None


def _parse_score_from_text(
    text: str, valid_scores: frozenset[int] = frozenset(range(4))
) -> int | None:
    """Extract a single integer score from free-form text, or None."""
    clean = text.strip()
    # Exact single-token match
    if clean in {str(i) for i in valid_scores}:
        return int(clean)
    # Find all standalone digits in range
    matches = [int(m) for m in re.findall(r'\b(\d)\b', clean) if int(m) in valid_scores]
    if len(matches) == 1:
        return matches[0]
    return None


# ── Low-level LLM call helpers ────────────────────────────────────────────────

def _call_judge_gemini(
    client: GeminiClient,
    model_name: str,
    user_msg: str,
    max_retries: int = 3,
) -> tuple[str | None, str, dict]:
    parser = GeminiResponseParser()
    _empty_tok = {"prompt": 0, "candidates": 0, "thoughts": 0, "total": 0}
    for attempt in range(max_retries):
        try:
            raw = client.query(
                model=model_name,
                system_message="",
                user_message=user_msg,
                temperature=0.1,
            )
            if raw is None:
                tqdm.write(
                    f"  WARN  Gemini judge: None response "
                    f"(attempt {attempt + 1}/{max_retries}), sleeping..."
                )
                time.sleep(5 * (attempt + 1))
                continue
            parsed = parser.parse(raw, parse_as_json=False, include_logprobs=False)
            tokens = {
                "prompt":     getattr(parsed.usage, "prompt_tokens",     0) or 0,
                "candidates": getattr(parsed.usage, "candidates_tokens", 0) or 0,
                "thoughts":   getattr(parsed.usage, "thoughts_tokens",   0) or 0,
                "total":      getattr(parsed.usage, "total_tokens",      0) or 0,
            }
            verdict = _parse_verdict_from_text(parsed.answer)
            if verdict is None:
                tqdm.write(
                    f"  WARN  Gemini judge: unexpected response "
                    f"{str(repr(parsed.answer))[:80]!s} "
                    f"(attempt {attempt + 1}/{max_retries}), retrying..."
                )
                continue
            return verdict, "", tokens
        except Exception as e:
            if attempt < max_retries - 1:
                tqdm.write(
                    f"  RETRY [{attempt + 1}/{max_retries}] Gemini judge: {e}"
                )
                time.sleep(2 ** attempt)
            else:
                tqdm.write(
                    f"  ERROR Gemini judge failed after {max_retries} attempts: {e}"
                )
    return None, "", _empty_tok


def _call_judge_openai(
    client,
    model_name: str,
    user_msg: str,
    max_retries: int = 3,
) -> tuple[str | None, str, dict]:
    _empty_tok = {"prompt": 0, "candidates": 0, "thoughts": 0, "total": 0}
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": user_msg}],
                temperature=0.0,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            )
            raw_text          = (response.choices[0].message.content or "").strip()
            usage_obj         = getattr(response, "usage", None)
            prompt_tok        = getattr(usage_obj, "prompt_tokens",     0) or 0
            completion_tok    = getattr(usage_obj, "completion_tokens", 0) or 0
            total_tok         = getattr(usage_obj, "total_tokens",      0) or 0
            details           = getattr(usage_obj, "completion_tokens_details", None)
            reasoning_tok     = getattr(details,   "reasoning_tokens",  0) or 0
            tokens = {
                "prompt":     prompt_tok,
                "candidates": completion_tok - reasoning_tok,
                "thoughts":   reasoning_tok,
                "total":      total_tok,
            }
            verdict = _parse_verdict_from_text(raw_text)
            if verdict is None:
                tqdm.write(
                    f"  WARN  Judge: unexpected response "
                    f"{str(repr(raw_text))[:80]!s} "
                    f"(attempt {attempt + 1}/{max_retries}), retrying..."
                )
                continue
            return verdict, "", tokens
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "rate_limit" in err.lower()
            if is_rate_limit and attempt < max_retries - 1:
                delay = 5.0 * (2 ** attempt) + random.uniform(0, 2)
                tqdm.write(
                    f"  [429] Judge rate-limited, retry {attempt + 1}/{max_retries} "
                    f"in {delay:.1f}s…"
                )
                time.sleep(delay)
            elif attempt < max_retries - 1:
                tqdm.write(f"  RETRY [{attempt + 1}/{max_retries}] Judge: {e}")
                time.sleep(2 ** attempt)
            else:
                tqdm.write(f"  ERROR Judge failed after {max_retries} attempts: {e}")
    return None, "", _empty_tok


def _call_metric_gemini(
    client: GeminiClient,
    model_name: str,
    thinking_level: ThinkingLevel,
    sys_msg: str,
    usr_msg: str,
    max_retries: int = 3,
) -> tuple[int | None, str, dict]:
    parser = GeminiResponseParser()
    _empty_tok = {"prompt": 0, "candidates": 0, "thoughts": 0, "total": 0}
    for attempt in range(max_retries):
        try:
            response = client.query(
                model_name, sys_msg, usr_msg, thinking_level=thinking_level
            )
            if response is None:
                tqdm.write(
                    f"  WARN  Gemini metric: None response "
                    f"(attempt {attempt + 1}/{max_retries}), sleeping..."
                )
                time.sleep(10)
                continue
            parsed = parser.parse(response, parse_as_json=True)
            tokens = {
                "prompt":     getattr(parsed.usage, "prompt_tokens",     0) or 0,
                "candidates": getattr(parsed.usage, "candidates_tokens", 0) or 0,
                "thoughts":   getattr(parsed.usage, "thoughts_tokens",   0) or 0,
                "total":      getattr(parsed.usage, "total_tokens",      0) or 0,
            }
            ans = parsed.answer_json if isinstance(parsed.answer_json, dict) else {}
            score = _parse_score_from_text(str(ans.get("score", "")))
            if score is None:
                tqdm.write(
                    f"  WARN  Gemini metric: unexpected response "
                    f"{str(repr(parsed.answer))[:80]!s} "
                    f"(attempt {attempt + 1}/{max_retries}), retrying..."
                )
                continue
            return score, parsed.reasoning or "", tokens
        except Exception as e:
            if attempt < max_retries - 1:
                tqdm.write(
                    f"  RETRY [{attempt + 1}/{max_retries}] Gemini metric: {e}"
                )
                time.sleep(2 ** attempt)
            else:
                tqdm.write(
                    f"  ERROR Gemini metric failed after {max_retries} attempts: {e}"
                )
    return None, "", _empty_tok


def _call_metric_openai(
    client,
    model_name: str,
    sys_msg: str,
    usr_msg: str,
    max_retries: int = 3,
) -> tuple[int | None, str, dict]:
    _empty_tok = {"prompt": 0, "candidates": 0, "thoughts": 0, "total": 0}
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": usr_msg},
                ],
                response_format={"type": "json_object"},
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            )
            raw_text          = (response.choices[0].message.content or "").strip()
            usage_obj         = getattr(response, "usage", None)
            prompt_tok        = getattr(usage_obj, "prompt_tokens",     0) or 0
            completion_tok    = getattr(usage_obj, "completion_tokens", 0) or 0
            total_tok         = getattr(usage_obj, "total_tokens",      0) or 0
            details           = getattr(usage_obj, "completion_tokens_details", None)
            reasoning_tok     = getattr(details,   "reasoning_tokens",  0) or 0
            tokens = {
                "prompt":     prompt_tok,
                "candidates": completion_tok - reasoning_tok,
                "thoughts":   reasoning_tok,
                "total":      total_tok,
            }
            ans = json.loads(raw_text) if raw_text else {}
            score = _parse_score_from_text(str(ans.get("score", "")))
            if score is None:
                tqdm.write(
                    f"  WARN  Metric: unexpected response "
                    f"{str(repr(raw_text))[:80]!s} "
                    f"(attempt {attempt + 1}/{max_retries}), retrying..."
                )
                continue
            return score, "", tokens
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "rate_limit" in err.lower()
            if is_rate_limit and attempt < max_retries - 1:
                delay = 5.0 * (2 ** attempt) + random.uniform(0, 2)
                tqdm.write(
                    f"  [429] Metric rate-limited, retry {attempt + 1}/{max_retries} "
                    f"in {delay:.1f}s…"
                )
                time.sleep(delay)
            elif attempt < max_retries - 1:
                tqdm.write(f"  RETRY [{attempt + 1}/{max_retries}] Metric: {e}")
                time.sleep(2 ** attempt)
            else:
                tqdm.write(f"  ERROR Metric failed after {max_retries} attempts: {e}")
    return None, "", _empty_tok


# ── Plot generation ──────────────────────────────────────────────────────────

def save_plots(df: pd.DataFrame, output_dir: Path, run_name: str = "") -> None:
    """
    Save two figures to *output_dir*:
      1. win_rate_bar[_<run_name>].png  — grouped bar chart of win/tie/loss counts per comparison
      2. metrics_bar[_<run_name>].png  — 2×2 subplot grid, one per metric, grouped bars per model
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _suffix = f"_{run_name}" if run_name else ""

    # ── 1. Win-rate bar chart ─────────────────────────────────────────────────
    wr_data = []
    for label, wi_col in [
        ("trained vs base",       "wr_vs_base_wi"),
        ("trained vs base_fs",    "wr_vs_base_fs_wi"),
        ("trained_fs vs base",    "wr_fs_vs_base_wi"),
        ("trained_fs vs base_fs", "wr_fs_vs_base_fs_wi"),
    ]:
        vals = pd.to_numeric(df.get(wi_col, pd.Series([])), errors="coerce").dropna()
        if not len(vals):
            continue
        wr_data.append({"Comparison": label, "Outcome": "Win (w>0.5)",   "Count": int((vals > 0.5).sum())})
        wr_data.append({"Comparison": label, "Outcome": "Tie (w=0.5)",   "Count": int((vals == 0.5).sum())})
        wr_data.append({"Comparison": label, "Outcome": "Loss (w<0.5)",  "Count": int((vals < 0.5).sum())})
        wr_data.append({"Comparison": label, "Outcome": "Win Rate",       "Count": round(float(vals.mean()), 3)})

    if wr_data:
        wr_df = pd.DataFrame(wr_data)
        count_df = wr_df[wr_df["Outcome"] != "Win Rate"]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        palette_wr = {"Win (w>0.5)": "#2196F3", "Tie (w=0.5)": "#9E9E9E", "Loss (w<0.5)": "#FF7043"}
        sns.barplot(
            data=count_df, x="Comparison", y="Count", hue="Outcome",
            hue_order=["Win (w>0.5)", "Tie (w=0.5)", "Loss (w<0.5)"],
            palette=palette_wr, ax=axes[0],
        )
        axes[0].set_title("Win / Tie / Loss Counts (trained model)", fontweight="bold")
        axes[0].set_xlabel("Comparison")
        axes[0].set_ylabel("Count")
        axes[0].legend(title="Outcome")

        # Win-rate values as horizontal bar
        wr_summary = [
            (label, float(pd.to_numeric(df.get(col, pd.Series([])), errors="coerce").dropna().mean()))
            for label, col in [
                ("trained vs base",       "wr_vs_base_wi"),
                ("trained vs base_fs",    "wr_vs_base_fs_wi"),
                ("trained_fs vs base",    "wr_fs_vs_base_wi"),
                ("trained_fs vs base_fs", "wr_fs_vs_base_fs_wi"),
            ]
            if len(df.get(col, pd.Series([])).dropna()) > 0
        ]
        if wr_summary:
            labels_wr, values_wr = zip(*wr_summary)
            colors = ["#2196F3" if v > 0.5 else "#FF7043" for v in values_wr]
            bars = axes[1].barh(labels_wr, values_wr, color=colors, height=0.4)
            axes[1].axvline(0.5, color="black", linestyle="--", linewidth=1, label="0.5 baseline")
            axes[1].set_xlim(0, 1)
            axes[1].set_xlabel("Win Rate")
            axes[1].set_title("Swap-Augmented Win Rate (trained model)", fontweight="bold")
            for bar, val in zip(bars, values_wr):
                axes[1].text(
                    val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=10,
                )
            axes[1].legend()

        fig.suptitle("Win-Rate Evaluation: Trained vs Reference Models", fontsize=14, fontweight="bold")
        plt.tight_layout()
        wr_path = output_dir / f"win_rate_bar{_suffix}.png"
        plt.savefig(wr_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved → {wr_path}")

    # ── 2. Metrics 2×2 grid ───────────────────────────────────────────────────
    metric_labels = {
        "clarity":           "Clarity (0–3)",
        "groundedness":      "Groundedness (0–4)",
        "relevancy":         "Relevancy (0–4)",
        "persona_adherence": "Persona Adherence (0–3)",
    }
    palette_m = {"trained": "#1565C0", "trained_fs": "#6A1B9A", "base": "#FF8F00", "base_fs": "#2E7D32"}

    # Build long-form DataFrame for seaborn
    metric_rows = []
    for model in MODELS:
        for metric in METRICS:
            col = f"{model}_{metric}_score"
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            for v in vals:
                metric_rows.append({"Model": model, "Metric": metric, "Score": v})

    if metric_rows:
        mdf = pd.DataFrame(metric_rows)
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        axes_flat = axes.flatten()

        for i, metric in enumerate(METRICS):
            ax = axes_flat[i]
            subset = mdf[mdf["Metric"] == metric]
            if subset.empty:
                ax.set_visible(False)
                continue
            sns.barplot(
                data=subset, x="Model", y="Score",
                hue="Model", hue_order=MODELS,
                palette=palette_m, ax=ax,
                order=MODELS, capsize=0.06, errorbar="sd", legend=False,
            )
            ax.set_title(metric_labels[metric], fontsize=12, fontweight="bold", pad=6)
            ax.set_xlabel("Model")
            ax.set_ylabel("Score")
            ax.set_ylim(0, METRIC_MAX_SCORES[metric] + 0.6)

        # shared legend
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=palette_m[m], label=m)
            for m in MODELS
        ]
        fig.legend(handles=handles, title="Model", loc="lower center",
                   ncol=len(MODELS), bbox_to_anchor=(0.5, -0.02), fontsize=10)
        fig.suptitle("Quality Metrics by Model", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        m_path = output_dir / f"metrics_bar{_suffix}.png"
        plt.savefig(m_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved → {m_path}")


# ── W&B logging helpers ───────────────────────────────────────────────────────

def _log_winrate_summary(run, comparison: str, wi_values: list[float]) -> None:
    if not wi_values:
        return
    win_rate = sum(wi_values) / len(wi_values)
    wins   = sum(1 for w in wi_values if w > 0.5)
    ties   = sum(1 for w in wi_values if w == 0.5)
    losses = sum(1 for w in wi_values if w < 0.5)
    n = len(wi_values)

    wandb.log({
        f"{comparison}/win_rate": win_rate,
        f"{comparison}/wins":     wins,
        f"{comparison}/ties":     ties,
        f"{comparison}/losses":   losses,
        f"{comparison}/n":        n,
        f"{comparison}/judgment_distribution": wandb.plot.bar(
            wandb.Table(
                data=[["Wins (trained)", wins], ["Ties", ties], ["Losses (ref)", losses]],
                columns=["label", "value"],
            ),
            "label", "value",
            title=f"{comparison} — Win Distribution",
        ),
    })
    run.summary[f"{comparison}/win_rate"] = win_rate
    run.summary[f"{comparison}/n"]        = n

    print(f"\n  [{comparison}]  WR = {win_rate:.3f} ({win_rate:.1%})")
    print(f"    Wins   (w > 0.5): {wins}/{n}")
    print(f"    Ties   (w = 0.5): {ties}/{n}")
    print(f"    Losses (w < 0.5): {losses}/{n}")


def _log_metrics_summary(run, df: pd.DataFrame) -> None:
    print("\n  [Metric Averages]")
    scalar_logs: dict = {}
    chart_logs: dict = {}
    for metric in METRICS:
        table_data = []
        for model in MODELS:
            col = f"{model}_{metric}_score"
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if not len(vals):
                continue
            avg = float(vals.mean())
            scalar_logs[f"metrics/{model}/{metric}"] = avg
            run.summary[f"metrics/{model}/{metric}_avg"] = avg
            table_data.append([model, round(avg, 4)])
            print(f"    {model}/{metric}: {avg:.3f} (n={len(vals)}, max={METRIC_MAX_SCORES[metric]})")
        if table_data:
            chart_logs[f"metrics/charts/{metric}"] = wandb.plot.bar(
                wandb.Table(data=table_data, columns=["Model", "Mean Score"]),
                "Model", "Mean Score",
                title=f"{metric.replace('_', ' ').title()} — Mean Score by Model",
            )
    if scalar_logs:
        wandb.log(scalar_logs)
    if chart_logs:
        wandb.log(chart_logs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    model_name     = args.model or _PROVIDER_DEFAULTS[args.provider]
    thinking_level = ThinkingLevel(args.thinking_level)

    # ── Resolve credentials ───────────────────────────────────────────────────
    if args.provider == "gemini":
        if not args.gemini_project_id:
            print("ERROR: --gemini-project-id is required when --provider=gemini")
            sys.exit(1)
        project_id = args.gemini_project_id
        api_key    = None
    else:
        project_id = None
        api_key    = _resolve_api_key(args.provider, args.env_file)

    # ── Load templates ────────────────────────────────────────────────────────
    judge_env = Environment(
        loader=FileSystemLoader(args.judge_templates_dir),
        autoescape=select_autoescape([]),
        keep_trailing_newline=True,
    )
    # judge_tpl = judge_env.get_template("judge_pairwise.j2")
    judge_tpl = judge_env.get_template("judge.j2")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    # df = df.groupby('user_id').sample(10, random_state=42)

    required_cols = {
        "user_id", "question", "context",
        "trained_response", "trained_fs_response", "base_response", "base_fs_response",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: CSV is missing required columns: {missing}")
        sys.exit(1)

    # ── Personas ──────────────────────────────────────────────────────────────
    # persona_lookup is only used as a plain-text fallback when no personas_path
    # is provided (i.e. persona_raw is empty and the CSV has a description column).
    persona_lookup: dict[str, str] = {}
    persona_raw:    dict[str, dict] = {}
    if args.personas_path:
        print(f"Loading personas from: {args.personas_path}")
        persona_raw = _load_personas_raw(args.personas_path)
        print(f"  → {len(persona_raw)} personas loaded.")
    elif "persona_description" in df.columns:
        persona_lookup = dict(zip(
            df["user_id"].astype(str),
            df["persona_description"].fillna("").astype(str),
        ))

    # ── Output path & resume ──────────────────────────────────────────────────
    output_path = (
        Path(args.output) if args.output
        else csv_path.with_stem(csv_path.stem + "_judged")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_keys = _load_processed_keys(output_path)
    all_rows       = df.to_dict("records")
    remaining      = [
        r for r in all_rows
        if (str(r.get("user_id", "")), str(r.get("question", ""))) not in processed_keys
    ]
    if processed_keys:
        print(f"Resuming — {len(processed_keys)} already done, {len(remaining)} remaining.")

    print(f"Input CSV  : {csv_path}  ({len(df)} rows, {len(remaining)} to process)")
    print(f"Output CSV : {output_path}")
    print(f"Provider   : {args.provider}  ({model_name})")
    print(f"Concurrency: {args.concurrency}")

    # ── Init W&B ──────────────────────────────────────────────────────────────
    use_wandb = bool(args.wandb_project)
    run = None
    if use_wandb:
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            tags=(args.wandb_tags or []) + ["judge-from-csv", args.provider],
            config={
                "provider":          args.provider,
                "model":             model_name,
                "input_csv":         str(csv_path),
                "personas_path":     args.personas_path,
                "gemini_project_id": args.gemini_project_id,
                "gemini_location":   args.gemini_location,
                "thinking_level":    args.thinking_level,
                "concurrency":       args.concurrency,
            },
        )

    # ── Thread-local clients and Jinja2 envs ──────────────────────────────────
    thread_local = threading.local()

    def _get_client():
        if not hasattr(thread_local, "client"):
            if args.provider == "gemini":
                thread_local.client = GeminiClient(
                    project_id=project_id, location=args.gemini_location
                )
            else:
                # base_url = #_DEEPSEEK_BASE_URL if args.provider == "deepseek" else None
                # thread_local.client = _OpenAI(api_key=api_key, base_url=base_url)
                # thread_local.client = _OpenAI(api_key=os.getenv("CLARIN_API_KEY"),
                #     base_url="https://services.clarin-pl.eu/api/v1/oapi")
                thread_local.client = _OpenAI(base_url="http://localhost:8000/v1")
        return thread_local.client

    def _get_metric_env():
        if not hasattr(thread_local, "metric_env"):
            thread_local.metric_env = Environment(
                loader=FileSystemLoader(args.metric_templates_dir),
                keep_trailing_newline=True,
            )
        return thread_local.metric_env

    def _get_persona_tpl():
        if not hasattr(thread_local, "persona_tpl"):
            env = Environment(
                loader=FileSystemLoader(str(_DATA_GEN_TPL)),
                keep_trailing_newline=True,
            )
            thread_local.persona_tpl = env.get_template("persona_template_2.j2")
        return thread_local.persona_tpl

    # ── Per-call helpers (closures capture thread-local accessors) ─────────────

    def _judge_once(
        hypothesis_a: str,
        hypothesis_b: str,
        persona_desc: str,
        question: str,
        uid: str = "",
        label: str = "",
    ) -> tuple[str | None, dict]:
        """Single deterministic judge call: A vs B (no internal shuffling)."""
        user_msg = judge_tpl.render(
            persona_description=persona_desc,
            research_question=question,
            hypothesis_a=hypothesis_a,
            hypothesis_b=hypothesis_b,
        )
        client = _get_client()
        if args.provider == "gemini":
            verdict, _, tokens = _call_judge_gemini(client, model_name, user_msg, args.max_retries)
        else:
            verdict, _, tokens = _call_judge_openai(client, model_name, user_msg, args.max_retries)
        tqdm.write(
            f"  [{uid:20s}][judge {label:20s}] verdict={verdict}  tok={tokens['total']}"
        )
        return verdict, tokens

    _VERDICT_FLIP = {"A": "B", "B": "A", "tie": "tie"}

    def _compare_pair(
        model_a: str,
        model_b: str,
        persona_desc: str,
        question: str,
        uid: str,
        label: str,
    ) -> tuple[str | None, str | None, float, dict]:
        """Judge model_a vs model_b, returning (verdict_normal, verdict_swapped, wi, tokens).

        Simple mode (--simple-winrate): one call with random position swap;
          verdict_swapped is None, wi is 1 / 0.5 / 0.
        Default mode: two calls (swap-augmented); wi = (s1 + 1 - s2) / 2.
        """
        if args.simple_winrate:
            swapped = random.random() < 0.5
            if swapped:
                v, tok = _judge_once(model_b, model_a, persona_desc, question,
                                     uid=uid, label=f"{label}[swap]")
                verdict = _VERDICT_FLIP.get(v) if v is not None else None
            else:
                verdict, tok = _judge_once(model_a, model_b, persona_desc, question,
                                           uid=uid, label=label)
            return verdict, None, _verdict_to_s(verdict), tok
        else:
            v1, tok1 = _judge_once(model_a, model_b, persona_desc, question,
                                   uid=uid, label=f"{label} normal ")
            v2, tok2 = _judge_once(model_b, model_a, persona_desc, question,
                                   uid=uid, label=f"{label} swapped")
            wi = _compute_wi(_verdict_to_s(v1), _verdict_to_s(v2))
            combined = {k: tok1.get(k, 0) + tok2.get(k, 0) for k in tok1}
            return v1, v2, wi, combined

    def _eval_metric(
        metric: str,
        hypothesis: str,
        context: str,
        question: str,
        persona_desc: str,
        uid: str = "",
        model_key: str = "",
    ) -> tuple[int | None, str, dict]:
        """Evaluate one metric for one hypothesis (answer text only, no reasoning)."""
        env     = _get_metric_env()
        sys_msg = env.get_template(f"{metric}_system.j2").render()
        kwargs: dict = {"hypothesis": hypothesis}
        if metric == "groundedness":
            kwargs["evidence"] = context
        elif metric == "relevancy":
            kwargs["query"] = question
        elif metric == "persona_adherence":
            kwargs["persona"] = persona_desc
        # Render with Jinja2 (handles {{ var }}), then replace any remaining
        # Python-format-style {var} placeholders (used in groundedness/relevancy).
        usr_msg = env.get_template(f"{metric}_user.j2").render(**kwargs)
        for _k, _v in kwargs.items():
            usr_msg = usr_msg.replace(f"{{{_k}}}", str(_v))

        client = _get_client()
        if args.provider == "gemini":
            score, reasoning, tokens = _call_metric_gemini(
                client, model_name, thinking_level, sys_msg, usr_msg, args.max_retries
            )
        else:
            score, reasoning, tokens = _call_metric_openai(
                client, model_name, sys_msg, usr_msg, args.max_retries
            )
        tqdm.write(
            f"  [{uid:20s}][{model_key:8s}][{metric:20s}] "
            f"score={score}  tok={tokens['total']}"
        )
        return score, reasoning, tokens

    # ── Row processor ─────────────────────────────────────────────────────────

    def process_row(row: dict) -> dict:
        uid        = str(row.get("user_id", ""))
        question   = str(row.get("question", ""))
        context    = str(row.get("context", ""))
        trained    = str(row.get("trained_response", ""))
        trained_fs = str(row.get("trained_fs_response", ""))
        base       = str(row.get("base_response", ""))
        base_fs    = str(row.get("base_fs_response", ""))

        # Build persona description — always from the raw persona dict when available
        persona_data = persona_raw.get(uid, {})
        if persona_data:
            persona_desc = _build_persona_text_from_dict(persona_data, _get_persona_tpl())
        else:
            persona_desc = persona_lookup.get(uid, "")

        out: dict = {"user_id": uid, "question": question, "context": context}
        row_prompt = row_candidates = row_thoughts = row_total = 0

        def _acc(tok: dict) -> None:
            nonlocal row_prompt, row_candidates, row_thoughts, row_total
            row_prompt     += tok.get("prompt", 0)
            row_candidates += tok.get("candidates", 0)
            row_thoughts   += tok.get("thoughts", 0)
            row_total      += tok.get("total", 0)

        # ── Win-rate comparisons ───────────────────────────────────────────
        for a, b, prefix, lbl in [
            (trained,    base,    "vs_base",       "vs_base      "),
            (trained,    base_fs, "vs_base_fs",    "vs_base_fs   "),
            (trained_fs, base,    "fs_vs_base",    "fs_vs_base   "),
            (trained_fs, base_fs, "fs_vs_base_fs", "fs_vs_base_fs"),
        ]:
            vn, vs, wi, tok = _compare_pair(a, b, persona_desc, question, uid=uid, label=lbl)
            _acc(tok)
            out[f"wr_{prefix}_verdict_normal"]  = vn
            out[f"wr_{prefix}_verdict_swapped"] = vs
            out[f"wr_{prefix}_wi"]              = wi

        # ── Quality metrics for each model (answer only, no reasoning) ─────
        for model_key, hypothesis in [
            ("trained",    trained),
            ("trained_fs", trained_fs),
            ("base",       base),
            ("base_fs",    base_fs),
        ]:
            for metric in METRICS:
                score, reasoning, tok = _eval_metric(
                    metric, hypothesis, context, question, persona_desc,
                    uid=uid, model_key=model_key,
                )
                _acc(tok)
                out[f"{model_key}_{metric}_score"]     = score
                out[f"{model_key}_{metric}_reasoning"] = reasoning

        tqdm.write(
            f"  [{uid:20s}] row done — "
            f"total={row_total} prompt={row_prompt} "
            f"candidates={row_candidates} thoughts={row_thoughts}"
        )
        row_tokens = {
            "prompt": row_prompt, "candidates": row_candidates,
            "thoughts": row_thoughts, "total": row_total,
        }
        return out, row_tokens

    # ── Concurrent evaluation loop ────────────────────────────────────────────

    # Global token counters (updated under token_lock)
    token_lock   = threading.Lock()
    g_prompt = g_candidates = g_thoughts = g_total = 0

    write_header = not output_path.exists() or output_path.stat().st_size == 0
    out_f     = open(output_path, "a", encoding="utf-8", newline="")
    writer    = csv.DictWriter(out_f, fieldnames=OUTPUT_COLS, extrasaction="ignore")
    write_lock = threading.Lock()
    if write_header:
        writer.writeheader()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.concurrency)
    ) as executor:
        futures = {executor.submit(process_row, row): row for row in remaining}
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Judging & evaluating",
            unit="row",
        ):
            try:
                out_row, row_tok = future.result()
            except Exception as e:
                print(f"Row processing failed: {e}")
                continue
            with write_lock:
                writer.writerow(out_row)
                out_f.flush()
            with token_lock:
                g_prompt     += row_tok.get("prompt", 0)
                g_candidates += row_tok.get("candidates", 0)
                g_thoughts   += row_tok.get("thoughts", 0)
                g_total      += row_tok.get("total", 0)

    out_f.close()
    print(
        f"\nResults saved → {output_path}"
        f"\nTotal tokens : {g_total} "
        f"(prompt={g_prompt}, candidates={g_candidates}, thoughts={g_thoughts})"
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    full_df = pd.read_csv(output_path)

    print("\n" + "=" * 72)
    print("Win-Rate Summary")
    print("=" * 72)
    comparisons = [
        ("trained vs base",       "wr_vs_base_wi",       "trained_vs_base"),
        ("trained vs base_fs",    "wr_vs_base_fs_wi",    "trained_vs_base_fs"),
        ("trained_fs vs base",    "wr_fs_vs_base_wi",    "trained_fs_vs_base"),
        ("trained_fs vs base_fs", "wr_fs_vs_base_fs_wi", "trained_fs_vs_base_fs"),
    ]
    for label, wi_col, wandb_key in comparisons:
        wi_vals = pd.to_numeric(full_df.get(wi_col, pd.Series([])), errors="coerce").dropna().tolist()
        if not wi_vals:
            print(f"\n  [{label}]  no data")
            continue
        wr = sum(wi_vals) / len(wi_vals)
        wins   = sum(1 for w in wi_vals if w > 0.5)
        ties   = sum(1 for w in wi_vals if w == 0.5)
        losses = sum(1 for w in wi_vals if w < 0.5)
        n = len(wi_vals)
        print(f"\n  [{label}]  WR = {wr:.3f} ({wr:.1%})")
        print(f"    Wins   (w > 0.5): {wins}/{n}")
        print(f"    Ties   (w = 0.5): {ties}/{n}")
        print(f"    Losses (w < 0.5): {losses}/{n}")
        if use_wandb and run:
            _log_winrate_summary(run, wandb_key, wi_vals)

    print("\n" + "=" * 72)
    print("Metric Averages")
    print("=" * 72)
    for model in MODELS:
        print(f"\n  [{model}]")
        for metric in METRICS:
            col = f"{model}_{metric}_score"
            if col not in full_df.columns:
                continue
            vals = pd.to_numeric(full_df[col], errors="coerce").dropna()
            if len(vals):
                print(
                    f"    {metric:20s}: {vals.mean():.3f} ± {vals.std():.3f}"
                    f"  (n={len(vals)}, max={METRIC_MAX_SCORES[metric]})"
                )

    run_name = ""
    if use_wandb and run:
        run_name = run.name or ""
        _log_metrics_summary(run, full_df)
        wandb.finish()

    # ── Plots ─────────────────────────────────────────────────────────────────
    save_plots(full_df, output_path.parent, run_name=run_name)

    print("=" * 72)


if __name__ == "__main__":
    main()
