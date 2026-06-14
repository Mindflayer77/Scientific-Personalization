#!/usr/bin/env python
"""
Evaluate groundedness, relevancy and persona_adherence metrics for trained and
trained_fs models from a generation CSV.

Three quality metrics evaluated per model (trained / trained_fs),
based on the answer text only (reasoning excluded):
  groundedness, relevancy, persona_adherence

Usage
-----
  python scripts/metrics_from_csv.py \\
      --csv hypotheses/final_3/clean/tmp/eval_results_generate_dpo.csv \\
      --provider gemini \\
      --gemini-project-id my-gcp-project \\
      --wandb-project personalization-dpo \\
      --wandb-run-name metrics-run-1 \\
      --concurrency 5

  python scripts/metrics_from_csv.py \\
      --csv eval_results/eval_results_run-name.csv \\
      --provider openai \\
      --model gpt-4o \\
      --env-file .env \\
      --wandb-project personalization-dpo

  python scripts/metrics_from_csv.py \\
      --csv eval_results/eval_results_run-name.csv \\
      --provider deepseek \\
      --model deepseek-chat \\
      --env-file .env \\
      --wandb-project personalization-dpo
"""

import csv
import json
import os
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
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR       = Path(__file__).resolve().parent
_REPO_ROOT        = _SCRIPT_DIR.parent
_EVAL3_TEMPLATES  = _REPO_ROOT / "prompt_templates" / "eval_3"
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
MODELS = ["trained", "trained_fs"]

_PROVIDER_DEFAULTS = {
    "openai":   "gpt-4o",
    "deepseek": "deepseek-chat",
    "gemini":   "gemini-2.5-pro",
}
_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


# ── Output columns ────────────────────────────────────────────────────────────

def _build_output_cols() -> list[str]:
    cols = ["user_id", "question", "context"]
    for model in MODELS:
        for metric in METRICS:
            cols += [f"{model}_{metric}_score", f"{model}_{metric}_reasoning"]
    return cols


OUTPUT_COLS = _build_output_cols()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate groundedness and relevancy metrics for trained and "
            "trained_fs models from a generation CSV, then log to W&B."
        )
    )
    p.add_argument("--csv", required=True,
                   help="Path to generation CSV. Required columns: user_id, question, "
                        "context, trained_response. trained_fs_response is optional.")
    p.add_argument("--personas-path", default=None,
                   help="Path to a personas JSON file or directory. When omitted, "
                        "falls back to 'persona_description' column in the CSV.")
    p.add_argument("--output", default=None,
                   help="Output CSV path. Defaults to <input>_metrics.csv.")
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
    p.add_argument("--metric-templates-dir", default=str(_METRIC_TEMPLATES),
                   help="Directory with per-metric templates (default: eval_3/).")
    p.add_argument("--max-retries", type=int, default=3,
                   help="Max retries per LLM call (default: 3).")
    p.add_argument("--concurrency", type=int, default=5,
                   help="Number of concurrent row evaluations (default: 5).")
    p.add_argument("--user-id", default=None,
                   help="Only evaluate rows for this user_id. Processes all users if omitted.")
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


def _parse_score_from_text(
    text: str, valid_scores: frozenset[int] = frozenset(range(4))
) -> int | None:
    """Extract a single integer score from free-form text, or None."""
    clean = text.strip()
    if clean in {str(i) for i in valid_scores}:
        return int(clean)
    matches = [int(m) for m in re.findall(r'\b(\d)\b', clean) if int(m) in valid_scores]
    if len(matches) == 1:
        return matches[0]
    return None


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


# ── Low-level LLM call helpers ────────────────────────────────────────────────

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
    import random
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


# ── Plot generation ───────────────────────────────────────────────────────────

def save_plots(
    df: pd.DataFrame,
    output_dir: Path,
    run_name: str = "",
    active_models: list[str] = MODELS,
) -> None:
    """
    Save a metrics bar chart to *output_dir*:
      metrics_bar[_<run_name>].png — grouped bars per model for each metric
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _suffix = f"_{run_name}" if run_name else ""

    metric_labels = {
        "groundedness":      "Groundedness (0–3)",
        "relevancy":         "Relevancy (0–3)",
        "persona_adherence": "Persona Adherence (0–3)",
    }
    palette_m = {"trained": "#1565C0", "trained_fs": "#6A1B9A"}

    metric_rows = []
    for model in active_models:
        for metric in METRICS:
            col = f"{model}_{metric}_score"
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            for v in vals:
                metric_rows.append({"Model": model, "Metric": metric, "Score": v})

    if not metric_rows:
        return

    mdf = pd.DataFrame(metric_rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, metric in enumerate(METRICS):
        ax = axes[i]
        subset = mdf[mdf["Metric"] == metric]
        if subset.empty:
            ax.set_visible(False)
            continue
        sns.barplot(
            data=subset, x="Model", y="Score",
            hue="Model", hue_order=active_models,
            palette=palette_m, ax=ax,
            order=active_models, capsize=0.06, errorbar="sd", legend=False,
        )
        ax.set_title(metric_labels[metric], fontsize=12, fontweight="bold", pad=6)
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_ylim(0, METRIC_MAX_SCORES[metric] + 0.6)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette_m[m], label=m)
        for m in active_models
    ]
    fig.legend(handles=handles, title="Model", loc="lower center",
               ncol=len(active_models), bbox_to_anchor=(0.5, -0.04), fontsize=10)
    fig.suptitle("Quality Metrics by Model", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    m_path = output_dir / f"metrics_bar{_suffix}.png"
    plt.savefig(m_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {m_path}")


# ── W&B logging helpers ───────────────────────────────────────────────────────

def _log_metrics_summary(run, df: pd.DataFrame, active_models: list[str] = MODELS) -> None:
    print("\n  [Metric Averages]")
    scalar_logs: dict = {}
    chart_logs: dict = {}
    for metric in METRICS:
        table_data = []
        for model in active_models:
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

    # ── Load CSV ──────────────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    required_cols = {"user_id", "question", "context", "trained_response"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: CSV is missing required columns: {missing}")
        sys.exit(1)

    active_models = [m for m in MODELS if f"{m}_response" in df.columns]
    if not active_models:
        print("ERROR: CSV must contain at least one of: trained_response, trained_fs_response")
        sys.exit(1)
    skipped_models = [m for m in MODELS if m not in active_models]
    if skipped_models:
        print(f"INFO: Column(s) missing — skipping model(s): {skipped_models}")

    # ── Personas ──────────────────────────────────────────────────────────────
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
        else csv_path.with_stem(csv_path.stem + "_metrics")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_keys = _load_processed_keys(output_path)
    all_rows       = df.to_dict("records")
    if args.user_id:
        all_rows = [r for r in all_rows if str(r.get("user_id", "")) == args.user_id]
        if not all_rows:
            print(f"ERROR: no rows found for user_id={args.user_id!r}")
            sys.exit(1)
    remaining      = [
        r for r in all_rows
        if (str(r.get("user_id", "")), str(r.get("question", ""))) not in processed_keys
    ]
    if processed_keys:
        print(f"Resuming — {len(processed_keys)} already done, {len(remaining)} remaining.")

    print(f"Input CSV  : {csv_path}  ({len(df)} rows total, {len(all_rows)} for user, {len(remaining)} to process)")
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
            tags=(args.wandb_tags or []) + ["metrics-from-csv", args.provider],
            config={
                "provider":          args.provider,
                "model":             model_name,
                "input_csv":         str(csv_path),
                "personas_path":     args.personas_path,
                "gemini_project_id": getattr(args, "gemini_project_id", None),
                "gemini_location":   getattr(args, "gemini_location", None),
                "thinking_level":    args.thinking_level,
                "concurrency":       args.concurrency,
                "metrics":           METRICS,
                "models":            MODELS,
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
                # thread_local.client = _OpenAI(base_url="http://localhost:8000/v1")
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

    # ── Per-call helpers ──────────────────────────────────────────────────────

    def _eval_metric(
        metric: str,
        hypothesis: str,
        context: str,
        question: str,
        persona_desc: str = "",
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

    def process_row(row: dict) -> tuple[dict, dict]:
        uid      = str(row.get("user_id", ""))
        question = str(row.get("question", ""))
        context  = str(row.get("context", ""))

        # Build persona description
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

        for model_key in active_models:
            hypothesis = str(row.get(f"{model_key}_response", ""))
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

    token_lock   = threading.Lock()
    g_prompt = g_candidates = g_thoughts = g_total = 0

    output_cols = ["user_id", "question", "context"]
    for _m in active_models:
        for _metric in METRICS:
            output_cols += [f"{_m}_{_metric}_score", f"{_m}_{_metric}_reasoning"]

    write_header = not output_path.exists() or output_path.stat().st_size == 0
    out_f      = open(output_path, "a", encoding="utf-8", newline="")
    writer     = csv.DictWriter(out_f, fieldnames=output_cols, extrasaction="ignore")
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
            desc="Evaluating metrics",
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
    print("Metric Averages")
    print("=" * 72)
    for model in active_models:
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
        _log_metrics_summary(run, full_df, active_models=active_models)
        wandb.finish()

    # ── Plots ─────────────────────────────────────────────────────────────────
    save_plots(full_df, output_path.parent, run_name=run_name, active_models=active_models)

    print("=" * 72)


if __name__ == "__main__":
    main()
