#!/usr/bin/env python3
"""Evaluate chosen and rejected hypotheses using 4 LLM-judge metrics.

For each sampled hypothesis, 3 sequential LLM calls are made — one per
metric:

  1. Groundedness     (0–3)  — concepts traceable to the evidence context
  2. Relevancy        (0–3)  — alignment with the research question
  3. Persona Adherence (0–3) — how well the hypothesis embodies the persona

Sampling:
  --n-per-persona chosen hypotheses are drawn per persona (chosen_persona col).
  --n-per-persona rejected hypotheses are drawn per persona (rejected_persona col).

Templates are read from prompt_templates/eval_2/.
Personas are loaded from personas/personas_all.json.

Results are appended to --output CSV (resume-safe).
A seaborn 2×2 subplot figure is saved to --plot after all evaluations finish.

Usage (Gemini):
    python scripts_2/evaluate_hypotheses.py \\
        --input  hypotheses/dpo_neighbourhood.csv \\
        --output hypotheses/eval_results.csv \\
        --plot   hypotheses/eval_metrics_plot.png \\
        --n-per-persona 10 \\
        --provider gemini \\
        --model gemini-2.5-pro \\
        --thinking-level high \\
        --concurrency 10

Usage (OpenAI):
    python scripts_2/evaluate_hypotheses.py \\
        --input  hypotheses/dpo_neighbourhood.csv \\
        --output hypotheses/eval_results.csv \\
        --provider openai \\
        --model o3 \\
        --thinking-level high \\
        --concurrency 5
"""

import argparse
import concurrent.futures
import csv
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api_client.gemini_client import GeminiClient, ThinkingLevel
from api_client.gemini_parser import GeminiResponseParser

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

from enum import StrEnum

class LLMProvider(StrEnum):
    gemini = "gemini"
    openai = "openai"

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERSONAS_FILE = PROJECT_ROOT / "personas" / "personas_all.json"
EVAL_TEMPLATES_DIR = PROJECT_ROOT / "prompt_templates" / "eval_3"
DATA_GEN_TEMPLATES_DIR = PROJECT_ROOT / "prompt_templates" / "data_generation"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METRICS = ["groundedness", "relevancy", "persona_adherence"]
METRIC_MAX_SCORES = {
    "groundedness": 3,
    "relevancy": 3,
    "persona_adherence": 3,
}
OUTPUT_COLS = [
    "node_id",
    "paper_id",
    "persona",
    "persona_display_name",
    "hypothesis_type",
    "hypothesis",
    "context",
    "question",
    "groundedness_score",
    "groundedness_reasoning",
    "relevancy_score",
    "relevancy_reasoning",
    "persona_adherence_score",
    "persona_adherence_reasoning",
]


# ---------------------------------------------------------------------------
# Persona helpers
# ---------------------------------------------------------------------------

def load_personas(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return {p["persona_id"]: p for p in raw}
    return raw


def build_persona_text(persona: dict, template) -> str:
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


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_hypotheses(
    df: pd.DataFrame,
    persona_ids: list[str],
    n: int,
    seed: int = 42,
) -> list[dict]:
    """Return a flat list of hypothesis dicts sampled from the DataFrame.

    For each persona: N rows from chosen_persona column (chosen hypothesis)
    and N rows from rejected_persona column (rejected hypothesis).
    """
    samples: list[dict] = []
    for pid in persona_ids:
        # --- chosen ---
        chosen_rows = df[df["chosen_persona"] == pid]
        chosen_sample = chosen_rows.sample(
            min(n, len(chosen_rows)), random_state=seed
        )
        for _, row in chosen_sample.iterrows():
            # "chosen" is the column written by the current generate scripts;
            # fall back to "chosen_hypothesis" for older CSVs.
            hyp = row.get("chosen") or row.get("chosen_hypothesis", "")
            if not hyp or (isinstance(hyp, float)):
                continue
            samples.append({
                "node_id": str(row.get("node_id", "")),
                "paper_id": str(row.get("paper_id", "")),
                "persona": pid,
                "hypothesis_type": "chosen",
                "hypothesis": str(hyp),
                "context": str(row.get("context", "")),
                "question": str(row.get("question", "")),
            })

        # --- rejected ---
        rejected_rows = df[df["rejected_persona"] == pid]
        rejected_sample = rejected_rows.sample(
            min(n, len(rejected_rows)), random_state=seed
        )
        for _, row in rejected_sample.iterrows():
            hyp = row.get("rejected", "")
            if not hyp or (isinstance(hyp, float)):
                continue
            samples.append({
                "node_id": str(row.get("node_id", "")),
                "paper_id": str(row.get("paper_id", "")),
                "persona": pid,
                "hypothesis_type": "rejected",
                "hypothesis": str(hyp),
                "context": str(row.get("context", "")),
                "question": str(row.get("question", "")),
            })
    return samples


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def load_processed_keys(output_path: Path) -> set[tuple]:
    """Return a set of (node_id, persona, hypothesis_type) already written."""
    if not output_path.exists():
        return set()
    processed: set[tuple] = set()
    with output_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row.get("node_id", ""),
                row.get("persona", ""),
                row.get("hypothesis_type", ""),
            )
            processed.add(key)
    return processed


# ---------------------------------------------------------------------------
# Metric evaluation
# ---------------------------------------------------------------------------

_METRIC_MAX_RETRIES = 3
_METRIC_RETRY_BASE_DELAY = 5.0


def call_metric(
    metric: str,
    client,
    parser: GeminiResponseParser,
    jinja_env,
    model: str,
    thinking_level: ThinkingLevel,
    provider: LLMProvider = LLMProvider.gemini,
    **template_kwargs,
) -> tuple[int | None, str, dict]:
    """Render templates, call the LLM judge, parse JSON response.

    Supports both Gemini (via GeminiClient) and OpenAI (via openai.OpenAI).
    Returns (score, reasoning, usage_dict).
    Retries up to _METRIC_MAX_RETRIES times on None response, JSON parse
    failure, or an out-of-range / missing score value.
    On exhausted retries returns (None, "", {}).
    """
    sys_tpl = jinja_env.get_template(f"{metric}_system.j2")
    usr_tpl = jinja_env.get_template(f"{metric}_user.j2")
    sys_msg = sys_tpl.render()
    usr_msg = usr_tpl.render(**template_kwargs)
    # Handle Python-format-style {var} placeholders not processed by Jinja2
    # (groundedness_user.j2 and relevancy_user.j2 use {hypothesis}/{evidence}/{query})
    for _k, _v in template_kwargs.items():
        usr_msg = usr_msg.replace(f"{{{_k}}}", str(_v))

    max_valid_score = METRIC_MAX_SCORES.get(metric, 3)

    for attempt in range(_METRIC_MAX_RETRIES):
        if attempt > 0:
            delay = _METRIC_RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 2)
            tqdm.write(
                f"[retry {attempt}/{_METRIC_MAX_RETRIES - 1}] metric={metric} "
                f"sleeping {delay:.1f}s before retry…"
            )
            time.sleep(delay)

        score: int | None = None
        reasoning: str = ""
        usage: dict = {}

        # -------------------------------------------------------------- Gemini
        if provider == LLMProvider.gemini:
            response = client.query(
                model, sys_msg, usr_msg, thinking_level=thinking_level
            )
            if response is None:
                tqdm.write(f"[retry] Gemini returned None for metric={metric}")
                continue

            parsed = parser.parse(response, parse_as_json=True)
            ans = parsed.answer_json if isinstance(parsed.answer_json, dict) else {}
            score = ans.get("score")
            reasoning = ans.get("reasoning") or parsed.reasoning or ""

            usage = {
                "prompt_tokens": parsed.usage.prompt_tokens,
                "candidates_tokens": parsed.usage.candidates_tokens,
                "thoughts_tokens": parsed.usage.thoughts_tokens,
                "total_tokens": parsed.usage.total_tokens,
            }

        # --------------------------------------------------------------- OpenAI
        else:
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg},
            ]
            call_kwargs: dict = {
                "model": model,
                "messages": messages,
                "response_format": {"type": "json_object"},
            }
            # reasoning_effort is only honoured by o-series reasoning models; passing
            # it to standard models is a no-op on most API versions but may raise on
            # older ones — the caller should pair reasoning models with a thinking level.
            # if thinking_level is not None:
            #     call_kwargs["reasoning_effort"] = thinking_level.value

            openai_max_retries = 8
            base_delay = 5.0
            response = None
            for oa_attempt in range(openai_max_retries):
                try:
                    response = client.chat.completions.create(
                        **call_kwargs,
                        # extra_body={
                        #     "chat_template_kwargs": {"enable_thinking": True}
                        # },
                        stream=False,
                        reasoning_effort="high",
                        extra_body={"thinking": {"type": "enabled"}}
                    )
                    break
                except Exception as e:
                    err_str = str(e)
                    is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
                    is_last_oa = oa_attempt == openai_max_retries - 1
                    if is_rate_limit and not is_last_oa:
                        oa_delay = base_delay * (2 ** oa_attempt) + random.uniform(0, 3)
                        tqdm.write(
                            f"[429] OpenAI rate limited. Retrying in {oa_delay:.1f}s "
                            f"(attempt {oa_attempt + 1}/{openai_max_retries})…"
                        )
                        time.sleep(oa_delay)
                    else:
                        tqdm.write(f"OpenAI call failed: {e}")
                        break

            if response is None:
                tqdm.write(f"[retry] OpenAI returned no response for metric={metric}")
                continue

            choice = response.choices[0] if response.choices else None
            message = choice.message if choice else None
            content = (message.content or "") if message else ""
            try:
                ans = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                tqdm.write(
                    f"[retry] metric={metric} could not parse JSON from response: {content[:200]!r}"
                )
                continue

            score = ans.get("score")
            message_reasoning = getattr(message, "reasoning", None) if message else None
            reasoning = message_reasoning or ans.get("reasoning", "")

            usage_obj = getattr(response, "usage", None)
            prompt_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage_obj, "completion_tokens", 0) or 0
            total_tokens = getattr(usage_obj, "total_tokens", 0) or 0
            details = getattr(usage_obj, "completion_tokens_details", None)
            reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

            usage = {
                "prompt_tokens": prompt_tokens,
                "candidates_tokens": completion_tokens - reasoning_tokens,
                "thoughts_tokens": reasoning_tokens,
                "total_tokens": total_tokens,
            }

        # --------------------------------------------------------- Validate score
        if score is None or not isinstance(score, (int, float)):
            tqdm.write(
                f"[retry] metric={metric} missing or non-numeric score={score!r}, retrying…"
            )
            continue
        score = int(score)
        if not (0 <= score <= max_valid_score):
            tqdm.write(
                f"[retry] metric={metric} score={score} out of range "
                f"[0, {max_valid_score}], retrying…"
            )
            continue

        return score, reasoning, usage

    tqdm.write(f"[error] metric={metric} failed after {_METRIC_MAX_RETRIES} attempts, skipping.")
    return None, "", {}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plot(output_df: pd.DataFrame, plot_path: Path) -> None:
    """Generate a 1×3 seaborn barplot grid, one subplot per metric."""
    score_cols = [f"{m}_score" for m in METRICS]
    plot_df = output_df.dropna(subset=score_cols).copy()
    for col in score_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    # Short persona labels (strip leading "The ")
    plot_df["Persona"] = (
        plot_df["persona_display_name"]
        .fillna(plot_df["persona"])
        .str.replace(r"^The\s+", "", regex=True)
    )

    metric_labels = {
        "groundedness": "Groundedness (0–3)",
        "relevancy": "Relevancy (0–3)",
        "persona_adherence": "Persona Adherence (0–3)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes_flat = axes.flatten()

    palette = {"chosen": "#2196F3", "rejected": "#FF7043"}

    for i, metric in enumerate(METRICS):
        ax = axes_flat[i]
        subset = plot_df[["Persona", "hypothesis_type", f"{metric}_score"]].copy()
        subset = subset.rename(columns={f"{metric}_score": "score"})
        subset = subset.dropna(subset=["score"])

        sns.barplot(
            data=subset,
            x="Persona",
            y="score",
            hue="hypothesis_type",
            hue_order=["chosen", "rejected"],
            palette=palette,
            ax=ax,
            capsize=0.06,
            errorbar="sd",
        )
        ax.set_title(metric_labels[metric], fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("Persona", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_ylim(0, METRIC_MAX_SCORES[metric] + 0.6)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Type", fontsize=9, title_fontsize=9)

    fig.suptitle(
        "Hypothesis Evaluation: Chosen vs Rejected by Metric & Persona",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate hypotheses with LLM judge metrics (one call per metric)."
    )
    p.add_argument(
        "--input",
        default="hypotheses/dpo_neighbourhood.csv",
        help="Path to the DPO CSV produced by generate_rejected_neighbourhood.py",
    )
    p.add_argument(
        "--output",
        default="hypotheses/eval_results.csv",
        help="Path to write evaluation results CSV",
    )
    p.add_argument(
        "--plot",
        default="hypotheses/eval_metrics_plot.png",
        help="Path to save the seaborn plot",
    )
    p.add_argument(
        "--n-per-persona",
        type=int,
        default=10,
        help="Number of chosen (and rejected) hypotheses to sample per persona (default: 10)",
    )
    p.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="Model name (default: gemini-2.5-pro)",
    )
    p.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "openai"],
        help="LLM provider to use: 'gemini' (default) or 'openai'",
    )
    # Gemini-only options
    p.add_argument(
        "--project-id",
        default=None,
        help="[Gemini] GCP project ID (default: $GEMINI_PROJECT_ID_2 env var)",
    )
    p.add_argument(
        "--location",
        default="global",
        help="[Gemini] Vertex AI location (default: global)",
    )
    # OpenAI-only options
    p.add_argument(
        "--openai-api-key",
        default=None,
        help="[OpenAI] API key (default: $OPENAI_API_KEY env var)",
    )
    p.add_argument(
        "--openai-base-url",
        default=None,
        help="[OpenAI] Custom base URL for the OpenAI-compatible endpoint (e.g. Azure, local server)",
    )
    p.add_argument(
        "--thinking-level",
        default="high",
        choices=["low", "medium", "high"],
        help="Thinking/reasoning level: maps to Gemini thinking_level or OpenAI reasoning_effort (default: high)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent hypothesis evaluations (default: 10)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    provider = LLMProvider(args.provider)
    thinking_level = ThinkingLevel(args.thinking_level)

    # ---------------------------------------------------------------- clients
    if provider == LLMProvider.gemini:
        project_id = (
            args.project_id
            or os.getenv("GEMINI_PROJECT_ID_2")
            or os.getenv("PROJECT_ID")
        )
        if not project_id:
            raise EnvironmentError(
                "GCP project ID not found. Use --project-id or set GEMINI_PROJECT_ID_2."
            )
    else:
        if OpenAI is None:
            raise ImportError(
                "openai package is not installed. Run: pip install openai"
            )
        openai_api_key = args.openai_api_key or os.getenv("CLARIN_API_KEY")
        if not openai_api_key:
            raise EnvironmentError(
                "OpenAI API key not found. Use --openai-api-key or set OPENAI_API_KEY."
            )
        project_id = None  # not used for OpenAI

    # Load personas
    personas = load_personas(PERSONAS_FILE)
    persona_ids = list(personas.keys())

    # Load input data
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Personas: {persona_ids}")

    # Sample hypotheses
    samples = sample_hypotheses(df, persona_ids, args.n_per_persona, args.seed)
    print(
        f"Sampled {len(samples)} hypotheses "
        f"(up to {args.n_per_persona} chosen + {args.n_per_persona} rejected per persona)"
    )

    # Resume support
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_keys = load_processed_keys(output_path)
    remaining = [
        s for s in samples
        if (s["node_id"], s["persona"], s["hypothesis_type"]) not in processed_keys
    ]
    if processed_keys:
        print(f"Resuming — {len(processed_keys)} already done, {len(remaining)} remaining.")

    # Thread-local clients
    response_parser = GeminiResponseParser()
    thread_local = threading.local()

    def get_thread_client():
        if not hasattr(thread_local, "client"):
            if provider == LLMProvider.gemini:
                thread_local.client = GeminiClient(
                    project_id=project_id, location=args.location
                )
            else:
                openai_kwargs: dict = {"api_key": openai_api_key}
                thread_local.client = OpenAI(
                    api_key="EMPTY",
                    base_url="http://localhost:8000/v1")
                # thread_local.client = OpenAI(
                #     api_key=os.getenv("DEEPSEEK_API_KEY"),
                #     base_url="https://api.deepseek.com")
                
        return thread_local.client

    # Thread-local Jinja2 environments (Environment is not thread-safe for rendering)
    def get_thread_eval_env():
        if not hasattr(thread_local, "eval_env"):
            thread_local.eval_env = Environment(
                loader=FileSystemLoader(str(EVAL_TEMPLATES_DIR)),
                keep_trailing_newline=True,
            )
        return thread_local.eval_env

    def get_thread_persona_tpl():
        if not hasattr(thread_local, "persona_tpl"):
            env = Environment(
                loader=FileSystemLoader(str(DATA_GEN_TEMPLATES_DIR)),
                keep_trailing_newline=True,
            )
            thread_local.persona_tpl = env.get_template("persona_template_2.j2")
        return thread_local.persona_tpl

    # Token counters (protected by a lock)
    token_lock = threading.Lock()
    total_tokens = total_prompt = total_candidates = total_thoughts = 0

    write_header = not output_path.exists() or output_path.stat().st_size == 0
    out_f = open(output_path, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_f, fieldnames=OUTPUT_COLS, extrasaction="ignore")
    write_lock = threading.Lock()
    if write_header:
        writer.writeheader()

    def process_sample(sample: dict) -> dict:
        """Evaluate all 4 metrics for a single hypothesis. Returns the output row."""
        client = get_thread_client()
        env = get_thread_eval_env()
        p_tpl = get_thread_persona_tpl()

        persona = personas[sample["persona"]]
        persona_text = build_persona_text(persona, p_tpl)

        out_row: dict = {
            "node_id": sample["node_id"],
            "paper_id": sample["paper_id"],
            "persona": sample["persona"],
            "persona_display_name": persona.get("display_name", sample["persona"]),
            "hypothesis_type": sample["hypothesis_type"],
            "hypothesis": sample["hypothesis"],
            "context": sample["context"],
            "question": sample["question"],
        }

        row_prompt = row_candidates = row_thoughts = row_total = 0

        for metric in METRICS:
            kwargs: dict = {"hypothesis": sample["hypothesis"]}
            if metric == "groundedness":
                kwargs["evidence"] = sample["context"]
            elif metric == "relevancy":
                kwargs["query"] = sample["question"]
            elif metric == "persona_adherence":
                kwargs["persona"] = persona_text

            score, reasoning, usage = call_metric(
                metric=metric,
                client=client,
                parser=response_parser,
                jinja_env=env,
                model=args.model,
                thinking_level=thinking_level,
                provider=provider,
                **kwargs,
            )

            out_row[f"{metric}_score"] = score
            out_row[f"{metric}_reasoning"] = reasoning

            tok = usage.get("total_tokens", 0)
            row_total += tok
            row_prompt += usage.get("prompt_tokens", 0)
            row_candidates += usage.get("candidates_tokens", 0)
            row_thoughts += usage.get("thoughts_tokens", 0)

            tqdm.write(
                f"  [{sample['hypothesis_type']:8s}][{sample['persona']:20s}] "
                f"{metric:20s}: score={score}  tokens={tok}"
            )

        return out_row, {
            "total": row_total, "prompt": row_prompt,
            "candidates": row_candidates, "thoughts": row_thoughts,
        }

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.concurrency)
    ) as executor:
        futures = [
            executor.submit(process_sample, sample) for sample in remaining
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Evaluating",
            unit="hyp",
        ):
            out_row, usage = future.result()
            with write_lock:
                writer.writerow(out_row)
                out_f.flush()
            with token_lock:
                total_tokens += usage["total"]
                total_prompt += usage["prompt"]
                total_candidates += usage["candidates"]
                total_thoughts += usage["thoughts"]

    out_f.close()

    print(
        f"\nEvaluation complete. Rows written: {len(remaining)}\n"
        f"Total tokens: {total_tokens} "
        f"(prompt={total_prompt}, candidates={total_candidates}, thoughts={total_thoughts})"
    )

    # Generate plot from full output file
    full_df = pd.read_csv(output_path)
    save_plot(full_df, Path(args.plot))


if __name__ == "__main__":
    main()
