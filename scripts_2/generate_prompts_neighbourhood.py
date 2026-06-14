#!/usr/bin/env python3
"""Generate research questions (prompts) for hypotheses produced by
generate_hypotheses_neighbourhood.py.

For each row in the input CSV (default: hypotheses/chosen_neighbourhood.csv):
  - Renders the question_system / question_user prompt templates using the
    hypothesis, context, and chosen persona details.
  - Calls the Gemini model and parses the JSON response.
  - Writes all original CSV columns plus:
      question    – extracted research question (research_question field)
      scope_notes – scope notes from the model response

Usage:
    python scripts_2/generate_prompts_neighbourhood.py \\
        --input  hypotheses/chosen_neighbourhood.csv \\
        --output hypotheses/prompts_neighbourhood.csv \\
        --model  gemini-2.5-pro
"""

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm

# Make sure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api_client.gemini_client import GeminiClient, ThinkingLevel
from api_client.gemini_parser import GeminiResponseParser

load_dotenv()
load_dotenv(".env_db")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERSONAS_FILE = PROJECT_ROOT / "personas" / "personas_all.json"
TEMPLATES_DIR = PROJECT_ROOT / "prompt_templates" / "data_generation"


# ---------------------------------------------------------------------------
# Persona helpers
# ---------------------------------------------------------------------------

def load_personas(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return {p["persona_id"]: p for p in raw}
    return raw


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def load_processed_ids(output_path: Path) -> set[tuple[str, str, str]]:
    if not output_path.exists():
        return set()
    processed: set[tuple[str, str, str]] = set()
    with output_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nid = row.get("node_id", "").strip()
            chosen = row.get("chosen_persona", "").strip()
            rejected = row.get("rejected_persona", "").strip()
            if nid:
                processed.add((nid, chosen, rejected))
    return processed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate research questions for neighbourhood hypotheses via Gemini."
    )
    parser.add_argument(
        "--input", default="hypotheses/chosen_neighbourhood.csv",
        help="Path to the hypothesis CSV (default: hypotheses/chosen_neighbourhood.csv)"
    )
    parser.add_argument(
        "--output", default="hypotheses/prompts_neighbourhood.csv",
        help="Path to write the output CSV (default: hypotheses/prompts_neighbourhood.csv)"
    )
    parser.add_argument(
        "--model", default="gemini-2.5-pro",
        help="Gemini model name (default: gemini-2.5-pro)"
    )
    parser.add_argument(
        "--project-id", default=None,
        help="GCP project ID (default: $GEMINI_PROJECT_ID_2 env var)"
    )
    parser.add_argument(
        "--thinking-level", default=None,
        choices=[None,"low", "medium", "high"],
        help="Thinking level for reasoning models (default: medium)"
    )
    parser.add_argument(
        "--location", default="global",
        help="Vertex AI location (default: global)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Number of concurrent requests (default: 10)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    project_id = args.project_id or os.getenv("GEMINI_PROJECT_ID_2")
    if not project_id:
        raise EnvironmentError(
            "GCP project ID not set. Use --project-id or set GEMINI_PROJECT_ID_2 env var."
        )

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── templates ──────────────────────────────────────────────────────────
    jinja_env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        keep_trailing_newline=True,
    )
    system_template = jinja_env.get_template("prompts_2/question_system.j2")
    user_template = jinja_env.get_template("prompts_2/question_user.j2")

    # ── personas ───────────────────────────────────────────────────────────
    personas = load_personas(PERSONAS_FILE)
    print(f"Loaded {len(personas)} personas from {PERSONAS_FILE}")

    # ── Gemini client (thread-local) ───────────────────────────────────────
    response_parser = GeminiResponseParser()
    thinking_level = ThinkingLevel(args.thinking_level)
    thread_local = threading.local()

    def get_thread_client() -> GeminiClient:
        if not hasattr(thread_local, "client"):
            thread_local.client = GeminiClient(project_id=project_id, location=args.location)
        return thread_local.client

    # ── load input CSV ─────────────────────────────────────────────────────
    import pandas as pd
    df = pd.read_csv(input_path)
    # df = df.sample(10).reset_index(drop=True)  # TEMP: limit to 10 rows for testing
    print(f"Loaded {len(df)} rows from {input_path}")

    # ── resume support ─────────────────────────────────────────────────────
    processed_ids = load_processed_ids(output_path)
    if processed_ids:
        print(f"Resuming: {len(processed_ids)} rows already processed.")

    # ── output CSV fieldnames ──────────────────────────────────────────────
    original_cols = list(df.columns)
    extra_cols = [
        "question",
        "scope_notes",
        "methodology_terms_avoided",
        "question_is_answerable",
        "question_unanswerable_reason",
        "reasoning_prompt",
    ]
    all_cols = original_cols + extra_cols

    write_header = not output_path.exists() or output_path.stat().st_size == 0
    out_file = output_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_file, fieldnames=all_cols, extrasaction="ignore")
    if write_header:
        writer.writeheader()

    # ── aggregate token counters ───────────────────────────────────────────
    total_prompt_tokens = 0
    total_candidates_tokens = 0
    total_thoughts_tokens = 0
    total_tokens = 0
    total_requests = 0

    # ── worker function ────────────────────────────────────────────────────
    def process_row(row_dict: dict) -> tuple[dict | None, dict, str | None]:
        node_id = str(row_dict["node_id"]).strip()

        hypothesis = str(row_dict.get("chosen_hypothesis") or "").strip()
        if not hypothesis:
            return None, {}, f"Empty hypothesis for node {node_id}. Skipping."

        context = str(row_dict.get("context") or "").strip()

        persona_id = str(row_dict.get("chosen_persona") or "").strip()
        persona = personas.get(persona_id)
        if persona is None:
            return None, {}, f"Unknown persona '{persona_id}' for node {node_id}. Skipping."

        system_msg = system_template.render()
        user_msg = user_template.render(
            hypothesis=hypothesis,
            context=context,
            chosen_persona_display_name=persona["display_name"],
            chosen_persona_core_philosophy=persona["core_philosophy"],
            chosen_persona_vocabulary_markers=persona["vocabulary_markers"],
            chosen_persona_what_i_look_for=persona['what_i_look_for'],
        )

        client = get_thread_client()

        def call_api():
            return client.query(
                model=args.model,
                system_message=system_msg,
                user_message=user_msg,
                thinking_level=thinking_level,
            )

        response = call_api()

        # One retry after 10-second delay if the call fails.
        if response is None:
            time.sleep(10)
            response = call_api()

        out_row = dict(row_dict)

        if response is None:
            out_row["question"] = ""
            out_row["scope_notes"] = ""
            out_row["methodology_terms_avoided"] = ""
            out_row["reasoning_prompt"] = ""
            return out_row, {
                "prompt": 0, "candidates": 0, "thoughts": 0, "total": 0, "count": 0,
            }, f"No response for node {node_id} after 1 retry."

        parsed = response_parser.parse(response, parse_as_json=True)
        ans = parsed.answer_json if isinstance(parsed.answer_json, dict) else {}

        out_row["question"] = ans.get("research_question", "") or parsed.answer
        out_row["scope_notes"] = ans.get("scope_notes", "")
        out_row["methodology_terms_avoided"] = json.dumps(
            ans.get("methodology_terms_avoided", []), ensure_ascii=False
        )
        out_row["question_is_answerable"] = ans.get("is_answerable", True)
        out_row["question_unanswerable_reason"] = ans.get("unanswerable_reason", "")
        out_row["reasoning_prompt"] = parsed.reasoning

        u = parsed.usage
        usage = {
            "prompt": u.prompt_tokens,
            "candidates": u.candidates_tokens,
            "thoughts": u.thoughts_tokens,
            "total": u.total_tokens,
            "count": 1,
        }
        return out_row, usage, None

    # ── build work list ────────────────────────────────────────────────────
    rows_to_process: list[dict] = []
    for _, row in df.iterrows():
        key = (
            str(row["node_id"]).strip(),
            str(row.get("chosen_persona", "")).strip(),
            str(row.get("rejected_persona", "")).strip(),
        )
        if key not in processed_ids:
            rows_to_process.append(row.to_dict())

    # ── concurrent execution ───────────────────────────────────────────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [executor.submit(process_row, row_dict) for row_dict in rows_to_process]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Generating questions",
        ):
            out_row, usage, err = future.result()

            if err:
                tqdm.write(f"[WARN] {err}")

            if out_row is not None:
                writer.writerow(out_row)
                out_file.flush()

            total_prompt_tokens += usage.get("prompt", 0)
            total_candidates_tokens += usage.get("candidates", 0)
            total_thoughts_tokens += usage.get("thoughts", 0)
            total_tokens += usage.get("total", 0)
            total_requests += usage.get("count", 0)

            if usage.get("count", 0) > 0:
                tqdm.write(
                    f"  tokens — prompt: {usage['prompt']}, "
                    f"candidates: {usage['candidates']}, "
                    f"thoughts: {usage['thoughts']}, "
                    f"total: {usage['total']}"
                )

    out_file.close()
    print("\nToken usage summary:")
    print(f"  requests:   {total_requests}")
    print(f"  prompt:     {total_prompt_tokens}")
    print(f"  candidates: {total_candidates_tokens}")
    print(f"  thoughts:   {total_thoughts_tokens}")
    print(f"  total:      {total_tokens}")
    print(f"\nDone. Output saved to {output_path}")


if __name__ == "__main__":
    main()
