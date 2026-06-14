#!/usr/bin/env python3
"""Generate chosen and rejected hypotheses for neighbourhood DPO pairs using Gemini.

Reads the prompts CSV produced by generate_prompts_neighbourhood.py
(which contains the research question, context, and persona information)
and generates BOTH a chosen hypothesis (from chosen_persona) and a
rejected hypothesis (from rejected_persona).

For each row:
  - Uses the `question` column as the research question.
  - Uses the `context` column as the scientific evidence.
  - Chosen hypothesis:  looks up `chosen_persona`, renders
      chosen_system_2.j2 / chosen_user_2.j2, calls Gemini.
  - Rejected hypothesis: looks up `rejected_persona`, renders
      rejected_system.j2 / rejected_user.j2, calls Gemini.

Old chosen-hypothesis columns from the input CSV are NOT forwarded to the
output; only freshly generated fields are saved.  All other input columns
are preserved unchanged.

Concurrency is split evenly between the two generation tasks:
  --concurrency N  →  N//2 chosen workers  +  (N - N//2) rejected workers

Column order in output CSV:
  [identity]  node_id, paper_id, chosen_persona, rejected_persona, rejected_score,
              abstract, summary, scores, neighbours
  [context]   context, n_articles
  [prompt]    question, scope_notes
  [chosen]    chosen, chosen_falsification_criteria,
              chosen_persona_signature_elements, chosen_evidence_grounding,
              chosen_is_answerable, reasoning_chosen
  [rejected]  rejected, reasoning_rejected, rejected_falsification_criteria,
              rejected_persona_signature_elements, rejected_evidence_grounding,
              rejected_is_answerable

Usage:
    python scripts_2/generate_rejected_neighbourhood.py \\
        --input  hypotheses/prompts_neighbourhood.csv \\
        --output hypotheses/dpo_neighbourhood.csv \\
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

# Make sure src/ and scripts_2/ are importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_few_shot import render_chosen, render_rejected

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
# Input chosen-hypothesis columns to exclude from output
# (replaced by freshly generated values)
# ---------------------------------------------------------------------------
OLD_CHOSEN_COLS: frozenset[str] = frozenset({
    "chosen",
    "chosen_hypothesis",
    "chosen_falsification_criteria",
    "chosen_persona_signature_elements",
    "chosen_evidence_grounding",
    "chosen_is_answerable",
    "reasoning_chosen",
})

# ---------------------------------------------------------------------------
# Explicit output column order
# ---------------------------------------------------------------------------
OUTPUT_COLS = [
    # identity / meta
    "node_id", "paper_id", "chosen_persona", "rejected_persona", "rejected_score",
    "abstract", "summary", "scores", "neighbours",
    # shared context
    "context", "n_articles",
    # prompt
    "question", "scope_notes",
    # chosen hypothesis (freshly generated)
    "chosen", "chosen_falsification_criteria",
    "chosen_persona_signature_elements", "chosen_evidence_grounding",
    "chosen_is_answerable", "reasoning_chosen",
    # rejected hypothesis (freshly generated)
    "rejected", "reasoning_rejected", "rejected_falsification_criteria",
    "rejected_persona_signature_elements", "rejected_evidence_grounding",
    "rejected_is_answerable",
]

# Sentinel: distinguishes "not yet arrived" from "arrived but empty"
_PENDING = object()


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


def sample_balanced_rows(
    df,
    n_per_chosen: int = 5,
    n_per_rejected: int = 5,
    n_trials: int = 2000,
    random_seed: int = 42,
):
    """Sample rows to balance both chosen and rejected persona counts.

    Strategy:
    - Enforce exactly `n_per_chosen` rows per chosen persona (where available).
    - Random-restart search to minimize deviation from `n_per_rejected` per rejected persona.
    - Return the best sample found; exact rejected balance is reached when feasible.
    """
    import random
    import pandas as pd

    if df.empty:
        return df

    required = {p: n_per_chosen for p in sorted(df["chosen_persona"].dropna().unique())}
    available = df["chosen_persona"].value_counts().to_dict()
    lacking = [p for p, need in required.items() if available.get(p, 0) < need]
    if lacking:
        missing_str = ", ".join(
            f"{p} (have {available.get(p, 0)}, need {required[p]})" for p in lacking
        )
        raise ValueError(
            "Cannot sample chosen personas with requested quota. "
            f"Insufficient rows for: {missing_str}"
        )

    chosen_groups = {
        p: grp.reset_index(drop=False)
        for p, grp in df.groupby("chosen_persona", sort=False)
    }
    target_rejected = {p: n_per_rejected for p in sorted(df["rejected_persona"].dropna().unique())}

    best_sample = None
    best_score = None
    rng = random.Random(random_seed)

    for _ in range(n_trials):
        picks = []
        # Build a candidate set with exact chosen-persona quota.
        for persona, grp in chosen_groups.items():
            idxs = list(range(len(grp)))
            chosen_local = rng.sample(idxs, required[persona])
            picks.append(grp.iloc[chosen_local])

        candidate = pd.concat(picks, ignore_index=True)
        rejected_counts = candidate["rejected_persona"].value_counts().to_dict()

        # Objective: minimize L1 distance from rejected targets.
        score = 0
        for persona, target in target_rejected.items():
            score += abs(rejected_counts.get(persona, 0) - target)

        if best_score is None or score < best_score:
            best_score = score
            best_sample = candidate
            if best_score == 0:
                break

    if best_sample is None:
        raise RuntimeError("Sampling failed unexpectedly.")

    return best_sample.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate chosen and rejected hypotheses for neighbourhood DPO pairs via Gemini."
    )
    parser.add_argument(
        "--input", default="hypotheses/prompts_neighbourhood.csv",
        help="Path to the prompts CSV (default: hypotheses/prompts_neighbourhood.csv)"
    )
    parser.add_argument(
        "--output", default="hypotheses/dpo_neighbourhood.csv",
        help="Path to write the output CSV (default: hypotheses/dpo_neighbourhood.csv)"
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
        "--thinking-level", default="medium",
        choices=["low", "medium", "high"],
        help="Thinking level for reasoning models (default: medium)"
    )
    parser.add_argument(
        "--location", default="global",
        help="Vertex AI location (default: global)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Total concurrent requests split evenly between chosen and rejected (default: 10)"
    )
    parser.add_argument(
        "--chosen-only", action="store_true", default=False,
        help="Generate only the chosen hypothesis; skip rejected generation (default: False)"
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
    chosen_system_template = jinja_env.get_template("hypotheses_2/chosen_system_2.j2")
    chosen_user_template = jinja_env.get_template("hypotheses_2/chosen_user_2.j2")
    rejected_system_template = jinja_env.get_template("hypotheses_2/rejected_system.j2")
    rejected_user_template = jinja_env.get_template("hypotheses_2/rejected_user.j2")
    persona_template = jinja_env.get_template("persona_template_2.j2")

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
    # df = sample_balanced_rows(
    #     df,
    #     n_per_chosen=25,
    #     n_per_rejected=15,
    #     n_trials=3000,
    #     random_seed=23,
    # )
    # df = df[df['chosen_persona'] == 'sota_chaser'] 
    chosen_counts = df["chosen_persona"].value_counts().sort_index().to_dict()
    rejected_counts = df["rejected_persona"].value_counts().sort_index().to_dict()
    print(f"Chosen counts per persona:   {chosen_counts}")
    print(f"Rejected counts per persona: {rejected_counts}")
    print(f"Loaded {len(df)} rows from {input_path}")

    # Build final column list: OUTPUT_COLS first, then any extra input columns
    # that are not in OUTPUT_COLS and not in OLD_CHOSEN_COLS, to avoid data loss.
    input_cols = list(df.columns)
    extra_input_cols = [
        c for c in input_cols
        if c not in set(OUTPUT_COLS) and c not in OLD_CHOSEN_COLS
    ]
    if args.chosen_only:
        rejected_cols = {
            "rejected", "reasoning_rejected", "rejected_falsification_criteria",
            "rejected_persona_signature_elements", "rejected_evidence_grounding",
            "rejected_is_answerable",
        }
        effective_output_cols = [c for c in OUTPUT_COLS if c not in rejected_cols]
    else:
        effective_output_cols = OUTPUT_COLS
    all_cols = effective_output_cols + extra_input_cols

    # ── resume support ─────────────────────────────────────────────────────
    processed_ids = load_processed_ids(output_path)
    if processed_ids:
        print(f"Resuming: {len(processed_ids)} rows already processed.")

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

    # ── worker functions ───────────────────────────────────────────────────

    def _call_with_retry(client: GeminiClient, system_msg: str, user_msg: str):
        """Call Gemini, retrying once after 10 s on failure."""
        response = client.query(
            model=args.model,
            system_message=system_msg,
            user_message=user_msg,
            thinking_level=thinking_level,
        )
        if response is None:
            time.sleep(10)
            response = client.query(
                model=args.model,
                system_message=system_msg,
                user_message=user_msg,
                thinking_level=thinking_level,
            )
        return response

    def process_chosen(row_dict: dict) -> tuple[dict, dict, str | None]:
        """Generate the chosen hypothesis for one row.

        Returns (fields_dict, usage_dict, error_message_or_None).
        fields_dict always contains all chosen_* keys (empty strings on failure).
        """
        node_id = str(row_dict["node_id"]).strip()
        question = str(row_dict["question"]).strip()
        context = str(row_dict["context"]).strip()
        persona_id = str(row_dict.get("chosen_persona") or "").strip()

        empty_fields = {
            "chosen": "",
            "chosen_falsification_criteria": "",
            "chosen_persona_signature_elements": "",
            "chosen_evidence_grounding": "",
            "chosen_is_answerable": "",
            "reasoning_chosen": "",
        }
        empty_usage = {"prompt": 0, "candidates": 0, "thoughts": 0, "total": 0, "count": 0}

        if not question:
            return empty_fields, empty_usage, f"[chosen] Empty question for node {node_id}. Skipping."

        persona = personas.get(persona_id)
        if persona is None:
            return empty_fields, empty_usage, (
                f"[chosen] Unknown chosen persona '{persona_id}' for node {node_id}. Skipping."
            )

        persona_text = build_persona_text(persona, persona_template)
        system_msg = chosen_system_template.render(persona=persona_text)
        user_msg = chosen_user_template.render(query=question, context=context)

        response = _call_with_retry(get_thread_client(), system_msg, user_msg)

        if response is None:
            return empty_fields, empty_usage, (
                f"[chosen] No response for node {node_id} after 1 retry."
            )

        parsed = response_parser.parse(response, parse_as_json=True)
        ans = parsed.answer_json if isinstance(parsed.answer_json, dict) else {}

        fields = {
            "chosen": ans.get("hypothesis") or ans.get("hypothesis_statement") or parsed.answer,
            "reasoning_chosen": ans.get("reasoning", "") or parsed.reasoning,
            "chosen_falsification_criteria": ans.get("falsification_criteria", ""),
            "chosen_persona_signature_elements": json.dumps(
                ans.get("persona_signature_elements", []), ensure_ascii=False
            ),
            "chosen_evidence_grounding": json.dumps(
                ans.get("evidence_grounding", []), ensure_ascii=False
            ),
            "chosen_is_answerable": ans.get("is_answerable", ""),
        }

        u = parsed.usage
        usage = {
            "prompt": u.prompt_tokens,
            "candidates": u.candidates_tokens,
            "thoughts": u.thoughts_tokens,
            "total": u.total_tokens,
            "count": 1,
        }
        return fields, usage, None

    def process_rejected(row_dict: dict) -> tuple[dict, dict, str | None]:
        """Generate the rejected hypothesis for one row.

        Returns (fields_dict, usage_dict, error_message_or_None).
        fields_dict always contains all rejected_* keys (empty strings on failure).
        """
        node_id = str(row_dict["node_id"]).strip()
        question = str(row_dict["question"]).strip()
        context = str(row_dict["context"]).strip()
        persona_id = str(row_dict.get("rejected_persona") or "").strip()

        empty_fields = {
            "rejected": "",
            "reasoning_rejected": "",
            "rejected_falsification_criteria": "",
            "rejected_persona_signature_elements": "",
            "rejected_evidence_grounding": "",
            "rejected_is_answerable": "",
        }
        empty_usage = {"prompt": 0, "candidates": 0, "thoughts": 0, "total": 0, "count": 0}

        if not question:
            return empty_fields, empty_usage, f"[rejected] Empty question for node {node_id}. Skipping."

        persona = personas.get(persona_id)
        if persona is None:
            return empty_fields, empty_usage, (
                f"[rejected] Unknown rejected persona '{persona_id}' for node {node_id}. Skipping."
            )

        persona_text = build_persona_text(persona, persona_template)
        # few_shot_block = render_rejected(persona_id)
        # system_msg = rejected_system_template.render(persona=persona_text, few_shot_block=few_shot_block)
        system_msg = rejected_system_template.render(persona=persona_text)

        user_msg = rejected_user_template.render(query=question, context=context)

        response = _call_with_retry(get_thread_client(), system_msg, user_msg)

        if response is None:
            return empty_fields, empty_usage, (
                f"[rejected] No response for node {node_id} after 1 retry."
            )

        parsed = response_parser.parse(response, parse_as_json=True)
        ans = parsed.answer_json if isinstance(parsed.answer_json, dict) else {}

        fields = {
            "rejected": ans.get("hypothesis") or ans.get("hypothesis_statement") or parsed.answer,
            "reasoning_rejected": ans.get("reasoning", "") or parsed.reasoning,
            "rejected_falsification_criteria": ans.get("falsification_criteria", ""),
            "rejected_persona_signature_elements": json.dumps(
                ans.get("persona_signature_elements", []), ensure_ascii=False
            ),
            "rejected_evidence_grounding": json.dumps(
                ans.get("evidence_grounding", []), ensure_ascii=False
            ),
            "rejected_is_answerable": ans.get("is_answerable", ""),
        }

        u = parsed.usage
        usage = {
            "prompt": u.prompt_tokens,
            "candidates": u.candidates_tokens,
            "thoughts": u.thoughts_tokens,
            "total": u.total_tokens,
            "count": 1,
        }
        return fields, usage, None

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

    print(f"Rows to process: {len(rows_to_process)}")

    # ── concurrency split ──────────────────────────────────────────────────
    if args.chosen_only:
        chosen_workers = args.concurrency
        rejected_workers = 1  # unused
        print(f"Workers — chosen: {chosen_workers} (chosen-only mode)")
    else:
        chosen_workers = max(1, args.concurrency // 2)
        rejected_workers = max(1, args.concurrency - chosen_workers)
        print(f"Workers — chosen: {chosen_workers}, rejected: {rejected_workers}")

    # pending[row_key] = {"chosen": _PENDING|fields, "rejected": _PENDING|fields, "base": base_row}
    # base_row = input row dict with OLD_CHOSEN_COLS removed
    pending: dict[tuple, dict] = {}
    for row_dict in rows_to_process:
        key = (
            str(row_dict["node_id"]).strip(),
            str(row_dict.get("chosen_persona", "")).strip(),
            str(row_dict.get("rejected_persona", "")).strip(),
        )
        base_row = {k: v for k, v in row_dict.items() if k not in OLD_CHOSEN_COLS}
        if args.chosen_only:
            pending[key] = {"chosen": _PENDING, "base": base_row}
        else:
            pending[key] = {"chosen": _PENDING, "rejected": _PENDING, "base": base_row}

    # ── concurrent execution ───────────────────────────────────────────────
    # future_meta maps each future to ("chosen"|"rejected", row_key)
    future_meta: dict[concurrent.futures.Future, tuple[str, tuple]] = {}

    if args.chosen_only:
        total_futures = len(rows_to_process)  # one chosen per row
    else:
        total_futures = len(rows_to_process) * 2  # one chosen + one rejected per row

    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=chosen_workers) as chosen_exec,
        concurrent.futures.ThreadPoolExecutor(max_workers=rejected_workers) as rejected_exec,
    ):
        for row_dict in rows_to_process:
            key = (
                str(row_dict["node_id"]).strip(),
                str(row_dict.get("chosen_persona", "")).strip(),
                str(row_dict.get("rejected_persona", "")).strip(),
            )
            cf = chosen_exec.submit(process_chosen, row_dict)
            future_meta[cf] = ("chosen", key)
            if not args.chosen_only:
                rf = rejected_exec.submit(process_rejected, row_dict)
                future_meta[rf] = ("rejected", key)

        for future in tqdm(
            concurrent.futures.as_completed(future_meta),
            total=total_futures,
            desc="Generating hypotheses",
        ):
            kind, key = future_meta[future]
            fields, usage, err = future.result()

            if err:
                tqdm.write(f"[WARN] {err}")

            # Store the result in the pending dict
            pending[key][kind] = fields

            # Accumulate token usage
            total_prompt_tokens += usage.get("prompt", 0)
            total_candidates_tokens += usage.get("candidates", 0)
            total_thoughts_tokens += usage.get("thoughts", 0)
            total_tokens += usage.get("total", 0)
            total_requests += usage.get("count", 0)

            if usage.get("count", 0) > 0:
                tqdm.write(
                    f"  [{kind}] tokens — prompt: {usage['prompt']}, "
                    f"candidates: {usage['candidates']}, "
                    f"thoughts: {usage['thoughts']}, "
                    f"total: {usage['total']}"
                )

            # Write the row once all expected results have arrived
            entry = pending[key]
            if args.chosen_only:
                ready = entry["chosen"] is not _PENDING
            else:
                ready = entry["chosen"] is not _PENDING and entry["rejected"] is not _PENDING
            if ready:
                out_row = dict(entry["base"])
                out_row.update(entry["chosen"])
                if not args.chosen_only:
                    out_row.update(entry["rejected"])
                writer.writerow(out_row)
                out_file.flush()
                del pending[key]

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
