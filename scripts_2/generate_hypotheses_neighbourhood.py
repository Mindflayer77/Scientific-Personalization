#!/usr/bin/env python3
"""Generate chosen hypotheses from neighbourhood context using Gemini.

For each row in neighbourhood_balanced.csv:
  - Fetches summaries + paper IDs for the 5 neighbour nodes from Neo4j.
  - Combines the row's own summary (with its paper_id) + 5 neighbour summaries
    to form a 6-article context block.
  - Renders the chosen_system / chosen_user prompt templates with the
    chosen_persona from personas/personas_all.json.
  - Calls the Gemini model and parses the JSON response.
  - Writes all original CSV columns plus:
      context           – the combined 6-summary context string
      chosen            – generated hypothesis
      reasoning_chosen  – model reasoning / thinking text
      n_articles        – number of articles in the context

Usage:
    python scripts_2/generate_hypotheses_neighbourhood.py \\
        --input  data/neighbourhood_balanced.csv \\
        --output hypotheses/chosen_neighbourhood.csv \\
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
from neo4j import GraphDatabase
from tqdm import tqdm

# Make sure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api_client.gemini_client import GeminiClient, ThinkingLevel
from api_client.gemini_parser import GeminiResponseParser

load_dotenv()
load_dotenv('.env_db')


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERSONAS_FILE = PROJECT_ROOT / "personas" / "personas_all.json"
TEMPLATES_DIR = PROJECT_ROOT / "prompt_templates" / "data_generation"


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

def get_neo4j_driver():
    vps_ip = os.getenv("VPS_IP")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    if not all([vps_ip, user, password]):
        raise EnvironmentError(
            "Missing Neo4j connection variables. "
            "Ensure VPS_IP, NEO4J_USER, and NEO4J_PASSWORD are set."
        )
    uri = f"bolt://{vps_ip}:7687"
    return GraphDatabase.driver(uri, auth=(user, password))


def fetch_node_data(driver, node_ids: list[str]) -> dict[str, dict]:
    """Return {node_id: {'summary': ..., 'paper_id': ...}} for the given ids."""
    cypher = """
        UNWIND $ids AS eid
        MATCH (n)
        WHERE elementId(n) = eid
        RETURN elementId(n) AS node_id,
               n.summary   AS summary,
               n.paperId   AS paper_id
    """
    result_map: dict[str, dict] = {}
    with driver.session() as session:
        records = session.run(cypher, ids=node_ids)
        for rec in records:
            result_map[rec["node_id"]] = {
                "summary": rec["summary"] or "",
                "paper_id": rec["paper_id"] or "",
            }
    return result_map


# ---------------------------------------------------------------------------
# Persona helpers
# ---------------------------------------------------------------------------

def load_personas(path: Path) -> dict[str, dict]:
    """Return {persona_id: persona_dict} from a JSON list file."""
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
# Context builder
# ---------------------------------------------------------------------------

def build_context(
    row_paper_id: str,
    row_summary: str,
    neighbour_data: list[dict],
) -> tuple[str, int]:
    """
    Build the formatted context string from the focal row + neighbours.

    Each entry is formatted as:
        [<paper_id>] <summary>

    Returns (context_string, n_articles).
    """
    entries: list[str] = []

    if row_summary.strip():
        entries.append(f"[{row_paper_id}] {row_summary.strip()}")

    for nd in neighbour_data:
        summary = (nd.get("summary") or "").strip()
        paper_id = (nd.get("paper_id") or "").strip()
        if summary:
            entries.append(f"[{paper_id}] \n {summary}")

    context = "\n\n".join(entries)
    return context, len(entries)


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def load_processed_ids(output_path: Path) -> set[tuple[str, str, str]]:
    """Return a set of (node_id, chosen_persona, rejected_persona) tuples already written."""
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
        description="Generate chosen hypotheses from neighbourhood context via Gemini."
    )
    parser.add_argument(
        "--input", default="data/neighbourhood_500.csv",
        help="Path to neighbourhood_balanced.csv (default: data/neighbourhood_balanced.csv)"
    )
    parser.add_argument(
        "--output", default="hypotheses/chosen_neighbourhood.csv",
        help="Path to write the output CSV"
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
        raise EnvironmentError("GCP project ID not set. Use --project-id or set GEMINI_PROJECT_ID_2 env var.")

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── templates ──────────────────────────────────────────────────────────
    jinja_env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        keep_trailing_newline=True,
    )
    system_template = jinja_env.get_template("hypotheses_2/chosen_system.j2")
    user_template = jinja_env.get_template("hypotheses_2/chosen_user.j2")
    persona_template = jinja_env.get_template("persona_template_2.j2")

    # ── personas ───────────────────────────────────────────────────────────
    personas = load_personas(PERSONAS_FILE)
    print(f"Loaded {len(personas)} personas from {PERSONAS_FILE}")

    # ── Gemini client ──────────────────────────────────────────────────────
    parser = GeminiResponseParser()
    thinking_level = ThinkingLevel(args.thinking_level)
    thread_local = threading.local()

    def get_thread_client() -> GeminiClient:
        if not hasattr(thread_local, "client"):
            thread_local.client = GeminiClient(project_id=project_id, location=args.location)
        return thread_local.client

    # ── Neo4j driver ───────────────────────────────────────────────────────
    print("Connecting to Neo4j...")
    driver = get_neo4j_driver()

    # ── load input CSV ─────────────────────────────────────────────────────
    import pandas as pd
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    # ── resume support ─────────────────────────────────────────────────────
    processed_ids = load_processed_ids(output_path)
    if processed_ids:
        print(f"Resuming: {len(processed_ids)} rows already processed.")

    # ── output CSV fieldnames ──────────────────────────────────────────────
    original_cols = list(df.columns)
    extra_cols = [
        "context",
        "n_articles",
        "chosen_hypothesis",
        "chosen_falsification_criteria",
        "chosen_persona_signature_elements",
        "chosen_evidence_grounding",
        "chosen_is_answerable",
        "reasoning_chosen",
    ]
    all_cols = original_cols + extra_cols



    write_header = not output_path.exists() or output_path.stat().st_size == 0
    out_file = output_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_file, fieldnames=all_cols, extrasaction="ignore")
    if write_header:
        writer.writeheader()

    # Aggregate token usage across all successful Gemini responses.
    total_prompt_tokens = 0
    total_candidates_tokens = 0
    total_thoughts_tokens = 0
    total_tokens = 0
    total_requests = 0

    def process_row(row_dict: dict) -> tuple[dict | None, dict, str | None]:
        node_id = str(row_dict["node_id"]).strip()

        # Parse neighbour IDs
        try:
            neighbour_ids: list[str] = json.loads(row_dict["neighbours"])
        except (json.JSONDecodeError, TypeError):
            return None, {}, f"Invalid JSON in neighbours column for node {node_id}: {row_dict.get('neighbours')}"

        # Fetch neighbour data from Neo4j
        try:
            neighbour_data_map = fetch_node_data(driver, neighbour_ids) if neighbour_ids else {}
        except Exception as e:
            return None, {}, f"Neo4j fetch failed for node {node_id}: {e}"

        neighbour_data = [
            neighbour_data_map.get(nid, {"summary": "", "paper_id": nid})
            for nid in neighbour_ids
        ]

        # Build context
        context, n_articles = build_context(
            row_paper_id=str(row_dict["paper_id"]),
            row_summary=str(row_dict["summary"]),
            neighbour_data=neighbour_data,
        )

        # Resolve persona
        persona_id = str(row_dict["chosen_persona"]).strip()
        persona = personas.get(persona_id)
        if persona is None:
            return None, {}, f"Unknown persona '{persona_id}' for node {node_id}."

        persona_text = build_persona_text(persona, persona_template)

        # Render prompts
        system_msg = system_template.render(persona=persona_text)
        user_msg = user_template.render(context=context)

        # Call Gemini
        client = get_thread_client()
        response = client.query(
            model=args.model,
            system_message=system_msg,
            user_message=user_msg,
            thinking_level=thinking_level,
        )

        # Retry once after a short delay when API call fails.
        if response is None:
            time.sleep(10)
            response = client.query(
                model=args.model,
                system_message=system_msg,
                user_message=user_msg,
                # thinking_level=thinking_level,
            )

        out_row = dict(row_dict)
        out_row["context"] = context
        out_row["n_articles"] = n_articles

        if response is None:
            out_row["chosen_hypothesis"] = ""
            out_row["chosen_falsification_criteria"] = ""
            out_row["chosen_persona_signature_elements"] = ""
            out_row["chosen_evidence_grounding"] = ""
            out_row["chosen_is_answerable"] = ""
            out_row["reasoning_chosen"] = ""
            return out_row, {
                "prompt": 0,
                "candidates": 0,
                "thoughts": 0,
                "total": 0,
                "count": 0,
            }, f"No response for node {node_id} after 1 retry."

        # Parse response
        parsed = parser.parse(response, parse_as_json=True)
        ans = parsed.answer_json if isinstance(parsed.answer_json, dict) else {}

        hypothesis = ans.get("hypothesis") or ans.get("hypothesis_statement") or parsed.answer
        sig_elements = ans.get("persona_signature_elements", [])
        evidence = ans.get("evidence_grounding", [])

        out_row["chosen_hypothesis"] = hypothesis
        out_row["chosen_falsification_criteria"] = ans.get("falsification_criteria", "")
        out_row["chosen_persona_signature_elements"] = json.dumps(sig_elements, ensure_ascii=False)
        out_row["chosen_evidence_grounding"] = json.dumps(evidence, ensure_ascii=False)
        out_row["chosen_is_answerable"] = ans.get("is_answerable", "")
        out_row["reasoning_chosen"] = parsed.reasoning

        u = parsed.usage
        usage = {
            "prompt": u.prompt_tokens,
            "candidates": u.candidates_tokens,
            "thoughts": u.thoughts_tokens,
            "total": u.total_tokens,
            "count": 1,
        }
        return out_row, usage, None

    rows_to_process: list[dict] = []
    for _, row in df.iterrows():
        key = (
            str(row["node_id"]).strip(),
            str(row["chosen_persona"]).strip(),
            str(row["rejected_persona"]).strip(),
        )
        if key in processed_ids:
            continue
        rows_to_process.append(row.to_dict())

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [executor.submit(process_row, row_dict) for row_dict in rows_to_process]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Generating hypotheses",
        ):
            try:
                out_row, usage, err = future.result()
            except Exception as exc:
                tqdm.write(f"[ERROR] Unhandled exception in worker: {exc}")
                continue

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
    driver.close()
    print("\nToken usage summary:")
    print(f"  requests: {total_requests}")
    print(f"  prompt: {total_prompt_tokens}")
    print(f"  candidates: {total_candidates_tokens}")
    print(f"  thoughts: {total_thoughts_tokens}")
    print(f"  total: {total_tokens}")
    print(f"\nDone. Output saved to {output_path}")


if __name__ == "__main__":
    main()
