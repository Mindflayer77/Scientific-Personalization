#!/usr/bin/env python3
"""Simulate a user persona explaining why a chosen hypothesis was preferred.

For each row in the input CL dataset the script:
  1. Resolves persona from ``chosen_persona`` or ``user_id``.
  2. Renders the persona description with persona_template_2.j2.
  3. Calls the LLM (Gemini/DeepSeek/OpenAI) with system+user templates from
      prompt_templates/simulation/.
  4. Appends the result (rationale) to the output CSV.

The prompts are intentionally short so the model focuses on persona simulation
for pairwise choice (chosen vs rejected), not generic hypothesis assessment.

Usage (Gemini):
    python scripts_2/simulate_user_rationale.py \\
        --csv hypotheses/final_3/clean/cl_data.csv \\
        --output hypotheses/final_3/cl_rationales.csv \\
        --provider gemini \\
        --gemini-project-id my-gcp-project \\
        --personas-path personas/personas_all.json \\
        --concurrency 5

Usage (DeepSeek):
    python scripts_2/simulate_user_rationale.py \\
        --csv hypotheses/final_3/clean/val_dpo.csv \\
        --output hypotheses/final_3/cl_rationales.csv \\
        --provider deepseek \\
        --env-file .env \\
        --concurrency 8

Usage (limit rows / sample):
    python scripts_2/simulate_user_rationale.py \\
        --csv hypotheses/final_3/clean/cl_data.csv \\
        --output hypotheses/final_3/cl_rationales.csv \\
        --provider gemini \\
        --gemini-project-id my-gcp-project \\
        --sample 50          # process at most N rows
        --seed 42
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import hashlib
import os
import random
import re
import sys
import time
import threading
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent
_DATA_GEN    = _REPO_ROOT / "prompt_templates" / "data_generation"
_SIM_TEMPLATES = _REPO_ROOT / "prompt_templates" / "simulation"
_SRC         = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from api_client.gemini_client import GeminiClient, ThinkingLevel      # noqa: E402
    from api_client.gemini_parser import GeminiResponseParser              # noqa: E402
except ImportError:
    GeminiClient = None  # type: ignore[assignment]
    ThinkingLevel = Any  # type: ignore[assignment]
    GeminiResponseParser = None  # type: ignore[assignment]

try:
    from openai import OpenAI as _OpenAI
except ImportError:
    _OpenAI = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_PROVIDER_DEFAULTS = {
    "gemini":   "gemini-2.5-pro",
    "deepseek": "deepseek-chat",
    "openai":   "gpt-4o",
}
_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

RATIONALE_COL = "rationale"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Simulate user persona preference for chosen vs rejected "
            "hypothesis and write rationale to output CSV."
        )
    )
    p.add_argument("--csv", required=True,
                   help="Path to CL dataset CSV (e.g. hypotheses/final_3/clean/cl_data.csv).")
    p.add_argument("--output", default=None,
                   help="Output CSV path. Defaults to <input>_rationales.csv.")
    p.add_argument("--provider", default="gemini",
                   choices=["gemini", "deepseek", "openai"],
                   help="LLM provider (default: gemini).")
    p.add_argument("--model", default=None,
                   help="Model override. Defaults: gemini→gemini-2.5-pro, "
                        "deepseek→deepseek-chat, openai→gpt-4o.")
    p.add_argument("--env-file", default=".env",
                   help="Path to .env file with API keys (default: .env).")
    p.add_argument("--gemini-project-id", default=None,
                   help="GCP project ID. Required when --provider=gemini.")
    p.add_argument("--gemini-location", default="global",
                   help="Vertex AI location (default: global).")
    p.add_argument("--thinking-level", default="low",
                   choices=["low", "medium", "high"],
                   help="Thinking effort for Gemini (default: low — rationale "
                        "generation does not require deep reasoning).")
    p.add_argument("--personas-path", default=None,
                   help="Path to personas JSON file (default: personas/personas_all.json).")
    p.add_argument("--persona-template", default=None,
                   help="Path to persona Jinja2 template "
                        "(default: prompt_templates/data_generation/persona_template_2.j2).")
    p.add_argument("--system-template", default=None,
                   help="Path to system Jinja2 template "
                        "(default: prompt_templates/simulation/rationale_system.j2).")
    p.add_argument("--user-template", default=None,
                   help="Path to user Jinja2 template "
                        "(default: prompt_templates/simulation/rationale_user.j2).")
    p.add_argument("--sample", type=int, default=None,
                   help="If set, randomly sample at most N rows before processing.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for sampling (default: 42).")
    p.add_argument("--max-retries", type=int, default=3,
                   help="Max retries per LLM call (default: 3).")
    p.add_argument("--concurrency", type=int, default=5,
                   help="Concurrent LLM calls (default: 5).")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="LLM temperature for rationale generation (default: 0.7).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_api_key(provider: str, env_file: str) -> str | None:
    load_dotenv(env_file)
    key_map = {"openai": "OPENAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}
    if provider in key_map:
        key = os.getenv(key_map[provider])
        if not key:
            print(f"ERROR: {key_map[provider]} not found in {env_file}")
            sys.exit(1)
        return key
    return None  # gemini uses project_id


def _load_personas(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return {p["persona_id"]: p for p in raw}
    return raw


def _build_persona_text(persona: dict, template) -> str:
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


def _build_row_key(row: dict, key_cols: list[str]) -> str:
    """Build a stable hash key from selected columns for resume support."""
    payload = {col: str(row.get(col, "")) for col in key_cols}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_processed_keys(output_path: Path, key_cols: list[str]) -> set[str]:
    """Return hashed row keys already written to output CSV."""
    if not output_path.exists():
        return set()
    processed: set[str] = set()
    with output_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(_build_row_key(row, key_cols))
    return processed


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def _resolve_schema(df: pd.DataFrame, personas: dict[str, dict]) -> tuple[str, str, str, str, list[str]]:
    """Resolve required persona/question/chosen/rejected columns and key columns."""
    columns = list(df.columns)

    persona_col = _first_existing(columns, ["chosen_persona", "user_id"])
    if persona_col is None:
        raise ValueError("Missing persona column. Expected one of: chosen_persona, user_id")

    # Validate persona values if we had to fall back to user_id.
    if persona_col == "user_id":
        vals = {
            str(v).strip()
            for v in df[persona_col].dropna().astype(str).tolist()
            if str(v).strip()
        }
        unknown = sorted([v for v in vals if v not in personas])
        if unknown:
            preview = ", ".join(unknown[:8])
            raise ValueError(
                "Detected personas in user_id that are not present in personas file: "
                f"{preview}"
            )

    question_col = _first_existing(columns, ["question"])
    if question_col is None:
        raise ValueError("Missing required question column: question")

    chosen_col = _first_existing(columns, ["chosen_hypothesis", "chosen"])
    if chosen_col is None:
        raise ValueError("Missing chosen hypothesis column. Expected one of: chosen_hypothesis, chosen")

    rejected_col = _first_existing(columns, ["rejected_hypothesis", "rejected"])
    if rejected_col is None:
        raise ValueError("Missing rejected hypothesis column. Expected one of: rejected_hypothesis, rejected")

    # Use a robust multi-column signature for resume behavior across schemas.
    key_cols = [col for col in ["node_id", "paper_id", persona_col, question_col, chosen_col, rejected_col] if col in columns]
    if not key_cols:
        # Worst-case fallback should still remain stable for same input rows.
        key_cols = columns

    return persona_col, question_col, chosen_col, rejected_col, key_cols


def _extract_json(text: str) -> dict:
    """Best-effort extraction of JSON from a model response."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown code blocks
    clean = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    clean = re.sub(r"\s*```$", "", clean.strip(), flags=re.MULTILINE)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    # Grab first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

_empty_tok = {"prompt": 0, "candidates": 0, "thoughts": 0, "total": 0}


def _call_gemini(
    client,
    model_name: str,
    thinking_level,
    temperature: float,
    sys_msg: str,
    usr_msg: str,
    max_retries: int = 3,
) -> tuple[dict, dict]:
    """Return (parsed_json, token_counts)."""
    if GeminiResponseParser is None:
        raise RuntimeError("Gemini parser dependency is missing. Install google-genai stack.")
    parser = GeminiResponseParser()
    for attempt in range(max_retries):
        try:
            raw = client.query(
                model=model_name,
                system_message=sys_msg,
                user_message=usr_msg,
                thinking_level=thinking_level,
                temperature=temperature,
            )
            if raw is None:
                tqdm.write(f"  WARN  Gemini: None response (attempt {attempt+1}/{max_retries})")
                time.sleep(5 * (attempt + 1))
                continue
            parsed = parser.parse(raw, parse_as_json=True)
            tokens = {
                "prompt":     getattr(parsed.usage, "prompt_tokens",     0) or 0,
                "candidates": getattr(parsed.usage, "candidates_tokens", 0) or 0,
                "thoughts":   getattr(parsed.usage, "thoughts_tokens",   0) or 0,
                "total":      getattr(parsed.usage, "total_tokens",      0) or 0,
            }
            data = parsed.answer_json if isinstance(parsed.answer_json, dict) else {}
            if not data:
                # Fall back to raw answer text
                data = _extract_json(parsed.answer or "")
            if not data.get("rationale"):
                tqdm.write(
                    f"  WARN  Gemini: missing 'rationale' in response "
                    f"(attempt {attempt+1}/{max_retries}): {str(repr(parsed.answer))[:120]}"
                )
                if attempt < max_retries - 1:
                    continue
            return data, tokens
        except Exception as e:
            if attempt < max_retries - 1:
                tqdm.write(f"  RETRY [{attempt+1}/{max_retries}] Gemini: {e}")
                time.sleep(2 ** attempt)
            else:
                tqdm.write(f"  ERROR Gemini failed after {max_retries} attempts: {e}")
    return {}, dict(_empty_tok)


def _call_openai_compat(
    client,
    model_name: str,
    temperature: float,
    sys_msg: str,
    usr_msg: str,
    max_retries: int = 3,
) -> tuple[dict, dict]:
    """OpenAI-compatible call (OpenAI or DeepSeek). Returns (parsed_json, token_counts)."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": usr_msg},
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            raw_text = (response.choices[0].message.content or "").strip()
            usage_obj      = getattr(response, "usage", None)
            prompt_tok     = getattr(usage_obj, "prompt_tokens",     0) or 0
            completion_tok = getattr(usage_obj, "completion_tokens", 0) or 0
            total_tok      = getattr(usage_obj, "total_tokens",      0) or 0
            details        = getattr(usage_obj, "completion_tokens_details", None)
            reasoning_tok  = getattr(details,   "reasoning_tokens",  0) or 0
            tokens = {
                "prompt":     prompt_tok,
                "candidates": completion_tok - reasoning_tok,
                "thoughts":   reasoning_tok,
                "total":      total_tok,
            }
            data = _extract_json(raw_text)
            if not data.get("rationale"):
                tqdm.write(
                    f"  WARN  LLM: missing 'rationale' in response "
                    f"(attempt {attempt+1}/{max_retries}): {raw_text[:120]!r}"
                )
                if attempt < max_retries - 1:
                    continue
            return data, tokens
        except Exception as e:
            err = str(e)
            is_rate = "429" in err or "rate_limit" in err.lower()
            if is_rate and attempt < max_retries - 1:
                delay = 5.0 * (2 ** attempt) + random.uniform(0, 2)
                tqdm.write(f"  [429] Rate-limited, retry {attempt+1}/{max_retries} in {delay:.1f}s")
                time.sleep(delay)
            elif attempt < max_retries - 1:
                tqdm.write(f"  RETRY [{attempt+1}/{max_retries}]: {e}")
                time.sleep(2 ** attempt)
            else:
                tqdm.write(f"  ERROR failed after {max_retries} attempts: {e}")
    return {}, dict(_empty_tok)


# ---------------------------------------------------------------------------
# Per-row processing
# ---------------------------------------------------------------------------

def _process_row(
    row: dict,
    persona: dict,
    persona_text: str,
    persona_col: str,
    question_col: str,
    chosen_col: str,
    rejected_col: str,
    sys_jinja,
    usr_jinja,
    call_fn,            # callable(sys_msg, usr_msg) → (dict, dict)
) -> dict:
    """Render prompts, call LLM, return result dict ready for CSV."""
    display_name = persona.get("display_name", row.get(persona_col, ""))

    sys_msg = sys_jinja.render(
        display_name=display_name,
        persona=persona_text,
    )
    usr_msg = usr_jinja.render(
        question=row.get(question_col, ""),
        chosen_hypothesis=row.get(chosen_col, ""),
        rejected_hypothesis=row.get(rejected_col, ""),
    )

    data, tokens = call_fn(sys_msg, usr_msg)

    rationale  = data.get("rationale", "")
    out = dict(row)
    out[RATIONALE_COL] = rationale
    out["_prompt_tokens"] = tokens.get("prompt", 0)
    out["_candidates_tokens"] = tokens.get("candidates", 0)
    out["_thoughts_tokens"] = tokens.get("thoughts", 0)
    out["_total_tokens"] = tokens.get("total", 0)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = _REPO_ROOT / csv_path
    if not csv_path.exists():
        print(f"ERROR: input CSV not found: {csv_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else csv_path.with_suffix("").with_name(
        csv_path.stem + "_rationales.csv"
    )
    if not output_path.is_absolute():
        output_path = _REPO_ROOT / output_path

    personas_path = Path(args.personas_path) if args.personas_path else (
        _REPO_ROOT / "personas" / "personas_all.json"
    )
    if not personas_path.exists():
        print(f"ERROR: personas file not found: {personas_path}")
        sys.exit(1)

    persona_tpl_path = Path(args.persona_template) if args.persona_template else (
        _DATA_GEN / "persona_template_2.j2"
    )
    if not persona_tpl_path.exists():
        print(f"ERROR: persona template not found: {persona_tpl_path}")
        sys.exit(1)

    system_tpl_path = Path(args.system_template) if args.system_template else (
        _SIM_TEMPLATES / "rationale_system.j2"
    )
    if not system_tpl_path.exists():
        print(f"ERROR: system template not found: {system_tpl_path}")
        sys.exit(1)

    user_tpl_path = Path(args.user_template) if args.user_template else (
        _SIM_TEMPLATES / "rationale_user.j2"
    )
    if not user_tpl_path.exists():
        print(f"ERROR: user template not found: {user_tpl_path}")
        sys.exit(1)

    # ── Jinja2 environments ───────────────────────────────────────────────────
    persona_env = Environment(loader=FileSystemLoader(str(persona_tpl_path.parent)))
    persona_tpl = persona_env.get_template(persona_tpl_path.name)

    system_env = Environment(loader=FileSystemLoader(str(system_tpl_path.parent)))
    sys_tpl = system_env.get_template(system_tpl_path.name)

    user_env = Environment(loader=FileSystemLoader(str(user_tpl_path.parent)))
    usr_tpl = user_env.get_template(user_tpl_path.name)

    # ── Load personas ─────────────────────────────────────────────────────────
    personas = _load_personas(personas_path)
    print(f"Loaded {len(personas)} personas from {personas_path}")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    try:
        persona_col, question_col, chosen_col, rejected_col, key_cols = _resolve_schema(df, personas)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(
        "Resolved schema: "
        f"persona_col='{persona_col}', "
        f"question_col='{question_col}', "
        f"chosen_col='{chosen_col}', "
        f"rejected_col='{rejected_col}'"
    )

    # Drop rows without usable required inputs
    required = [persona_col, question_col, chosen_col, rejected_col]
    for col in required:
        df = df[df[col].notna()]
        df = df[df[col].astype(str).str.strip() != ""]
    print(f"After filtering: {len(df)} rows with valid persona/question/chosen/rejected")

    # Optional random sample
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)
        print(f"Sampled {len(df)} rows (seed={args.seed})")

    # ── Resume: skip already processed rows ───────────────────────────────────
    processed_keys = _load_processed_keys(output_path, key_cols)
    rows_to_process = [
        row.to_dict()
        for _, row in df.iterrows()
        if _build_row_key(row.to_dict(), key_cols) not in processed_keys
    ]
    if processed_keys:
        print(f"Resuming: {len(processed_keys)} already done, {len(rows_to_process)} remaining")

    if not rows_to_process:
        print("Nothing to do — all rows already processed.")
        return

    # ── Build LLM client & call function ─────────────────────────────────────
    model_name = args.model or _PROVIDER_DEFAULTS[args.provider]

    if args.provider == "gemini":
        if GeminiClient is None:
            print(
                "ERROR: Gemini dependencies are not installed. "
                "Install required packages (google-genai/google-cloud-aiplatform)."
            )
            sys.exit(1)
        load_dotenv(args.env_file)
        project_id = args.gemini_project_id or os.getenv("PROJECT_ID") or os.getenv("GEMINI_PROJECT_ID_2")
        if not project_id:
            print("ERROR: --gemini-project-id required (or set PROJECT_ID in .env)")
            sys.exit(1)
        thinking = ThinkingLevel[args.thinking_level]
        gemini_client = GeminiClient(project_id, location=args.gemini_location)

        def call_fn(sys_msg: str, usr_msg: str) -> tuple[dict, dict]:
            return _call_gemini(
                gemini_client, model_name, thinking,
                args.temperature, sys_msg, usr_msg, args.max_retries,
            )
    else:
        api_key = _resolve_api_key(args.provider, args.env_file)
        if _OpenAI is None:
            print("ERROR: openai package not installed. pip install openai")
            sys.exit(1)
        base_url = _DEEPSEEK_BASE_URL if args.provider == "deepseek" else None
        oa_client = _OpenAI(api_key=api_key, base_url=base_url)

        def call_fn(sys_msg: str, usr_msg: str) -> tuple[dict, dict]:
            return _call_openai_compat(
                oa_client, model_name, args.temperature,
                sys_msg, usr_msg, args.max_retries,
            )

    # ── CSV writer ────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    output_cols = list(df.columns)
    if RATIONALE_COL not in output_cols:
        output_cols.append(RATIONALE_COL)
    csv_lock = threading.Lock()
    out_file = output_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_file, fieldnames=output_cols, extrasaction="ignore")
    if write_header:
        writer.writeheader()

    # ── Token counters ────────────────────────────────────────────────────────
    total_tokens: dict[str, int] = {"prompt": 0, "candidates": 0, "thoughts": 0, "total": 0}
    tokens_lock = threading.Lock()

    # ── Concurrent processing ─────────────────────────────────────────────────
    def _process(row: dict) -> None:
        persona_id = str(row.get(persona_col, ""))
        persona = personas.get(persona_id)
        if persona is None:
            tqdm.write(f"  WARN  Unknown persona '{persona_id}' — skipping row")
            return

        persona_text = _build_persona_text(persona, persona_tpl)
        result = _process_row(
            row=row,
            persona=persona,
            persona_text=persona_text,
            persona_col=persona_col,
            question_col=question_col,
            chosen_col=chosen_col,
            rejected_col=rejected_col,
            sys_jinja=sys_tpl,
            usr_jinja=usr_tpl,
            call_fn=call_fn,
        )

        with tokens_lock:
            total_tokens["prompt"] += int(result.get("_prompt_tokens", 0) or 0)
            total_tokens["candidates"] += int(result.get("_candidates_tokens", 0) or 0)
            total_tokens["thoughts"] += int(result.get("_thoughts_tokens", 0) or 0)
            total_tokens["total"] += int(result.get("_total_tokens", 0) or 0)

        result.pop("_prompt_tokens", None)
        result.pop("_candidates_tokens", None)
        result.pop("_thoughts_tokens", None)
        result.pop("_total_tokens", None)

        with csv_lock:
            writer.writerow(result)
            out_file.flush()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_process, row): row for row in rows_to_process}
        with tqdm(total=len(futures), desc="Generating rationales") as pbar:
            for fut in concurrent.futures.as_completed(futures):
                pbar.update(1)
                exc = fut.exception()
                if exc:
                    row = futures[fut]
                    tqdm.write(f"  ERROR row {row.get('node_id')}: {exc}")

    out_file.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nDone. Results written to: {output_path}")
    print(f"Token usage — prompt: {total_tokens['prompt']:,}  "
          f"candidates: {total_tokens['candidates']:,}  "
          f"thoughts: {total_tokens['thoughts']:,}  "
          f"total: {total_tokens['total']:,}")


if __name__ == "__main__":
    main()
