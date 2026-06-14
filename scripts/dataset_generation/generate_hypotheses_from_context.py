"""
generate_hypotheses_from_context.py

Reads a pre-built context CSV (chosen/rejected prompts + retrieved chunks) and
generates DPO hypothesis pairs.

Backends
--------
--backend vllm      Use a locally vLLM-served model (default).
--backend gemini    Use the Gemini API via GeminiClient.
--backend deepseek  Use the DeepSeek API (OpenAI-compatible).

Generation modes
----------------
--mode accepted_only   Generate only accepted (chosen) hypotheses.
--mode all             Generate both accepted and rejected hypotheses using
                       two complementary rejection strategies (default).

Rejection strategies (when --mode all)
---------------------------------------
Strategy 1  system_persona=chosen,   question=chosen_prompt,   context=rejected_context
            Applied to ALL rows (synthetic=True and False).

Strategy 2  system_persona=rejected, question=chosen_prompt,   context=chosen_context
            Applied only to rows where synthetic=False.

Strategy 3  system_persona=rejected, question=rejected_prompt, context=rejected_context
            Applied only to rows where synthetic=True.

Config keys used
----------------
files.retrieved_context  → input context CSV (relative to base_path)
files.hypotheses         → output CSV (relative to hypotheses_dir)
files.prompts            → DPO prompts CSV for persona look-up
"""

import os
import re
import csv
import json
import asyncio
import random
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

from dotenv import load_dotenv
from tqdm import tqdm
from openai import AsyncOpenAI

from schemas.data_generation import HypothesisResponse
from jinja2 import Environment, FileSystemLoader, PrefixLoader
from utils.config import CONFIG

load_dotenv()

# ---------------------------------------------------------------------------
# Paths from existing config keys
# ---------------------------------------------------------------------------
RAW_BASE_PATH = CONFIG['paths']['base_path']
ABS_BASE_PATH = os.path.expanduser(RAW_BASE_PATH)

PERSONAS_DIR  = os.path.join(ABS_BASE_PATH, CONFIG['paths']['personas_dir'])
PERSONAS_FILE = os.path.join(PERSONAS_DIR, CONFIG['files']['personas'])

HYPOTHESES_DIR = os.path.join(ABS_BASE_PATH, CONFIG['paths']['hypotheses_dir'])

# Re-use existing config keys
_CONTEXT_KEY  = CONFIG['files']['retrieved_context']
CONTEXT_FILE  = os.path.join(HYPOTHESES_DIR, _CONTEXT_KEY)
HYPOTHESES_FILE = os.path.join(HYPOTHESES_DIR, CONFIG['files']['hypotheses'])

# ---------------------------------------------------------------------------
# Models / API — vLLM backend
# ---------------------------------------------------------------------------
GEN_MODEL         = CONFIG['models']['generation_model']
VLLM_BASE_URL     = CONFIG['models']['vllm_api_url']
VLLM_API_KEY      = CONFIG['models']['vllm_api_key']
_GENERATOR_PORT   = CONFIG['models'].get('generator_port', 8003)
GENERATOR_API_URL = f"{VLLM_BASE_URL}:{_GENERATOR_PORT}/v1"

# ---------------------------------------------------------------------------
# Models / API — DeepSeek backend
# ---------------------------------------------------------------------------
DEEPSEEK_API_URL = CONFIG['models'].get('deepseek_api_url', 'https://api.deepseek.com/v1')
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', CONFIG['models'].get('deepseek_api_key', ''))

# ---------------------------------------------------------------------------
# Models / API — Gemini backend
# ---------------------------------------------------------------------------
GEMINI_PROJECT_ID = os.environ.get('PROJECT_ID', CONFIG['models'].get('gemini_project_id', ''))

# ---------------------------------------------------------------------------
# Inference settings
# ---------------------------------------------------------------------------
TEMPERATURE        = CONFIG['inference']['temperature']
TOP_K              = CONFIG['inference']['top_k']
TOP_P              = CONFIG['inference']['top_p']
MIN_P              = CONFIG['inference'].get('min_p', 0)
MAX_TOKENS         = CONFIG['inference']['max_tokens']
SEMAPHORE_SIZE     = CONFIG['inference']['semaphore_size']
REPETITION_PENALTY = CONFIG['inference'].get('repetition_penalty', 1.0)
PRESENCE_PENALTY   = CONFIG['inference'].get('presence_penalty', 0.0)
# Number of top-token alternatives to request for log-probability output.
# Set to 0 in config (inference.logprobs_top_k) to disable logprobs entirely.
LOGPROBS_TOP_K     = int(CONFIG['inference'].get('logprobs_top_k', 5))

# ---------------------------------------------------------------------------
# Jinja2 templates
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PROMPT_TEMPLATES_DIR = PROJECT_ROOT / 'prompt_templates'
DATA_GENERATION_TEMPLATES_DIR = PROMPT_TEMPLATES_DIR / 'data_generation'
HYPOTHESES_GENERATION_TEMPLATES_DIR = DATA_GENERATION_TEMPLATES_DIR / 'hypotheses_generation'

env = Environment(loader=PrefixLoader({
    "data_generation":       FileSystemLoader(str(DATA_GENERATION_TEMPLATES_DIR)),
    "hypotheses_generation": FileSystemLoader(str(HYPOTHESES_GENERATION_TEMPLATES_DIR)),
}))

system_template  = env.get_template("hypotheses_generation/system.j2")
user_template    = env.get_template("hypotheses_generation/user.j2")
persona_template = env.get_template("data_generation/persona_template.j2")

# ---------------------------------------------------------------------------
# Output CSV columns
# ---------------------------------------------------------------------------
CSV_FIELDNAMES = [
    # Pass-through from retrieval context CSV
    'pair_id',
    'initial_node',
    'neighbor_node',
    'initial_persona',
    'target_persona',
    'initial_persona_score',
    'target_persona_score',
    'relationship_type',
    'chosen_prompt',
    'chosen_concepts',
    'chosen_context',
    'chosen_context_meta',
    'chosen_hypothesis',
    'chosen_reasoning',
    'rejected_hypothesis',
    'rejected_reasoning',
    'chosen_is_answerable',
    'rejected_is_answerable',
    'chosen_falsification_criteria',
    'rejected_falsification_criteria',
    'chosen_n_chunks',
    'chosen_average_rerank_score',
    'chosen_average_rerank_score_per_concept',
    'n_excluded_nodes',
    'chosen_abstract',
    'chosen_summary',
    'chosen_paper_id',
    'initial_scores',
    'neighbor_scores',

    'synthetic',
    # Generated fields
    'strategy',
    'backend',
    'error',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_persona_desc(persona: dict) -> str:
    return persona_template.render(
        display_name=persona['display_name'],
        core_philosophy=persona['core_philosophy'],
        areas_of_expertise=persona.get('areas_of_expertise', []),
        communication_style=persona['communication_style'],
        what_i_look_for=persona.get('what_i_look_for', []),
        what_i_reject=persona.get('what_i_reject', []),
    )


def _split_reasoning_and_answer(content: str) -> Tuple[str, str]:
    """
    Extract <think>…</think> reasoning from the content string.
    Returns (reasoning, answer_json_str).
    """
    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    if think_match:
        return think_match.group(1).strip(), content[think_match.end():].strip()
    return '', content.strip()


def _is_synthetic(value: str) -> bool:
    return str(value).strip().lower() in ('true', '1', 'yes')


def _normalize_hypothesis_dict(d: dict) -> dict:
    """
    Normalise common model output variations before Pydantic validation.

    Handles:
    - 'hypothesis_statement' returned instead of 'hypothesis'
    - 'falsification_criteria' returned as a list instead of a string
    """
    d = dict(d)  # shallow copy — avoid mutating the original
    if 'hypothesist' not in d and 'hypothesis_statement' in d:
        d['hypothesis'] = d.pop('hypothesis_statement')
    if isinstance(d.get('falsification_criteria'), list):
        d['falsification_criteria'] = '; '.join(str(x) for x in d['falsification_criteria'])
    return d

def _serialize_logprobs_openai(logprobs_obj) -> tuple[str, str, str]:
    """
    Serialize OpenAI-compatible logprobs to three CSV strings:
      (logprobs_chosen, logprobs_top, avg_log_prob)

    Accepts either:
    - A plain list of token logprob objects (Responses API: part.content[0].logprobs)
    - A wrapper object with a .content list (Chat Completions API: choice.logprobs)

    Each token logprob object has .token, .logprob, .top_logprobs.
    """
    if logprobs_obj is None:
        return '', '', ''
    content = logprobs_obj if isinstance(logprobs_obj, list) else (getattr(logprobs_obj, 'content', None) or [])
    if not content:
        return '', '', ''

    chosen = json.dumps(
        [{'token': t.token, 'logprob': round(t.logprob, 4)} for t in content],
        ensure_ascii=False,
    )
    # Store top alternatives as compact float-only list-of-lists (no token
    # strings) — sufficient for entropy calculation and ~10x smaller.
    top = json.dumps(
        [
            [round(alt.logprob, 4) for alt in t.top_logprobs]
            for t in content
        ],
        ensure_ascii=False,
    )
    # Average log-prob over generated tokens
    lps = [t.logprob for t in content if t.logprob is not None]
    avg = str(round(sum(lps) / len(lps), 6)) if lps else ''
    return chosen, top, avg


def _serialize_logprobs_gemini(logprobs_result) -> tuple[str, str, str]:
    """
    Serialize a GeminiResponseParser LogprobsResult to three CSV strings:
      (logprobs_chosen, logprobs_top, avg_log_prob)
    """
    if logprobs_result is None:
        return '', '', ''

    chosen = json.dumps(
        [{'token': t.token, 'log_probability': round(t.log_probability, 4)}
         for t in logprobs_result.chosen_candidates],
        ensure_ascii=False,
    )
    # Store top alternatives as compact float-only list-of-lists.
    top = json.dumps(
        [[round(t.log_probability, 4) for t in step]
         for step in logprobs_result.top_candidates],
        ensure_ascii=False,
    )
    avg = '' if logprobs_result.avg_log_prob is None else str(round(logprobs_result.avg_log_prob, 6))
    return chosen, top, avg


def _make_openai_client(backend: str) -> AsyncOpenAI:
    if backend == 'deepseek':
        return AsyncOpenAI(base_url=DEEPSEEK_API_URL, api_key=DEEPSEEK_API_KEY)
    # vllm
    return AsyncOpenAI(base_url=GENERATOR_API_URL, api_key=VLLM_API_KEY)


# ---------------------------------------------------------------------------
# Per-backend generation implementations
# ---------------------------------------------------------------------------

async def _generate_openai_compat(
    persona_desc: str,
    question: str,
    context: str,
    client: AsyncOpenAI,
) -> dict:
    """Shared generation logic for vLLM and DeepSeek (both OpenAI-compatible)."""
    system_prompt = system_template.render(persona=persona_desc)
    user_prompt   = user_template.render(query=question, context=context)
    print(f"\nSystem prompt:\n{system_prompt}\n")
    print(f"User prompt:\n{user_prompt}\n")

    lp_kwargs: dict = {}
    if LOGPROBS_TOP_K > 0:
        lp_kwargs = {'logprobs': True, 'top_logprobs': LOGPROBS_TOP_K}

    response = await client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name":   "hypothesis-response",
                "schema": HypothesisResponse.model_json_schema(),
            },
        },
        **lp_kwargs,
        extra_body={
            "top_k":              TOP_K,
            "min_p":              MIN_P,
            "repetition_penalty": REPETITION_PENALTY,
            "presence_penalty":   PRESENCE_PENALTY,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )

    choice  = response.choices[0]
    message = choice.message
    # vLLM exposes reasoning via .reasoning (preferred) or .reasoning_content
    raw_reasoning: str = (
        getattr(message, 'reasoning', None)
        or getattr(message, 'reasoning_content', None)
        or ''
    )
    content_str: str = message.content or ''

    if raw_reasoning:
        reasoning  = raw_reasoning.strip()
        answer_str = content_str.strip()
    else:
        reasoning, answer_str = _split_reasoning_and_answer(content_str)

    data = HypothesisResponse.model_validate_json(answer_str)

    usage = response.usage
    usage_dict = {
        'prompt_tokens':     usage.prompt_tokens     if usage else 0,
        'completion_tokens': usage.completion_tokens if usage else 0,
        'total_tokens':      usage.total_tokens      if usage else 0,
    }

    lp_chosen, lp_top, avg_lp = _serialize_logprobs_openai(
        getattr(choice, 'logprobs', None)
    )

    return {
        'hypothesis':             data.hypothesis,
        'is_answerable':          data.is_answerable,
        'falsification_criteria': data.falsification_criteria,
        'reasoning':              reasoning,
        'usage':                  json.dumps(usage_dict),
        'logprobs_chosen':        lp_chosen,
        'logprobs_top':           lp_top,
        'avg_log_prob':           avg_lp,
        'status':                 'success',
    }


async def _generate_gemini(
    persona_desc: str,
    question: str,
    context: str,
    gemini_client,
    gemini_parser,
) -> dict:
    """Generation via the Gemini API."""
    system_prompt = system_template.render(persona=persona_desc)
    user_prompt   = user_template.render(query=question, context=context)

    lp_kwargs: dict = {}
    if LOGPROBS_TOP_K > 0:
        lp_kwargs = {'response_logprobs': True, 'logprobs': LOGPROBS_TOP_K}

    raw_response = await gemini_client.async_query(
        model=GEN_MODEL,
        system_message=system_prompt,
        user_message=user_prompt,
        response_schema=HypothesisResponse,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_output_tokens=MAX_TOKENS,
        **lp_kwargs,
    )

    include_lp = LOGPROBS_TOP_K > 0
    parsed = gemini_parser.parse(raw_response, parse_as_json=True, include_logprobs=include_lp)
    data   = HypothesisResponse.model_validate(_normalize_hypothesis_dict(parsed.answer_json))

    usage_dict = {
        'prompt_tokens':     parsed.usage.prompt_tokens,
        'completion_tokens': parsed.usage.candidates_tokens,
        'thoughts_tokens':   parsed.usage.thoughts_tokens,
        'total_tokens':      parsed.usage.total_tokens,
    }

    lp_chosen, lp_top, avg_lp = _serialize_logprobs_gemini(
        parsed.logprobs if include_lp else None
    )

    return {
        'hypothesis':             data.hypothesis,
        'is_answerable':          data.is_answerable,
        'falsification_criteria': data.falsification_criteria,
        'reasoning':              parsed.reasoning,
        'usage':                  json.dumps(usage_dict),
        'logprobs_chosen':        lp_chosen,
        'logprobs_top':           lp_top,
        'avg_log_prob':           avg_lp,
        'status':                 'success',
    }


# ---------------------------------------------------------------------------
# Unified entry point with retry
# ---------------------------------------------------------------------------

async def generate_hypothesis(
    persona_desc: str,
    question: str,
    context: str,
    backend: str,
    semaphore: asyncio.Semaphore,
    *,
    openai_client: AsyncOpenAI | None = None,
    gemini_client=None,
    gemini_parser=None,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> dict:

    async with semaphore:
        for attempt in range(max_retries):
            try:
                if backend == 'gemini':
                    return await _generate_gemini(
                        persona_desc, question, context, gemini_client, gemini_parser
                    )
                else:  # vllm or deepseek
                    return await _generate_openai_compat(
                        persona_desc, question, context, openai_client
                    )
            except Exception as exc:
                err = str(exc)
                if '429' in err or 'rate limit' in err.lower() or 'resource exhausted' in err.lower():
                    wait = (base_delay ** attempt) + random.uniform(0, 1)
                    print(
                        f"\nRate limited — retrying in {wait:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait)
                else:
                    return {'error': err, 'status': 'failed'}

    return {'error': 'Max retries exceeded', 'status': 'failed'}


# ---------------------------------------------------------------------------
# Row processor
# ---------------------------------------------------------------------------

async def process_row(
    row: dict,
    personas: Dict[str, dict],
    backend: str,
    semaphore: asyncio.Semaphore,
    mode: str,
    *,
    openai_client: AsyncOpenAI | None = None,
    gemini_client=None,
    gemini_parser=None,
) -> List[dict]:
    """
    Process one context CSV row.

    Returns a list of output dicts (1 row for accepted_only, 1-2 rows for all).
    """
    syn_raw = row.get('synthetic', 'False')

    # Pass through all fields from the retrieval context CSV
    _RETRIEVAL_FIELDS = [
        'pair_id', 'initial_node', 'neighbor_node', 'initial_persona', 'target_persona',
        'initial_persona_score', 'target_persona_score', 'relationship_type',
        'chosen_prompt', 'chosen_concepts', 'chosen_context', 'chosen_context_meta',
        'chosen_n_chunks', 'chosen_average_rerank_score', 'chosen_average_rerank_score_per_concept',
        'n_excluded_nodes', 'chosen_abstract', 'chosen_summary',
        'chosen_paper_id', 'initial_scores', 'neighbor_scores', 'synthetic',
    ]
    base = {field: row.get(field, '') for field in _RETRIEVAL_FIELDS}
    base['backend'] = backend

    # -- Persona look-up directly from context row --
    chosen_pid   = row.get('initial_persona', '')
    rejected_pid = row.get('target_persona', '')

    chosen_obj   = personas.get(chosen_pid)
    rejected_obj = personas.get(rejected_pid)

    if not chosen_obj or not rejected_obj:
        return [{**base,
                 'error': f'Persona object missing: chosen={chosen_pid}, rejected={rejected_pid}'}]

    chosen_desc   = _build_persona_desc(chosen_obj)
    rejected_desc = _build_persona_desc(rejected_obj)

    chosen_prompt    = row.get('chosen_prompt', '')
    chosen_context   = row.get('chosen_context', '')

    # ------------------------------------------------------------------
    # accepted_only: one hypothesis, no rejected
    # ------------------------------------------------------------------
    gen_kwargs = dict(
        backend=backend,
        semaphore=semaphore,
        openai_client=openai_client,
        gemini_client=gemini_client,
        gemini_parser=gemini_parser,
    )

    if mode == 'accepted_only':
        accepted = await generate_hypothesis(chosen_desc, chosen_prompt, chosen_context, **gen_kwargs)
        if accepted.get('status') == 'failed':
            return [{**base, 'strategy': '', 'error': accepted.get('error', '')}]

        return [{
            **base,
            'strategy':            '',
            'chosen_hypothesis':   accepted['hypothesis'],
            'chosen_reasoning':    accepted['reasoning'],
            # 'chosen_is_answerable': accepted['is_answerable'],
            'chosen_falsification_criteria': accepted['falsification_criteria'],
            'rejected_hypothesis': '',
            'rejected_reasoning':  '',
            'rejected_is_answerable': '',
            'rejected_falsification_criteria': '',
            '_chosen_usage':       accepted.get('usage', ''),
            '_rejected_usage':     '',
            'error':               '',
        }]

    # ------------------------------------------------------------------
    # all: generate accepted first, then rejected (strategies 1, 2, and 3)
    # Strategy 2 only for non-synthetic pairs.
    # Strategy 3 only for synthetic pairs.
    # If accepted.is_answerable is empty, skip rejected generation entirely.
    # ------------------------------------------------------------------
    synthetic_flag = _is_synthetic(syn_raw)

    accepted = await generate_hypothesis(chosen_desc, chosen_prompt, chosen_context, **gen_kwargs)
    if accepted.get('status') == 'failed':
        return [{**base, 'strategy': '', 'error': f"accepted: {accepted.get('error', '')}"}]

    # if not accepted.get('is_answerable', ''):
    #     # is_answerable is empty — return a single row with no rejected fields
    #     return [{
    #         **base,
    #         'strategy':            '',
    #         'chosen_hypothesis':   accepted['hypothesis'],
    #         'chosen_reasoning':    accepted['reasoning'],
    #         'chosen_is_answerable': accepted['is_answerable'],
    #         'chosen_falsification_criteria': accepted['falsification_criteria'],
    #         'rejected_hypothesis': '',
    #         'rejected_reasoning':  '',
    #         'rejected_is_answerable': '',
    #         'rejected_falsification_criteria': '',
    #         '_chosen_usage':       accepted.get('usage', ''),
    #         '_rejected_usage':     '',
    #         'error':               'skipped: accepted is_answerable is empty',
    #     }]

    rej_tasks: dict = {
        # 'strat1': generate_hypothesis(chosen_desc, chosen_prompt, rejected_context, **gen_kwargs),
    }
    # if synthetic_flag:
    rej_tasks['strat2'] = generate_hypothesis(
        rejected_desc, chosen_prompt, chosen_context, **gen_kwargs
    )
    # if synthetic_flag:
    #     rejected_prompt = row.get('rejected_prompt', '')
    #     rej_tasks['strat3'] = generate_hypothesis(
    #         rejected_desc, rejected_prompt, rejected_context, **gen_kwargs
    #     )

    rej_results = dict(
        zip(rej_tasks.keys(), await asyncio.gather(*rej_tasks.values()))
    )

    output_rows: List[dict] = []

    # Strategy 1 row (all rows)
    # strat1 = rej_results['strat1']
    # output_rows.append({
    #     **base,
    #     'strategy':            1,
    #     'chosen_hypothesis':   accepted['hypothesis'],
    #     'rejected_hypothesis': strat1.get('hypothesis', ''),
    #     'rejected_reasoning':  strat1.get('reasoning', ''),
    #     'error':               strat1.get('error', '') if strat1.get('status') == 'failed' else '',
    # })

    # Strategy 2 row (only for synthetic pairs)
    # if synthetic_flag:
    strat2 = rej_results['strat2']
    output_rows.append({
        **base,
        'strategy':            2,
        'chosen_hypothesis':   accepted.get('hypothesis', ''),
        'chosen_reasoning':    accepted.get('reasoning', ''),
        'chosen_is_answerable': accepted.get('is_answerable', ''),
        'chosen_falsification_criteria': accepted['falsification_criteria'],
        'rejected_hypothesis': strat2.get('hypothesis', ''),
        'rejected_reasoning':  strat2.get('reasoning', ''),
        'rejected_is_answerable': strat2.get('is_answerable'),
        'rejected_falsification_criteria': strat2.get('falsification_criteria', ''),
        'error':               strat2.get('error', '') if strat2.get('status') == 'failed' else '',
    })

    # Strategy 3 row (only for synthetic pairs)
    # if synthetic_flag:
    #     strat3 = rej_results['strat3']
    #     output_rows.append({
    #         **base,
    #         'strategy':            3,
    #         'chosen_hypothesis':   accepted['hypothesis'],
    #         'rejected_hypothesis': strat3.get('hypothesis', ''),
    #         'rejected_reasoning':  strat3.get('reasoning', ''),
    #         'error':               strat3.get('error', '') if strat3.get('status') == 'failed' else '',
    #     })

    return output_rows


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_context_csv(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Context CSV not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_personas(path: str) -> Dict[str, dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return {p['persona_id']: p for p in json.load(f)}


def _expected_keys(row: dict, mode: str) -> Set[tuple]:
    """Return the set of (pair_id, strategy) keys expected for this source row."""
    pid = row['pair_id']
    if mode == 'accepted_only':
        return {(pid, '')}
    expected = {(pid, '1')}
    synthetic = _is_synthetic(row.get('synthetic', 'False'))
    if not synthetic:
        expected.add((pid, '2'))
    if synthetic:
        expected.add((pid, '3'))
    return expected


def load_done_pairs(output_path: str) -> Set[Tuple[str, str]]:
    """
    Return a set of (pair_id, strategy) tuples already present in the output CSV.
    For accepted_only rows the strategy field is an empty string.
    """
    done: Set[Tuple[str, str]] = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            pid = row.get('pair_id', '')
            if pid:
                done.add((pid, str(row.get('strategy', ''))))
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DPO hypotheses from a pre-built context CSV."
    )
    parser.add_argument(
        '--backend',
        choices=['vllm', 'gemini', 'deepseek'],
        default='vllm',
        help="Generation backend (default: vllm).",
    )
    parser.add_argument(
        '--mode',
        choices=['accepted_only', 'all'],
        default='all',
        help=(
            "'accepted_only': generate only accepted hypotheses. "
            "'all': generate both accepted and rejected using strategies 1 and 2 "
            "(default: all)."
        ),
    )

    args = parser.parse_args()

    input_file  = CONTEXT_FILE
    output_file = HYPOTHESES_FILE

    if not input_file:
        parser.error(
            "No input file specified. Set files.retrieved_context in config.yaml "
            "or pass --input <path>."
        )

    print(f"Backend          : {args.backend}")
    print(f"Mode             : {args.mode}")
    print(f"Generation model : {GEN_MODEL}")
    print(f"Input context    : {input_file}")
    print(f"Output file      : {output_file}")
    print(f"Working dir      : {os.getcwd()}")
    print(f"Project root     : {PROJECT_ROOT}")
    print(f"Template root    : {PROMPT_TEMPLATES_DIR}")
    print(f"System template  : {getattr(system_template, 'filename', 'unknown')}")
    print(f"User template    : {getattr(user_template, 'filename', 'unknown')}")
    print(f"Persona template : {getattr(persona_template, 'filename', 'unknown')}")

    print("\nLoading context CSV …")
    rows = load_context_csv(input_file)
    print(f"  Total rows: {len(rows)}")

    print("Loading personas …")
    personas = load_personas(PERSONAS_FILE)
    print(f"  Personas loaded: {len(personas)}")

    os.makedirs(HYPOTHESES_DIR, exist_ok=True)

    done_keys = load_done_pairs(output_file)
    print(f"Already completed: {len(done_keys)}")

    rows_to_run = [
        r for r in rows
        if not _expected_keys(r, args.mode).issubset(done_keys)
    ]

    rows_to_run = random.sample(rows_to_run, min(len(rows_to_run), 500))

    print(f"Remaining rows   : {len(rows_to_run)}\n")

    if not rows_to_run:
        print("Nothing to do — all rows already processed.")
        return

    # ------------------------------------------------------------------
    # Instantiate backend clients
    # ------------------------------------------------------------------
    openai_client = None
    gemini_client = None
    gemini_parser = None

    if args.backend == 'gemini':
        from api_client.gemini_client import GeminiClient
        from api_client.gemini_parser import GeminiResponseParser
        if not GEMINI_PROJECT_ID:
            raise EnvironmentError(
                "Gemini backend requires PROJECT_ID env var or "
                "models.gemini_project_id in config."
            )
        gemini_client = GeminiClient(project_id=GEMINI_PROJECT_ID)
        gemini_parser = GeminiResponseParser()
        print(f"Gemini project   : {GEMINI_PROJECT_ID}")
    elif args.backend == 'deepseek':
        if not DEEPSEEK_API_KEY:
            raise EnvironmentError(
                "DeepSeek backend requires DEEPSEEK_API_KEY env var or "
                "models.deepseek_api_key in config."
            )
        openai_client = _make_openai_client('deepseek')
        print(f"DeepSeek API URL : {DEEPSEEK_API_URL}")
    else:  # vllm
        openai_client = _make_openai_client('vllm')
        print(f"vLLM API URL     : {GENERATOR_API_URL}")

    semaphore = asyncio.Semaphore(SEMAPHORE_SIZE)

    file_exists = os.path.exists(output_file) and os.path.getsize(output_file) > 0

    tasks = [
        process_row(
            row, personas,
            backend=args.backend,
            semaphore=semaphore,
            mode=args.mode,
            openai_client=openai_client,
            gemini_client=gemini_client,
            gemini_parser=gemini_parser,
        )
        for row in rows_to_run
    ]

    success_count = 0
    fail_count    = 0
    total_prompt_tokens     = 0
    total_completion_tokens = 0

    with open(output_file, 'a', newline='', encoding='utf-8', errors='replace') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
            csv_file.flush()

        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            output_rows = await future

            for out_row in output_rows:
                writer.writerow({k: out_row.get(k, '') for k in CSV_FIELDNAMES})
                for usage_field in ('_chosen_usage', '_rejected_usage'):
                    raw = out_row.get(usage_field, '')
                    if raw:
                        try:
                            u = json.loads(raw)
                            total_prompt_tokens     += u.get('prompt_tokens', 0) or u.get('input_tokens', 0)
                            total_completion_tokens += u.get('completion_tokens', 0) or u.get('output_tokens', 0)
                        except (json.JSONDecodeError, AttributeError):
                            pass
            csv_file.flush()

            any_error = any(r.get('error') for r in output_rows)
            if any_error:
                for r in output_rows:
                    if r.get('error'):
                        print(
                            f"\nError  pair_id={r.get('pair_id')}  "
                            f"strategy={r.get('strategy')}: {r['error']}"
                        )
                fail_count += 1
            else:
                success_count += 1

    print("\n" + "-" * 40)
    print("Summary:")
    print(f"  Source rows processed : {len(tasks)}")
    print(f"  Successful            : {success_count}")
    print(f"  Failed                : {fail_count}")
    print(f"  Input tokens          : {total_prompt_tokens:,}")
    print(f"  Output tokens         : {total_completion_tokens:,}")
    print(f"  Total tokens          : {total_prompt_tokens + total_completion_tokens:,}")
    print(f"  Output saved to       : {output_file}")
    print("-" * 40)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FileNotFoundError as exc:
        print(exc)
    except Exception as exc:
        print(f"\nCritical error: {exc}")
        raise
