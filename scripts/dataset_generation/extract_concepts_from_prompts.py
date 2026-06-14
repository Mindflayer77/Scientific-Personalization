import os
import json
import csv
import asyncio
import random
from tqdm import tqdm
from openai import AsyncOpenAI
from schemas.data_generation import ExtractedConcepts
from jinja2 import Environment, FileSystemLoader, PrefixLoader
from utils.config import CONFIG


RAW_BASE_PATH = CONFIG['paths']['base_path']
ABS_BASE_PATH = os.path.expanduser(RAW_BASE_PATH)

GEN_MODEL     = CONFIG['models']['generation_model']
VLLM_BASE_URL = CONFIG['models']['vllm_api_url']
VLLM_API_KEY  = CONFIG['models']['vllm_api_key']

SEMAPHORE_SIZE     = CONFIG['inference']['semaphore_size']
TEMPERATURE        = CONFIG['inference']['temperature']
TOP_P              = CONFIG['inference']['top_p']
MAX_TOKENS         = CONFIG['inference']['max_tokens']
TOP_K              = CONFIG['inference']['top_k']
PRESENCE_PENALTY   = CONFIG['inference']['presence_penalty']
REPETITION_PENALTY = CONFIG['inference']['repetition_penalty']

PROMPTS_DIR   = os.path.join(ABS_BASE_PATH, CONFIG['paths']['prompts_dir'])
PROMPTS_FILE  = os.path.join(PROMPTS_DIR, CONFIG['files']['prompts'])
CONCEPTS_FILE = os.path.join(PROMPTS_DIR, CONFIG['files']['concepts'])

client = AsyncOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
)

env = Environment(loader=PrefixLoader({
    "concept_extraction": FileSystemLoader(
        "prompt_templates/data_generation/prompt_concept_extraction/"
    ),
}))

system_template = env.get_template("concept_extraction/system.j2")
user_template   = env.get_template("concept_extraction/user.j2")

CSV_FIELDNAMES = [
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
    'chosen_reasoning',
    'chosen_abstract',
    'chosen_summary',
    'chosen_paper_id',
    'initial_scores',
    'neighbor_scores',
    'synthetic',
]


def load_pairs(pairs_path: str) -> list[dict]:
    """Load all rows from the DPO prompts CSV."""
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Prompts file not found at: {pairs_path}")
    with open(pairs_path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_completed_pairs(output_file: str) -> set[str]:
    """Return a set of pair_ids already written to the output CSV."""
    done: set[str] = set()
    if not os.path.exists(output_file):
        return done
    with open(output_file, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('pair_id'):
                done.add(row['pair_id'])
    return done


def to_json_list(items) -> str:
    """Serialize a list to a JSON string for safe CSV storage."""
    if isinstance(items, list):
        return json.dumps(items, ensure_ascii=False)
    return json.dumps([str(items)], ensure_ascii=False)


async def extract_concepts(prompt_text: str, cache: dict, semaphore: asyncio.Semaphore) -> dict:
    """
    Extract concepts from a single prompt text via the LLM.
    Results are cached by prompt text to avoid redundant API calls.
    Returns a dict with concepts, reasoning, and token counts.
    """
    if prompt_text in cache:
        return cache[prompt_text]

    max_retries = 3
    base_delay  = 2

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.responses.parse(
                    model=GEN_MODEL,
                    instructions=system_template.render(),
                    input=user_template.render(generated_prompt=prompt_text),
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_output_tokens=MAX_TOKENS,
                    text_format=ExtractedConcepts,
                    extra_body={
                        "top_k": TOP_K,
                        "presence_penalty": PRESENCE_PENALTY,
                        "repetition_penalty": REPETITION_PENALTY,
                        "chat_template_kwargs": {"enable_thinking": True},
                    },
                )

                reasoning = ""
                answer = ""
                for mess in response.output:
                    if mess.type == "message":
                        answer = mess.content[0].text
                    elif mess.type == "reasoning":
                        reasoning = mess.content[0].text

                concepts = ExtractedConcepts.model_validate_json(answer)

                usage = response.usage
                prompt_tokens    = usage.input_tokens if usage else 0
                reasoning_tokens = (
                    usage.output_tokens_details.reasoning_tokens
                    if usage and usage.output_tokens_details else 0
                )
                completion_tokens = (usage.output_tokens - reasoning_tokens) if usage else 0

                result = {
                    'concepts':           to_json_list(concepts.core_concepts),
                    'reasoning':          reasoning,
                    'prompt_tokens':      prompt_tokens,
                    'completion_tokens':  completion_tokens,
                    'reasoning_tokens':   reasoning_tokens,
                    'status':             'success',
                }
                cache[prompt_text] = result
                return result

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate limit" in err_str.lower():
                    wait_time = (base_delay ** attempt) + random.uniform(0, 1)
                    print(
                        f"\nRate limited. Retrying in {wait_time:.2f}s "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    return {'error': err_str, 'status': 'failed'}

        return {'error': 'Max retries exceeded', 'status': 'failed'}


async def process_pair(pair: dict, cache: dict, semaphore: asyncio.Semaphore) -> dict:
    """
    Process one pair: extract concepts for the chosen prompt.
    """
    chosen_result = await extract_concepts(pair['chosen_prompt'], cache, semaphore)

    if chosen_result.get('status') == 'failed':
        return {
            'initial_node':  pair['initial_node'],
            'neighbor_node': pair['neighbor_node'],
            'error':         chosen_result.get('error'),
            'status':        'failed',
        }

    return {
        'pair_id':                  pair.get('pair_id', ''),
        'initial_node':             pair['initial_node'],
        'neighbor_node':            pair['neighbor_node'],
        'initial_persona':          pair['initial_persona'],
        'target_persona':           pair['target_persona'],
        'initial_persona_score':    pair.get('initial_persona_score', ''),
        'target_persona_score':     pair.get('target_persona_score', ''),
        'relationship_type':        pair['relationship_type'],
        'chosen_prompt':            pair['chosen_prompt'],
        'chosen_concepts':          chosen_result['concepts'],
        'chosen_reasoning':         chosen_result['reasoning'],
        'chosen_abstract':          pair['chosen_abstract'],
        'chosen_summary':           pair['chosen_summary'],
        'chosen_paper_id':          pair['chosen_paper_id'],
        'initial_scores':           pair.get('initial_scores', ''),
        'neighbor_scores':          pair.get('neighbor_scores', ''),
        'synthetic':                pair['synthetic'],
        'chosen_prompt_tokens':     chosen_result['prompt_tokens'],
        'chosen_completion_tokens': chosen_result['completion_tokens'],
        'chosen_reasoning_tokens':  chosen_result['reasoning_tokens'],
        'status':                   'success',
    }


async def main():
    print(f"Model            : {GEN_MODEL}")
    print(f"vLLM base URL    : {VLLM_BASE_URL}")
    print(f"Loading prompts from: {PROMPTS_FILE}")

    all_pairs = load_pairs(PROMPTS_FILE)
    print(f"Total pairs loaded   : {len(all_pairs)}")

    done_pairs = load_completed_pairs(CONCEPTS_FILE)
    print(f"Already completed    : {len(done_pairs)}")

    pairs_to_run = [
        row for row in all_pairs
        if row.get('pair_id') not in done_pairs
    ]
    # pairs_to_run = random.sample(pairs_to_run, min(50, len(pairs_to_run)))
     
    print(f"Remaining to run     : {len(pairs_to_run)}")

    if not pairs_to_run:
        print("All tasks completed. Nothing to do.")
        return

    file_exists = os.path.exists(CONCEPTS_FILE)
    csv_file    = open(CONCEPTS_FILE, 'a', newline='', encoding='utf-8')
    writer      = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)

    if not file_exists or os.path.getsize(CONCEPTS_FILE) == 0:
        writer.writeheader()
        csv_file.flush()

    semaphore     = asyncio.Semaphore(SEMAPHORE_SIZE)
    concepts_cache: dict = {}
    tasks = [process_pair(row, concepts_cache, semaphore) for row in pairs_to_run]

    success_count    = 0
    fail_count       = 0
    total_prompt_tok = 0
    total_comp_tok   = 0
    total_reason_tok = 0

    try:
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await future

            if result['status'] == 'success':
                writer.writerow({k: result[k] for k in CSV_FIELDNAMES})
                csv_file.flush()
                success_count    += 1
                total_prompt_tok += result['chosen_prompt_tokens']     or 0
                total_comp_tok   += result['chosen_completion_tokens'] or 0
                total_reason_tok += result['chosen_reasoning_tokens']  or 0
            else:
                print(
                    f"\nFailed  initial={result.get('initial_node')}  "
                    f"neighbor={result.get('neighbor_node')}  — {result.get('error')}"
                )
                fail_count += 1
    finally:
        csv_file.close()

    print("-" * 40)
    print("Summary:")
    print(f"  Total attempted    : {len(tasks)}")
    print(f"  Successful         : {success_count}")
    print(f"  Failed             : {fail_count}")
    print(f"  Prompt tokens      : {total_prompt_tok:,}")
    print(f"  Completion tokens  : {total_comp_tok:,}")
    print(f"  Reasoning tokens   : {total_reason_tok:,}")
    print("-" * 40)


if __name__ == "__main__":
    try:
        asyncio.run(main())
        print(f"Done! Results saved to {CONCEPTS_FILE}")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"\nCritical error during execution: {e}")
