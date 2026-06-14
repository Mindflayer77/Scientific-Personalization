import os
import json
import csv
import asyncio
import random
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from openai import AsyncOpenAI
from neo4j import GraphDatabase
from schemas.data_generation import SyntheticUserPrompt
from jinja2 import Environment, FileSystemLoader, PrefixLoader
from utils.config import CONFIG

RAW_BASE_PATH = CONFIG['paths']['base_path']
ABS_BASE_PATH = os.path.expanduser(RAW_BASE_PATH)

PERSONAS_DIR = os.path.join(ABS_BASE_PATH, CONFIG['paths']['personas_dir'])
PERSONAS_FILE = os.path.join(PERSONAS_DIR, CONFIG['files']['personas'])

GEN_MODEL = CONFIG['models']['generation_model']
VLLM_BASE_URL = CONFIG['models']['vllm_api_url']
VLLM_API_KEY = CONFIG['models']['vllm_api_key']

load_dotenv(Path(__file__).resolve().parents[2] / ".env_db")
NEO4J_URI = os.getenv("NEO4J_URI")
if not NEO4J_URI:
    vps_ip = os.getenv("VPS_IP")
    if not vps_ip:
        raise EnvironmentError("Missing NEO4J_URI or VPS_IP in .env_db")
    NEO4J_URI = f"bolt://{vps_ip}:7687"

NEO4J_USER = os.getenv("NEO4J_USER") or CONFIG['neo4j']['user']
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or CONFIG['neo4j']['password']

SEMAPHORE_SIZE = CONFIG['inference']['semaphore_size']

TEMPERATURE = CONFIG['inference']['temperature']
TOP_P = CONFIG['inference']['top_p']
MAX_TOKENS = CONFIG['inference']['max_tokens']
TOP_K = CONFIG['inference']['top_k']
PRESENCE_PENALTY = CONFIG['inference']['presence_penalty']
REPETITION_PENALTY = CONFIG['inference']['repetition_penalty']

ARTICLES_DIR = os.path.join(ABS_BASE_PATH, CONFIG['paths']['articles_dir'])
PROMPTS_DIR = os.path.join(ABS_BASE_PATH, CONFIG['paths']['prompts_dir'])
PAIRS_FILE = os.path.join(ARTICLES_DIR, CONFIG['files']['pairs'])
OUTPUT_FILE = os.path.join(PROMPTS_DIR, CONFIG['files']['prompts'])

client = AsyncOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
)

env = Environment(loader=PrefixLoader({
    "persona_prompts": FileSystemLoader("prompt_templates/data_generation/persona_prompts"),
    "data_generation": FileSystemLoader("prompt_templates/data_generation/"),
}))

system_template = env.get_template("persona_prompts/system.j2")
user_template = env.get_template("persona_prompts/user.j2")
persona_template = env.get_template("data_generation/persona_template.j2")

token_stats = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}
token_lock = asyncio.Lock()

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
    'chosen_reasoning',
    'chosen_abstract',
    'chosen_summary',
    'chosen_paper_id',
    'initial_scores',
    'neighbor_scores',
    'synthetic',
]

def load_personas(personas_path: str):
    with open(personas_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_pairs(pairs_path: str):
    pairs = []
    with open(pairs_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(row)
    return pairs

def load_processed_pairs(output_path: str):
    """Load pair_ids that have already been processed from the output CSV."""
    processed_pairs = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row and row.get('pair_id'):
                        processed_pairs.add(row['pair_id'])
        except Exception as e:
            print(f"Warning: Could not read processed pairs from {output_path}: {e}")
    return processed_pairs

def get_node_data(driver, node_id, cache):
    if node_id in cache:
        return cache[node_id]
    
    if node_id.isdigit():
        query = """
        MATCH (n)
        WHERE id(n) = $id
        RETURN n.abstract AS abstract, n.summary AS summary, n.paperId AS paperId
        """
        params = {'id': int(node_id)}
    else:
        query = """
        MATCH (n)
        WHERE elementId(n) = $id
        RETURN n.abstract AS abstract, n.summary AS summary, n.paperId AS paperId
        """
        params = {'id': node_id}

    with driver.session() as session:
        result = session.run(query, **params)
        record = result.single()
        if record:
            data = {
                'abstract': record['abstract'],
                'summary': record['summary'],
                'paperId': record['paperId']
            }
            cache[node_id] = data
            return data
        else:
            cache[node_id] = None
            return None

async def generate_prompt(node_id: str, persona_obj: dict, node_data_cache: dict, prompt_cache: dict, driver, semaphore: asyncio.Semaphore) -> dict:
    cache_key = (node_id, persona_obj['persona_id'])
    if cache_key in prompt_cache:
        return prompt_cache[cache_key]
    
    node_data = get_node_data(driver, node_id, node_data_cache)
    if not node_data:
        return {'error': 'Node data not found', 'generated_prompt_text': None, 'usage': None}
    
    persona_rendered = persona_template.render(
        display_name=persona_obj['display_name'],
        core_philosophy=persona_obj['core_philosophy'],
        areas_of_expertise=persona_obj['areas_of_expertise'],
        communication_style=persona_obj['communication_style'],
        what_i_look_for=persona_obj['what_i_look_for'],
        what_i_reject=persona_obj['what_i_reject'],
    )
    
    instructions = system_template.render(persona=persona_rendered)
    input_text = user_template.render(
        persona=persona_rendered,
        abstract=node_data['abstract'],
        summary=node_data['summary'],
    )
    
    max_retries = 3
    base_delay = 2
    
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=GEN_MODEL,
                    messages=[
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": input_text},
                    ],
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_completion_tokens=MAX_TOKENS,
                    presence_penalty=PRESENCE_PENALTY,
                    # response_format={"type": "json_object"},
                    extra_body={
                        "top_k": TOP_K,
                        "repetition_penalty": REPETITION_PENALTY,
                        "chat_template_kwargs": {"enable_thinking": True},
                    }
                )

                answer = response.choices[0].message.content
                reasoning = getattr(response.choices[0].message, 'reasoning_content', '') or ''

                # prompt_obj = SyntheticUserPrompt.model_validate_json(answer)

                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                total_tokens = usage.total_tokens if usage else 0
                reasoning_tokens = usage.completion_tokens_details.reasoning_tokens if usage and usage.completion_tokens_details else 0

                async with token_lock:
                    token_stats["input_tokens"] += input_tokens
                    token_stats["output_tokens"] += output_tokens
                    token_stats["total_tokens"] += total_tokens
                    token_stats["reasoning_tokens"] += reasoning_tokens

                result = {
                    'generated_prompt_text': answer,
                    'reasoning': reasoning,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'reasoning_tokens': reasoning_tokens,
                    'total_tokens': total_tokens,
                    'abstract': node_data['abstract'],
                    'summary': node_data['summary'],
                    'paper_id': node_data['paperId'],
                }
                prompt_cache[cache_key] = result
                return result
    
            except Exception as e:
                if "429" in str(e) or "Resource exhausted" in str(e):
                    wait_time = (base_delay ** attempt) + random.uniform(0, 1)
                    print(f"\nRate limited. Retrying in {wait_time:.2f}s "
                          f"(attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(wait_time)
                else:
                    return {
                        'error': str(e),
                        'generated_prompt_text': None,
                        'usage': None
                    }
    
        return {
            'error': 'Max retries exceeded',
            'generated_prompt_text': None,
            'usage': None
        }

async def process_pair(pair: dict, personas: list, node_data_cache: dict, prompt_cache: dict, driver, semaphore: asyncio.Semaphore) -> dict:
    initial_persona_obj = next((p for p in personas if p['persona_id'] == pair['initial_persona']), None)
    target_persona_obj = next((p for p in personas if p['persona_id'] == pair['target_persona']), None)
    
    if not initial_persona_obj or not target_persona_obj:
        return {
            'initial_node': pair['initial_node'],
            'neighbor_node': pair['neighbor_node'],
            'error': 'Persona not found',
            'status': 'failed'
        }
    
    chosen_result = await generate_prompt(pair['initial_node'], initial_persona_obj, node_data_cache, prompt_cache, driver, semaphore)
    
    if chosen_result.get('error'):
        return {
            'initial_node': pair['initial_node'],
            'neighbor_node': pair['neighbor_node'],
            'error': chosen_result.get('error'),
            'status': 'failed'
        }
    
    return {
        'pair_id': pair.get('pair_id', ''),
        'initial_node': pair['initial_node'],
        'neighbor_node': pair['neighbor_node'],
        'initial_persona': pair['initial_persona'],
        'target_persona': pair['target_persona'],
        'initial_persona_score': pair.get('initial_persona_score', ''),
        'target_persona_score': pair.get('target_persona_score', ''),
        'chosen_prompt': chosen_result['generated_prompt_text'],
        'chosen_abstract': chosen_result['abstract'],
        'chosen_summary': chosen_result['summary'],
        'chosen_paper_id': chosen_result['paper_id'],
        'chosen_reasoning': chosen_result['reasoning'],
        'initial_scores': pair.get('initial_scores', ''),
        'neighbor_scores': pair.get('neighbor_scores', ''),
        'relationship_type': pair['relationship_type'],
        'synthetic': pair['synthetic'],
        'status': 'success',
    }

async def main():
    personas = load_personas(PERSONAS_FILE)
    pairs = load_pairs(PAIRS_FILE)
    
    # Load already processed pairs and filter them out
    processed_pairs = load_processed_pairs(OUTPUT_FILE)
    original_count = len(pairs)
    pairs = [p for p in pairs if p.get('pair_id') not in processed_pairs]
    # pairs = random.sample(pairs, min(50, len(pairs)))

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    node_data_cache = {}
    prompt_cache = {}
    
    print(f"Total pairs in input file: {original_count}")
    print(f"Already processed pairs: {len(processed_pairs)}")
    print(f"Pairs to process: {len(pairs)}")
    
    file_exists = os.path.exists(OUTPUT_FILE)
    csv_file = open(OUTPUT_FILE, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
    if not file_exists or os.path.getsize(OUTPUT_FILE) == 0:
        writer.writeheader()
        csv_file.flush()
    
    semaphore = asyncio.Semaphore(SEMAPHORE_SIZE)
    tasks = [process_pair(pair, personas, node_data_cache, prompt_cache, driver, semaphore) for pair in pairs]
    

    success_count = 0
    fail_count = 0
    
    try:
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await future
    
            if result['status'] == 'success':
                writer.writerow({k: result[k] for k in CSV_FIELDNAMES})
                csv_file.flush()
                success_count += 1
            else:
                print(f"\nFailed pair {result.get('initial_node')} -> {result.get('neighbor_node')} — {result.get('error')}")
                fail_count += 1
    finally:
        csv_file.close()
        driver.close()
    
    print("-" * 30)
    print(f"Summary:")
    print(f"  Total Attempted : {len(tasks)}")
    print(f"  Successful      : {success_count}")
    print(f"  Failed          : {fail_count}")
    print(f"Token Usage:")
    print(f"  Input tokens    : {token_stats['input_tokens']:,}")
    print(f"  Output tokens   : {token_stats['output_tokens']:,}")
    print(f"  Reasoning tokens: {token_stats['reasoning_tokens']:,}")
    print(f"  Total tokens    : {token_stats['total_tokens']:,}")
    print("-" * 30)

if __name__ == "__main__":
    try:
        asyncio.run(main())
        print(f"Done! Results saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"\nCritical error during execution: {e}")