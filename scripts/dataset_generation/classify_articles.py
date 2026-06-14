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
from schemas.data_generation import PersonaClassification
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
NODES_FILE = os.path.join(ARTICLES_DIR, 'neo4j_sample_nodes.csv')
REFERENCES_FILE = os.path.join(ARTICLES_DIR, 'neo4j_sample_references.csv')
OUTPUT_FILE = os.path.join(ARTICLES_DIR, 'classifications.csv')

client = AsyncOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
)

env = Environment(loader=PrefixLoader({
    "persona_prompts": FileSystemLoader("prompt_templates/data_generation/persona_prompts"),
    "data_generation": FileSystemLoader("prompt_templates/data_generation/"),
}))

system_template = env.get_template("persona_prompts/classify_system.j2")
user_template = env.get_template("persona_prompts/classify_user.j2")

token_stats = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}
token_lock = asyncio.Lock()

CSV_FIELDNAMES = [
    'paperId',
    'abstract',
    'summary',
    'node_id',
    'relationships',
    'scores',
    'reasoning',
    'input_tokens',
    'output_tokens',
    'reasoning_tokens',
    'total_tokens',
]


def load_personas(personas_path: str):
    with open(personas_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_node_ids(nodes_path: str) -> list[str]:
    """
    Load the canonical list of sampled node IDs from neo4j_sample_nodes.csv.
    Using this as the task source guarantees exactly one LLM call per node
    regardless of how many relationships that node has, and captures nodes
    that may have no relationships at all after the deduplication fix.
    """
    with open(nodes_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [row['node_id'] for row in reader]


def load_references(references_path: str) -> dict[str, list[dict]]:
    """
    Build a node_id -> relationships lookup from neo4j_sample_references.csv.

    The new CSV (produced by the fixed collect_references query) has three
    columns only: source_node_id, relationship_type, target_node_id.
    There is no direction column — each physical edge appears exactly once
    with source = startNode(r) and target = endNode(r).

    Direction is computed from position: for the source node the edge is
    outgoing, for the target node it is incoming. Both nodes receive an
    entry so that each can look up its full neighbourhood when the
    classification result is stored.
    """
    references: dict[str, list[dict]] = {}

    with open(references_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = row['source_node_id']
            target = row['target_node_id']
            rel_type = row['relationship_type']

            if source not in references:
                references[source] = []
            references[source].append({
                'relationship_type': rel_type,
                'source_node_id': source,
                'target_node_id': target,
                'direction': 'outgoing',
            })

            if target not in references:
                references[target] = []
            references[target].append({
                'relationship_type': rel_type,
                'source_node_id': source,
                'target_node_id': target,
                'direction': 'incoming',
            })

    return references


def load_processed_nodes(output_path: str) -> set[str]:
    processed = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row and row.get('node_id'):
                        processed.add(row['node_id'])
        except Exception as e:
            print(f"Warning: could not read existing output file: {e}")
    return processed


def get_node_data(driver, node_id: str) -> dict | None:
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
            return {
                'abstract': record['abstract'],
                'summary': record['summary'],
                'paperId': record['paperId'],
            }
        return None


async def process_node(
    node_id: str,
    references: list,
    driver,
    personas: list,
    semaphore: asyncio.Semaphore,
) -> dict:
    node_data = get_node_data(driver, node_id)
    if not node_data:
        return {'node_id': node_id, 'error': 'Node data not found', 'status': 'failed'}

    shuffled_personas = personas.copy()
    random.shuffle(shuffled_personas)

    instructions = system_template.render(personas=shuffled_personas)
    input_text = user_template.render(**node_data)

    max_retries = 3
    base_delay = 2

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.responses.parse(
                    model=GEN_MODEL,
                    instructions=instructions,
                    input=input_text,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_output_tokens=MAX_TOKENS,
                    text_format=PersonaClassification,
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

                classification = PersonaClassification.model_validate_json(answer)

                usage = response.usage
                input_tokens = usage.input_tokens if usage else 0
                output_tokens = usage.output_tokens if usage else 0
                total_tokens = usage.total_tokens if usage else 0
                reasoning_tokens = (
                    usage.output_tokens_details.reasoning_tokens
                    if usage and usage.output_tokens_details else 0
                )

                async with token_lock:
                    token_stats["input_tokens"] += input_tokens
                    token_stats["output_tokens"] += output_tokens
                    token_stats["total_tokens"] += total_tokens
                    token_stats["reasoning_tokens"] += reasoning_tokens

                return {
                    'paperId': node_data['paperId'],
                    'abstract': node_data['abstract'],
                    'summary': node_data['summary'],
                    'node_id': node_id,
                    'relationships': json.dumps(references),
                    'scores': json.dumps(classification.scores),
                    'reasoning': reasoning,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'reasoning_tokens': reasoning_tokens,
                    'total_tokens': total_tokens,
                    'status': 'success',
                }

            except Exception as e:
                if "429" in str(e) or "Resource exhausted" in str(e):
                    wait_time = (base_delay ** attempt) + random.uniform(0, 1)
                    print(f"\nRate limited. Retrying in {wait_time:.2f}s "
                          f"(attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(wait_time)
                else:
                    return {'node_id': node_id, 'error': str(e), 'status': 'failed'}

        return {'node_id': node_id, 'error': 'Max retries exceeded', 'status': 'failed'}


async def main():
    personas = load_personas(PERSONAS_FILE)

    # Load canonical node list — one entry per sampled node, no duplicates.
    # This is the single source of truth for which nodes need classification.
    all_node_ids = load_node_ids(NODES_FILE)

    # Load relationships as a lookup keyed by node_id.
    # Nodes with no relationships get an empty list via .get() below.
    references = load_references(REFERENCES_FILE)

    processed_nodes = load_processed_nodes(OUTPUT_FILE)

    # Union of initial nodes and every node that appears in any reference edge,
    # preserving the original order (initial nodes first, then reference-only nodes).
    reference_node_ids = [nid for nid in references if nid not in set(all_node_ids)]
    candidate_node_ids = all_node_ids + reference_node_ids

    tasks_to_run = [
        (node_id, references.get(node_id, []))
        for node_id in candidate_node_ids
        if node_id not in processed_nodes
    ]

    print(f"Total sampled nodes            : {len(all_node_ids)}")
    print(f"Total reference nodes          : {len(reference_node_ids)}")
    print(f"Total candidate nodes          : {len(candidate_node_ids)}")
    print(f"Already processed              : {len(processed_nodes)}")
    print(f"Nodes to process               : {len(tasks_to_run)}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    file_exists = os.path.exists(OUTPUT_FILE)
    csv_file = open(OUTPUT_FILE, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
    if not file_exists or os.path.getsize(OUTPUT_FILE) == 0:
        writer.writeheader()
        csv_file.flush()

    semaphore = asyncio.Semaphore(SEMAPHORE_SIZE)
    tasks = [
        process_node(node_id, refs, driver, personas, semaphore)
        for node_id, refs in tasks_to_run
    ]

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
                print(f"\nFailed node={result.get('node_id')} — {result.get('error')}")
                fail_count += 1
    finally:
        csv_file.close()
        driver.close()

    print("-" * 30)
    print(f"Summary:")
    print(f"  Total attempted : {len(tasks)}")
    print(f"  Successful      : {success_count}")
    print(f"  Failed          : {fail_count}")
    print(f"Token usage:")
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