import os
import asyncio
import json
from typing import Dict
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import json_repair
import random
from utils.config import CONFIG

RAW_BASE_PATH = CONFIG['paths']['base_path']
ABS_BASE_PATH = os.path.expanduser(RAW_BASE_PATH)

GEN_MODEL = CONFIG['models']['generation_model']
VLLM_API_URL = CONFIG['models']['vllm_api_url']
VLLM_API_KEY = CONFIG['models']['vllm_api_key']

TEMPERATURE = CONFIG['inference']['temperature']
TOP_K = CONFIG['inference']['top_k']
TOP_P = CONFIG['inference']['top_p']
MIN_P = CONFIG['inference']['min_p']
MAX_TOKENS = CONFIG['inference']['max_tokens']
SEMAPHORE_SIZE = CONFIG['inference']['semaphore_size']

PROMPTS_DIR = os.path.join(ABS_BASE_PATH, CONFIG['paths']['prompts_dir'])
INPUT_FILE = os.path.join(PROMPTS_DIR, CONFIG['files']['taxonomy_phase1'])
OUTPUT_FILE = os.path.join(PROMPTS_DIR, CONFIG['files']['taxonomy_phase2'])

system_prompt_text = """
    You are a specialized Professor and Curriculum Designer at a top-tier research institute.
    You are currently designing the curriculum for the specific research track: {DISCIPLINE}.
    Your goal is to prepare a PhD-level course list.
    
    IMPORTANT: You output ONLY raw valid JSON. Do not use Markdown code blocks (e.g., ```json).
    Do not provide explanations or introductions.

    Example Output Structure (Follow this level of depth):
    {{
      "subjects": [
        {{
          "name": "State Space Models and Linear Recurrent Dualities",
          "level": "Frontier",
          "description": "Investigation of the intersection between classical State Space Models (SSMs) and modern deep learning recurrence.",
          "key_subtopics": [
            "Structured State Space Sequence Models (S4)",
            "HiPPO (High-order Polynomial Projection Operators)",
            "Recurrent-Convolutional Duality and Parallelization",
            "Associative Scan Algorithms for Linear Recurrence",
            "Selective State Spaces and Hardware-Aware Gating (Mamba)",
            "Discretization Schemes (Bilinear, Zero-Order Hold)"
          ]
        }}
      ]
    }}
"""

user_prompt_text = """
    Context: This discipline is part of the broader sub-field '{SUBFIELD}' within '{FIELD}'.

    Task: Generate a comprehensive list of 8 Academic Subjects (Courses) that a PhD student must
    master to become a world-class expert in '{DISCIPLINE}'.

    Guidelines:
    1. Granularity & Specificity: Avoid generic titles like 'Intro to RL'. Instead, use specific,
    research-oriented titles like 'Sparse Reward Environments' or 'Curiosity-Driven Exploration'.
    2. Theoretical & Applied Balance: Include subjects covering not only mathematical proofs (theorems), 
    but also algorithmic and system implementations and emerging research frontiers.
    3. Concept Density: For 'key_subtopics', list at least 5-8 specific concepts per subject. They must be detailed.
    4. Be Exhaustive: Include all important subjects and subtopics to cover each discipline.
    5. Correctness over exhaustiveness: Make sure to prioritize correctness and do NOT invent new non-existent topics.
    You can include less than 5 subtopics for a given subject if the subject is already exhausted.
    If you can't find subtopics for a subject, return message *N/A* in key_subtopics field.
    6. Redundancy check: Minimize conceptual overlap between closely related disciplines.
    7. Use only the following three levels for subject leveling:
        1) Foundational: Established methods and core algorithms.
        2) Advanced: Current state-of-the-art and complex methods and systems.
        3) Frontier: Unresolved research gaps and theoretical breaking points.
    
    Generate json now:
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt_text),
    ("user", user_prompt_text)
])


# llm_generator = ChatOpenAI(
#     model=GEN_MODEL, 
#     openai_api_base=VLLM_API_URL, 
#     openai_api_key=VLLM_API_KEY,
#     max_tokens=MAX_TOKENS, 
#     # temperature=TEMPERATURE,
#     # top_p=TOP_P,
#     extra_body={
#         # "top_k": TOP_K,
#         # "min_p": MIN_P,
#         "thinking_level": "high"
#     }
# )

llm_generator = ChatGoogleGenerativeAI(
    model=GEN_MODEL, 
    # openai_api_base=VLLM_API_URL, 
    api_key=VLLM_API_KEY,
    max_tokens=MAX_TOKENS, 
    temperature=TEMPERATURE,
    # top_p=TOP_P,
    thinking_level="high",
)

async def process_discipline(item: Dict, semaphore: asyncio.Semaphore) -> Dict:
    
    discipline = item['discipline']
    subfield = item['subfield']
    field = item['field']

    inputs = {
        "DISCIPLINE": discipline,
        "SUBFIELD": subfield,
        "FIELD": field
    }

    max_retries = 3
    base_delay = 2

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await (prompt_template | llm_generator).ainvoke(inputs)
                parsed_data = json_repair.loads(response.text)
                
                subjects_list = []
                if isinstance(parsed_data, dict):
                    subjects_list = parsed_data.get("subjects", [])
                elif isinstance(parsed_data, list):
                    if len(parsed_data) > 0 and "subjects" in parsed_data[0]:
                        subjects_list = parsed_data[0]["subjects"]
                    else:
                        subjects_list = parsed_data

                return {
                    "discipline": discipline,
                    "subfield": subfield,
                    "field": field,
                    "subjects": subjects_list,
                    "status": "success"
                }

            except Exception as e:
                # Check if it's a rate limit error (usually 429)
                if "429" in str(e) or "Resource exhausted" in str(e):
                    wait_time = (base_delay ** attempt) + random.uniform(0, 1)
                    print(f"Rate limited. Retrying in {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    return {"discipline": item['discipline'], "error": str(e), "status": "failed"}
        
        return {"discipline": item['discipline'], "error": "Max retries exceeded", "status": "failed"}

async def main():
    print(f"Loading Input Taxonomy: {INPUT_FILE}")
    try:
        with open(INPUT_FILE, "r") as f:
            taxonomy_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Input file {INPUT_FILE} not found.")
        raise e

    work_queue = []
    for field in taxonomy_data:
        field_name = field.get('Field')
        for subfield in field.get('Sub-fields', []):
            subfield_name = subfield.get('Name')
            for discipline in subfield.get('Disciplines', []):
                work_queue.append({
                    "field": field_name,
                    "subfield": subfield_name,
                    "discipline": discipline
                })

    done_disciplines = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get('discipline'):
                        done_disciplines.add(data['discipline'])
                except json.JSONDecodeError:
                    continue
    
    tasks_to_run = [item for item in work_queue if item['discipline'] not in done_disciplines]
    print(f"Total Disciplines: {len(work_queue)}")
    print(f"Already Done: {len(done_disciplines)}")
    print(f"Remaining to Process: {len(tasks_to_run)}")

    if not tasks_to_run:
        print("All tasks completed.")
        return

    semaphore = asyncio.Semaphore(SEMAPHORE_SIZE)
    
    success_count = 0
    fail_count = 0

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        
        tasks = [process_discipline(item, semaphore) for item in tasks_to_run]
        
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await future
            
            if result['status'] == 'success':
                f_out.write(json.dumps(result) + "\n")
                f_out.flush()
                success_count += 1
            else:
                print(f"\nFailed: {result.get('discipline')} - {result.get('error')}")
                fail_count += 1

    print("-" * 30)
    print(f"Summary:")
    print(f"  Total Attempted: {len(tasks)}")
    print(f"  Successful:      {success_count}")
    print(f"  Failed:          {fail_count}")
    print("-" * 30)

if __name__ == "__main__":
    try:
        asyncio.run(main())
        print(f"Done! Results saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"\nCritical error during execution: {e}")
