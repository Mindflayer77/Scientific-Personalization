import os
import asyncio
import json
import re
from typing import Dict
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json_repair
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
INPUT_FILE = os.path.join(PROMPTS_DIR, CONFIG['files']['taxonomy_phase2'])
OUTPUT_FILE = os.path.join(PROMPTS_DIR, CONFIG['files']['taxonomy_phase3'])

system_prompt_text = (
    "You are a Distinguished Research Professor and Curriculum Architect at a world-class institute (e.g., MIT, ETH Zurich). "
    "You are designing a rigorous, PhD-level syllabus for a specialized course in '{DISCIPLINE}'. "
    "Your goal is to break down complex research subjects into their constituent 'Atomic Concepts'—the fundamental "
    "theorems, equations, algorithms, and physical mechanisms that drive the field. "
    "CRITICAL: You output ONLY raw valid JSON. Do not use Markdown code blocks (e.g., ```json). "
    "Do not include any conversational text."
)

user_prompt_text = (
    "Context:\n"
    "Subject: '{SUBJECT_NAME}'\n"
    "Level: {LEVEL}\n"
    "Description: {DESCRIPTION}\n"
    "Key Subtopics to Cover: {SUBTOPICS}\n\n"
    
    "Task:\n"
    "Create a detailed, strictly technical syllabus containing 8 to 12 'Class Sessions'.\n"
    "For each Class Session, list 6 to 10 'Atomic Concepts'.\n\n"
    
    "Guidelines for High-Quality Hypothesis Generation:\n"
    "1. NO INTRODUCTIONS: Do not create sessions like 'Introduction to X' or 'History of Y'. Start immediately with technical content.\n"
    "2. ATOMIC CONCEPTS: A concept is 'atomic' if it can be a variable in a mathematical equation or a specific component of an algorithm. "
    "   - BAD: 'Optimization methods', 'Deep learning basics'.\n"
    "   - GOOD: 'Nesterov Accelerated Gradient', 'Hessian Eigenvalue Spectrum', 'Lipchitz Continuity Condition'.\n"
    "3. COHERENCE: Ensure the flow of sessions matches the 'Key Subtopics' provided, but expands on them significantly.\n\n"
    
    "Output Format (Strict JSON):\n"
    "{{\n"
    '  "subject": "{SUBJECT_NAME}",\n'
    '  "syllabus": [\n'
    '    {{\n'
    '      "session_title": "Session 1: [Technical Title]",\n'
    '      "atomic_concepts": [\n'
    '        "Specific Theorem A",\n'
    '        "Algorithm B",\n'
    '        "Mathematical Condition C"\n'
    '      ]\n'
    '    }}\n'
    '  ]\n'
    "}}\n\n"
    
    "Generate the JSON for '{SUBJECT_NAME}' now."
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt_text),
    ("user", user_prompt_text)
])


def normalize_atomic_concepts(concepts_value):
    if isinstance(concepts_value, list):
        return [str(c).strip() for c in concepts_value if isinstance(c, (str, int, float)) and str(c).strip()]
    
    if isinstance(concepts_value, str):
        bullets = re.findall(r'(?:^|\n)\s*(?:[-*•]|\d+\.)\s+([^\n]+)', concepts_value)
        if bullets:
            return [b.strip() for b in bullets]
        if '\n\n' in concepts_value:
             return [line.strip() for line in concepts_value.split('\n\n') if line.strip()]
        lines = [line.strip() for line in concepts_value.split('\n') if line.strip()]
        if len(lines) > 1:
            return lines
    return []

def find_syllabus_list(data):
    if isinstance(data, dict):
        if "syllabus" in data:
            return find_syllabus_list(data["syllabus"])
        if "session_title" in data and "atomic_concepts" in data:
            return [data]
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                result = find_syllabus_list(value)
                if result: return result
    elif isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict):
            if "session_title" in data[0] or "atomic_concepts" in data[0]:
                return data
            if "syllabus" in data[0]:
                return find_syllabus_list(data[0]["syllabus"])
    return []


llm_generator = ChatOpenAI(
    model=GEN_MODEL, 
    openai_api_base=VLLM_API_URL, 
    openai_api_key=VLLM_API_KEY,
    max_tokens=MAX_TOKENS, 
    temperature=TEMPERATURE,
    top_p=TOP_P,
    extra_body={
        "top_k": TOP_K,
        "min_p": MIN_P,
    }
)

async def process_subject(item: Dict, semaphore: asyncio.Semaphore) -> Dict:
    subject_name = item.get('name')
    discipline = item.get('discipline', 'Advanced AI')
    level = item.get('level', 'Advanced')
    description = item.get('description', '')
    
    raw_subtopics = item.get('key_subtopics', [])
    subtopics_str = ", ".join(raw_subtopics) if isinstance(raw_subtopics, list) else str(raw_subtopics)

    inputs = {
        "DISCIPLINE": discipline,
        "SUBJECT_NAME": subject_name,
        "LEVEL": level,
        "DESCRIPTION": description,
        "SUBTOPICS": subtopics_str
    }

    async with semaphore:
        try:
            response = await (prompt_template | llm_generator).ainvoke(inputs)
            raw_text = response.content
            
            parsed_data = json_repair.loads(raw_text)
            
            syllabus_sessions = find_syllabus_list(parsed_data)
            
            clean_sessions = []
            for session in syllabus_sessions:
                if not isinstance(session, dict): continue
                
                raw_concepts = session.get("atomic_concepts", [])
                clean_concepts = normalize_atomic_concepts(raw_concepts)
                
                if not clean_concepts: continue
                
                clean_sessions.append({
                    "session_title": session.get("session_title", "Untitled Session"),
                    "atomic_concepts": clean_concepts
                })
            
            if not clean_sessions:
                return {
                    "subject_name": subject_name,
                    "error": "Parsed JSON but found no valid sessions.",
                    "status": "failed",
                    "raw_output": raw_text[:200]
                }

            return {
                "subject_name": subject_name,
                "original_level": level,
                "parent_discipline": discipline,
                "syllabus": clean_sessions,
                "status": "success"
            }

        except Exception as e:
            return {
                "subject_name": subject_name,
                "error": str(e),
                "status": "failed"
            }

async def main():
    print(f"Loading Input Phase 2 Data: {INPUT_FILE}")
    
    work_queue = []
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            
            if first_char == '[':
                # Standard JSON Array
                data = json.load(f)
                for discipline_obj in data:
                    parent_discipline = discipline_obj.get('discipline')
                    for sub in discipline_obj.get('subjects', []):
                        sub['discipline'] = parent_discipline 
                        work_queue.append(sub)
            else:
                for line in f:
                    if not line.strip(): continue
                    discipline_obj = json.loads(line)
                    parent_discipline = discipline_obj.get('discipline')
                    for sub in discipline_obj.get('subjects', []):
                        sub['discipline'] = parent_discipline
                        work_queue.append(sub)
                        
    except FileNotFoundError as e:
        print(f"Error: Input file {INPUT_FILE} not found.")
        raise e

    done_subjects = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get('subject_name'):
                        done_subjects.add(data['subject_name'])
                except: continue

    tasks_to_run = [item for item in work_queue if item.get('name') not in done_subjects]
    
    print(f"Total Subjects Found: {len(work_queue)}")
    print(f"Already Completed:    {len(done_subjects)}")
    print(f"Remaining to Process: {len(tasks_to_run)}")

    if not tasks_to_run:
        print("All tasks completed.")
        return

    semaphore = asyncio.Semaphore(SEMAPHORE_SIZE)
    success_count = 0
    fail_count = 0

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        tasks = [process_subject(item, semaphore) for item in tasks_to_run]
        
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await future
            
            if result['status'] == 'success':
                f_out.write(json.dumps(result) + "\n")
                f_out.flush()
                success_count += 1
            else:
                print(f"\nFailed: {result.get('subject_name')} - {result.get('error')}")
                fail_count += 1

    print("-" * 30)
    print(f"Summary:")
    print(f"  Successful: {success_count}")
    print(f"  Failed:     {fail_count}")
    print("-" * 30)

if __name__ == "__main__":
    try:
        asyncio.run(main())
        print(f"Process completed. Results saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"\nCritical error during execution: {e}")
