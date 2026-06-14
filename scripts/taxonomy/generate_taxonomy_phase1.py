import os
import json
from openai import OpenAI
import dotenv
from utils.config import CONFIG


RAW_BASE_PATH = CONFIG['paths']['base_path']
ABS_BASE_PATH = os.path.expanduser(RAW_BASE_PATH)
PROMPTS_DIR = os.path.join(ABS_BASE_PATH, CONFIG['paths']['prompts_dir'])
OUTPUT_FILE = os.path.join(PROMPTS_DIR, CONFIG['files']['taxonomy_phase1'])
GEN_MODEL = CONFIG['models']['generation_model']
RAW_RESPONSE_FILE = os.path.join(PROMPTS_DIR, f"{GEN_MODEL}_response.txt")


system_prompt = (
    "You are the Chief AI Architect and Lead Researcher at a world-class artificial "
    "intelligence institute. You possess an encyclopedic knowledge of the entire history, "
    "current state, and theoretical underpinnings of Machine Learning, AI and Data Science."
)

user_prompt = (
    "Task: Construct a comprehensive, hierarchical taxonomy of the entire domain of Artificial "
    "Intelligence and Machine Learning. "
    "Goal: Create a structured tree that breaks down the high-level concept of 'AI' into Fields, "
    "Sub-fields, and Disciplines. This taxonomy will be used to generate a curriculum for training "
    "advanced AI agents, so it must be exhaustive, covering both foundational theories and cutting-edge "
    "applications. "
    "Instructions: "
    "Level 1 (Fields): Identify the major pillars of AI (e.g., Computer Vision, Natural Language "
    "Processing, Reinforcement Learning, Symbolic AI, Optimization Theory, AI Ethics & Safety, "
    "Representation Learning, Robotics, Audio/Speech Processing). "
    "Level 2 (Sub-fields): For each Field, break it down into specific branches (e.g., Under NLP: Syntax "
    "Parsing, Semantics, Machine Translation, Large Language Models, Text Generation). "
    "Level 3 (Disciplines): For each Sub-field, list the specific 'Disciplines' or 'Course Titles' that "
    "one would study. These should be granular enough to have a distinct syllabus (e.g., Under Large "
    "Language Models: 'Instruction Tuning Paradigms', 'PEFT Techniques', 'Context Window Management', "
    "'Hallucination Mitigation'). "
    "Constraints for High Quality: "
    "1. Breadth: Do not limit yourself to popular topics. Include niche areas like Neuromorphic Computing, "
    "2. Federated Learning, Causal Inference, and Multi-Agent Systems. "
    "3. Depth: Ensure you cover the mathematical foundations (e.g., Information Theory, Linear Algebra for ML) "
    "as their own fields or sub-fields. "
    "4. Be exhaustive. I expect 20+ Fields and hundreds of Disciplines in total."
    "Output Format: Provide the output ONLY in a structured JSON format where each key is a Field containing a "
    "list of Sub-fields, each containing a list of Disciplines. Do NOT add any addtional tokens."
    "Example Structure (JSON): "
    "{\n"
    '    "Field": "Computer Vision",\n'
    '    "Sub-fields": [\n'
    "        {\n"
    '            "Name": "Generative Vision",\n'
    '            "Disciplines": ["Diffusion Models", "GAN Architectures", "Neural Style Transfer"]\n'
    "        }\n"
    "    ]\n"
    "}"
)



dotenv.load_dotenv()
api_key = os.environ['API_KEY']
    
os.environ["OPENAI_API_KEY"] = api_key
    
client = OpenAI()

response = client.responses.create(
            model=GEN_MODEL, #gpt-5.1
            reasoning={"effort": "high"},
            input=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
            ],
        )

try:
    answer = json.loads(response.output_text)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(answer, f, indent=2)
except Exception as e:
    print("Error while saving to json: ", e)
    print(f"Saving raw response to {RAW_RESPONSE_FILE}")
    with open(RAW_RESPONSE_FILE, "w") as f:
        f.write(response.output_text)

    
