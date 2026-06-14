import json
import random
import os
from utils.config import CONFIG

RAW_BASE_PATH = CONFIG['paths']['base_path']
ABS_BASE_PATH = os.path.expanduser(RAW_BASE_PATH)

PROMPTS_DIR = os.path.join(ABS_BASE_PATH, CONFIG['paths']['prompts_dir'])
INPUT_FILE = os.path.join(PROMPTS_DIR, CONFIG['files']['taxonomy_phase2'])
OUTPUT_FILE = os.path.join(PROMPTS_DIR, CONFIG['files']['filtered_taxonomy_phase2'])

KEEP_LEVELS = {
    'frontier', 'advanced frontier', 'emerging research', 'borderline frontier', 
    'cutting edge', 'research frontier', 'frontier research', 'frontiers', 
    'emerging', 'emergent', 'thesis',
    'advanced', 'advanced research', 'advanced theory', 'theoretical', 
    'theoretical foundations', 'specialized', 'expert', 'interdisciplinary',
    'experimental', 'algorithmic', 'methodological', 'applied/frontier'
}

FRONTIER_LABELS = {
    'frontier', 'advanced frontier', 'emerging research', 'borderline frontier', 
    'cutting edge', 'research frontier', 'frontier research', 'frontiers', 
    'emerging', 'emergent', 'thesis'
}

MAX_FRONTIER_PER_DISC = 4
MAX_ADVANCED_PER_DISC = 2

def is_high_value(subject):
    """
    Determines if a subject is worth keeping based on Level and Content Density.
    """
    # 1. Filter by Level
    level = subject.get('level', '').lower()
    if level not in KEEP_LEVELS:
        return False
    
    # 2. Filter by Subtopic Density (Must have at least 5 subtopics)
    subtopics = subject.get('key_subtopics', [])
    if isinstance(subtopics, list) and len(subtopics) < 5:
        return False
    
    return True

def filter_subjects(subjects_list):
    """
    Groups subjects by Discipline and selects a balanced mix of Frontier/Advanced.
    """
    grouped = {}
    for s in subjects_list:
        disc = s.get('discipline', 'General')
        if disc not in grouped:
            grouped[disc] = []
        grouped[disc].append(s)
        
    final_list = []
    
    print(f"{'Discipline':<50} | {'Frontier':<10} | {'Advanced':<10} | {'Selected'}")
    print("-" * 90)

    for disc, items in grouped.items():
        # Split into Frontier vs Advanced
        f_list = [x for x in items if x.get('level', '').lower() in FRONTIER_LABELS]
        a_list = [x for x in items if x.get('level', '').lower() not in FRONTIER_LABELS]
        
        # Selection Logic
        # 1. Take Frontiers (up to limit)
        if len(f_list) > MAX_FRONTIER_PER_DISC:
            selected_f = random.sample(f_list, MAX_FRONTIER_PER_DISC)
        else:
            selected_f = f_list
            
        # 2. Take Advanced (up to limit)
        if len(a_list) > MAX_ADVANCED_PER_DISC:
            selected_a = random.sample(a_list, MAX_ADVANCED_PER_DISC)
        else:
            selected_a = a_list
            
        batch = selected_f + selected_a
        final_list.extend(batch)
        
        # Print stats for active disciplines
        if len(batch) > 0:
             print(f"{disc[:50]:<50} | {len(f_list):<10} | {len(a_list):<10} | {len(batch)}")

    return final_list


def main():
    print(f"Reading from: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File {INPUT_FILE} does not exist.")
        return

    # 1. LOAD & FLATTEN (JSONL Handler)
    all_candidates = []
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            
            try:
                # Parse Discipline object
                discipline_obj = json.loads(line)
                
                # Context info
                parent_discipline = discipline_obj.get('discipline', 'Unknown')
                parent_field = discipline_obj.get('field', 'Unknown')
                parent_subfield = discipline_obj.get('subfield', 'Unknown')

                # Extract Subjects
                if 'subjects' in discipline_obj and isinstance(discipline_obj['subjects'], list):
                    for subj in discipline_obj['subjects']:
                        # Validate
                        if is_high_value(subj):
                            # Inject Context (Critical for downstream phases)
                            subj['discipline'] = parent_discipline
                            subj['field'] = parent_field
                            subj['subfield'] = parent_subfield
                            
                            all_candidates.append(subj)
                            
            except json.JSONDecodeError:
                continue

    print(f"\nCandidates Found: {len(all_candidates)}")

    # 2. APPLY SAMPLING LOGIC
    gold_set = filter_subjects(all_candidates)
    
    # Assign global IDs
    for i, s in enumerate(gold_set):
        s['id'] = i + 1  # 1-based index

    # 3. SAVE AS JSONL
    print(f"\nWriting {len(gold_set)} selected subjects to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in gold_set:
            f.write(json.dumps(item) + "\n")

    print("Done.")

if __name__ == "__main__":
    main()