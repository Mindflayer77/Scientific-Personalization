# Taxonomy generation scripts
## Running the scripts
In order to run scripts on WCSS cluster, you need to execute this command from the repository root:
```bash
sbatch server_scripts/<script_name>
```

You can monitor the script execution by executing:
```bash
tail -f out/<job_name>
```

In order to stop the job early, you first have to check the `JOBID`:
```bash
squeue -u ${USER}
```

And then cancel the job:
```bash
scancel <JOBID>
```

For now, only the `generate_taxonomy_phase2.py` and `generate_taxonomy_phase3.py` scripts are adjusted to run on wcss.

## Script descriptions
### `generate_taxonomy_phase1.py`

This script initiates the curriculum generation process by constructing a high-level, hierarchical taxonomy of the entire Artificial Intelligence and Machine Learning domain. It uses an LLM (OpenAI API) to generate a structured tree of Fields (e.g., "Computer Vision"), Sub-fields (e.g., "Generative Vision"), and specific Disciplines (e.g., "Diffusion Models"). This output serves as the structural backbone for the entire dataset.

### `generate_taxonomy_phase2.py`

Taking the disciplines from Phase 1 as input, this script expands each discipline into a list of specific, PhD-level academic subjects (courses). For every discipline, it generates approximately 8 distinct subjects, enriching them with metadata such as:

    Level: (e.g., Foundational, Applied, Frontier).

    Description: A technical summary of the course.

    Key Subtopics: A list of specific concepts covered in the course.

### `filter_taxonomy_phase2.py`

A curation utility that processes the raw output from Phase 2. It applies heuristic filters to select only the highest-value subjects for hypothesis generation. Key filtering logic includes:

    Level Filtering: Prioritizes "Frontier", "Advanced", and "Emerging" topics while limiting standard "Foundational" subjects.

    Density Check: Ensures subjects have a sufficient number of subtopics.

    Balancing: Randomly samples subjects to ensure a diverse distribution across disciplines (e.g., capping the number of "Frontier" subjects per discipline).

Output: `CONFIG['files']['filtered_taxonomy_phase_2]`.jsonl  
(This file is the direct input for `generate_prompts.py`).

### `generate_taxonomy_phase3.py`

The final stage of the deep-dive curriculum design. It takes the academic subjects (from Phase 2) and generates a rigorous, session-by-session syllabus for each. It breaks down the subject into 8–12 class sessions, and further decomposes each session into "Atomic Concepts"—the fundamental theorems, equations, and mechanisms that drive the field. This granular data provides the deep technical context required for high-quality scientific generation.

## Parameters

Below you can read the description of all configurable parameters. In order to use the default config, you need to create a config.yaml file by copying the example config:

```bash
cp config/config.yaml.example config/config.yaml
```

To change the default parameters, you can edit the config.yaml file. You don't have to change the example file as it will not be utilized by the scripts.

### 1. Taxonomy Files & Filtering

These parameters control the input/output paths for the taxonomy pipeline and the filtering logic used in `filter_taxonomy_phase2.py`.
File Paths

- **taxonomy_phase1:** `taxonomy_phase1_gpt-5.1.json` – Output of the high-level hierarchy generation.
- **taxonomy_phase2:** `taxonomy_phase2_Qwen3-32B.jsonl` – Output of the subject expansion phase.
- **filtered_taxonomy_phase2:** `filtered_taxonomy_phase2_Qwen3-32B.jsonl` – The curated list of subjects used for prompt generation.
- **taxonomy_phase3:** `taxonomy_phase3_Qwen3-32B.jsonl` – Output of the syllabus generation phase.

### 2. Filtering Logic (Internal)

Note: These parameters are defined as constants within `filter_taxonomy_phase2.py` and are not currently exposed in `config.yaml`.
- **KEEP_LEVELS:** A set of keywords (e.g., `frontier, advanced, emerging`) used to identify high-value subjects.
- **MAX_FRONTIER_PER_DISC:** `4` – Maximum number of "Frontier" level subjects to retain per discipline to prevent data skew.
- **MAX_ADVANCED_PER_DISC:** `2` – Maximum number of "Advanced" level subjects to retain per discipline.
- **Density Check:** Subjects with fewer than `5` key_subtopics are automatically discarded.