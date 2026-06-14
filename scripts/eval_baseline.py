#!/usr/bin/env python
"""
Evaluation script for DPO with personalization.

Usage:
    python scripts/eval_baseline.py --config configs/default.yaml
"""

import os
import sys
import argparse
from personalization.evaluation.eval_baseline import evaluate
from personalization.personalization_config import PersonalizationConfig
from dotenv import load_dotenv


def parse_args():
    parser = argparse.ArgumentParser(description="Train Baseline personalization model")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file")
    return parser.parse_args()


def main():
    args = parse_args()

    config = PersonalizationConfig.from_yaml(args.config)
    config.print_config()
    load_dotenv(config.evaluation.env_file)

    provider = config.evaluation.judge_provider
    api_key: str | None = None

    if config.evaluation.generation_only:
        print("generation_only=True — skipping API key validation and LLM judge.")
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(f"ERROR: OPENAI_API_KEY not found in {config.evaluation.env_file}")
            sys.exit(1)
    elif provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print(f"ERROR: DEEPSEEK_API_KEY not found in {config.evaluation.env_file}")
            sys.exit(1)
    elif provider == "gemini":
        if not config.evaluation.gemini_project_id:
            print("ERROR: evaluation.gemini_project_id must be set in the config when using provider='gemini'")
            sys.exit(1)
        # Gemini uses Application Default Credentials via the google-genai SDK;
        # no explicit API key is required.
    else:
        print(f"ERROR: Unknown judge_provider '{provider}'. Choose 'openai', 'deepseek', or 'gemini'.")
        sys.exit(1)

    evaluate(config, api_key=api_key)
    
    print("\n" + "=" * 80)
    print("Evaluation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
