#!/usr/bin/env python
"""
Training script for DPO with personalization.

Usage:
    python scripts/train_baseline.py --config configs/default.yaml
"""

#import unsloth
import argparse

from personalization.train.train_baselines import train
from personalization.personalization_config import PersonalizationConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train Baseline personalization model")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    config = PersonalizationConfig.from_yaml(args.config)
    config.print_config()
    
    # Train
    train(config)
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
