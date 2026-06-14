"""
Personalization configuration class for managing training and evaluation parameters.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class DataBaselineConfig:
    """Baseline data configuration."""
    sft_train: str = "data/sft.csv"
    sft_val: str = "data/val_sft.csv"
    sft_test: str = "data/test.csv"
    dpo_train: str = "data/dpo.csv"
    dpo_val: str = "data/val_dpo.csv"
    dpo_test: str = "data/test.csv"
    personas_path: str = "personas/personas_all.json"
    prompt_templates_dir: str = "prompt_templates/data_generation"
    # Rationale-format training/eval (use_rationale=True enables new data format)
    use_rationale: bool = False
    rationale_dpo_train: str = "data/rationale/dpo_train.csv"
    rationale_dpo_val: str = "data/rationale/val_dpo.csv"
    rationale_dpo_test: str = "data/rationale/test.csv"
    rationale_sft_train: str = "data/rationale/sft_train.csv"
    rationale_sft_val: str = "data/rationale/val_sft.csv"
    rationale_sft_test: str = "data/rationale/test.csv"

    # Backward-compatible dict-style access (cfg.data.baseline["key"])
    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

@dataclass
class DataConfig:
    """Data-related configuration."""
    baseline: DataBaselineConfig = field(default_factory=DataBaselineConfig)

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    cache_dir: str = "model_cache"
    use_unsloth: bool = True

@dataclass
class TrainingBaselineConfig:
    """Baseline training configuration"""
    strategy: str = "sft"
    adapter_path: Optional[str] = None
    cl_adapter_path: Optional[str] = None
    precompute_ref_log_probs: bool = True

    # Backward-compatible dict-style access (cfg.training.baseline["key"])
    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)
    
@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    learning_rate: float = 1e-5
    lr_scheduler: str = "cosine"
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.03
    max_length: int = 1024
    gradient_checkpointing: bool = False
    beta: float = 0.1
    batch_size: int = 2
    max_epochs: int = 1
    seed: int = 42
    baseline: TrainingBaselineConfig = field(default_factory=TrainingBaselineConfig)

@dataclass
class LoggingConfig:
    """Logging and checkpointing configuration."""
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    save_top_k: int = 3
    early_stopping_patience: Optional[int] = None
    wandb_project: str = "personalization-dpo"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[list] = None


@dataclass
class EvaluationConfig:
    """Evaluation-specific configuration."""
    personalized_checkpoint: Optional[str] = None
    max_new_tokens: int = 8192
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    device: str = "cuda"
    train_only: bool = False
    # Win ratio evaluation specific
    # Provider selection: "openai" | "deepseek" | "gemini"
    judge_provider: str = "openai"
    # Model name for the selected provider.
    # Defaults applied per provider when None: openai→gpt-4o, deepseek→deepseek-chat, gemini→gemini-2.5-pro
    judge_model: Optional[str] = None
    # Kept for backwards compatibility; used as judge_model when judge_provider="openai"
    env_file: str = ".env"
    output_dir: str = "eval_results"
    eval: bool = False
    n_few_shots: int = 1

    # Batch size used during generation. Overrides training.batch_size so
    # eval can use large batches (e.g. 16) while training stays at 1.
    eval_batch_size: Optional[int] = None
    # When True, only generate responses and save them to CSV; skip LLM judging entirely.
    generation_only: bool = False
    # When True, the judge receives "<think>reasoning</think>\nanswer" instead of answer only.
    judge_use_reasoning: bool = False
    # Gemini / Vertex AI settings (only required when judge_provider="gemini")
    gemini_project_id: Optional[str] = None
    gemini_location: str = "global"

@dataclass
class PersonalizationConfig:
    """
    Main configuration class combining all sub-configurations.
    
    This class can be instantiated from a YAML file or from individual parameters.
    """
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PersonalizationConfig':
        """
        Load configuration from a YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            PersonalizationConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Extract sub-dicts as copies so we can pop 'baseline' without mutating
        data_dict = dict(config_dict.get('data', {}))
        training_dict = dict(config_dict.get('training', {}))

        data_baseline = DataBaselineConfig(**data_dict.pop('baseline', {}))
        training_baseline = TrainingBaselineConfig(**training_dict.pop('baseline', {}))

        return cls(
            data=DataConfig(baseline=data_baseline, **data_dict),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(baseline=training_baseline, **training_dict),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {}))
        )
    
    def to_yaml(self, yaml_path: str) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration file
        """
        config_dict = {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'logging': asdict(self.logging),
            'evaluation': asdict(self.evaluation)
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'logging': asdict(self.logging),
            'evaluation': asdict(self.evaluation)
        }
    
    def print_config(self) -> None:
        """Print configuration in a readable format."""
        print("=" * 80)
        print("Personalization Configuration")
        print("=" * 80)
        
        sections = [
            ("Data Configuration", self.data),
            ("Model Configuration", self.model),
            ("Training Configuration", self.training),
            ("Logging Configuration", self.logging),
            ("Evaluation Configuration", self.evaluation)
        ]
        
        for section_name, section_config in sections:
            print(f"\n{section_name}:")
            for key, value in asdict(section_config).items():
                print(f"  {key}: {value}")
        
        print("=" * 80)
