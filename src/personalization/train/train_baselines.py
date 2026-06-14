import os
import wandb
from personalization.models.baselines.factory import BaselineFactory
from personalization.train.baselines.strategies import SFTStrategy, DPOStrategy
from personalization.dataset.baselines.data_loader import DatasetLoader
from personalization.dataset.baselines.data_processor import DataProcessor
from personalization.personalization_config import PersonalizationConfig
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

login(token=os.environ.get("HF_TOKEN"))


def train(cfg: PersonalizationConfig):
    print(cfg.training.baseline["strategy"])
    adapter_to_load = cfg.training.baseline["adapter_path"] if cfg.training.baseline["strategy"] in ("sft+dpo", "cl") else None
    use_rationale = cfg.data.baseline.get("use_rationale", False)

    model, tokenizer, peft_config = BaselineFactory.get_model_and_tokenizer(cfg, adapter_to_load)

    if cfg.training.baseline["strategy"] == "sft":
        sft_train = cfg.data.baseline.get("rationale_sft_train", cfg.data.baseline["sft_train"]) if use_rationale else cfg.data.baseline["sft_train"]
        sft_val   = cfg.data.baseline.get("rationale_sft_val",   cfg.data.baseline["sft_val"])   if use_rationale else cfg.data.baseline["sft_val"]
        sft_test  = cfg.data.baseline.get("rationale_sft_test",  cfg.data.baseline["sft_test"])  if use_rationale else cfg.data.baseline["sft_test"]
        loader = DatasetLoader(sft_train, sft_val, sft_test, cfg.data.baseline["personas_path"])
    elif cfg.training.baseline["strategy"] in ("dpo", "sft+dpo", "cl"):
        dpo_train = cfg.data.baseline.get("rationale_dpo_train", cfg.data.baseline["dpo_train"]) if use_rationale else cfg.data.baseline["dpo_train"]
        dpo_val   = cfg.data.baseline.get("rationale_dpo_val",   cfg.data.baseline["dpo_val"])   if use_rationale else cfg.data.baseline["dpo_val"]
        dpo_test  = cfg.data.baseline.get("rationale_dpo_test",  cfg.data.baseline["dpo_test"])  if use_rationale else cfg.data.baseline["dpo_test"]
        loader = DatasetLoader(dpo_train, dpo_val, dpo_test, cfg.data.baseline["personas_path"])

    personas = loader.load_persona_map()
    raw_ds = loader.load_train_dataset()

    processor = DataProcessor(
        tokenizer,
        personas,
        templates_dir=cfg.data.baseline["prompt_templates_dir"]
    )

    if cfg.training.baseline["strategy"] == "sft":
        processed_ds = raw_ds.map(
            processor.batch_format_sft_rationale if use_rationale else processor.batch_format_sft,
            batched=True,
        )
        processed_ds = processed_ds.map(
            lambda batch: {
                "text": tokenizer.apply_chat_template(
                    batch["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            },
            batched=True
        )

        sample = processed_ds["train"][0]

        print("Keys:", list(sample.keys()))
        print("\nTEXT:", sample["text"])

        # processed_ds["train"] = processed_ds["train"].select(
        #     range(min(25, len(processed_ds["train"])))
        # )
        # processed_ds["val"] = processed_ds["val"].select(
        #     range(min(25, len(processed_ds["val"])))
        # )
        strategy = SFTStrategy(cfg)
    else:
        processed_ds = raw_ds.map(
            processor.batch_format_dpo_rationale if use_rationale else processor.batch_format_dpo,
            batched=True,
        )
        # processed_ds["train"] = processed_ds["train"].select(
        #     range(min(200, len(processed_ds["train"])))
        # )
        # processed_ds["val"] = processed_ds["val"].select(
        #     range(min(200, len(processed_ds["val"])))
        # )
        sample = processed_ds["train"][0]

        print("Keys:", list(sample.keys()))
        print("\nPROMPT:", sample["prompt"])
        print("\nCHOSEN:", sample["chosen"])
        print("\nREJECTED:", sample["rejected"])

        sample = processed_ds["val"][0]

        print("Keys:", list(sample.keys()))
        print("\nPROMPT:", sample["prompt"])
        print("\nCHOSEN:", sample["chosen"])
        print("\nREJECTED:", sample["rejected"])

        strategy = DPOStrategy(cfg)
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if local_rank <= 0:
        wandb.init(
            entity=cfg.logging.wandb_entity,
            project=cfg.logging.wandb_project,
            name=cfg.logging.wandb_run_name,
            tags=cfg.logging.wandb_tags,
            config=cfg,                    
            reinit=True                  
    )

    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_PROJECT"]=cfg.logging.wandb_project
    os.environ["WANDB_WATCH"]="false"
    
    strategy.train(model, processed_ds, tokenizer, peft_config)

    # wandb.finish()
