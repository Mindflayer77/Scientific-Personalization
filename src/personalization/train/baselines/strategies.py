import os
from abc import ABC, abstractmethod
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
from transformers import DataCollatorForLanguageModeling, EarlyStoppingCallback, TrainingArguments
from personalization.personalization_config import PersonalizationConfig

class BaseStrategy(ABC):
    @abstractmethod
    def train(self, model, dataset): 
        pass

class SFTStrategy(BaseStrategy):
    def __init__(self, config: PersonalizationConfig):
        self.cfg = config

    def train(self, model, dataset, tokenizer, peft_config=None):
        trainer = SFTTrainer(
            model=model, 
            peft_config=peft_config,
            #processing_class=tokenizer,
            train_dataset=dataset['train'], 
            eval_dataset=dataset['val'],
            args=SFTConfig(
                dataset_text_field="text",
                output_dir=os.path.join(self.cfg.logging.checkpoint_dir, self.cfg.logging.wandb_run_name),
                max_length=self.cfg.training.max_length,
                report_to="wandb",
                run_name=self.cfg.logging.wandb_run_name,
                logging_strategy="steps",
                logging_steps=4,
                per_device_train_batch_size=self.cfg.training.batch_size,
                per_device_eval_batch_size=self.cfg.training.batch_size,
                gradient_accumulation_steps=4,
                optim="paged_adamw_32bit",
                num_train_epochs=self.cfg.training.max_epochs,
                warmup_ratio=self.cfg.training.warmup_ratio,
                learning_rate=self.cfg.training.learning_rate,
                lr_scheduler_type=self.cfg.training.lr_scheduler,
                weight_decay=self.cfg.training.weight_decay,
                gradient_checkpointing=self.cfg.training.gradient_checkpointing,
                max_grad_norm=1.0,
                # label_smoothing=self.cfg.training.label_smoothing,
                #gradient_checkpointing_kwargs={"use_reentrant": False},
                #fp16=False,
                bf16=True,
                save_strategy="steps",
                save_steps=16,
                save_total_limit=self.cfg.logging.save_top_k,
                eval_steps=16,
                eval_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="loss",
                greater_is_better=False,
                seed=self.cfg.training.seed,
                save_safetensors=False,
                #use_liger_kernel=False,
                eos_token=tokenizer.eos_token,
            ),
#            data_collator=DataCollatorForLanguageModeling(
#                tokenizer=tokenizer,
#                mlm=False
#            ),
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=self.cfg.logging.early_stopping_patience, 
            )]
        )
        trainer.train()

        output_dir = os.path.join(self.cfg.logging.checkpoint_dir, self.cfg.logging.wandb_run_name)
        final_adapter_path = os.path.join(output_dir, "final_adapter")
        trainer.save_model(final_adapter_path)
        print(f"[SFT] Final adapter saved to: {final_adapter_path}")

class DPOStrategy(BaseStrategy):
    def __init__(self, config: PersonalizationConfig):
        self.cfg = config

    def train(self, model, dataset, tokenizer, peft_config=None):
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"Trainable params: {len(trainable)}")
        print(trainable[:5])
        print("Active adapter:", model.active_adapters)

        trainer = DPOTrainer(
            model=model,
            processing_class=tokenizer,
            #peft_config=peft_config,
            #ref_model=None,
            args=DPOConfig(
                #ddp_find_unused_parameters=False,
                model_adapter_name="train_lora",
                ref_adapter_name="reference_lora",
                # dataset_text_field="text",
                output_dir=os.path.join(self.cfg.logging.checkpoint_dir, self.cfg.logging.wandb_run_name),
                max_prompt_length=10000,
                max_length=self.cfg.training.max_length,
                report_to="none",
                # run_name=self.cfg.logging.wandb_run_name,
                logging_strategy="steps",
                logging_steps=2,
                per_device_train_batch_size=self.cfg.training.batch_size,
                per_device_eval_batch_size=self.cfg.training.batch_size,
                gradient_accumulation_steps=4,
                optim="paged_adamw_32bit",
                beta=self.cfg.training.beta,
                num_train_epochs=self.cfg.training.max_epochs,
                warmup_ratio=self.cfg.training.warmup_ratio,
                learning_rate=self.cfg.training.learning_rate,
                lr_scheduler_type=self.cfg.training.lr_scheduler,
                weight_decay=self.cfg.training.weight_decay,
                gradient_checkpointing=self.cfg.training.gradient_checkpointing,
                precompute_ref_log_probs=self.cfg.training.baseline['precompute_ref_log_probs'],
                max_grad_norm=1.0,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                #fp16=False,
                bf16=True,
                save_strategy="steps",
                save_steps=8,
                save_total_limit=self.cfg.logging.save_top_k,
                eval_strategy="steps",
                eval_steps=8,
                load_best_model_at_end=True,
                metric_for_best_model="rewards/margins",
                greater_is_better=True,
                seed=self.cfg.training.seed,
                save_safetensors=False,
                #use_liger_kernel=False,
                #eos_token=tokenizer.eos_token,
            ),
            train_dataset=dataset['train'],
            eval_dataset=dataset['val'],
        )
        trainer.train()

        output_dir = os.path.join(self.cfg.logging.checkpoint_dir, self.cfg.logging.wandb_run_name)
        final_adapter_path = os.path.join(output_dir, "final_adapter")
        trainer.save_model(final_adapter_path)
        print(f"[DPO] Final adapter saved to: {final_adapter_path}")