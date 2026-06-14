import torch
from typing import Dict, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape

class DataProcessor:
    """Transforms raw data into model-ready formats."""
    
    def __init__(self, tokenizer, persona_map: dict, templates_dir: str):
        self.tokenizer = tokenizer
        self.persona_map = persona_map
        env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape([]),
            keep_trailing_newline=True,
        )
        self.persona_alingment_template = env.get_template("persona_template_2.j2")
        self.system_prompt_template = env.get_template("hypotheses_2/system.j2")
        self.system_inference_template = env.get_template("hypotheses_2/system.j2")
        self.system_few_shot_template = env.get_template("hypotheses_2/system_few_shot.j2")
        self.user_prompt_template = env.get_template("hypotheses_2/user.j2")
        # Rationale-aware templates (hypotheses_rationale/)
        self.system_rationale_template = env.get_template("hypotheses_rationale/system.j2")
        self.system_rationale_few_shot_template = env.get_template("hypotheses_rationale/system_few_shot.j2")

    def _build_persona_instr(self, user_id: str) -> str:
        """Render persona details using persona_template.j2."""
        p = self.persona_map.get(user_id, {})
        return self.persona_alingment_template.render(
            display_name=p.get("display_name", "Researcher"),
            core_philosophy=p.get("core_philosophy", "Scientific Rigor"),
            areas_of_expertise=p.get("areas_of_expertise", ["General Science"]),
            communication_style=p.get("communication_style", "Formal"),
            what_i_look_for=p.get("what_i_look_for", []),
            what_i_reject=p.get("what_i_reject", []),
            hypothesis_signature=p.get("hypothesis_signature", []),
            vocabulary_markers=p.get("vocabulary_markers", []),
        )

    def _format_single_sft_messages(self, user_id, question, context, reasoning, answer):
        """Creates a list of dictionaries for one conversation."""
        persona_str = self._build_persona_instr(user_id)

        return [
            {
                "role": "system", 
                "content": self.system_prompt_template.render(persona=persona_str)
            },
            {
                "role": "user", 
                "content": self.user_prompt_template.render(query=question, context=context)
            },
            {
                "role": "assistant", 
                # "content": f"<think>{reasoning}</think>{answer}" - change this for finetuning reasoning model
                "content": f"{answer}"
            }
        ]

    def batch_format_sft(self, examples):
        """Processes a batch and returns a list of message lists."""
        all_conversations = []
        
        for uid, q, c, r, a in zip(
            examples["user_id"], 
            examples["question"], 
            examples["context"], 
            examples["reasoning"], 
            examples["answer"]
        ):
            messages = self._format_single_sft_messages(uid, q, c, r, a)
            all_conversations.append(messages)
        
        return {"messages": all_conversations}
    
    def _format_single_dpo_messages(self, user_id, question, context, chosen_ans, rejected_ans):
        """Helper for a single DPO pair using conversational format."""
        persona_str = self._build_persona_instr(user_id)
        
        prompt_msgs = [
            {
                "role": "system", 
                "content": self.system_prompt_template.render(persona=persona_str)
            },
            {
                "role": "user", 
                "content": self.user_prompt_template.render(query=question, context=context)
            },
        ]

        return {
            "prompt": prompt_msgs,
            "chosen": [{"role": "assistant", "content": chosen_ans}],
            "rejected": [{"role": "assistant", "content": rejected_ans}]
        }

    def batch_format_dpo(self, examples):
        """Batched DPO processor for dataset.map(batched=True)."""
        res_prompts = []
        res_chosens = []
        res_rejecteds = []
        
        for uid, question, context, reasoning_chosen, chosen, reasoning_rejected, rejected in zip(
            examples["user_id"], 
            examples["question"], 
            examples["context"], 
            examples.get("reasoning_chosen", None),
            examples["chosen"], 
            examples.get("reasoning_rejected", None), 
            examples["rejected"]
        ):
            # chosen = f"<think>{reasoning_chosen}</think> {chosen}" if reasoning_chosen else chosen
            # rejected = f"<think>{reasoning_rejected}</think> {rejected}" if reasoning_rejected else rejected
            
            chosen = f"{chosen}"
            rejected = f"{rejected}"
            
            dpo_data = self._format_single_dpo_messages(uid, question, context, chosen, rejected)
            
            res_prompts.append(dpo_data["prompt"])
            res_chosens.append(dpo_data["chosen"])
            res_rejecteds.append(dpo_data["rejected"])
            
        return {
            "prompt": res_prompts,
            "chosen": res_chosens,
            "rejected": res_rejecteds
        }
    
    def _format_single_inference_messages(self, user_id, question, context):
        """Creates the prompt structure for generation (no assistant role)."""
        persona_str = self._build_persona_instr(user_id)
        
        return [
            {
                "role": "system", 
                "content": self.system_inference_template.render(persona=persona_str)
            },
            {
                "role": "user", 
                "content": self.user_prompt_template.render(query=question, context=context)
            }
        ]
    def _format_single_inference_few_shot_messages(self, user_id, question, context, few_shot_examples):
        """Creates the prompt structure for few-shot generation (no assistant role)."""
        persona_str = self._build_persona_instr(user_id)
        return [
            {
                "role": "system",
                "content": self.system_few_shot_template.render(
                    persona=persona_str,
                    few_shot_examples=few_shot_examples,
                ),
            },
            {
                "role": "user",
                "content": self.user_prompt_template.render(query=question, context=context),
            },
        ]

    def batch_format_inference_few_shot(self, examples, few_shot_examples: list):
        """Batched processor for few-shot inference."""
        all_messages = []
        all_formatted_strings = []
        all_user_prompts = []
        for uid, q, c in zip(examples["user_id"], examples["question"], examples["context"]):
            messages = self._format_single_inference_few_shot_messages(uid, q, c, few_shot_examples)
            tokenized_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            all_messages.append(messages)
            all_formatted_strings.append(tokenized_prompt)
            all_user_prompts.append(self.user_prompt_template.render(query=q, context=c))
        return {
            "messages": all_messages,
            "formatted_prompt": all_formatted_strings,
            "user_prompt": all_user_prompts,
        }

    def batch_format_inference(self, examples):
        """
        Batched processor for inference. 
        Returns both the message list and the pre-formatted string.
        """
        all_messages = []
        all_formatted_strings = []
        all_user_prompts = []
        for uid, q, c in zip(examples["user_id"], examples["question"], examples["context"]):
            messages = self._format_single_inference_messages(uid, q, c)
            
            # apply_chat_template with add_generation_prompt=True is CRITICAL.
            # It adds the "<|im_start|>assistant\n" header so the model knows to start.
            tokenized_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                # enable_thinking=True,
            )
            
            all_messages.append(messages)
            all_formatted_strings.append(tokenized_prompt)
            all_user_prompts.append(self.user_prompt_template.render(query=q, context=c))
            
        return {
            "messages": all_messages, 
            "formatted_prompt": all_formatted_strings,
            "user_prompt": all_user_prompts,
        }

    # ------------------------------------------------------------------
    # Rationale-aware methods (use_rationale=True in config)
    # ------------------------------------------------------------------

    def _format_single_sft_messages_rationale(
        self, user_id, question, context, answer,
        rationale_question, rationale_chosen, rationale_rejected, rationale_train,
    ):
        """SFT message list with rationale injected into the system prompt."""
        persona_str = self._build_persona_instr(user_id)
        return [
            {
                "role": "system",
                "content": self.system_rationale_template.render(
                    persona=persona_str,
                    rationale_question=rationale_question,
                    rationale_chosen=rationale_chosen,
                    rationale_rejected=rationale_rejected,
                    rationale_train=rationale_train,
                ),
            },
            {
                "role": "user",
                "content": self.user_prompt_template.render(query=question, context=context),
            },
            {
                "role": "assistant",
                "content": f"{answer}",
            },
        ]

    def batch_format_sft_rationale(self, examples):
        """Batched SFT processor for rationale data (chosen/rationale_* columns)."""
        all_conversations = []
        n = len(examples["user_id"])
        for uid, q, c, answer, rq, rch, rrj, rt in zip(
            examples["user_id"],
            examples["question"],
            examples["context"],
            examples["chosen"],
            examples["rationale_question"],
            examples["rationale_chosen"],
            examples["rationale_rejected"],
            examples["rationale_train"],
        ):
            messages = self._format_single_sft_messages_rationale(
                uid, q, c, answer, rq, rch, rrj, rt
            )
            all_conversations.append(messages)
        return {"messages": all_conversations}

    def _format_single_dpo_messages_rationale(
        self, user_id, question, context, chosen_ans, rejected_ans,
        rationale_question, rationale_chosen, rationale_rejected, rationale_train,
    ):
        """DPO prompt/chosen/rejected with rationale injected into the system prompt."""
        persona_str = self._build_persona_instr(user_id)
        prompt_msgs = [
            {
                "role": "system",
                "content": self.system_rationale_template.render(
                    persona=persona_str,
                    rationale_question=rationale_question,
                    rationale_chosen=rationale_chosen,
                    rationale_rejected=rationale_rejected,
                    rationale_train=rationale_train,
                ),
            },
            {
                "role": "user",
                "content": self.user_prompt_template.render(query=question, context=context),
            },
        ]
        return {
            "prompt": prompt_msgs,
            "chosen": [{"role": "assistant", "content": chosen_ans}],
            "rejected": [{"role": "assistant", "content": rejected_ans}],
        }

    def batch_format_dpo_rationale(self, examples):
        """Batched DPO processor for rationale data."""
        res_prompts, res_chosens, res_rejecteds = [], [], []
        for uid, q, c, chosen, rejected, rq, rch, rrj, rt, reasoning_chosen, reasoning_rejected in zip(
            examples["user_id"],
            examples["question"],
            examples["context"],
            examples["chosen"],
            examples["rejected"],
            examples["rationale_question"],
            examples["rationale_chosen"],
            examples["rationale_rejected"],
            examples["rationale_train"],
            examples.get("reasoning_chosen", None),
            examples.get("reasoning_rejected", None), 
        ):
            chosen = f"<think>{reasoning_chosen}</think> {chosen}" if reasoning_chosen else chosen
            rejected = f"<think>{reasoning_rejected}</think> {rejected}" if reasoning_rejected else rejected
            
            dpo_data = self._format_single_dpo_messages_rationale(
                uid, q, c, chosen, rejected, rq, rch, rrj, rt
            )
            res_prompts.append(dpo_data["prompt"])
            res_chosens.append(dpo_data["chosen"])
            res_rejecteds.append(dpo_data["rejected"])
        return {"prompt": res_prompts, "chosen": res_chosens, "rejected": res_rejecteds}

    def _format_single_inference_messages_rationale(
        self, user_id, question, context,
        rationale_question, rationale_chosen, rationale_rejected, rationale_train,
    ):
        """Inference message list with rationale in system prompt (no few-shot)."""
        persona_str = self._build_persona_instr(user_id)
        return [
            {
                "role": "system",
                "content": self.system_rationale_template.render(
                    persona=persona_str,
                    rationale_question=rationale_question,
                    rationale_chosen=rationale_chosen,
                    rationale_rejected=rationale_rejected,
                    rationale_train=rationale_train,
                ),
            },
            {
                "role": "user",
                "content": self.user_prompt_template.render(query=question, context=context),
            },
        ]

    def _format_single_inference_few_shot_messages_rationale(
        self, user_id, question, context,
        rationale_question, rationale_chosen, rationale_rejected, rationale_train,
        few_shot_examples,
    ):
        """Inference message list with rationale + few-shot examples in system prompt."""
        persona_str = self._build_persona_instr(user_id)
        return [
            {
                "role": "system",
                "content": self.system_rationale_few_shot_template.render(
                    persona=persona_str,
                    rationale_question=rationale_question,
                    rationale_chosen=rationale_chosen,
                    rationale_rejected=rationale_rejected,
                    rationale_train=rationale_train,
                    few_shot_examples=few_shot_examples,
                ),
            },
            {
                "role": "user",
                "content": self.user_prompt_template.render(query=question, context=context),
            },
        ]
