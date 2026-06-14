"""
Generation utilities for model evaluation.

This module provides functions for generating responses from personalized
and baseline models during evaluation.
"""

import torch
from typing import List, Dict, Tuple
from transformers import PreTrainedTokenizer

# Qwen3 </think> end-of-thinking token ID
_THINK_END_TOKEN_ID = 151668


def _split_thinking(output_ids: List[int], tokenizer) -> Tuple[str, str]:
    """
    Split raw generated token IDs into (reasoning, answer).

    Uses the last occurrence of the Qwen3 </think> token (151668) as the
    boundary. If no </think> token is present, reasoning is empty and the
    full decoded text is returned as the answer.
    """
    try:
        # rindex equivalent: find last occurrence of </think>
        index = len(output_ids) - output_ids[::-1].index(_THINK_END_TOKEN_ID)
    except ValueError:
        index = 0

    reasoning = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    answer    = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return reasoning, answer

def generate_with_baseline_model(
    model,
    baseline_prompts: List[str],
    tokenizer: PreTrainedTokenizer,
    max_new_tokens: int = 8192,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    min_p: float = 0.0,
    device: str = "cuda"
) -> Tuple[List[str], List[str]]:
    """
    Generate responses using the baseline pretrained model.
    
    Args:
        model: Pretrained model (e.g., Qwen3ForCausalLM)
        baseline_prompts: List of prompts with context
        tokenizer: Tokenizer
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling probability threshold
        top_k: Top-k sampling parameter
        min_p: Minimum probability threshold for sampling
        device: Device to use
        
    Returns:
        Tuple of (answers, reasonings, raw_responses) where each is a list of
        strings. ``answers`` contains only the text after </think>;
        ``reasonings`` contains the thinking content between <think> and
        </think> (empty string when no thinking block is present);
        ``raw_responses`` is the full decoded generation before any splitting.
    """
    model.eval()
    
    with torch.no_grad():
        # Tokenize
        inputs = tokenizer(
            baseline_prompts,
            return_tensors='pt',
            padding=True,
            truncation=False,
            #max_length=2048
        ).to(device)
        
        # Collect all stop token IDs (Qwen3 uses both <|im_end|> and <|endoftext|>)
        # stop_token_ids = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
        # stop_token_ids = [t for t in stop_token_ids if t is not None and t != tokenizer.unk_token_id]
        # if not stop_token_ids:
        #     stop_token_ids = tokenizer.eos_token_id

        # Generate
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            # eos_token_id=stop_token_ids,
        )
        
        # Decode
        prompt_len = inputs['input_ids'].shape[1]
        answers       = []
        reasonings    = []
        raw_responses = []

        for i in range(outputs.shape[0]):
            # Extract only the newly generated token IDs (exclude prompt)
            new_ids = outputs[i, prompt_len:].tolist()
            raw = tokenizer.decode(new_ids, skip_special_tokens=False).strip()
            reasoning, answer = _split_thinking(new_ids, tokenizer)
            answers.append(answer)
            reasonings.append(reasoning)
            raw_responses.append(raw)
        
    return answers, reasonings, raw_responses
