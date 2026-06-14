import wandb
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader, select_autoescape

from personalization.models.baselines.factory import BaselineFactory
from personalization.dataset.baselines.data_loader import DatasetLoader
from personalization.dataset.baselines.data_processor import DataProcessor
from personalization.personalization_config import PersonalizationConfig
from personalization.evaluation.generation import generate_with_baseline_model
from personalization.evaluation.gpt_judge import judge

# Paths resolved relative to this file: src/personalization/evaluation/eval_baseline.py
_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVAL_DIR.parents[2]          # …/pers2/Personalization
_TEMPLATES_DATA = _PROJECT_ROOT / "prompt_templates" / "data_generation"
_TEMPLATES_EVAL = _PROJECT_ROOT / "prompt_templates" / "evaluation"

# Evaluation modes: (name, use_few_shot_for_base, use_few_shot_for_trained)
_EVAL_MODES = [
    ("base_vs_trained",                    False, False),
    ("base_few_shot_vs_trained",           True,  False),
    ("base_few_shot_vs_trained_few_shot",  True,  True),
]


def _select_few_shot_pool(
    val_df: pd.DataFrame,
    pool_size: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select the first `pool_size` rows as the few-shot pool and guarantee that
    at least one example per persona is included. Any persona not covered by
    the top-`pool_size` rows gets its first occurrence appended from the
    remainder. Returns (few_shot_df, eval_df) where eval_df is the rest of
    the validation set used for evaluation.
    """
    top = val_df.iloc[:pool_size]
    covered = set(top["user_id"].unique())
    all_personas = set(val_df["user_id"].unique())

    extra_indices = []
    for uid in all_personas - covered:
        matches = val_df.iloc[pool_size:][val_df.iloc[pool_size:]["user_id"] == uid]
        if not matches.empty:
            extra_indices.append(matches.index[0])

    pool_indices = list(top.index) + extra_indices
    few_shot_df = val_df.loc[pool_indices].reset_index(drop=True)
    eval_df     = val_df.drop(index=pool_indices).reset_index(drop=True)
    return few_shot_df, eval_df


def _build_few_shot_examples(
    val_df: pd.DataFrame,
    processor: DataProcessor,
    max_per_user: int = 3,
    seed: int = 42,
    answer_col: str = "answer",
) -> dict:
    """Return {user_id: [{"persona_description", "question", "hypothesis"}, …]}.

    For each persona, randomly samples up to ``max_per_user`` examples from
    the rows belonging to that persona using ``seed`` for reproducibility.
    """
    few_shots: dict = {}
    for uid, group in val_df.groupby("user_id"):
        n = min(max_per_user, len(group))
        sampled = group.sample(n=n, random_state=seed)
        few_shots[uid] = [
            {
                # "persona_description": processor._build_persona_instr(uid),
                "question": row["question"],
                "hypothesis": row[answer_col],
            }
            for _, row in sampled.iterrows()
        ]
    return few_shots


def _format_sample(
    processor: DataProcessor,
    tokenizer,
    user_id: str,
    question: str,
    context: str,
    few_shot_examples=None,
):
    """Return (formatted_prompt_str, user_prompt_str) for a single sample."""
    if few_shot_examples:
        messages = processor._format_single_inference_few_shot_messages(
            user_id, question, context, few_shot_examples
        )
    else:
        messages = processor._format_single_inference_messages(user_id, question, context)

    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, 
        # enable_thinking=True
    )
    # formatted += "<think>\n"
    user_prompt = processor.user_prompt_template.render(query=question, context=context)
    return formatted, user_prompt


def _format_sample_rationale(
    processor: DataProcessor,
    tokenizer,
    user_id: str,
    question: str,
    context: str,
    rationale_question: str,
    rationale_chosen: str,
    rationale_rejected: str,
    rationale_train: str,
    few_shot_examples=None,
):
    """Return (formatted_prompt_str, user_prompt_str) using rationale-aware templates."""
    if few_shot_examples:
        messages = processor._format_single_inference_few_shot_messages_rationale(
            user_id, question, context,
            rationale_question, rationale_chosen, rationale_rejected, rationale_train,
            few_shot_examples,
        )
    else:
        messages = processor._format_single_inference_messages_rationale(
            user_id, question, context,
            rationale_question, rationale_chosen, rationale_rejected, rationale_train,
        )

    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    user_prompt = processor.user_prompt_template.render(query=question, context=context)
    return formatted, user_prompt


def evaluate(cfg: PersonalizationConfig, api_key: str = None):
    generation_only: bool = getattr(cfg.evaluation, "generation_only", False)
    train_only: bool = getattr(cfg.evaluation, "train_only", False)
    judge_use_reasoning: bool = getattr(cfg.evaluation, "judge_use_reasoning", False)
    use_rationale: bool = cfg.data.baseline.get("use_rationale", False)
    # Column name for the reference answer in val/test data
    answer_col: str = "chosen" if use_rationale else "answer"

    # ── Load models ──────────────────────────────────────────────────────────
    adapter_to_load = cfg.evaluation.personalized_checkpoint 
    model, tokenizer, _ = BaselineFactory.get_model_and_tokenizer(cfg, adapter_to_load)
    if not train_only:
        base_model, _, _ = BaselineFactory.get_model_and_tokenizer(cfg)
    else:
        base_model = None

    tokenizer.padding_side = "left"
    model.eval()
    if not train_only:
        base_model.eval()

    # ── Dataset loader ────────────────────────────────────────────────────────
    strategy = cfg.training.baseline["strategy"]
    if use_rationale:
        loader = DatasetLoader(
            cfg.data.baseline.get("rationale_dpo_train", cfg.data.baseline["dpo_train"]),
            cfg.data.baseline.get("rationale_dpo_val",   cfg.data.baseline["dpo_val"]),
            cfg.data.baseline.get("rationale_dpo_test",  cfg.data.baseline["dpo_test"]),
            cfg.data.baseline["personas_path"],
        )
    elif strategy == "sft":
        loader = DatasetLoader(
            cfg.data.baseline["sft_train"],
            cfg.data.baseline["sft_val"],
            cfg.data.baseline["sft_test"],
            cfg.data.baseline["personas_path"],
        )
    else:  # dpo / sft+dpo
        loader = DatasetLoader(
            cfg.data.baseline["dpo_train"],
            cfg.data.baseline["sft_val"],
            cfg.data.baseline["dpo_test"],
            cfg.data.baseline["personas_path"],
        )

    personas_map = loader.load_persona_map()

    # ── DataProcessor (loads Jinja2 templates) ────────────────────────────────
    processor = DataProcessor(tokenizer, personas_map, str(_TEMPLATES_DATA))

    # ── Judge template ────────────────────────────────────────────────────────
    if not generation_only:
        judge_env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_EVAL)),
            autoescape=select_autoescape([]),
            keep_trailing_newline=True,
        )
        judge_template = judge_env.get_template("judge.j2")
    else:
        judge_template = None

    # ── Few-shot pool (first 30 val rows, ≥1 per persona) + eval remainder ────
    val_df = pd.read_csv(loader.val_path)
    test_df = pd.read_csv(loader.test_path)
    few_shot_df, _ = _select_few_shot_pool(val_df, pool_size=30)
    n_few_shots = cfg.evaluation.n_few_shots if hasattr(cfg.evaluation, "n_few_shots") else 3
    few_shot_seed = cfg.training.seed if hasattr(cfg.training, "seed") else 42
    few_shots_per_user = _build_few_shot_examples(
        few_shot_df, processor, max_per_user=n_few_shots, seed=few_shot_seed, answer_col=answer_col,
    )

    # test_df = test_df.sample(n=10, random_state=42).reset_index(drop=True)  # TEMP: subsample for faster eval during development
    # test_df = test_df.groupby("user_id").sample(10).reset_index(drop=True)
    # test_df = test_df[test_df['user_id'] == 'efficient_compute']
    # test_df = test_df[test_df['user_id'] == 'sota_chaser']  # TEMP: limit to 2 samples per user for faster eval during development
    print(f"Few-shot pool: {len(few_shot_df)} examples | Test set: {len(test_df)} examples")

    #val_df = pd.read_csv(loader.val_path)
    # n_few_shots = cfg.evaluation.n_few_shots if hasattr(cfg.evaluation, "n_few_shots") else 1
    # few_shots_per_user = _build_few_shot_examples(val_df, processor, max_per_user=n_few_shots)

    # ── Test data ─────────────────────────────────────────────────────────────
    test_df = loader.load_test_dataset()["test"].to_pandas()
    # ── W&B run ───────────────────────────────────────────────────────────────
    run = wandb.init(
        entity=cfg.logging.wandb_entity,
        project=cfg.logging.wandb_project,
        name=cfg.logging.wandb_run_name,
        tags=["baseline", "evaluation"],
        config=cfg,
    )

    # ── Output directory ──────────────────────────────────────────────────────
    output_dir = Path(cfg.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = cfg.logging.wandb_run_name or "run"
    results_csv_path = output_dir / f"eval_results_{run_name}.csv"

    all_results: list[dict] = []
    mode_stats: dict = {}

    if generation_only:
        print("\ngeneration_only=True — LLM judging is disabled. Generating responses only.")

    # ── Pre-generate trained-model responses (reused across both eval modes) ──
    batch_size = getattr(cfg.evaluation, "eval_batch_size", None) or cfg.training.batch_size
    print("\nPre-generating trained model responses (shared across both eval modes)…")
    trained_cache_responses: list[str] = []
    trained_cache_reasonings: list[str] = []
    trained_cache_raws: list[str] = []
    _first_trained_prompt_printed = False
    for batch_start in tqdm(range(0, len(test_df), batch_size), desc="[trained pre-gen]"):
        batch_df = test_df.iloc[batch_start : batch_start + batch_size]
        if use_rationale:
            trained_prompts = [
                _format_sample_rationale(
                    processor, tokenizer, str(row["user_id"]), row["question"], row["context"],
                    row["rationale_question"], row["rationale_chosen"],
                    row["rationale_rejected"], row["rationale_train"],
                )[0]
                for _, row in batch_df.iterrows()
            ]
        else:
            trained_prompts = [
                _format_sample(processor, tokenizer, str(row["user_id"]), row["question"], row["context"])[0]
                for _, row in batch_df.iterrows()
            ]
        if not _first_trained_prompt_printed:
            print("\n" + "="*60)
            print("[DEBUG] First trained-model prompt after apply_chat_template:")
            print("="*60)
            print(trained_prompts[0])
            print("="*60 + "\n")
            _first_trained_prompt_printed = True
        _responses, _reasonings, _raws = generate_with_baseline_model(
            model=model,
            baseline_prompts=trained_prompts,
            tokenizer=tokenizer,
            max_new_tokens=cfg.evaluation.max_new_tokens,
            temperature=cfg.evaluation.temperature,
            top_p=cfg.evaluation.top_p,
            top_k=cfg.evaluation.top_k,
            min_p=cfg.evaluation.min_p,
            device=cfg.evaluation.device,
        )
        trained_cache_responses.extend(_responses)
        trained_cache_reasonings.extend(_reasonings)
        trained_cache_raws.extend(_raws)

    # ── Pre-generate trained-model responses WITH few-shot (for new mode) ────
    print("\nPre-generating trained model responses (with few-shot)…")
    trained_fs_cache_responses: list[str] = []
    trained_fs_cache_reasonings: list[str] = []
    trained_fs_cache_raws: list[str] = []
    _first_trained_fs_prompt_printed = False
    for batch_start in tqdm(range(0, len(test_df), batch_size), desc="[trained few-shot pre-gen]"):
        batch_df = test_df.iloc[batch_start : batch_start + batch_size]
        trained_fs_prompts = []
        for _, row in batch_df.iterrows():
            uid = str(row["user_id"])
            shots = few_shots_per_user.get(uid, [])
            if use_rationale:
                fmt, _ = _format_sample_rationale(
                    processor, tokenizer, uid, row["question"], row["context"],
                    row["rationale_question"], row["rationale_chosen"],
                    row["rationale_rejected"], row["rationale_train"],
                    few_shot_examples=shots if shots else None,
                )
            else:
                fmt, _ = _format_sample(
                    processor, tokenizer, uid, row["question"], row["context"],
                    few_shot_examples=shots if shots else None,
                )
            trained_fs_prompts.append(fmt)
        if not _first_trained_fs_prompt_printed:
            print("\n" + "="*60)
            print("[DEBUG] First trained-model prompt (few-shot) after apply_chat_template:")
            print("="*60)
            print(trained_fs_prompts[0])
            print("="*60 + "\n")
            _first_trained_fs_prompt_printed = True
        _r, _rs, _raw = generate_with_baseline_model(
            model=model,
            baseline_prompts=trained_fs_prompts,
            tokenizer=tokenizer,
            max_new_tokens=cfg.evaluation.max_new_tokens,
            temperature=cfg.evaluation.temperature,
            top_p=cfg.evaluation.top_p,
            top_k=cfg.evaluation.top_k,
            min_p=cfg.evaluation.min_p,
            device=cfg.evaluation.device,
        )
        trained_fs_cache_responses.extend(_r)
        trained_fs_cache_reasonings.extend(_rs)
        trained_fs_cache_raws.extend(_raw)

    # ── Early-return: generation-only mode ───────────────────────────────────
    if generation_only:
        if train_only:
            # Skip base model generation — only save trained responses
            gen_columns = [
                "user_id", "question", "context",
                "trained_response", "trained_reasoning", "trained_raw",
                "trained_fs_response", "trained_fs_reasoning", "trained_fs_raw",
            ]
            gen_table = wandb.Table(columns=gen_columns)
            gen_results: list[dict] = []
            for i, (_, row) in enumerate(test_df.iterrows()):
                uid = str(row["user_id"])
                gen_table.add_data(
                    uid, row["question"], row["context"],
                    trained_cache_responses[i], trained_cache_reasonings[i], trained_cache_raws[i],
                    # trained_fs_cache_responses[i], trained_fs_cache_reasonings[i], trained_fs_cache_raws[i],
                )
                gen_results.append({
                    "user_id":              uid,
                    "question":             row["question"],
                    "context":              row["context"],
                    "trained_response":     trained_cache_responses[i],
                    "trained_reasoning":    trained_cache_reasonings[i],
                    "trained_raw":          trained_cache_raws[i],
                    "trained_fs_response":  trained_fs_cache_responses[i],
                    "trained_fs_reasoning": trained_fs_cache_reasonings[i],
                    "trained_fs_raw":       trained_fs_cache_raws[i],
                })
            wandb.log({"evaluation/generation_only/results_table": gen_table})
            run.summary["total_generated"] = len(gen_results)
            pd.DataFrame(gen_results).to_csv(results_csv_path, index=False)
            print(f"\nResults saved to: {results_csv_path}")
            print("\n" + "=" * 80)
            print("Train-only run completed — trained responses saved.")
            print("=" * 80)
            run.finish()
            return

        # ── Generate base (no few-shot) responses ────────────────────────────
        print("\nGenerating base model responses (no few-shot)…")
        base_cache_responses: list[str] = []
        base_cache_reasonings: list[str] = []
        base_cache_raws: list[str] = []
        _first_base_printed = False
        for batch_start in tqdm(range(0, len(test_df), batch_size), desc="[base no-shot]"):
            batch_df = test_df.iloc[batch_start : batch_start + batch_size]
            if use_rationale:
                base_prompts = [
                    _format_sample_rationale(
                        processor, tokenizer, str(row["user_id"]), row["question"], row["context"],
                        row["rationale_question"], row["rationale_chosen"],
                        row["rationale_rejected"], row["rationale_train"],
                    )[0]
                    for _, row in batch_df.iterrows()
                ]
            else:
                base_prompts = [
                    _format_sample(processor, tokenizer, str(row["user_id"]), row["question"], row["context"])[0]
                    for _, row in batch_df.iterrows()
                ]
            if not _first_base_printed:
                print("\n" + "="*60)
                print("[DEBUG] First base-model prompt (no few-shot) after apply_chat_template:")
                print("="*60)
                print(base_prompts[0])
                print("="*60 + "\n")
                _first_base_printed = True
            _r, _rs, _raw = generate_with_baseline_model(
                model=base_model,
                baseline_prompts=base_prompts,
                tokenizer=tokenizer,
                max_new_tokens=cfg.evaluation.max_new_tokens,
                temperature=cfg.evaluation.temperature,
                top_p=cfg.evaluation.top_p,
                top_k=cfg.evaluation.top_k,
                min_p=cfg.evaluation.min_p,
                device=cfg.evaluation.device,
            )
            base_cache_responses.extend(_r)
            base_cache_reasonings.extend(_rs)
            base_cache_raws.extend(_raw)

        # ── Generate base (few-shot) responses ───────────────────────────────
        print("\nGenerating base model responses (with few-shot)…")
        base_fs_cache_responses: list[str] = []
        base_fs_cache_reasonings: list[str] = []
        base_fs_cache_raws: list[str] = []
        _first_base_fs_printed = False
        for batch_start in tqdm(range(0, len(test_df), batch_size), desc="[base few-shot]"):
            batch_df = test_df.iloc[batch_start : batch_start + batch_size]
            base_fs_prompts = []
            for _, row in batch_df.iterrows():
                uid = str(row["user_id"])
                shots = few_shots_per_user.get(uid, [])
                if use_rationale:
                    fmt, _ = _format_sample_rationale(
                        processor, tokenizer, uid, row["question"], row["context"],
                        row["rationale_question"], row["rationale_chosen"],
                        row["rationale_rejected"], row["rationale_train"],
                        few_shot_examples=shots if shots else None,
                    )
                else:
                    fmt, _ = _format_sample(
                        processor, tokenizer, uid, row["question"], row["context"],
                        few_shot_examples=shots if shots else None,
                    )
                base_fs_prompts.append(fmt)
            if not _first_base_fs_printed:
                print("\n" + "="*60)
                print("[DEBUG] First base-model prompt (few-shot) after apply_chat_template:")
                print("="*60)
                print(base_fs_prompts[0])
                print("="*60 + "\n")
                _first_base_fs_printed = True
            _r, _rs, _raw = generate_with_baseline_model(
                model=base_model,
                baseline_prompts=base_fs_prompts,
                tokenizer=tokenizer,
                max_new_tokens=cfg.evaluation.max_new_tokens,
                temperature=cfg.evaluation.temperature,
                top_p=cfg.evaluation.top_p,
                top_k=cfg.evaluation.top_k,
                min_p=cfg.evaluation.min_p,
                device=cfg.evaluation.device,
            )
            base_fs_cache_responses.extend(_r)
            base_fs_cache_reasonings.extend(_rs)
            base_fs_cache_raws.extend(_raw)

        # ── Build table & CSV ─────────────────────────────────────────────────
        gen_columns = [
            "user_id", "question", "context",
            "trained_response", "trained_reasoning", "trained_raw",
            "trained_fs_response", "trained_fs_reasoning", "trained_fs_raw",
            "base_response", "base_reasoning", "base_raw",
            "base_fs_response", "base_fs_reasoning", "base_fs_raw",
        ]
        gen_table = wandb.Table(columns=gen_columns)
        gen_results: list[dict] = []
        for i, (_, row) in enumerate(test_df.iterrows()):
            uid = str(row["user_id"])
            gen_table.add_data(
                uid, row["question"], row["context"],
                trained_cache_responses[i], trained_cache_reasonings[i], trained_cache_raws[i],
                trained_fs_cache_responses[i], trained_fs_cache_reasonings[i], trained_fs_cache_raws[i],
                base_cache_responses[i], base_cache_reasonings[i], base_cache_raws[i],
                base_fs_cache_responses[i], base_fs_cache_reasonings[i], base_fs_cache_raws[i],
            )
            gen_results.append({
                "user_id":              uid,
                "question":             row["question"],
                "context":              row["context"],
                "trained_response":     trained_cache_responses[i],
                "trained_reasoning":    trained_cache_reasonings[i],
                "trained_raw":          trained_cache_raws[i],
                "trained_fs_response":  trained_fs_cache_responses[i],
                "trained_fs_reasoning": trained_fs_cache_reasonings[i],
                "trained_fs_raw":       trained_fs_cache_raws[i],
                "base_response":        base_cache_responses[i],
                "base_reasoning":       base_cache_reasonings[i],
                "base_raw":             base_cache_raws[i],
                "base_fs_response":     base_fs_cache_responses[i],
                "base_fs_reasoning":    base_fs_cache_reasonings[i],
                "base_fs_raw":          base_fs_cache_raws[i],
            })
        wandb.log({"evaluation/generation_only/results_table": gen_table})
        run.summary["total_generated"] = len(gen_results)
        pd.DataFrame(gen_results).to_csv(results_csv_path, index=False)
        print(f"\nResults saved to: {results_csv_path}")
        print("\n" + "=" * 80)
        print("Generation-only run completed — responses saved, no judging performed.")
        print("=" * 80)
        run.finish()
        return

    # ── Main evaluation loop (two modes) ──────────────────────────────────────
    for mode_name, use_few_shot, use_few_shot_trained in _EVAL_MODES:
        print(f"\n{'='*60}\nEvaluation mode: {mode_name}\n{'='*60}")

        wins = losses = ties = errors = 0
        wb_columns = [
            "eval_mode", "user_id", "question",
            "trained_response", "trained_reasoning", "trained_raw",
            "base_response", "base_reasoning", "base_raw", "judgment",
        ]
        results_table = wandb.Table(columns=wb_columns)

        _first_base_prompt_printed = False
        for batch_start in tqdm(
            range(0, len(test_df), batch_size), desc=f"[{mode_name}]"
        ):
            batch_df = test_df.iloc[batch_start : batch_start + batch_size]
            batch_end = batch_start + len(batch_df)

            base_prompts = []
            user_prompts_list, user_ids, questions, contexts = [], [], [], []

            for _, row in batch_df.iterrows():
                uid = str(row["user_id"])
                q   = row["question"]
                c   = row["context"]

                user_prompt = processor.user_prompt_template.render(query=q, context=c)

                # Base model: persona only (mode 1) OR persona + few-shot (mode 2)
                shots = few_shots_per_user.get(uid, []) if use_few_shot else []
                if use_rationale:
                    base_fmt, _ = _format_sample_rationale(
                        processor, tokenizer, uid, q, c,
                        row["rationale_question"], row["rationale_chosen"],
                        row["rationale_rejected"], row["rationale_train"],
                        few_shot_examples=shots if shots else None,
                    )
                else:
                    base_fmt, _ = _format_sample(
                        processor, tokenizer, uid, q, c,
                        few_shot_examples=shots if shots else None,
                    )

                base_prompts.append(base_fmt)
                user_prompts_list.append(user_prompt)
                user_ids.append(uid)
                questions.append(q)
                contexts.append(c)

            if not _first_base_prompt_printed:
                print(f"\n" + "="*60)
                print(f"[DEBUG] First base-model prompt ({mode_name}) after apply_chat_template:")
                print("="*60)
                print(base_prompts[0])
                print("="*60 + "\n")
                _first_base_prompt_printed = True

            # Trained responses: reuse pre-generated cache (with or without few-shot)
            if use_few_shot_trained:
                trained_responses  = trained_fs_cache_responses[batch_start:batch_end]
                trained_reasonings = trained_fs_cache_reasonings[batch_start:batch_end]
                trained_raws       = trained_fs_cache_raws[batch_start:batch_end]
            else:
                trained_responses  = trained_cache_responses[batch_start:batch_end]
                trained_reasonings = trained_cache_reasonings[batch_start:batch_end]
                trained_raws       = trained_cache_raws[batch_start:batch_end]

            base_responses, base_reasonings, base_raws = generate_with_baseline_model(
                model=base_model,
                baseline_prompts=base_prompts,
                tokenizer=tokenizer,
                max_new_tokens=cfg.evaluation.max_new_tokens,
                temperature=cfg.evaluation.temperature,
                top_p=cfg.evaluation.top_p,
                top_k=cfg.evaluation.top_k,
                min_p=cfg.evaluation.min_p,
                device=cfg.evaluation.device,
            )

            # Judge (skipped in generation_only mode)
            for i in range(len(trained_responses)):
                uid = user_ids[i]

                if generation_only:
                    judgment = None
                else:
                    persona_desc = processor._build_persona_instr(uid)
                    # Compose judge inputs: answer only OR thinking + answer
                    def _compose(answer: str, reasoning: str) -> str:
                        if judge_use_reasoning and reasoning:
                            return f"<think>\n{reasoning}\n</think>\n{answer}"
                        return answer

                    judgment = judge(
                        prompt=user_prompts_list[i],
                        response_a=_compose(trained_responses[i], trained_reasonings[i]),
                        response_b=_compose(base_responses[i], base_reasonings[i]),
                        persona_description=persona_desc,
                        provider=cfg.evaluation.judge_provider,
                        api_key=api_key,
                        model_name=cfg.evaluation.judge_model,
                        judge_template=judge_template,
                        gemini_project_id=cfg.evaluation.gemini_project_id,
                        gemini_location=cfg.evaluation.gemini_location,
                    )

                if judgment == "A":
                    wins += 1
                elif judgment == "B":
                    losses += 1
                elif judgment == "tie":
                    ties += 1
                elif judgment is not None:
                    errors += 1

                row_result = {
                    "eval_mode":          mode_name,
                    "user_id":            uid,
                    "question":           questions[i],
                    "context":            contexts[i],
                    "trained_response":   trained_responses[i],
                    "trained_reasoning":  trained_reasonings[i],
                    "trained_raw":        trained_raws[i],
                    "base_response":      base_responses[i],
                    "base_reasoning":     base_reasonings[i],
                    "base_raw":           base_raws[i],
                    "judgment":           judgment,
                    "trained_wins":       int(judgment == "A") if judgment is not None else None,
                }
                all_results.append(row_result)
                results_table.add_data(
                    mode_name, uid, questions[i],
                    trained_responses[i], trained_reasonings[i], trained_raws[i],
                    base_responses[i], base_reasonings[i], base_raws[i], judgment,
                )

            total_so_far = wins + losses + ties
            if not generation_only and total_so_far > 0:
                wandb.log({
                    f"eval/{mode_name}/running_win_rate": wins / total_so_far,
                    f"eval/{mode_name}/wins":   wins,
                    f"eval/{mode_name}/losses": losses,
                    f"eval/{mode_name}/ties":   ties,
                })

        total    = wins + losses + ties
        win_rate = wins / total if total > 0 else 0.0
        mode_stats[mode_name] = {
            "wins": wins, "losses": losses, "ties": ties,
            "errors": errors, "total": total, "win_rate": win_rate,
        }

        wandb.log({f"evaluation/{mode_name}/results_table": results_table})
        if not generation_only:
            wandb.log({
                f"evaluation/{mode_name}/win_loss_dist": wandb.plot.bar(
                    wandb.Table(
                        data=[["Wins", wins], ["Losses", losses], ["Ties", ties]],
                        columns=["label", "value"],
                    ),
                    "label", "value",
                    title=f"[{mode_name}] Judgment Distribution",
                ),
            })
            run.summary[f"{mode_name}/win_rate"]     = win_rate
            run.summary[f"{mode_name}/total_samples"] = total

    # ── Save all results to CSV ───────────────────────────────────────────────
    pd.DataFrame(all_results).to_csv(results_csv_path, index=False)
    print(f"\nResults saved to: {results_csv_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    for mode_name, stats in mode_stats.items():
        wr = stats["win_rate"]
        total = stats["total"]
        print(f"\n  [{mode_name}]")
        print(f"  Wins (Trained):  {stats['wins']:>4}  ({wr:.1%})")
        print(f"  Losses (Base):   {stats['losses']:>4}  ({stats['losses']/total:.1%})" if total else "")
        print(f"  Ties:            {stats['ties']:>4}")
        print(f"  Errors:          {stats['errors']:>4}")
        print(f"  Total Evaluated: {total:>4}")
    print("=" * 80)
    run.finish()

