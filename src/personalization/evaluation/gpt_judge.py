"""
Judge utilities for win ratio evaluation.

Supports three LLM providers for judging response quality:
  - openai   : OpenAI API (GPT models)
  - deepseek : DeepSeek API (OpenAI-compatible)
  - gemini   : Google Gemini via Vertex AI (google-genai SDK)
"""

import json
import re
import time
import random
from typing import Optional, Tuple
from jinja2 import Template
import openai

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _build_judge_prompt(
    judge_template: Optional[Template],
    persona_description: str,
    prompt: str,
    first_response: str,
    second_response: str,
) -> str:
    if judge_template is not None:
        return judge_template.render(
            persona_description=persona_description,
            research_question=prompt,
            hypothesis_a=first_response,
            hypothesis_b=second_response,
        )
    return (
        "You are an expert evaluator of scientific hypotheses. "
        "Determine which hypothesis better represents the researcher's persona and preferences.\n\n"
        f"Researcher Persona:\n{persona_description}\n\n"
        f"Research Question:\n{prompt}\n\n"
        f"Hypothesis A:\n{first_response}\n\n"
        f"Hypothesis B:\n{second_response}\n\n"
        "Judge primarily on persona alignment: domain vocabulary, philosophy, "
        "interests, communication style, and avoiding rejected topics.\n\n"
        'Respond with ONLY "A", "B", or "tie".\n\nYour answer:'
    )


def _parse_judge_choice(result: str, randomize_order: bool) -> Optional[str]:
    result = result.strip().lower()
    if "a" in result and "b" not in result:
        judge_choice = "A"
    elif "b" in result and "a" not in result:
        judge_choice = "B"
    else:
        return "tie"
    if randomize_order:
        return "B" if judge_choice == "A" else "A"
    return judge_choice


def judge_with_openai_compatible(
    prompt: str,
    response_a: str,
    response_b: str,
    persona_description: str,
    api_key: str,
    model_name: str = "gpt-4",
    max_retries: int = 3,
    judge_template: Optional[Template] = None,
    base_url: Optional[str] = None,
) -> Optional[str]:
    """Judge using any OpenAI-compatible API (OpenAI or DeepSeek)."""
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.OpenAI(**client_kwargs)

    randomize_order = random.choice([True, False])
    first_response  = response_b if randomize_order else response_a
    second_response = response_a if randomize_order else response_b

    judge_prompt = _build_judge_prompt(
        judge_template, persona_description, prompt, first_response, second_response
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": judge_prompt}],
            )
            result = response.choices[0].message.content or ""
            return _parse_judge_choice(result, randomize_order)
        except Exception as e:
            print(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


def judge_with_gemini(
    prompt: str,
    response_a: str,
    response_b: str,
    persona_description: str,
    gemini_project_id: str,
    model_name: str = "gemini-2.5-pro",
    gemini_location: str = "global",
    max_retries: int = 3,
    judge_template: Optional[Template] = None,
) -> Optional[str]:
    """Judge using Google Gemini via Vertex AI."""
    from personalization.api_client.gemini_client import GeminiClient
    from personalization.api_client.gemini_parser import GeminiResponseParser

    client = GeminiClient(project_id=gemini_project_id, location=gemini_location)
    parser = GeminiResponseParser()

    randomize_order = random.choice([True, False])
    first_response  = response_b if randomize_order else response_a
    second_response = response_a if randomize_order else response_b

    judge_prompt = _build_judge_prompt(
        judge_template, persona_description, prompt, first_response, second_response
    )

    system_message = (
        "You are an expert evaluator of scientific hypotheses. "
        'Respond with ONLY "A", "B", or "tie".'
    )

    for attempt in range(max_retries):
        try:
            raw = client.query(
                model=model_name,
                system_message=system_message,
                user_message=judge_prompt,
            )
            parsed = parser.parse(raw)
            return _parse_judge_choice(parsed.answer, randomize_order)
        except Exception as e:
            print(f"Gemini API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


def judge(
    prompt: str,
    response_a: str,
    response_b: str,
    persona_description: str,
    provider: str = "openai",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    judge_template: Optional[Template] = None,
    gemini_project_id: Optional[str] = None,
    gemini_location: str = "global",
    max_retries: int = 3,
) -> Optional[str]:
    """
    Unified judge dispatcher.

    Parameters
    ----------
    provider:
        One of "openai", "deepseek", or "gemini".
    api_key:
        API key for OpenAI or DeepSeek. Not used for Gemini (uses ADC / Vertex AI).
    model_name:
        Model identifier. Defaults per provider:
          openai   → "gpt-4o"
          deepseek → "deepseek-chat"
          gemini   → "gemini-2.5-pro"
    gemini_project_id:
        GCP project ID. Required when provider="gemini".
    gemini_location:
        Vertex AI region. Defaults to "global".
    """
    if provider == "openai":
        return judge_with_openai_compatible(
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
            persona_description=persona_description,
            api_key=api_key,
            model_name=model_name or "gpt-4o",
            max_retries=max_retries,
            judge_template=judge_template,
        )
    elif provider == "deepseek":
        return judge_with_openai_compatible(
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
            persona_description=persona_description,
            api_key=api_key,
            model_name=model_name or "deepseek-chat",
            max_retries=max_retries,
            judge_template=judge_template,
            base_url=_DEEPSEEK_BASE_URL,
        )
    elif provider == "gemini":
        if not gemini_project_id:
            raise ValueError("gemini_project_id is required when provider='gemini'")
        return judge_with_gemini(
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
            persona_description=persona_description,
            gemini_project_id=gemini_project_id,
            model_name=model_name or "gemini-2.5-pro",
            gemini_location=gemini_location,
            max_retries=max_retries,
            judge_template=judge_template,
        )
    else:
        raise ValueError(f"Unknown judge provider: {provider!r}. Choose 'openai', 'deepseek', or 'gemini'.")


# ---------------------------------------------------------------------------
# Legacy shim – keeps existing callers working unchanged
# ---------------------------------------------------------------------------

def judge_with_gpt(
    prompt: str,
    response_a: str,
    response_b: str,
    persona_description: str,
    openai_api_key: str,
    model_name: str = "gpt-4",
    max_retries: int = 3,
    judge_template: Optional[Template] = None,
) -> Optional[str]:
    """Deprecated shim. Use ``judge()`` with provider='openai' instead."""
    return judge(
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
        persona_description=persona_description,
        provider="openai",
        api_key=openai_api_key,
        model_name=model_name,
        judge_template=judge_template,
        max_retries=max_retries,
    )


def judge_with_llm(
    prompt: str,
    response_a: str,
    response_b: str,
    persona_description: str,
    api_key: Optional[str],
    system_msg: str,
    user_tpl: Template,
    model_name: str,
    provider: str,
    max_retries: int = 3,
    gemini_location: str = "global",
) -> Tuple[Optional[str], str]:
    """
    Judge two responses using separate system/user templates.

    Parameters
    ----------
    api_key:
        API key for OpenAI / DeepSeek.  For Gemini, pass the GCP project ID.
    system_msg:
        Pre-rendered system message string (from system.j2).
    user_tpl:
        Jinja2 Template for the user message (user.j2). Expected variables:
        ``persona_description``, ``prompt``, ``first_response``, ``second_response``.
    provider:
        One of "gpt", "openai", "deepseek", or "gemini".

    Returns
    -------
    (verdict, justification)
        ``verdict`` is "A", "B", "tie", or None on unrecoverable error.
        ``justification`` is the raw explanation from the model (empty string
        when the model returned only a bare verdict).
    """
    randomize_order = random.choice([True, False])
    first_response  = response_b if randomize_order else response_a
    second_response = response_a if randomize_order else response_b

    user_msg = user_tpl.render(
        persona_description=persona_description,
        # old variable names (win_ratio/user.j2)
        prompt=prompt,
        first_response=first_response,
        second_response=second_response,
        # new variable names (evaluation/judge.j2)
        research_question=prompt,
        hypothesis_a=first_response,
        hypothesis_b=second_response,
    )

    def _call(system: str, user: str) -> str:
        if provider in ("gpt", "openai"):
            client = openai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            return resp.choices[0].message.content or ""
        elif provider == "deepseek":
            client = openai.OpenAI(api_key=api_key, base_url=_DEEPSEEK_BASE_URL)
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            return resp.choices[0].message.content or ""
        elif provider == "gemini":
            from personalization.api_client.gemini_client import GeminiClient
            from personalization.api_client.gemini_parser import GeminiResponseParser
            client = GeminiClient(project_id=api_key, location=gemini_location)
            parser = GeminiResponseParser()
            raw = client.query(model=model_name, system_message=system, user_message=user)
            return parser.parse(raw).answer
        else:
            raise ValueError(f"Unknown provider: {provider!r}")

    for attempt in range(max_retries):
        try:
            raw = _call(system_msg, user_msg)

            # Try JSON parse first ({"winner": "A", "justification": "..."})
            try:
                clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
                data  = json.loads(clean)
                winner        = str(data.get("winner", "")).strip().upper()
                justification = str(data.get("justification", ""))
            except (json.JSONDecodeError, AttributeError):
                winner        = raw.strip().upper()
                justification = ""

            if winner == "A":
                verdict = "B" if randomize_order else "A"
            elif winner == "B":
                verdict = "A" if randomize_order else "B"
            elif "TIE" in winner:
                verdict = "tie"
            else:
                verdict = None

            return verdict, justification

        except Exception as e:
            print(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return None, ""
