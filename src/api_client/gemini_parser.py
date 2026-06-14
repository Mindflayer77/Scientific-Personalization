"""
gemini_parser.py
~~~~~~~~~~~~~~~~
A structured parser for GenerateContentResponse objects returned by the
Google Gen AI / Vertex AI SDK (google-genai / google-cloud-aiplatform).

Handles:
  • Reasoning / thinking parts  (part.thought == True)
  • Answer parts                (part.thought == False)
  • JSON or plain-text answer   (controlled by `parse_as_json`)
  • Log-probabilities           (optional, controlled by `include_logprobs`)
  • Token-usage statistics
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data-transfer objects
# ---------------------------------------------------------------------------

@dataclass
class TokenUsage:
    """Token counts reported by the model."""
    prompt_tokens: int = 0
    candidates_tokens: int = 0
    thoughts_tokens: int = 0          # Only populated for thinking models
    total_tokens: int = 0

    def __repr__(self) -> str:
        return (
            f"TokenUsage(prompt={self.prompt_tokens}, "
            f"candidates={self.candidates_tokens}, "
            f"thoughts={self.thoughts_tokens}, "
            f"total={self.total_tokens})"
        )


@dataclass
class LogprobToken:
    """Log-probability entry for a single token."""
    token: str
    log_probability: float

    def __repr__(self) -> str:
        return f"LogprobToken(token={self.token!r}, log_prob={self.log_probability:.4f})"


@dataclass
class LogprobsResult:
    """
    Parsed representation of a candidate's logprobsResult.

    chosen_candidates  – the token actually selected at each step.
    top_candidates     – the top-N alternatives at each step (list of lists).
    avg_log_prob       – average log-probability of the whole candidate.
    """
    chosen_candidates: list[LogprobToken] = field(default_factory=list)
    top_candidates: list[list[LogprobToken]] = field(default_factory=list)
    avg_log_prob: float | None = None


@dataclass
class ParsedResponse:
    """
    The fully parsed result of one GenerateContentResponse candidate.

    reasoning   – concatenated text from thought=True parts (may be empty).
    answer      – the model's actual answer as a string.
    answer_json – answer parsed as JSON (dict / list / scalar), or None if
                  `parse_as_json=False` or parsing failed.
    logprobs    – LogprobsResult if `include_logprobs=True`, else None.
    usage       – TokenUsage from usageMetadata.
    model_version – model + version string from the response.
    finish_reason – finish-reason enum string of the first candidate.
    raw_response  – the original SDK response object for further inspection.
    """
    reasoning: str = ""
    answer: str = ""
    answer_json: Any = None
    logprobs: LogprobsResult | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    model_version: str = ""
    finish_reason: str = ""
    raw_response: Any = None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class GeminiResponseParser:
    """
    Parse a ``GenerateContentResponse`` (from google-genai or google-cloud-aiplatform)
    into a tidy :class:`ParsedResponse`.

    Usage
    -----
    >>> parser = GeminiResponseParser()
    >>> parsed = parser.parse(response, include_logprobs=True, parse_as_json=True)
    >>> print(parsed.reasoning)
    >>> print(parsed.answer_json)
    >>> print(parsed.usage)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        response: Any,
        *,
        include_logprobs: bool = False,
        parse_as_json: bool = False,
        candidate_index: int = 0,
    ) -> ParsedResponse:
        """
        Parse *response* and return a :class:`ParsedResponse`.

        Parameters
        ----------
        response:
            A ``GenerateContentResponse`` object from the SDK.
        include_logprobs:
            When True, extract log-probability data from the candidate.
            The model must have been called with ``responseLogprobs=True``;
            otherwise the fields will be empty.
        parse_as_json:
            When True, attempt to parse the answer text as JSON. The result
            is stored in ``ParsedResponse.answer_json``. If parsing fails,
            ``answer_json`` is set to ``None`` and no exception is raised.
        candidate_index:
            Which candidate to parse (default 0 = first).
        """
        result = ParsedResponse(raw_response=response)

        # ── model version ───────────────────────────────────────────────
        result.model_version = getattr(response, "model_version", "")

        # ── usage metadata ──────────────────────────────────────────────
        result.usage = self._parse_usage(response)

        # ── candidates ──────────────────────────────────────────────────
        candidates = getattr(response, "candidates", None) or []
        if not candidates or candidate_index >= len(candidates):
            return result

        candidate = candidates[candidate_index]
        result.finish_reason = self._enum_to_str(
            getattr(candidate, "finish_reason", "")
        )

        # ── parts: separate reasoning from answer ───────────────────────
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []

        reasoning_parts: list[str] = []
        answer_parts: list[str] = []

        for part in parts:
            text = getattr(part, "text", None) or ""
            is_thought = getattr(part, "thought", False)
            if is_thought:
                reasoning_parts.append(text)
            else:
                answer_parts.append(text)

        result.reasoning = "\n\n".join(filter(None, reasoning_parts))
        result.answer = "\n\n".join(filter(None, answer_parts))

        # ── JSON parsing ─────────────────────────────────────────────────
        if parse_as_json and result.answer:
            result.answer_json = self._try_parse_json(result.answer)

        # ── log-probabilities ────────────────────────────────────────────
        if include_logprobs:
            result.logprobs = self._parse_logprobs(candidate)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_usage(response: Any) -> TokenUsage:
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return TokenUsage()

        return TokenUsage(
            prompt_tokens=getattr(meta, "prompt_token_count", 0) or 0,
            candidates_tokens=getattr(meta, "candidates_token_count", 0) or 0,
            thoughts_tokens=getattr(meta, "thoughts_token_count", 0) or 0,
            total_tokens=getattr(meta, "total_token_count", 0) or 0,
        )

    @staticmethod
    def _parse_logprobs(candidate: Any) -> LogprobsResult:
        result = LogprobsResult()

        # avg_logprobs lives directly on the candidate
        avg = getattr(candidate, "avg_logprobs", None)
        if avg is not None:
            result.avg_log_prob = float(avg)

        lp_result = getattr(candidate, "logprobs_result", None)
        if lp_result is None:
            return result

        # chosen_candidates  →  one token per generation step
        for entry in getattr(lp_result, "chosen_candidates", None) or []:
            result.chosen_candidates.append(
                LogprobToken(
                    token=getattr(entry, "token", ""),
                    log_probability=float(getattr(entry, "log_probability", 0.0)),
                )
            )

        # top_candidates  →  list[TopCandidates], each has .candidates list
        for top in getattr(lp_result, "top_candidates", None) or []:
            step: list[LogprobToken] = []
            for entry in getattr(top, "candidates", None) or []:
                step.append(
                    LogprobToken(
                        token=getattr(entry, "token", ""),
                        log_probability=float(
                            getattr(entry, "log_probability", 0.0)
                        ),
                    )
                )
            result.top_candidates.append(step)

        return result

    @staticmethod
    def _try_parse_json(text: str) -> Any:
        """
        Attempt to parse *text* as JSON.

        Strips common markdown fences (```json … ```) before attempting.
        Returns the parsed value on success, or ``None`` on failure.
        """
        cleaned = text.strip()

        # Remove optional ```json … ``` fences
        fence_re = re.compile(r"^```(?:json)?\s*([\s\S]*?)\s*```$", re.MULTILINE)
        match = fence_re.match(cleaned)
        if match:
            cleaned = match.group(1).strip()

        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _enum_to_str(value: Any) -> str:
        """Convert a proto/enum value to a plain string."""
        if isinstance(value, str):
            return value
        # SDK enums have a .name attribute
        return getattr(value, "name", str(value))


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def parse_gemini_response(
    response: Any,
    *,
    include_logprobs: bool = False,
    parse_as_json: bool = False,
    candidate_index: int = 0,
) -> ParsedResponse:
    """
    Module-level shortcut – equivalent to ``GeminiResponseParser().parse(...)``.

    Example
    -------
    >>> from gemini_parser import parse_gemini_response
    >>> parsed = parse_gemini_response(response, include_logprobs=True, parse_as_json=True)
    """
    return GeminiResponseParser().parse(
        response,
        include_logprobs=include_logprobs,
        parse_as_json=parse_as_json,
        candidate_index=candidate_index,
    )


# ---------------------------------------------------------------------------
# Quick smoke-test against the sample response in the docstring
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    #  Reconstruct a minimal mock of the SDK response shown in the prompt #
    # ------------------------------------------------------------------ #
    from types import SimpleNamespace

    def ns(**kwargs):
        return SimpleNamespace(**kwargs)

    mock_response = ns(
        model_version="gemini-2.5-pro",
        usage_metadata=ns(
            prompt_token_count=488,
            candidates_token_count=172,
            thoughts_token_count=989,
            total_token_count=1649,
        ),
        candidates=[
            ns(
                avg_logprobs=-1.2799084685569586,
                finish_reason=ns(name="STOP"),
                content=ns(
                    role="model",
                    parts=[
                        ns(
                            thought=True,
                            text=(
                                "Alright, let's get down to it. They've given me a "
                                "hypothesis and a profile – the Data-Centric Engineer…"
                            ),
                        ),
                        ns(
                            thought=False,
                            text=json.dumps(
                                {
                                    "persona_score": 5,
                                    "relevance_score": 5,
                                    "persona_justification": "The hypothesis is a perfect fit.",
                                    "relevance_justifaction": "Highly relevant.",
                                }
                            ),
                        ),
                    ],
                ),
                logprobs_result=ns(
                    chosen_candidates=[
                        ns(token="H", log_probability=-0.028251097),
                        ns(token="ere", log_probability=-0.006435601),
                        ns(token=" is", log_probability=-9.537907e-07),
                    ],
                    top_candidates=[
                        ns(
                            candidates=[
                                ns(token="H", log_probability=-0.028251097),
                                ns(token="The", log_probability=-0.15),
                            ]
                        ),
                    ],
                ),
            )
        ],
    )

    parser = GeminiResponseParser()

    print("=" * 60)
    print("  TEXT MODE  (parse_as_json=False, include_logprobs=False)")
    print("=" * 60)
    parsed = parser.parse(mock_response)
    print(f"finish_reason : {parsed.finish_reason}")
    print(f"model_version : {parsed.model_version}")
    print(f"reasoning     : {parsed.reasoning[:80]}…")
    print(f"answer        : {parsed.answer[:80]}…")
    print(f"answer_json   : {parsed.answer_json}")
    print(f"usage         : {parsed.usage}")
    print()

    print("=" * 60)
    print("  JSON MODE  (parse_as_json=True, include_logprobs=True)")
    print("=" * 60)
    parsed = parser.parse(mock_response, parse_as_json=True, include_logprobs=True)
    print(f"answer_json   : {parsed.answer_json}")
    print(f"logprobs.avg  : {parsed.logprobs.avg_log_prob:.6f}")
    print(f"chosen[0]     : {parsed.logprobs.chosen_candidates[0]}")
    print(f"top[0][0]     : {parsed.logprobs.top_candidates[0][0]}")
    print(f"usage         : {parsed.usage}")