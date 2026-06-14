from enum import StrEnum

from pydantic import BaseModel, Field


class PersonaAnswerRelevance(BaseModel):
    persona_relevance_score: int
    answer_relevance_score: int
    persona_relevance_justification: str
    answer_relevance_justifaction: str

class ContextRelevanceFaithfulness(BaseModel):
    context_relevance_score: int
    faithfulness_score: int
    context_relevance_justification: str
    faithfulness_justification: str


class JudgmentWinner(StrEnum):
    A = "A"
    B = "B"
    tie = "tie"


class JudgmentResponse(BaseModel):
    winner: JudgmentWinner = Field(
        ...,
        description=(
            "'A' if Hypothesis A better fits the researcher persona, "
            "'B' if Hypothesis B is better, 'tie' if both are equal."
        ),
    )
    justification: str = Field(
        ...,
        description=(
            "One or two sentences explaining why the chosen hypothesis "
            "better matches the persona's interests and style."
        ),
    )