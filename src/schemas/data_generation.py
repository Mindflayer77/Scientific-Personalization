from pydantic import BaseModel, Field

class SyntheticUserPrompt(BaseModel):
    generated_prompt_text: str = Field(..., description="The actual text prompt the user types into the system")
    prompt_type: str = Field(..., description="Whether the prompt is single-hop or multi-hop.")

class ExtractedConcepts(BaseModel):
    core_concepts: list[str] = Field(..., description="List of core concepts extracted from the user prompt.")

class HypothesisResponse(BaseModel):
    is_answerable: bool = Field(..., description="True if a valid hypothesis can be formed based on provided context.")
    hypothesis_statement: str = Field(..., description="The full scientific statement.")
    falsification_criteria: str = Field(..., description="Experiment outcome that disproves this hypothesis.")
    
    
class PersonaClassification(BaseModel):
    scores: dict[str, int] = Field(..., description="Dictionary mapping persona_id to a discrete score (0-5) indicating how well the article matches the persona.")

