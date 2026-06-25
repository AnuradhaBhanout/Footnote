from pydantic import BaseModel,Field
from typing import Optional

class TriageAssessment(BaseModel):
    """
    Structured judgment of whether a user's request is specific enough to act on,
    or needs a clarifying question first. Used via .with_structured_output() so the
    model returns this shape directly - no manual JSON parsing,
    no markdown-fence stripping,
    no silent failures on malformed text.

    """
    is_clear: bool = Field(
        description="True if the request is specific enough to act on right away"
        "False if it is ambiguous, vague,or missing a detail that would change the answer."
    )

    reason: str = Field(
        description="One plain-language sentence explain why the request is clear or unclear"
    )

    clarifying_question: Optional[str] = Field(
        default=None,
        description="""If is_clear is False, a short, friendly question to ask to the user
        to narrow down what they meant. Write it for a non-technical person - no jargon, no assumptions 
        about prior context. Null if is_clear is True
        """
    )

    possible_interpretations: list[str] = Field(
        default_factory=list,
        description="If is_clear is False, list 2-4 short, plain-language phrases describing"
        "the different things the request could plausibly mean.Empty list if is_cleat is True"
    )



class QueryReformulation(BaseModel):
    """
    Used when a search returned poor/insufficient results - ask the
    model to retry with better terms.

    """

    reformulated_query: str = Field(
        description="A rewritten version of the original query using different terms,"
        "broader or narrower scope , or synonyms - aimed at finding better results than the original phrasing did"
    )

    reasoning: str = Field(
        description="One sentence on what changed and why it might work better"
    )