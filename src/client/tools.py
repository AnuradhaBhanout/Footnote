from langchain_core.tools import tool
 
 
@tool
def ask_clarification(question: str, options: list[str] | None = None) -> str:
    """Call this INSTEAD of any search  tool when the user's request is ambiguous,
       uses an unclear abbreviation/name, or is missing a detail needed  to act.
       Do NOT call this together with any other tool in the same turn.
       Args:
          questions: A plain-language clarifying question for the user.
          options: 2-4 short possible interpretations to help the user answer quickly.
 
    """
    options = options or []
    return "CLARIFICATION_REQUESTED"