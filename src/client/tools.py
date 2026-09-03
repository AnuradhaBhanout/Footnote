from langchain_core.tools import tool
 
 
@tool
def ask_clarification(question: str) -> str:
    """Call this INSTEAD of any search  tool when the user's request is ambiguous,
       uses an unclear abbreviation/name, or is missing a detail needed  to act.
       Do NOT call this together with any other tool in the same turn.
       Args:
          questions: A plain-language clarifying question for the user.
 
    """
    return "CLARIFICATION_REQUESTED"