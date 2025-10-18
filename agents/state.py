# agents/state.py
from typing import TypedDict, List, Dict, Optional, Any

class AgentState(TypedDict):
    """
    Represents the state of the social support application workflow.
    """
    application_id: int
    applicant_id: int
    uploaded_files: Dict[str, str] # { "emirates_id": "/path/to/id.png", "bank_statement": "/path/to/statement.pdf", ... }
    
    # Data extracted by the extraction agent
    extracted_data: Optional[Dict[str, Any]]
    
    # Errors encountered during extraction or validation
    error_message: Optional[str]

    # Final validated data (set by persistence agent)
    validated_data: Optional[Dict[str, Any]]

    # Eligibility decision
    eligibility_decision: Optional[str] # e.g., "Approve", "Decline", "Review"
    eligibility_score: Optional[float]

    # Final recommendation message
    final_recommendation: Optional[str]