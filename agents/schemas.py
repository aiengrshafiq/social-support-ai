# agents/schemas.py
from pydantic import BaseModel, Field, validator, EmailStr
from typing import Optional, List, Union, Any, Dict
import datetime

class FamilyMember(BaseModel):
    name: Optional[str] = None
    relation: Optional[str] = None

class ValidatedApplicationData(BaseModel):
    # Basic Info (required)
    full_name: str = Field(..., min_length=1, description="Applicant's full name")
    emirates_id: str = Field(..., description="Applicant's Emirates ID number") # Add regex later if needed

    # Contact (optional but recommended)
    email: Optional[EmailStr] = None
    phone_number: Optional[str] = None

    # Extracted Details (potentially optional depending on source docs)
    address: Optional[str] = None
    employer: Optional[str] = None
    total_income: Optional[float] = Field(None, ge=0, description="Estimated monthly income")
    # Add fields for assets, liabilities if extracted from Excel
    assets_liabilities: Optional[List[Dict[str, Any]]] = None
    family_members: Optional[List[FamilyMember]] = []

    # --- Basic Validation Example ---
    @validator('total_income', pre=True, always=True)
    def check_income_plausibility(cls, v):
        if v is not None and v > 1_000_000: # Example: Flag unusually high income for review
             print(f"Warning: High income detected: {v}")
             # In a real system, might raise a specific validation warning
             # or require manual review, but for now just print.
        return v

    class Config:
        extra = 'ignore' # Ignore fields in extracted_data not defined here