# apps/api/schemas/application.py
from pydantic import BaseModel, EmailStr, ConfigDict
from typing import Optional

class ApplicantBase(BaseModel):
    full_name: str
    email: EmailStr
    emirates_id: str
    phone_number: Optional[str] = None

class ApplicantCreate(ApplicantBase):
    pass

class ApplicantResponse(ApplicantBase):
    id: int
    
    # --- THIS IS THE FIX ---
    model_config = ConfigDict(from_attributes=True)


class ApplicationBase(BaseModel):
    applicant_id: int

class ApplicationCreate(ApplicationBase):
    pass

class ApplicationResponse(ApplicationBase):
    id: int
    status: str

    # --- THIS IS THE FIX ---
    model_config = ConfigDict(from_attributes=True)