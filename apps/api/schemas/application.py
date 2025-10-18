# apps/api/schemas/application.py
from pydantic import BaseModel, EmailStr
from typing import Optional

class ApplicantCreate(BaseModel):
    full_name: str
    email: EmailStr
    emirates_id: str
    phone_number: Optional[str] = None

class ApplicantResponse(ApplicantCreate):
    id: int

    class Config:
        orm_mode = True

class ApplicationCreate(BaseModel):
    applicant_id: int

class ApplicationResponse(BaseModel):
    id: int
    applicant_id: int
    status: str

    class Config:
        orm_mode = True