# apps/api/routes/applications.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
import shutil
import os

from storage import models, database # We need to create database.py to manage sessions
from apps.api.schemas import application as app_schemas

router = APIRouter(
    prefix="/api/v1/applications",
    tags=["Applications"]
)

# --- Define upload directory ---
UPLOAD_DIR = "/app/uploads" # This path must be a volume in Docker
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --- Dependency to get DB session ---
def get_db():
    db = database.SessionLocal() # Assuming SessionLocal is defined in storage/database.py
    try:
        yield db
    finally:
        db.close()


@router.post("/applicant", response_model=app_schemas.ApplicantResponse)
def create_applicant(applicant: app_schemas.ApplicantCreate, db: Session = Depends(get_db)):
    # Check if applicant exists
    db_applicant = db.query(models.Applicant).filter(models.Applicant.emirates_id == applicant.emirates_id).first()
    if db_applicant:
        raise HTTPException(status_code=400, detail="Applicant with this Emirates ID already exists")
    
    new_applicant = models.Applicant(**applicant.dict())
    db.add(new_applicant)
    db.commit()
    db.refresh(new_applicant)
    return new_applicant

@router.post("/", response_model=app_schemas.ApplicationResponse)
def create_application(application: app_schemas.ApplicationCreate, db: Session = Depends(get_db)):
    # Check if applicant exists
    db_applicant = db.query(models.Applicant).filter(models.Applicant.id == application.applicant_id).first()
    if not db_applicant:
        raise HTTPException(status_code=404, detail="Applicant not found")
        
    new_application = models.Application(applicant_id=application.applicant_id, status="AWAITING_DOCUMENTS")
    db.add(new_application)
    db.commit()
    db.refresh(new_application)
    return new_application


@router.post("/{application_id}/upload", status_code=201)
async def upload_document(application_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 1. Check if application exists
    db_application = db.query(models.Application).filter(models.Application.id == application_id).first()
    if not db_application:
        raise HTTPException(status_code=404, detail="Application not found")

    # 2. Create a unique path for the file
    # In production, we'd save this to S3 or MongoDB GridFS
    # For this prototype, saving to a named volume is fine.
    file_location = os.path.join(UPLOAD_DIR, f"app_{application_id}_{file.filename}")

    # 3. Save the file
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    # 4. (Next Step) Here, we would trigger the LangGraph workflow
    # For now, we just log it and update status
    
    # We'd also log this document in a 'documents' table in Postgres
    
    print(f"File {file.filename} uploaded for application {application_id}")
    
    # This is a placeholder; eventually, the agent will manage status
    db_application.status = "PENDING"
    db.commit()

    return {"filename": file.filename, "location": file_location}