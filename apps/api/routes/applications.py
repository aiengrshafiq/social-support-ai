# apps/api/routes/applications.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
import shutil
import os
import uuid # For unique filenames
from storage import database

from storage import models, database
from apps.api.schemas import application as app_schemas
from agents.graph import app_graph, get_langfuse_trace, langfuse_client # Import the compiled graph and trace helper
from agents.state import AgentState # Import the state definition




router = APIRouter(
    prefix="/api/v1/applications",
    tags=["Applications"]
)

# --- Define upload directory ---
# Make sure this directory exists and is accessible by the container
UPLOAD_DIR = "/app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --- Dependency to get DB session ---
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Function to run the graph in the background ---
def run_graph_background(initial_state: AgentState):
    """Invokes the LangGraph app asynchronously."""
    print(f"Starting background graph execution for App ID: {initial_state['application_id']}")
    
    # --- Langfuse Trace ---
    trace = get_langfuse_trace(initial_state['application_id'])
    
    # Configuration for the graph run, including the trace context
    config = {}
    if trace:
        config["configurable"] = {"trace": trace} # Pass trace object if available

    try:
        # Stream events (good for debugging, optional for background task)
        # for event in app_graph.stream(initial_state, config=config):
        #     print(f"Graph Event: {event}")
            
        # Or just invoke and get the final state (simpler for background)
        final_state = app_graph.invoke(initial_state, config=config)
        print(f"Graph execution finished for App ID: {initial_state['application_id']}. Final state keys: {final_state.keys()}")
        
        # --- Update DB based on final state (Optional Here) ---
        # You might update the application status in Postgres here based on final_state
        # db = database.SessionLocal()
        # try:
        #    db_app = db.query(models.Application).filter(models.Application.id == initial_state['application_id']).first()
        #    if db_app:
        #        if final_state.get("error_message"):
        #            db_app.status = models.ApplicationStatus.VALIDATION_FAILED # Or a new EXTRACTION_FAILED status
        #        # Add more status updates based on graph progress
        #        db.commit()
        # finally:
        #    db.close()

    except Exception as e:
        print(f"Error during graph execution for App ID {initial_state['application_id']}: {e}")
        # Optionally update DB status to indicate failure
    finally:
        # --- Ensure Langfuse trace is ended and client flushed ---
        status_message = "completed" if final_state is not None else "error"
        if trace:
            trace.update(output={"status": status_message})
        # --- FLUSH THE IMPORTED CLIENT ---
        if langfuse_client: # Check if client was initialized successfully
            langfuse_client.flush()
            print("Langfuse client flushed.")

@router.post("/{application_id}/upload")
# Add BackgroundTasks dependency
async def upload_document(application_id: int, file_type: str, background_tasks: BackgroundTasks, file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 1. Check if application exists
    db_application = db.query(models.Application).filter(models.Application.id == application_id).first()
    if not db_application:
        raise HTTPException(status_code=404, detail="Application not found")
    if not db_application.applicant_id:
         raise HTTPException(status_code=400, detail="Application is missing applicant ID")


    # 2. Save the file with a unique name to prevent overwrites
    _, file_extension = os.path.splitext(file.filename)
    unique_filename = f"app_{application_id}_{file_type}_{uuid.uuid4()}{file_extension}"
    file_location = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        await file.close() # Use await for async file closing

    print(f"File {unique_filename} uploaded for application {application_id} as type {file_type}")

    # --- Store file info temporarily (e.g., in Redis or add to Postgres JSON) ---
    # For V1, we'll assume we know the necessary files are uploaded
    # In a real app, you'd track uploads until all required docs are present
    
    # --- Check if this is the 'last' required file (Simplified Check for V1) ---
    # Assume for now that uploading any file triggers the processing
    # A real check would look at db_application's status or stored file list
    
    # --- Prepare Initial State & Trigger Graph ---
    # We need to gather *all* uploaded file paths for this application ID.
    # This is tricky without a proper document tracking table or storing paths in the Application model.
    # **V1 Simplification:** Assume for now we only process the *currently uploaded* file.
    # **Better V2:** Query a 'documents' table or update Application.uploaded_files JSON field.
    
    initial_files = {file_type: file_location} # V1: Just process the current file
    
    initial_state = AgentState(
        application_id=application_id,
        applicant_id=db_application.applicant_id,
        uploaded_files=initial_files,
        extracted_data=None,
        error_message=None,
        validated_data=None,
        eligibility_decision=None,
        eligibility_score=None,
        final_recommendation=None
    )

    # Add the graph execution to background tasks
    background_tasks.add_task(run_graph_background, initial_state)
    print(f"Added graph execution to background tasks for App ID: {application_id}")

    # Update application status
    db_application.status = models.ApplicationStatus.PENDING # Or PROCESSING
    db.commit()

    return {"filename": unique_filename, "file_type": file_type, "message": "File uploaded and processing started."}

# --- Add other endpoints (create_applicant, create_application) as before ---
@router.post("/applicant", response_model=app_schemas.ApplicantResponse)
def create_applicant(applicant: app_schemas.ApplicantCreate, db: Session = Depends(get_db)):
    db_applicant = db.query(models.Applicant).filter(models.Applicant.emirates_id == applicant.emirates_id).first()
    if db_applicant:
        raise HTTPException(status_code=400, detail="Applicant with this Emirates ID already exists")
    
    new_applicant = models.Applicant(**applicant.model_dump()) # Use model_dump for Pydantic v2
    db.add(new_applicant)
    db.commit()
    db.refresh(new_applicant)
    return new_applicant

@router.post("/", response_model=app_schemas.ApplicationResponse)
def create_application(application: app_schemas.ApplicationCreate, db: Session = Depends(get_db)):
    db_applicant = db.query(models.Applicant).filter(models.Applicant.id == application.applicant_id).first()
    if not db_applicant:
        raise HTTPException(status_code=404, detail="Applicant not found")
        
    new_application = models.Application(applicant_id=application.applicant_id, status=models.ApplicationStatus.AWAITING_DOCUMENTS)
    db.add(new_application)
    db.commit()
    db.refresh(new_application)
    return new_application