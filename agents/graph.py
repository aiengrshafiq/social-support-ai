# agents/graph.py
import os
import json
import httpx
from langgraph.graph import StateGraph, END
from pymongo import MongoClient
from langfuse import Langfuse # Import Langfuse client
from langfuse.decorators import langfuse_context, observe # Import decorators

# Assuming your parser is in parsers/document_parser.py
from parsers.document_parser import extract_data_from_document
from agents.state import AgentState
from apps.api.core.config import settings # For Langfuse keys/host

from pydantic import ValidationError
from agents.schemas import ValidatedApplicationData


from neo4j import GraphDatabase # <-- Add Neo4j driver
from qdrant_client import QdrantClient, models # <-- Add Qdrant client
from sentence_transformers import SentenceTransformer # For embeddings
import numpy as np # For embedding conversion if needed


# --- Initialize Neo4j Connection ---
NEO4J_URI = "bolt://neo4j:7687" # Use Docker service name
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = settings.NEO4J_AUTH.split('/')[1] if settings.NEO4J_AUTH else "password" # Extract password
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
print("Neo4j driver initialized.")

# --- Initialize Qdrant Connection ---
QDRANT_HOST = "qdrant" # Use Docker service name
QDRANT_PORT = 6333
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
print("Qdrant client initialized.")

# --- Initialize Embedding Model (using sentence-transformers) ---
# This will run locally on the CPU within the API container.
# Ensure 'sentence-transformers' is added to requirements.txt if not already present
# Or alternatively, use Ollama's nomic-embed-text via the ollama client
try:
    # Using a small, efficient model suitable for CPU
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Sentence Transformer embedding model loaded.")
    # --- Create Qdrant Collection (if it doesn't exist) ---
    try:
        qdrant_client.create_collection(
            collection_name="applicant_resumes",
            vectors_config=models.VectorParams(size=embedding_model.get_sentence_embedding_dimension(), distance=models.Distance.COSINE)
        )
        print("Created Qdrant collection 'applicant_resumes'.")
    except Exception as e:
        if "already exists" in str(e):
             print("Qdrant collection 'applicant_resumes' already exists.")
        else:
             print(f"Error creating/checking Qdrant collection: {e}")

except Exception as e:
    print(f"Error loading Sentence Transformer model: {e}. Embeddings will not be generated.")
    embedding_model = None

from storage.database import SessionLocal

# --- Initialize MongoDB Connection ---
# Use environment variables for connection string in production
mongo_client = MongoClient("mongodb://mongo:27017/") # Use Docker service name
db = mongo_client["social_support_raw_data"] # Database name
raw_extractions_collection = db["raw_extractions"] # Collection name


@observe()
def persist_data(state: AgentState) -> AgentState:
    """
    Saves the validated application data to PostgreSQL, Neo4j, and Qdrant.
    """
    print(f"--- Running Data Persistence for App ID: {state['application_id']} ---")
    application_id = state["application_id"]
    validated_data = state.get("validated_data")
    applicant_id = state["applicant_id"]
    errors = []

    current_span = langfuse_context.get_current_span()
    if current_span:
        current_span.input({"application_id": application_id, "validated_data_keys": list(validated_data.keys()) if validated_data else []})

    if not validated_data:
        error_msg = "Persistence failed: No validated data found."
        print(error_msg)
        if current_span:
            current_span.level="ERROR"
            current_span.status_message = error_msg
            current_span.output({"status": "failed", "errors": [error_msg]})
        return {**state, "error_message": error_msg}

    # --- 1. Persist to PostgreSQL (Main Record) ---
    pg_session = SessionLocal()
    try:
        # Find the existing application record
        db_application = pg_session.query(models.Application).filter(models.Application.id == application_id).first()
        if db_application:
            # Update the application with the validated data JSON blob
            db_application.validated_data = validated_data
            # Potentially update applicant details if needed (e.g., phone number)
            db_applicant = pg_session.query(models.Applicant).filter(models.Applicant.id == applicant_id).first()
            if db_applicant:
                 # Update fields if they exist in validated_data and are different
                 if 'full_name' in validated_data and validated_data['full_name'] != db_applicant.full_name:
                     db_applicant.full_name = validated_data['full_name']
                 if 'email' in validated_data and validated_data['email'] != db_applicant.email:
                     db_applicant.email = validated_data['email']
                 # ... add other fields like phone_number ...

            pg_session.commit()
            print(f"Application {application_id} updated in PostgreSQL.")
        else:
            errors.append(f"Application {application_id} not found in PostgreSQL.")

    except Exception as e:
        pg_session.rollback()
        error_msg = f"Error persisting to PostgreSQL: {str(e)}"
        print(error_msg)
        errors.append(error_msg)
    finally:
        pg_session.close()

    # --- 2. Persist to Neo4j (Relationships) ---
    try:
        with neo4j_driver.session() as session:
            # Create/Update Applicant Node
            session.run("""
                MERGE (p:Person {emiratesId: $emirates_id})
                ON CREATE SET p.name = $name, p.applicationId = $app_id
                ON MATCH SET p.name = $name, p.applicationId = $app_id
                """,
                emirates_id=validated_data.get('emirates_id'),
                name=validated_data.get('full_name'),
                app_id=application_id)

            # Create Address Node and Relationship (Example)
            if validated_data.get('address'):
                session.run("""
                    MERGE (a:Address {fullAddress: $address})
                    WITH a
                    MATCH (p:Person {emiratesId: $emirates_id})
                    MERGE (p)-[:LIVES_AT]->(a)
                    """,
                    address=validated_data.get('address'),
                    emirates_id=validated_data.get('emirates_id'))

            # Create Family Member Nodes and Relationships (Example)
            for member in validated_data.get('family_members', []):
                 if member.get('name') and member.get('relation'):
                     session.run("""
                         MERGE (fm:Person {name: $member_name}) // Simplistic merge, needs better ID
                         WITH fm
                         MATCH (applicant:Person {emiratesId: $applicant_id})
                         MERGE (applicant)-[:HAS_FAMILY_MEMBER {relation: $relation}]->(fm)
                         """,
                         member_name=member['name'],
                         applicant_id=validated_data.get('emirates_id'),
                         relation=member['relation'])

            print(f"Data for applicant {validated_data.get('emirates_id')} updated in Neo4j.")

    except Exception as e:
        error_msg = f"Error persisting to Neo4j: {str(e)}"
        print(error_msg)
        errors.append(error_msg)

    # --- 3. Persist Embeddings to Qdrant (Example: Resume Text) ---
    # This assumes 'resume_text' is extracted somehow (modify parser/state if needed)
    resume_text = validated_data.get("resume_text", "") # Placeholder
    if resume_text and embedding_model:
        try:
            # Generate embedding
            vector = embedding_model.encode(resume_text).tolist()

            # Upsert into Qdrant
            qdrant_client.upsert(
                collection_name="applicant_resumes",
                points=[
                    models.PointStruct(
                        id=application_id, # Use application ID or a specific resume ID
                        vector=vector,
                        payload={ # Store metadata alongside the vector
                            "applicant_id": applicant_id,
                            "application_id": application_id,
                            "file_type": "resume" # Or identify source
                        }
                    )
                ],
                wait=True # Ensure operation completes
            )
            print(f"Resume embedding for application {application_id} saved to Qdrant.")

        except Exception as e:
            error_msg = f"Error saving embedding to Qdrant: {str(e)}"
            print(error_msg)
            errors.append(error_msg)


    # --- Update State ---
    final_message = "Persistence completed."
    if errors:
        final_message = "Persistence finished with errors:\n" + "\n".join(errors)
        print(final_message)
        if current_span:
            current_span.level="ERROR"
            current_span.status_message = "Persistence Errors"
            current_span.output({"status": "errors", "errors": errors})
        # Keep existing error message if validation failed, otherwise add persistence errors
        current_error = state.get("error_message")
        return {**state, "error_message": f"{current_error}\n{final_message}" if current_error else final_message}
    else:
        print(final_message)
        if current_span:
            current_span.output({"status": "success"})
        # Clear error state if persistence was successful after validation was successful
        return {**state, "error_message": None}

# --- Initialize Langfuse Client ---
# Use the client directly from settings if available, else initialize here
# Ensure .env has LANGFUSE_HOST, LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY for cloud
langfuse_client = None
if settings.LANGFUSE_HOST and settings.LANGFUSE_SECRET_KEY and settings.LANGFUSE_PUBLIC_KEY:
    try:
        langfuse_client = Langfuse(
            host=str(settings.LANGFUSE_HOST), # Ensure host is string
            secret_key=settings.LANGFUSE_SECRET_KEY,
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            release="social-support-ai-v1.0"
        )
        print("Langfuse client for Graph initialized.")
    except Exception as e:
        print(f"Failed to initialize Langfuse client in Graph: {e}")
        langfuse_client = None
else:
    print("Langfuse env vars not set in Graph, tracing disabled.")

# --- Helper to get Langfuse context ---
def get_langfuse_trace(application_id):
    if langfuse_client:
        # Create a unique trace for each application run
        return langfuse_client.trace(name="application-workflow", user_id=f"app-{application_id}")
    return None

# --- Data Extraction Agent Node ---
# Use @observe for automatic Langfuse tracing if client is available
@observe()
def run_data_extraction(state: AgentState) -> AgentState:
    """
    Extracts data from uploaded documents using the multimodal parser.
    """
    print(f"--- Running Data Extraction for App ID: {state['application_id']} ---")
    application_id = state["application_id"]
    uploaded_files = state["uploaded_files"]
    all_extracted_data = {}
    errors = []

    # Get the current Langfuse trace/span context if available
    current_span = langfuse_context.get_current_span()
    
    # Input metadata for Langfuse
    if current_span:
        current_span.input({"application_id": application_id, "files": list(uploaded_files.keys())})

    for file_type, file_path in uploaded_files.items():
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path} for type {file_type}"
            print(error_msg)
            errors.append(error_msg)
            continue

        print(f"Processing {file_type}: {file_path}...")
        
        # --- Langfuse Step for each document ---
        extraction_step = None
        if current_span:
             extraction_step = current_span.step(name=f"extract-{file_type}", input={"file_path": file_path})

        try:
            # Call the parser function (sync version for simplicity now)
            # In production, use async/await if parser supports it
            extracted = extract_data_from_document(file_path)
            
            # --- Update Langfuse Step ---
            if extraction_step:
                extraction_step.output(extracted)
                extraction_step.end() # Mark step as completed

            if "_error" in extracted:
                error_msg = f"Error extracting {file_type}: {extracted['_error']}"
                print(error_msg)
                errors.append(error_msg)
                # Store the raw error output too
                all_extracted_data[f"{file_type}_raw_error"] = extracted.get("_raw_output", extracted['_error'])
            else:
                all_extracted_data[file_type] = extracted
                print(f"Successfully extracted data for {file_type}.")

        except Exception as e:
            error_msg = f"Unhandled exception processing {file_type}: {str(e)}"
            print(error_msg)
            errors.append(error_msg)
            # --- Update Langfuse Step on unhandled error ---
            if extraction_step:
                 extraction_step.level = "ERROR"
                 extraction_step.status_message = error_msg
                 extraction_step.end() # Mark step as ended (with error)


    # --- Store Raw Results in MongoDB ---
    try:
        raw_extractions_collection.insert_one({
            "application_id": application_id,
            "extracted_data": all_extracted_data,
            "errors": errors,
            "timestamp": datetime.datetime.utcnow()
        })
        print("Raw extraction results saved to MongoDB.")
    except Exception as e:
        mongo_error = f"Failed to save raw extractions to MongoDB: {str(e)}"
        print(mongo_error)
        errors.append(mongo_error) # Add DB error to the list

    # --- Update State ---
    # Merge extracted data into a single dictionary if needed, or keep separate
    final_data = {}
    for key, data in all_extracted_data.items():
        if isinstance(data, dict): # Avoid merging raw error strings
            final_data.update(data) # Simple merge for V1
            
    # Output metadata for Langfuse
    if current_span:
        current_span.output({"extracted_data_keys": list(final_data.keys()), "errors": errors})

    # Decide next step based on errors
    if errors:
        print("Errors occurred during extraction.")
        return {**state, "error_message": "\n".join(errors), "extracted_data": final_data} # Pass partial data
    else:
        print("Data extraction completed successfully.")
        return {**state, "extracted_data": final_data, "error_message": None}


# --- Data Validation Agent Node (NEW) ---
@observe() # Trace this node with Langfuse
def run_data_validation(state: AgentState) -> AgentState:
    """
    Validates the extracted data against schema and business rules.
    """
    print(f"--- Running Data Validation for App ID: {state['application_id']} ---")
    application_id = state["application_id"]
    extracted_data = state.get("extracted_data")
    validation_errors = []

    current_span = langfuse_context.get_current_span()
    if current_span:
        current_span.input({"application_id": application_id, "extracted_data_keys": list(extracted_data.keys()) if extracted_data else []})

    if not extracted_data:
        error_msg = "Validation failed: No extracted data found in state."
        print(error_msg)
        if current_span:
            current_span.level = "ERROR"
            current_span.status_message = error_msg
            current_span.output({"validation_status": "failed", "errors": [error_msg]})
        return {**state, "error_message": error_msg}

    # 1. Pydantic Schema Validation
    try:
        # Attempt to parse the extracted data using our Pydantic model
        # This checks required fields, types, and basic validators
        validated_pydantic = ValidatedApplicationData(**extracted_data)
        print("Pydantic schema validation successful.")
        # Store the Pydantic-validated data (includes defaults, type coercion)
        validated_data_dict = validated_pydantic.model_dump(exclude_unset=True) # Use model_dump

    except ValidationError as e:
        error_msg = f"Pydantic validation failed: {e}"
        print(error_msg)
        validation_errors.append(error_msg)
        # Even if Pydantic fails, keep the raw extracted data for potential partial use or debugging
        validated_data_dict = extracted_data # Fallback to raw data on schema failure

    # 2. Custom Business Rule Checks (Add more as needed)
    # Example: Check if income is present if an employer is listed
    if validated_data_dict.get("employer") and not validated_data_dict.get("total_income"):
        warning_msg = "Validation Warning: Employer listed but no income extracted."
        print(warning_msg)
        # For V1, we'll just log warnings, not fail the validation
        # validation_errors.append(warning_msg)

    # Example: Check for consistency between different document sources (if available)
    # This requires modifying the state/parser to store source info
    # e.g., if 'address_from_id' in validated_data_dict and 'address_from_statement' in validated_data_dict:
    #    if validated_data_dict['address_from_id'] != validated_data_dict['address_from_statement']:
    #        error_msg = "Address mismatch between ID and bank statement."
    #        validation_errors.append(error_msg)


    # --- Update State ---
    if validation_errors:
        final_error_message = "Validation Failed:\n" + "\n".join(validation_errors)
        print(final_error_message)
        if current_span:
            current_span.level = "ERROR"
            current_span.status_message = "Validation Failed"
            current_span.output({"validation_status": "failed", "errors": validation_errors, "validated_data": validated_data_dict}) # Log partially validated data
        # Keep partially validated data, set error
        return {**state, "validated_data": validated_data_dict, "error_message": final_error_message}
    else:
        print("Data validation completed successfully.")
        if current_span:
             current_span.output({"validation_status": "success", "validated_data": validated_data_dict})
        # Store fully validated data, clear error
        return {**state, "validated_data": validated_data_dict, "error_message": None}



# --- Eligibility Check Agent Node (NEW) ---
@observe()
async def check_eligibility(state: AgentState) -> AgentState:
    """
    Prepares features and calls the ML service to get an eligibility prediction.
    """
    print(f"--- Running Eligibility Check for App ID: {state['application_id']} ---")
    application_id = state["application_id"]
    validated_data = state.get("validated_data")
    errors = []

    current_span = langfuse_context.get_current_span()
    if current_span:
        current_span.input({"application_id": application_id, "validated_data_keys": list(validated_data.keys()) if validated_data else []})

    if not validated_data:
        error_msg = "Eligibility check failed: No validated data found."
        print(error_msg)
        if current_span:
            current_span.level="ERROR"; current_span.status_message = error_msg
            current_span.output({"status": "failed", "errors": [error_msg]})
        return {**state, "error_message": error_msg}

    # --- 1. Prepare Features ---
    # Extract features needed by the model from validated_data
    # Use default values or handle missing data appropriately
    features = {
        "income": validated_data.get("total_income", 0.0), # Default to 0 if missing
        "family_size": len(validated_data.get("family_members", [])) + 1, # Applicant + members
        # Add other features based on EXPECTED_FEATURES in serve_model.py
    }
    print(f"Prepared features: {features}")

    # --- 2. Call ML Service ---
    ml_service_url = "http://ml_service:8001/predict" # Use Docker service name
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(ml_service_url, json=features)
            response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
            prediction_result = response.json()
            print(f"ML Service Response: {prediction_result}")

            eligibility_decision = prediction_result.get("decision", "Error")
            eligibility_score = prediction_result.get("probability")

            if current_span:
                 current_span.output({"status": "success", "prediction": prediction_result})

            # Update state with decision and score
            return {
                **state,
                "eligibility_decision": eligibility_decision,
                "eligibility_score": eligibility_score,
                "error_message": None # Clear previous errors if check is successful
            }

    except httpx.RequestError as e:
        error_msg = f"Error calling ML service: {e}"
        print(error_msg)
        errors.append(error_msg)
    except Exception as e:
        error_msg = f"Error processing ML service response: {e}"
        print(error_msg)
        errors.append(error_msg)

    # --- Handle errors ---
    final_error_message = "Eligibility check failed:\n" + "\n".join(errors)
    if current_span:
        current_span.level="ERROR"; current_span.status_message = "Eligibility Check Failed"
        current_span.output({"status": "failed", "errors": errors})
    current_error = state.get("error_message")
    return {**state, "error_message": f"{current_error}\n{final_error_message}" if current_error else final_error_message}

# --- Update Workflow Graph Definition ---
workflow = StateGraph(AgentState)

# Define the nodes
workflow.add_node("extract_data", run_data_extraction)
workflow.add_node("validate_data", run_data_validation) # Add the new node
workflow.add_node("persist_data", persist_data)
workflow.add_node("check_eligibility", check_eligibility) # Add eligibility check node

# Define the entry point
workflow.set_entry_point("extract_data")

# Define edges
# After extraction, decide where to go based on errors
def decide_after_extraction(state: AgentState) -> str:
    if state.get("error_message") and not state.get("extracted_data"): # Major extraction failure
        print("Extraction failed critically. Ending workflow.")
        return "end_workflow"
    elif state.get("error_message"): # Partial extraction success
        print("Extraction had errors, proceeding to validation with partial data.")
        return "validate_data"
    else:
        print("Extraction successful. Proceeding to validation.")
        return "validate_data"

workflow.add_conditional_edges(
    "extract_data",
    decide_after_extraction,
    {
        "validate_data": "validate_data",
        "end_workflow": END
    }
)



# Edges after validation
def decide_after_validation(state: AgentState) -> str:
    if state.get("error_message"):
        print("Validation failed. Ending workflow.")
        # Optionally, still try to persist the (partially) validated data for logging/review
        # return "persist_data" # Uncomment to persist even on validation failure
        return "end_workflow" # End immediately on validation failure
    else:
        print("Validation successful. Proceeding to persistence.")
        return "persist_data" # <-- Go to persist_data on success

workflow.add_conditional_edges(
    "validate_data",
    decide_after_validation,
    {
        "persist_data": "persist_data", # <-- New success path
        "end_workflow": END
    }
)

# Edge after persistence -> Go to eligibility check
workflow.add_conditional_edges(
    "persist_data",
    lambda state: "error" if state.get("error_message") else "success",
    {
        "success": "check_eligibility", # <-- Go to eligibility on success
        "error": END # End if persistence failed
    }
)

# Edge after persistence
# For now, end the workflow after attempting persistence
workflow.add_edge("check_eligibility", END)

# Compile the graph
app_graph = workflow.compile()

# --- Ensure all imports are present ---
import datetime # Should be at the top already
from storage import models # Import your SQLAlchemy models for Postgres query