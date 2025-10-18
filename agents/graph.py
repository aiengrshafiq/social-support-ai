# agents/graph.py
import os
import json
import httpx
import datetime
import numpy as np # For embedding conversion if needed
from typing import TypedDict, List, Dict, Optional, Any # Ensure Any is imported

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END

# --- Database Imports ---
from pymongo import MongoClient
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from sqlalchemy.orm import Session # For type hinting if needed later
from storage.database import SessionLocal # For getting PG session
from storage import models as pg_models # Import SQLAlchemy models

# --- Langfuse Imports ---

import langfuse # <-- Import the main library
from langfuse.decorators import observe # <-- Keep observe decorator
from langfuse import Langfuse as LangfuseClient # Rename client class to avoid conflict


# --- Project Imports ---
from parsers.document_parser import extract_data_from_document
from agents.state import AgentState
from agents.schemas import ValidatedApplicationData
from apps.api.core.config import settings

# --- Pydantic Imports ---
from pydantic import ValidationError

# --- Embedding Imports ---
# Choose one method:
# 1. Sentence Transformers (local CPU/GPU)
try:
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()
    print("Sentence Transformer embedding model loaded.")
except ImportError:
    print("Sentence Transformers not installed. Run 'pip install sentence-transformers'. Embeddings disabled.")
    embedding_model = None
    EMBEDDING_DIM = 384 # Default dimension for 'all-MiniLM-L6-v2' if model fails to load but collection exists
except Exception as e:
    print(f"Error loading Sentence Transformer model: {e}. Embeddings disabled.")
    embedding_model = None
    EMBEDDING_DIM = 384

# --- Database Client Initializations ---

# MongoDB
mongo_client = MongoClient("mongodb://mongo:27017/")
db = mongo_client["social_support_raw_data"]
raw_extractions_collection = db["raw_extractions"]
print("MongoDB client initialized.")

# Neo4j
NEO4J_URI = "bolt://neo4j:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = settings.NEO4J_AUTH.split('/')[1] if settings.NEO4J_AUTH and '/' in settings.NEO4J_AUTH else "password" # Safer split
try:
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_driver.verify_connectivity() # Check connection on startup
    print("Neo4j driver initialized and connected.")
except Exception as e:
    print(f"Error initializing Neo4j driver: {e}")
    neo4j_driver = None # Handle potential connection errors

# Qdrant
QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "applicant_resumes"
try:
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10) # Add timeout
    # Check connection / collection existence
    try:
         qdrant_client.get_collection(collection_name=QDRANT_COLLECTION)
         print(f"Qdrant collection '{QDRANT_COLLECTION}' found.")
    except Exception as e_coll:
        if "404" in str(e_coll): # Check specific error for collection not found
            print(f"Qdrant collection '{QDRANT_COLLECTION}' not found. Creating...")
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE)
            )
            print(f"Created Qdrant collection '{QDRANT_COLLECTION}'.")
        else:
            print(f"Error checking Qdrant collection: {e_coll}")
            # Potentially raise error or disable Qdrant functionality
    print("Qdrant client initialized.")
except Exception as e:
    print(f"Error initializing Qdrant client: {e}. Qdrant operations disabled.")
    qdrant_client = None


# --- Langfuse Client Initialization ---
langfuse_client = None
if settings.LANGFUSE_HOST and settings.LANGFUSE_SECRET_KEY and settings.LANGFUSE_PUBLIC_KEY:
    try:
        langfuse_client = LangfuseClient(
            host=str(settings.LANGFUSE_HOST),
            secret_key=settings.LANGFUSE_SECRET_KEY,
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            release="social-support-ai-v1.0",
            flush_interval=1 # Flush more frequently for background tasks
        )
        print("Langfuse client for Graph initialized.")
    except Exception as e:
        print(f"Failed to initialize Langfuse client in Graph: {e}")
        langfuse_client = None
else:
    print("Langfuse env vars not set in Graph, tracing disabled.")

# --- Langfuse Helper ---
def get_langfuse_trace(application_id):
    if langfuse_client:
        return langfuse_client.trace(name="application-workflow", user_id=f"app-{application_id}")
    return None

# ==================================
# == Agent Nodes ==
# ==================================

@observe() # Decorator handles main span
def run_data_extraction(state: AgentState) -> Dict[str, Any]: # Return explicit Dict for clarity
    """
    Extracts data from uploaded documents.
    Uses sub-steps for each document.
    """
    print(f"--- Running Data Extraction for App ID: {state['application_id']} ---")
    application_id = state["application_id"]
    uploaded_files = state["uploaded_files"]
    all_extracted_data = {}
    errors = []

    for file_type, file_path in uploaded_files.items():
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path} for type {file_type}"
            print(error_msg)
            errors.append(error_msg)
            continue

        print(f"Processing {file_type}: {file_path}...")

        try:
            # Use context manager for step - automatically linked to parent @observe span
            with langfuse.get_current_observation().step(name=f"extract-{file_type}", input={"file_path": file_path}) as extraction_step:
                try:
                    extracted = extract_data_from_document(file_path)
                    extraction_step.output(extracted) # Log output for this step

                    if "_error" in extracted:
                        error_msg = f"Error extracting {file_type}: {extracted['_error']}"
                        print(error_msg)
                        errors.append(error_msg)
                        all_extracted_data[f"{file_type}_raw_error"] = extracted.get("_raw_output", extracted['_error'])
                        extraction_step.level = "ERROR"
                        extraction_step.status_message = error_msg
                    else:
                        all_extracted_data[file_type] = extracted
                        print(f"Successfully extracted data for {file_type}.")

                except Exception as e:
                    error_msg = f"Unhandled exception processing {file_type}: {str(e)}"
                    print(error_msg)
                    errors.append(error_msg)
                    extraction_step.level = "ERROR"
                    extraction_step.status_message = error_msg
                    # Optionally re-raise if needed: raise e

        except Exception as e_ctx:
             print(f"Error setting up Langfuse step context for {file_type}: {e_ctx}")
             # Log error but continue with other files if possible
             errors.append(f"Langfuse context error for {file_type}: {e_ctx}")


    # --- Store Raw Results in MongoDB ---
    try:
        raw_extractions_collection.insert_one({
            "application_id": application_id,
            "extracted_data": all_extracted_data, # Store potentially mixed results
            "errors": errors, # Store errors encountered
            "timestamp": datetime.datetime.utcnow()
        })
        print("Raw extraction results saved to MongoDB.")
    except Exception as e:
        mongo_error = f"Failed to save raw extractions to MongoDB: {str(e)}"
        print(mongo_error)
        errors.append(mongo_error)

    # --- Prepare return state ---
    final_data = {}
    for key, data in all_extracted_data.items():
        if isinstance(data, dict):
            final_data.update(data) # Simple merge

    # @observe automatically logs return value as output
    if errors:
        print("Errors occurred during extraction.")
        # Combine errors into a single message for the state
        error_summary = "Extraction Errors:\n" + "\n".join(errors)
        return {"extracted_data": final_data, "error_message": error_summary}
    else:
        print("Data extraction completed successfully.")
        return {"extracted_data": final_data, "error_message": None} # Clear errors


@observe()
def run_data_validation(state: AgentState) -> Dict[str, Any]:
    """
    Validates the extracted data against schema and business rules.
    """
    print(f"--- Running Data Validation for App ID: {state['application_id']} ---")
    extracted_data = state.get("extracted_data")
    validation_errors = []
    validated_data_dict = {} # Initialize

    if not extracted_data:
        error_msg = "Validation failed: No extracted data found in state."
        print(error_msg)
        langfuse.update_current_observation(level="ERROR", status_message=error_msg)
        return {"error_message": error_msg} # Return only error

    try:
        # Pydantic Validation
        validated_pydantic = ValidatedApplicationData(**extracted_data)
        validated_data_dict = validated_pydantic.model_dump(exclude_unset=True)
        print("Pydantic schema validation successful.")
    except ValidationError as e:
        error_msg = f"Pydantic validation failed: {e}"
        print(error_msg)
        validation_errors.append(error_msg)
        validated_data_dict = extracted_data # Fallback to raw on schema failure for potential partial persistence

    # --- Custom Business Rule Checks ---
    if validated_data_dict.get("employer") and not validated_data_dict.get("total_income"):
        warning_msg = "Validation Warning: Employer listed but no income extracted."
        print(warning_msg)
        # We don't add warnings to errors for now

    # ... Add more consistency checks here (e.g., address matching) ...

    # --- Prepare return state ---
    if validation_errors:
        final_error_message = "Validation Failed:\n" + "\n".join(validation_errors)
        print(final_error_message)
        langfuse.update_current_observation(level="ERROR", status_message="Validation Failed", output={"errors": validation_errors})
        # Return partially validated data along with error
        return {"validated_data": validated_data_dict, "error_message": final_error_message}
    else:
        print("Data validation completed successfully.")
        # Return fully validated data, clear errors from this stage
        return {"validated_data": validated_data_dict, "error_message": None}


@observe()
def persist_data(state: AgentState) -> Dict[str, Any]:
    """
    Saves the validated application data to PostgreSQL, Neo4j, and Qdrant.
    """
    print(f"--- Running Data Persistence for App ID: {state['application_id']} ---")
    application_id = state["application_id"]
    validated_data = state.get("validated_data")
    applicant_id = state["applicant_id"]
    errors = []

    if not validated_data:
        error_msg = "Persistence failed: No validated data found."
        print(error_msg)
        langfuse.update_current_observation(level="ERROR", status_message=error_msg)
        return {"error_message": error_msg}

    # --- 1. Persist to PostgreSQL ---
    pg_session = SessionLocal()
    try:
        with pg_session.begin(): # Use transaction block
            db_application = pg_session.query(pg_models.Application).filter(pg_models.Application.id == application_id).with_for_update().first() # Lock row
            if db_application:
                db_application.validated_data = validated_data
                db_applicant = pg_session.query(pg_models.Applicant).filter(pg_models.Applicant.id == applicant_id).first()
                if db_applicant:
                     # Update fields safely
                     db_applicant.full_name = validated_data.get('full_name', db_applicant.full_name)
                     db_applicant.email = validated_data.get('email', db_applicant.email)
                     db_applicant.phone_number = validated_data.get('phone_number', db_applicant.phone_number)
                print(f"Application {application_id} updated in PostgreSQL.")
            else:
                errors.append(f"Application {application_id} not found in PostgreSQL.")
    except Exception as e:
        error_msg = f"Error persisting to PostgreSQL: {str(e)}"
        print(error_msg)
        errors.append(error_msg)
    finally:
        pg_session.close()

    # --- 2. Persist to Neo4j ---
    if neo4j_driver: # Check if driver initialized correctly
        try:
            with neo4j_driver.session() as session:
                # Use write_transaction for atomicity
                session.write_transaction(update_neo4j_graph, validated_data, application_id)
                print(f"Data for applicant {validated_data.get('emirates_id')} updated in Neo4j.")
        except Exception as e:
            error_msg = f"Error persisting to Neo4j: {str(e)}"
            print(error_msg)
            errors.append(error_msg)
    else:
        errors.append("Neo4j driver not available.")

    # --- 3. Persist Embeddings to Qdrant ---
    resume_text = validated_data.get("resume_text", "") # Placeholder - Need actual resume data
    if resume_text and embedding_model and qdrant_client:
        try:
            vector = embedding_model.encode(resume_text).tolist()
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=[
                    models.PointStruct(
                        id=application_id, vector=vector,
                        payload={"applicant_id": applicant_id, "application_id": application_id}
                    )
                ],
                wait=True
            )
            print(f"Resume embedding for application {application_id} saved to Qdrant.")
        except Exception as e:
            error_msg = f"Error saving embedding to Qdrant: {str(e)}"
            print(error_msg)
            errors.append(error_msg)
    elif resume_text and not embedding_model:
        errors.append("Resume text found but embedding model not loaded.")
    elif resume_text and not qdrant_client:
        errors.append("Resume text found but Qdrant client not available.")


    # --- Prepare return state ---
    if errors:
        final_message = "Persistence finished with errors:\n" + "\n".join(errors)
        print(final_message)
        langfuse.update_current_observation(level="ERROR", status_message="Persistence Errors", output={"errors": errors})
        # Keep previous error message if validation failed, otherwise use persistence errors
        current_error = state.get("error_message")
        return {"error_message": f"{current_error}\n{final_message}" if current_error else final_message}
    else:
        print("Persistence completed successfully.")
        # Clear error if persistence succeeds
        return {"error_message": None}

# Helper Function for Neo4j Transaction
def update_neo4j_graph(tx, validated_data, application_id):
    # Applicant Node
    tx.run("""
        MERGE (p:Person {emiratesId: $emirates_id})
        ON CREATE SET p.name = $name, p.applicationId = $app_id
        ON MATCH SET p.name = $name, p.applicationId = $app_id
        """,
        emirates_id=validated_data.get('emirates_id'),
        name=validated_data.get('full_name'),
        app_id=application_id)

    # Address Node and Relationship
    if validated_data.get('address'):
        tx.run("""
            MERGE (a:Address {fullAddress: $address})
            WITH a
            MATCH (p:Person {emiratesId: $emirates_id})
            MERGE (p)-[:LIVES_AT]->(a)
            """,
            address=validated_data.get('address'),
            emirates_id=validated_data.get('emirates_id'))

    # Family Member Nodes and Relationships
    for member in validated_data.get('family_members', []):
         if member.get('name') and member.get('relation'):
             tx.run("""
                 MERGE (fm:Person {name: $member_name}) // Simplistic merge
                 WITH fm
                 MATCH (applicant:Person {emiratesId: $applicant_id})
                 MERGE (applicant)-[:HAS_FAMILY_MEMBER {relation: $relation}]->(fm)
                 """,
                 member_name=member['name'],
                 applicant_id=validated_data.get('emirates_id'),
                 relation=member['relation'])


@observe()
async def check_eligibility(state: AgentState) -> Dict[str, Any]:
    """
    Prepares features and calls the ML service for eligibility prediction.
    """
    print(f"--- Running Eligibility Check for App ID: {state['application_id']} ---")
    validated_data = state.get("validated_data")
    errors = []

    if not validated_data:
        error_msg = "Eligibility check failed: No validated data found."
        print(error_msg)
        langfuse.update_current_observation(level="ERROR", status_message=error_msg)
        return {"error_message": error_msg}

    # Prepare Features
    features = {
        "income": validated_data.get("total_income", 0.0),
        "family_size": len(validated_data.get("family_members", [])) + 1,
    }
    print(f"Prepared features: {features}")

    # Call ML Service
    ml_service_url = "http://ml_service:8001/predict"
    prediction_result = None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client: # Add timeout
            response = await client.post(ml_service_url, json=features)
            response.raise_for_status()
            prediction_result = response.json()
            print(f"ML Service Response: {prediction_result}")

            # Update observation output on success
            langfuse.update_current_observation(output=prediction_result)

            return {
                "eligibility_decision": prediction_result.get("decision", "Error"),
                "eligibility_score": prediction_result.get("probability"),
                "error_message": None # Clear previous errors
            }

    except httpx.RequestError as e:
        error_msg = f"Error calling ML service ({e.request.url}): {e}"
        print(error_msg)
        errors.append(error_msg)
    except httpx.HTTPStatusError as e:
        error_msg = f"ML service returned error status {e.response.status_code}: {e.response.text}"
        print(error_msg)
        errors.append(error_msg)
    except Exception as e:
        error_msg = f"Error processing ML service response: {e}"
        print(error_msg)
        errors.append(error_msg)

    # Handle errors
    final_error_message = "Eligibility check failed:\n" + "\n".join(errors)
    langfuse.update_current_observation(level="ERROR", status_message="Eligibility Check Failed", output={"errors": errors})
    current_error = state.get("error_message")
    return {"error_message": f"{current_error}\n{final_error_message}" if current_error else final_error_message}


@observe()
async def generate_recommendation(state: AgentState) -> Dict[str, Any]:
    """
    Generates final recommendations using eligibility and RAG.
    """
    print(f"--- Generating Recommendation for App ID: {state['application_id']} ---")
    application_id = state["application_id"]
    applicant_id = state["applicant_id"]
    eligibility_decision = state.get("eligibility_decision")
    eligibility_score = state.get("eligibility_score")
    validated_data = state.get("validated_data", {})
    errors = []

    if not eligibility_decision:
        error_msg = "Recommendation failed: Eligibility decision not found."
        print(error_msg)
        langfuse.update_current_observation(level="ERROR", status_message=error_msg)
        current_error = state.get("error_message")
        return {"error_message": f"{current_error}\n{error_msg}" if current_error else error_msg}

    # Financial Recommendation
    financial_recommendation = ""
    score_str = f"(Score: {eligibility_score:.2f})" if eligibility_score is not None else ""
    applicant_name = validated_data.get('full_name', 'Applicant')

    if eligibility_decision == "Approve":
        financial_recommendation = f"Congratulations {applicant_name}! Your application for financial support has been approved {score_str}. Further details will follow."
    elif eligibility_decision == "Decline":
        financial_recommendation = f"We regret to inform you {applicant_name} that your application for financial support could not be approved at this time {score_str}. Please see economic enablement options below:"
    else:
         financial_recommendation = f"Your application (ID: {application_id}) requires further review {score_str}. We will update you soon."
         # Skip RAG if under review
         return {"final_recommendation": financial_recommendation}


    # Economic Enablement (RAG)
    economic_recommendation = ""
    vector_id = application_id # Using application_id as the Qdrant point ID

    if eligibility_decision in ["Approve", "Decline"] and embedding_model and qdrant_client:
        print("Searching Qdrant for economic enablement opportunities...")
        try:
             # Use context manager for the RAG step
            with langfuse.get_current_observation().step(name="rag-economic-enablement", input={"vector_id_searched": vector_id}) as rag_step:
                try:
                    # V1: Retrieve the applicant's own vector/payload
                    search_result = qdrant_client.retrieve(
                        collection_name=QDRANT_COLLECTION,
                        ids=[vector_id],
                        with_payload=True,
                        with_vectors=False
                    )

                    if search_result:
                        # V1 Placeholder: Just confirm retrieval
                        economic_recommendation = "\n\nWe have identified potential economic enablement opportunities based on your profile (details would follow via LLM in V2)."
                        print("Placeholder RAG search successful (retrieved own vector).")
                        rag_step.output({"search_results_count": len(search_result), "placeholder_message": economic_recommendation})
                        # V2: Here you would ideally search a DIFFERENT collection (jobs/courses)
                        # using the applicant's vector as the query vector.
                        # query_vector = embedding_model.encode(validated_data.get("resume_text","")).tolist()
                        # hits = qdrant_client.search(collection_name="jobs_collection", query_vector=query_vector, limit=3)
                        # Then format 'hits' using an LLM.
                    else:
                        economic_recommendation = "\n\nWe recommend exploring general resources."
                        print("Applicant vector not found in Qdrant.")
                        rag_step.output({"search_results_count": 0, "message": "Vector not found"})

                except Exception as e_rag:
                    error_msg = f"Error during Qdrant search: {e_rag}"
                    print(error_msg)
                    errors.append(error_msg)
                    economic_recommendation = "\n\nThere was an issue searching for opportunities."
                    rag_step.level="ERROR"; rag_step.status_message=error_msg; rag_step.output({"error": error_msg})

        except Exception as e_step:
            error_msg = f"Error setting up RAG step: {e_step}"
            print(error_msg); errors.append(error_msg)
            economic_recommendation = "\n\nIssue preparing opportunity search."

    elif eligibility_decision in ["Approve", "Decline"]:
         economic_recommendation = "\n\nEnablement search skipped (embedding model or Qdrant client unavailable)."


    # Combine and Finalize
    final_recommendation = financial_recommendation + economic_recommendation

    if errors:
        final_message = "Recommendation generation finished with RAG errors:\n" + "\n".join(errors)
        print(final_message)
        langfuse.update_current_observation(level="WARNING", status_message="Recommendation Errors (RAG)", output={"recommendation": final_recommendation, "errors": errors})
        current_error = state.get("error_message")
        # Still return the recommendation, but include errors
        return {"final_recommendation": final_recommendation, "error_message": f"{current_error}\n{final_message}" if current_error else final_message}
    else:
        print("Recommendation generated successfully.")
        # @observe logs return value automatically
        return {"final_recommendation": final_recommendation, "error_message": None} # Clear errors


# ==================================
# == Graph Definition ==
# ==================================

workflow = StateGraph(AgentState)

# --- Add Nodes ---
workflow.add_node("extract_data", run_data_extraction)
workflow.add_node("validate_data", run_data_validation)
workflow.add_node("persist_data", persist_data)
workflow.add_node("check_eligibility", check_eligibility)
workflow.add_node("generate_recommendation", generate_recommendation)

# --- Set Entry Point ---
workflow.set_entry_point("extract_data")

# --- Define Edges ---

# After Extraction
def decide_after_extraction(state: AgentState) -> str:
    # Check if 'extracted_data' exists and is not empty, AND if there's an error message
    has_data = bool(state.get("extracted_data"))
    has_error = bool(state.get("error_message"))

    if not has_data and has_error:
        print("Decision: Extraction failed critically. Ending.")
        return "end_workflow"
    # Proceed to validation even if there were partial errors but some data was extracted
    else: # Covers (has_data and has_error) OR (has_data and not has_error) OR (not has_data and not has_error - unlikely)
        print("Decision: Proceeding to validation.")
        return "validate_data"

workflow.add_conditional_edges(
    "extract_data",
    decide_after_extraction,
    {"validate_data": "validate_data", "end_workflow": END}
)

# After Validation
def decide_after_validation(state: AgentState) -> str:
    # Error message specifically set by the validation node indicates failure
    validation_failed = "Validation Failed" in state.get("error_message", "")

    if validation_failed:
        print("Decision: Validation failed. Ending.")
        return "end_workflow" # Stop if core validation fails
    else:
        print("Decision: Validation successful. Proceeding to persistence.")
        return "persist_data"

workflow.add_conditional_edges(
    "validate_data",
    decide_after_validation,
    {"persist_data": "persist_data", "end_workflow": END}
)

# After Persistence
def decide_after_persistence(state: AgentState) -> str:
    # Error message specifically set by the persistence node indicates failure
    persistence_failed = "Persistence finished with errors" in state.get("error_message", "")

    if persistence_failed:
        print("Decision: Persistence failed. Ending.")
        return "end_workflow" # Stop if critical data couldn't be saved
    else:
        print("Decision: Persistence successful. Proceeding to eligibility check.")
        return "check_eligibility"

workflow.add_conditional_edges(
    "persist_data",
    decide_after_persistence,
    {"check_eligibility": "check_eligibility", "end_workflow": END}
)

# After Eligibility Check
def decide_after_eligibility(state: AgentState) -> str:
    # Error message specifically set by the eligibility node indicates failure
    eligibility_failed = "Eligibility check failed" in state.get("error_message", "")

    if eligibility_failed:
        print("Decision: Eligibility check failed. Ending.")
        return "end_workflow"
    else:
        print("Decision: Eligibility check successful. Proceeding to recommendation.")
        return "generate_recommendation"

workflow.add_conditional_edges(
    "check_eligibility",
    decide_after_eligibility,
    {"generate_recommendation": "generate_recommendation", "end_workflow": END}
)

# After Recommendation Generation - Always End
workflow.add_edge("generate_recommendation", END)

# --- Compile Graph ---
try:
    app_graph = workflow.compile()
    print("LangGraph workflow compiled successfully.")
except Exception as e:
    print(f"Error compiling LangGraph workflow: {e}")
    app_graph = None # Set to None on failure