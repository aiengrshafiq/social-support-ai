# agents/graph.py (Corrected & Simplified Langfuse)
import os
import json
import httpx
import datetime
import numpy as np
from typing import TypedDict, List, Dict, Optional, Any

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END

# --- Database Imports ---
from pymongo import MongoClient
from neo4j import GraphDatabase, exceptions as neo4j_exceptions
from qdrant_client import QdrantClient, models
from sqlalchemy.orm import Session
from storage.database import SessionLocal
from storage import models as pg_models

# --- Langfuse Imports ---
# ONLY import observe and the client class
from langfuse.decorators import observe
from langfuse import Langfuse as LangfuseClient

# --- Project Imports ---
from parsers.document_parser import extract_data_from_document
from agents.state import AgentState
from agents.schemas import ValidatedApplicationData
from apps.api.core.config import settings

# --- Pydantic Imports ---
from pydantic import ValidationError

# --- Embedding Imports ---
try:
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./embedding_cache')
    EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()
    print("Sentence Transformer embedding model loaded.")
except ImportError:
    print("Sentence Transformers not installed. Embeddings disabled.")
    embedding_model = None; EMBEDDING_DIM = 384
except Exception as e:
    print(f"Error loading Sentence Transformer model: {e}. Embeddings disabled.")
    embedding_model = None; EMBEDDING_DIM = 384

# --- Database Client Initializations (with error handling) ---
mongo_client = None
try:
    mongo_client_init = MongoClient("mongodb://mongo:27017/", serverSelectionTimeoutMS=5000)
    mongo_client_init.admin.command('ping')
    db = mongo_client_init["social_support_raw_data"]
    raw_extractions_collection = db["raw_extractions"]
    mongo_client = mongo_client_init # Assign only on success
    print("MongoDB client initialized and connected.")
except Exception as e:
    print(f"Error initializing MongoDB client: {e}")

neo4j_driver = None
try:
    NEO4J_URI = "bolt://neo4j:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = settings.NEO4J_AUTH.split('/')[1] if settings.NEO4J_AUTH and '/' in settings.NEO4J_AUTH else "password"
    neo4j_driver_init = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), max_connection_lifetime=3600)
    neo4j_driver_init.verify_connectivity()
    neo4j_driver = neo4j_driver_init # Assign only on success
    print("Neo4j driver initialized and connected.")
except (neo4j_exceptions.AuthError, neo4j_exceptions.ServiceUnavailable, Exception) as e:
    print(f"Error initializing Neo4j driver: {e}")


QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "applicant_resumes"
try:
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=20)
    try:
         qdrant_client.get_collection(collection_name=QDRANT_COLLECTION)
         print(f"Qdrant collection '{QDRANT_COLLECTION}' found.")
    # --- Catch General Exception for Collection Check ---
    except Exception as e_coll: # <-- CHANGED THIS LINE
        # Check common error messages for "not found"
        if "not found" in str(e_coll).lower() or "404" in str(e_coll):
            print(f"Qdrant collection '{QDRANT_COLLECTION}' not found. Creating...")
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE)
            )
            print(f"Created Qdrant collection '{QDRANT_COLLECTION}'.")
        else:
            # Log other errors but proceed if possible
            print(f"Warning: Error checking Qdrant collection (but client connected): {e_coll}")
    print("Qdrant client initialized.")
except Exception as e:
    print(f"Error initializing Qdrant client: {e}. Qdrant operations disabled.")
    qdrant_client = None


# --- Langfuse Client Initialization ---
# This instance is used by the @observe decorator implicitly
langfuse_client = None
if settings.LANGFUSE_HOST and settings.LANGFUSE_SECRET_KEY and settings.LANGFUSE_PUBLIC_KEY:
    try:
        langfuse_client = LangfuseClient( # Use aliased name
            host=str(settings.LANGFUSE_HOST), secret_key=settings.LANGFUSE_SECRET_KEY,
            public_key=settings.LANGFUSE_PUBLIC_KEY, release="social-support-ai-v1.0",
            flush_interval=1 # Flush frequently
        )
        print("Langfuse client for Graph initialized (used by @observe).")
    except Exception as e:
        print(f"Failed to initialize Langfuse client in Graph: {e}")
else:
    print("Langfuse env vars not set in Graph, tracing disabled.")

# --- Langfuse Helper for manual trace start (used in background task) ---
def get_langfuse_trace(application_id):
    if langfuse_client:
        try:
            # We explicitly pass the global client instance here if needed elsewhere
            return langfuse_client.trace(name="application-workflow", user_id=f"app-{application_id}")
        except Exception as e: print(f"Error creating Langfuse trace: {e}")
    return None

# ==================================
# == Agent Nodes ==
# ==================================

@observe() # Let decorator handle trace/span for this function
def run_data_extraction(state: AgentState) -> Dict[str, Any]:
    print(f"--- Running Data Extraction for App ID: {state['application_id']} ---")
    application_id = state["application_id"]; uploaded_files = state["uploaded_files"]
    all_extracted_data = {}; errors = []

    for file_type, file_path in uploaded_files.items():
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"; print(error_msg); errors.append(error_msg)
            continue
        print(f"Processing {file_type}: {file_path}...")
        try:
            # --- Call parser directly - NO manual step creation ---
            extracted = extract_data_from_document(file_path)
            if "_error" in extracted:
                error_msg = f"Error extracting {file_type}: {extracted['_error']}"
                print(error_msg); errors.append(error_msg)
                all_extracted_data[f"{file_type}_raw_error"] = extracted.get("_raw_output", extracted['_error'])
            else:
                all_extracted_data[file_type] = extracted
                print(f"Successfully extracted data for {file_type}.")
        except Exception as e:
            error_msg = f"Unhandled exception processing {file_type}: {str(e)}"
            print(error_msg); errors.append(error_msg)

    # --- Store Raw Results in MongoDB ---
    if mongo_client:
        try:
            raw_extractions_collection.insert_one({
                "application_id": application_id, "extracted_data": all_extracted_data,
                "errors": errors, "timestamp": datetime.datetime.utcnow()
            })
            print("Raw extraction results saved to MongoDB.")
        except Exception as e:
            mongo_error = f"MongoDB save error: {str(e)}"; print(mongo_error); errors.append(mongo_error)
    else: errors.append("MongoDB client not available.")

    # --- Prepare return state ---
    final_data = {k: v for k, v in all_extracted_data.items() if isinstance(v, dict)} # Simple merge
    if errors:
        error_summary = "Extraction Errors:\n" + "\n".join(errors); print(error_summary)
        # @observe logs return value (including error_message) as output
        return {"extracted_data": final_data, "error_message": error_summary}
    else:
        print("Data extraction completed successfully.")
        return {"extracted_data": final_data, "error_message": None}


@observe()
def run_data_validation(state: AgentState) -> Dict[str, Any]:
    print(f"--- Running Data Validation for App ID: {state['application_id']} ---")
    extracted_data = state.get("extracted_data"); validation_errors = []; validated_data_dict = {}

    if not extracted_data:
        error_msg = "Validation failed: No extracted data found."; print(error_msg)
        # @observe will log the return value as output, including error_message
        return {"error_message": error_msg}

    try:
        validated_pydantic = ValidatedApplicationData(**extracted_data)
        validated_data_dict = validated_pydantic.model_dump(exclude_unset=True)
        print("Pydantic schema validation successful.")
    except ValidationError as e:
        error_msg = f"Pydantic validation failed: {e}"; print(error_msg); validation_errors.append(error_msg)
        validated_data_dict = extracted_data # Fallback

    # --- Custom Business Rule Checks ---
    if validated_data_dict.get("employer") and validated_data_dict.get("total_income") is None:
        print("Validation Warning: Employer listed but no income extracted.")
        # We don't fail validation for this warning in V1

    if validation_errors:
        final_error_message = "Validation Failed:\n" + "\n".join(validation_errors); print(final_error_message)
        return {"validated_data": validated_data_dict, "error_message": final_error_message}
    else:
        print("Data validation completed successfully.")
        return {"validated_data": validated_data_dict, "error_message": None}


@observe()
def persist_data(state: AgentState) -> Dict[str, Any]:
    print(f"--- Running Data Persistence for App ID: {state['application_id']} ---")
    application_id = state["application_id"]; validated_data = state.get("validated_data")
    applicant_id = state["applicant_id"]; errors = []

    if not validated_data:
        error_msg = "Persistence failed: No validated data found."; print(error_msg)
        return {"error_message": error_msg}

    # --- PostgreSQL ---
    pg_session = SessionLocal()
    try:
        with pg_session.begin():
            db_application = pg_session.query(pg_models.Application).filter(pg_models.Application.id == application_id).with_for_update().first()
            if db_application:
                db_application.validated_data = validated_data
                db_applicant = pg_session.query(pg_models.Applicant).filter(pg_models.Applicant.id == applicant_id).first()
                if db_applicant:
                     db_applicant.full_name = validated_data.get('full_name', db_applicant.full_name)
                     db_applicant.email = validated_data.get('email', db_applicant.email)
                     db_applicant.phone_number = validated_data.get('phone_number', db_applicant.phone_number)
                print(f"App {application_id} updated in PostgreSQL.")
            else: errors.append(f"App {application_id} not found in PostgreSQL.")
    except Exception as e:
        error_msg = f"PostgreSQL error: {str(e)}"; print(error_msg); errors.append(error_msg)
    finally: pg_session.close()

    # --- Neo4j ---
    if neo4j_driver:
        try:
            with neo4j_driver.session(database="neo4j") as session:
                session.write_transaction(update_neo4j_graph, validated_data, application_id)
                print(f"Data for App {application_id} updated in Neo4j.")
        except Exception as e:
            error_msg = f"Neo4j error: {str(e)}"; print(error_msg); errors.append(error_msg)
    else: errors.append("Neo4j driver unavailable.")

    # --- Qdrant ---
    resume_text = validated_data.get("resume_text", "")
    if resume_text and embedding_model and qdrant_client:
        try:
            vector = embedding_model.encode(resume_text).tolist()
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=[models.PointStruct(id=application_id, vector=vector, payload={"applicant_id": applicant_id})],
                wait=True
            )
            print(f"Embedding for App {application_id} saved to Qdrant.")
        except Exception as e:
            error_msg = f"Qdrant error: {str(e)}"; print(error_msg); errors.append(error_msg)
    elif resume_text: errors.append("Resume text found but embedding/Qdrant unavailable.")

    if errors:
        final_message = "Persistence Errors:\n" + "\n".join(errors); print(final_message)
        current_error = state.get("error_message")
        return {"error_message": f"{current_error}\n{final_message}" if current_error else final_message}
    else:
        print("Persistence completed successfully.")
        return {"error_message": None}


# Helper Function for Neo4j Transaction
def update_neo4j_graph(tx, validated_data, application_id):
    emirates_id = validated_data.get('emirates_id'); full_name = validated_data.get('full_name')
    address = validated_data.get('address'); family_members = validated_data.get('family_members', [])
    if emirates_id:
        tx.run("MERGE (p:Person {emiratesId: $eid}) SET p.name = $name, p.applicationId = $appId",
               eid=emirates_id, name=full_name, appId=application_id)
        if address:
            tx.run("MATCH (p:Person {emiratesId: $eid}) MERGE (a:Address {fullAddress: $addr}) MERGE (p)-[:LIVES_AT]->(a)",
                   eid=emirates_id, addr=address)
        for member in family_members:
             member_name = member.get('name'); relation = member.get('relation')
             if member_name and relation:
                 tx.run("MATCH (applicant:Person {emiratesId: $eid}) MERGE (fm:Person {name: $memName}) MERGE (applicant)-[:HAS_FAMILY_MEMBER {relation: $rel}]->(fm)",
                        eid=emirates_id, memName=member_name, rel=relation)


@observe()
async def check_eligibility(state: AgentState) -> Dict[str, Any]:
    print(f"--- Running Eligibility Check for App ID: {state['application_id']} ---")
    validated_data = state.get("validated_data"); errors = []

    if not validated_data:
        error_msg = "Eligibility check failed: No validated data."; print(error_msg)
        return {"error_message": error_msg}

    features = {"income": validated_data.get("total_income", 0.0),
                "family_size": len(validated_data.get("family_members", [])) + 1}
    print(f"Prepared features: {features}")

    ml_service_url = "http://ml_service:8001/predict"; prediction_result = None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(ml_service_url, json=features)
            response.raise_for_status(); prediction_result = response.json()
            print(f"ML Service Response: {prediction_result}")
            # @observe logs return value as output automatically
            return {"eligibility_decision": prediction_result.get("decision", "Error"),
                    "eligibility_score": prediction_result.get("probability"), "error_message": None}
    except (httpx.RequestError, httpx.HTTPStatusError, Exception) as e:
        error_msg = f"Eligibility check failed: {e}"; print(error_msg); errors.append(error_msg)
        current_error = state.get("error_message")
        final_error_message = "Eligibility Check Errors:\n" + "\n".join(errors)
        return {"error_message": f"{current_error}\n{final_error_message}" if current_error else final_error_message}


@observe()
async def generate_recommendation(state: AgentState) -> Dict[str, Any]:
    print(f"--- Generating Recommendation for App ID: {state['application_id']} ---")
    application_id = state["application_id"]; eligibility_decision = state.get("eligibility_decision")
    eligibility_score = state.get("eligibility_score"); validated_data = state.get("validated_data", {})
    errors = []

    if not eligibility_decision:
        error_msg = "Recommendation failed: Eligibility decision missing."; print(error_msg)
        current_error = state.get("error_message")
        return {"error_message": f"{current_error}\n{error_msg}" if current_error else error_msg}

    # Financial Recommendation
    score_str = f"(Score: {eligibility_score:.2f})" if eligibility_score is not None else ""
    applicant_name = validated_data.get('full_name', 'Applicant')
    if eligibility_decision == "Approve":
        financial_recommendation = f"Congratulations {applicant_name}! Support approved {score_str}. Details follow."
    elif eligibility_decision == "Decline":
        financial_recommendation = f"Regretfully {applicant_name}, support could not be approved {score_str}. See options below:"
    else: # Review state
         financial_recommendation = f"Application {application_id} needs review {score_str}. Status update soon."
         return {"final_recommendation": financial_recommendation} # End here if review needed

    # Economic Enablement (RAG)
    economic_recommendation = ""
    if eligibility_decision in ["Approve", "Decline"] and embedding_model and qdrant_client:
        print("Searching Qdrant for opportunities...")
        try:
            # --- NO manual step creation, let @observe handle the main span ---
            vector_id = application_id
            search_result = qdrant_client.retrieve( # V1: Retrieve own vector
                collection_name=QDRANT_COLLECTION, ids=[vector_id],
                with_payload=True, with_vectors=False
            )
            if search_result:
                economic_recommendation = "\n\nPotential economic enablement opportunities identified (V1 Placeholder)."
                print("Placeholder RAG search successful.")
                # Optional: Log RAG success to main span metadata if needed
                # obs = langfuse.get_current_observation();
                # if obs: obs.metadata = {"rag_status": "success", "results_found": True}
            else:
                economic_recommendation = "\n\nRecommend exploring general resources."
                print("Applicant vector not found in Qdrant.")
                # Optional: Log RAG failure to metadata
                # obs = langfuse.get_current_observation();
                # if obs: obs.metadata = {"rag_status": "vector_not_found", "results_found": False}
        except Exception as e_rag:
            error_msg = f"Error during Qdrant search: {e_rag}"; print(error_msg); errors.append(error_msg)
            economic_recommendation = "\n\nIssue searching for opportunities."
            # Optional: Log RAG error to main span metadata
            # obs = langfuse.get_current_observation();
            # if obs: obs.metadata = {"rag_status": "error", "error_message": error_msg}
    elif eligibility_decision in ["Approve", "Decline"]:
         economic_recommendation = "\n\nEnablement search skipped (setup incomplete)."

    # Combine and Finalize
    final_recommendation = financial_recommendation + economic_recommendation
    if errors: # Only log if RAG errors occurred, main decision already made
        final_message = "Recommendation generated with RAG errors:\n" + "\n".join(errors); print(final_message)
        obs = langfuse.get_current_observation();
        if obs: obs.level = "WARNING"; obs.status_message = "Recommendation Errors (RAG)"; # Update status, output is return value
        current_error = state.get("error_message")
        return {"final_recommendation": final_recommendation, "error_message": f"{current_error}\n{final_message}" if current_error else final_message}
    else:
        print("Recommendation generated successfully.")
        return {"final_recommendation": final_recommendation, "error_message": None}

# ==================================
# == Graph Definition ==
# ==================================

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("extract_data", run_data_extraction)
workflow.add_node("validate_data", run_data_validation)
workflow.add_node("persist_data", persist_data)
workflow.add_node("check_eligibility", check_eligibility)
workflow.add_node("generate_recommendation", generate_recommendation)

# Set Entry Point
workflow.set_entry_point("extract_data")

# Define Edges (using clearer error checks)
def decide_after_extraction(state: AgentState) -> str:
    has_data = bool(state.get("extracted_data"))
    has_extraction_error = "Extraction Errors" in state.get("error_message", "")
    if not has_data and has_extraction_error:
        print("Decision: Extraction failed critically. Ending."); return "end_workflow"
    else: print("Decision: Proceeding to validation."); return "validate_data"
workflow.add_conditional_edges("extract_data", decide_after_extraction, {"validate_data": "validate_data", "end_workflow": END})

def decide_after_validation(state: AgentState) -> str:
    validation_failed = "Validation Failed" in state.get("error_message", "")
    if validation_failed: print("Decision: Validation failed. Ending."); return "end_workflow"
    else: print("Decision: Validation successful. Proceeding to persistence."); return "persist_data"
workflow.add_conditional_edges("validate_data", decide_after_validation, {"persist_data": "persist_data", "end_workflow": END})

def decide_after_persistence(state: AgentState) -> str:
    persistence_failed = "Persistence Errors" in state.get("error_message", "") # Use specific error marker
    if persistence_failed: print("Decision: Persistence failed. Ending."); return "end_workflow"
    else: print("Decision: Persistence successful. Proceeding to eligibility check."); return "check_eligibility"
workflow.add_conditional_edges("persist_data", decide_after_persistence, {"check_eligibility": "check_eligibility", "end_workflow": END})

def decide_after_eligibility(state: AgentState) -> str:
    eligibility_failed = "Eligibility Check Errors" in state.get("error_message", "") # Use specific error marker
    if eligibility_failed: print("Decision: Eligibility check failed. Ending."); return "end_workflow"
    else: print("Decision: Eligibility check successful. Proceeding to recommendation."); return "generate_recommendation"
workflow.add_conditional_edges("check_eligibility", decide_after_eligibility, {"generate_recommendation": "generate_recommendation", "end_workflow": END})

workflow.add_edge("generate_recommendation", END)

# Compile Graph
try:
    app_graph = workflow.compile()
    print("LangGraph workflow compiled successfully.")
except Exception as e:
    print(f"Error compiling LangGraph workflow: {e}"); app_graph = None