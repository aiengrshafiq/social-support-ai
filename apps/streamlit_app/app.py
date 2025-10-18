# apps/streamlit_app/app.py

import streamlit as st
import requests # To call the FastAPI backend
import os
from typing import Dict, Any

# --- Configuration ---
API_BASE_URL = "http://api:8000/api/v1" # Use Docker service name 'api'

# --- Helper Functions ---
def create_applicant(name: str, email: str, emirates_id: str, phone: str) -> Dict[str, Any]:
    """Calls the backend to create a new applicant."""
    url = f"{API_BASE_URL}/applications/applicant"
    payload = {
        "full_name": name,
        "email": email,
        "emirates_id": emirates_id,
        "phone_number": phone
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error creating applicant: {e}")
        if hasattr(e, 'response') and e.response is not None:
             st.error(f"Backend Response: {e.response.text}")
        return {}

def create_application(applicant_id: int) -> Dict[str, Any]:
    """Calls the backend to create a new application."""
    url = f"{API_BASE_URL}/applications/"
    payload = {"applicant_id": applicant_id}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error creating application: {e}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Backend Response: {e.response.text}")
        return {}

def upload_file(application_id: int, file_type: str, uploaded_file_obj) -> Dict[str, Any]:
    """Calls the backend to upload a document."""
    url = f"{API_BASE_URL}/applications/{application_id}/upload"
    params = {"file_type": file_type}
    files = {"file": (uploaded_file_obj.name, uploaded_file_obj, uploaded_file_obj.type)}
    try:
        response = requests.post(url, params=params, files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading {file_type}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Backend Response: {e.response.text}")
        return {}


# --- Streamlit UI ---

st.set_page_config(page_title="Social Support AI Assistant", layout="wide")
st.title("🤖 Social Support Application Assistant")

# --- Initialize Session State ---
# Store applicant and application IDs across interactions
if "applicant_id" not in st.session_state:
    st.session_state.applicant_id = None
if "application_id" not in st.session_state:
    st.session_state.application_id = None
if "final_recommendation" not in st.session_state:
    st.session_state.final_recommendation = None # Store the final message
if "upload_status" not in st.session_state:
    st.session_state.upload_status = {} # Track uploads


# --- Step 1: Applicant Creation ---
st.header("Step 1: Applicant Information")

if not st.session_state.applicant_id:
    with st.form("applicant_form"):
        st.write("Please enter your details:")
        name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        emirates_id = st.text_input("Emirates ID (Format: 784-XXXX-XXXXXXX-X)")
        phone = st.text_input("Phone Number (Optional)")
        submitted = st.form_submit_button("Create Applicant Record")

        if submitted:
            if name and email and emirates_id:
                with st.spinner("Creating applicant..."):
                    applicant_data = create_applicant(name, email, emirates_id, phone)
                    if applicant_data and "id" in applicant_data:
                        st.session_state.applicant_id = applicant_data["id"]
                        st.success(f"Applicant record created successfully! Applicant ID: {st.session_state.applicant_id}")
                        st.rerun() # Rerun to move to the next step
                    else:
                        st.error("Failed to create applicant record.")
            else:
                st.warning("Please fill in all required fields.")
else:
    st.success(f"Applicant ID: {st.session_state.applicant_id}")

# --- Step 2: Application Creation ---
st.header("Step 2: Start New Application")

if st.session_state.applicant_id and not st.session_state.application_id:
    if st.button("Start New Application"):
        with st.spinner("Creating application..."):
            application_data = create_application(st.session_state.applicant_id)
            if application_data and "id" in application_data:
                st.session_state.application_id = application_data["id"]
                st.session_state.upload_status = {} # Reset upload status
                st.session_state.final_recommendation = None # Reset recommendation
                st.success(f"Application started successfully! Application ID: {st.session_state.application_id}")
                st.rerun()
            else:
                st.error("Failed to start application.")

elif st.session_state.application_id:
     st.success(f"Current Application ID: {st.session_state.application_id}")


# --- Step 3: Document Upload ---
st.header("Step 3: Upload Documents")

if st.session_state.application_id and not st.session_state.final_recommendation:
    st.write("Please upload the required documents.")

    # Define required documents (can be dynamic later)
    required_docs = {
        "emirates_id": "Emirates ID (Image: PNG, JPG)",
        "bank_statement": "Bank Statement (PDF)",
        "resume": "Resume/CV (PDF)",
        "assets_liabilities": "Assets/Liabilities (Excel: XLSX, XLS)"
        # "credit_report": "Credit Report (PDF)" # Add if needed
    }

    cols = st.columns(len(required_docs))

    for i, (doc_key, doc_desc) in enumerate(required_docs.items()):
        with cols[i]:
            uploaded_file = st.file_uploader(f"Upload {doc_desc}", key=f"upload_{doc_key}")
            if uploaded_file is not None:
                # Display upload button only if not already uploaded successfully
                if doc_key not in st.session_state.upload_status or not st.session_state.upload_status[doc_key].get("success"):
                    if st.button(f"Submit {doc_key}", key=f"submit_{doc_key}"):
                        with st.spinner(f"Uploading {doc_key}..."):
                            upload_result = upload_file(st.session_state.application_id, doc_key, uploaded_file)
                            if upload_result and "filename" in upload_result:
                                st.session_state.upload_status[doc_key] = {"success": True, "filename": upload_result["filename"]}
                                st.success(f"{doc_key} uploaded!")
                                st.rerun() # Rerun to update status display
                            else:
                                st.session_state.upload_status[doc_key] = {"success": False}
                                st.error(f"Failed to upload {doc_key}.")
                # Display status if already uploaded
                elif st.session_state.upload_status[doc_key].get("success"):
                    st.success(f"{doc_key} uploaded ({st.session_state.upload_status[doc_key].get('filename', '')}).")


    # --- Placeholder for Final Result ---
    # In a real app, you'd poll the backend or use websockets to get the final status
    # For V1, we assume the user uploads all docs, and the last upload triggers processing.
    # We don't have a way yet to retrieve the final_recommendation from the background task.
    # We will add a button to manually check status/result as a V1 workaround.

    st.markdown("---")
    if st.button("Check Application Result (Manual V1)"):
         st.warning("Feature not implemented: Cannot retrieve result from background task yet.")
         # V2: Implement an endpoint like GET /api/v1/applications/{application_id}/status
         # which checks the DB for the final recommendation stored by the agent.
         # For now, check Langfuse Cloud for the result.
         st.info("Please check the Langfuse trace for the final recommendation.")


elif st.session_state.final_recommendation:
    st.header("Application Result")
    st.info(st.session_state.final_recommendation)
    if st.button("Start Another Application"):
        # Reset state partially
        st.session_state.application_id = None
        st.session_state.final_recommendation = None
        st.session_state.upload_status = {}
        st.rerun()