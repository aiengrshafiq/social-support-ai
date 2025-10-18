# apps/api/main.py
import os
from fastapi import FastAPI
from apps.api.routes import applications
from storage import database, models  # Make sure database.py is correctly set up
from apps.api.core.config import settings
# --- THIS IS THE MOST LIKELY CORRECT IMPORT ---
from langfuse import Langfuse # Import the main Langfuse class

app = FastAPI(
    title="Social Support AI Workflow API",
    description="API for managing and processing social support applications.",
    version="0.1.0"
)

# Initialize Langfuse - **Manual Integration is more robust**
# Instead of relying on the changing FastAPIIntegration class,
# we'll initialize the core Langfuse object here.
# Tracing will be added manually to endpoints later using decorators or dependencies.
langfuse_client = None
if settings.LANGFUSE_HOST and settings.LANGFUSE_SECRET_KEY and settings.LANGFUSE_PUBLIC_KEY:
    try:
        langfuse_client = Langfuse(
            host=settings.LANGFUSE_HOST,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            release="social-support-ai-v1.0" # Example release name
            # Add other config as needed, e.g., flush_at, flush_interval
        )
        print("Langfuse client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Langfuse client: {e}")
        langfuse_client = None # Ensure client is None if init fails
else:
    print("Langfuse environment variables not fully set. Skipping Langfuse initialization.")

# --- Make Langfuse client available (e.g., via app state or dependency injection) ---
# This allows your endpoints/agents to access it for tracing later
app.state.langfuse_client = langfuse_client


# Include our API routes
app.include_router(applications.router)

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "API is running"}

# --- Add shutdown event for Langfuse ---
@app.on_event("shutdown")
def shutdown_event():
    if app.state.langfuse_client:
        print("Shutting down Langfuse client...")
        app.state.langfuse_client.shutdown()
        print("Langfuse client shut down.")

# You'll need storage/database.py to define SessionLocal and engine
# Example content for storage/database.py:
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# from apps.api.core.config import settings
#
# engine = create_engine(str(settings.DATABASE_URL))
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)