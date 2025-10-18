# apps/api/main.py
from fastapi import FastAPI
from apps.api.routes import applications
from storage import database, models
from apps.api.core.config import settings # pydantic-settings loads .env
from langfuse.fastapi import LangfuseIntegration

# Create the database tables (if not using Alembic)
# models.Base.metadata.create_all(bind=database.engine) # Alembic handles this now

app = FastAPI(
    title="Social Support AI Workflow API",
    description="API for managing and processing social support applications.",
    version="0.1.0"
)

# Initialize Langfuse
if settings.LANGFUSE_HOST:
    LangfuseIntegration(
        host=settings.LANGFUSE_HOST,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        release="1.0.0"
    )

# Include our API routes
app.include_router(applications.router)

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "API is running"}