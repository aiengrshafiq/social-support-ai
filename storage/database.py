# storage/database.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from apps.api.core.config import settings # Import your settings to get DB URL

# Create the SQLAlchemy engine using the DATABASE_URL from settings
# We use connect_args={} for potential future configurations
engine = create_engine(
    settings.DATABASE_URL,
    # connect_args={"check_same_thread": False} # Only needed for SQLite
    pool_pre_ping=True # Helps prevent connection errors after idle time
)

# Create a configured "Session" class
# autocommit=False and autoflush=False are standard for web apps
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for your SQLAlchemy models (already defined in storage/models.py)
# You might import it here or ensure your models inherit from a Base defined elsewhere
# For simplicity, let's redefine it here if not already easily importable
Base = declarative_base()

# --- Optional: Function to create tables ---
# Not strictly needed if using Alembic, but can be useful for initial setup/testing
# def init_db():
#    # Import all modules here that define models so that
#    # they will be registered properly on the metadata. Otherwise
#    # you will have to import them first before calling init_db()
#    import storage.models
#    print("Creating database tables...")
#    Base.metadata.create_all(bind=engine)
#    print("Database tables created.")