# apps/api/core/config.py

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import PostgresDsn, computed_field, AnyHttpUrl
from typing import Optional # Import Optional

# Define the path to the .env file relative to this config file's location
env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')

class Settings(BaseSettings):
    """
    Loads all environment variables from the .env file.
    """
    model_config = SettingsConfigDict(env_file=env_path, extra='ignore', env_file_encoding='utf-8')

    # --- PostgreSQL Config ---
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int = 5432

    @computed_field(return_type=str) # Specify return type for clarity
    @property
    def DATABASE_URL(self) -> str: # Use str type hint
        """
        Generates the full database connection string.
        NOTE: We build as string because PostgresDsn validation can be strict
              and sometimes fails with docker hostnames initially.
        """
        return (f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
                f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}")

    # --- Ollama Config ---
    OLLAMA_HOST: str = "http://ollama:11434"

    # --- Langfuse Config ---  <-- ADD THESE LINES
    LANGFUSE_HOST: Optional[AnyHttpUrl] = None # Use AnyHttpUrl for validation
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_PUBLIC_KEY: Optional[str] = None

    # --- API Config ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000


# Create a single, importable instance
settings = Settings()

# --- Print loaded DB URL for verification during startup ---
print(f"Database URL loaded: {settings.DATABASE_URL}")
print(f"Langfuse Host loaded: {settings.LANGFUSE_HOST}") # Optional: verify Langfuse vars