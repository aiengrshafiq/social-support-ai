# Social Support AI Workflow Automation

## Project Overview

This project implements an AI-powered workflow to automate the application process for a government social security department. The goal is to drastically reduce the assessment and approval time from the current 5-20 working days to potentially minutes, leveraging multimodal AI, agentic orchestration, and local large language models (LLMs).

The current manual process suffers from data entry errors, inconsistencies across documents, time-consuming reviews, and subjective decision-making. This solution aims to address these pain points by:

1.  **Ingesting** applicant data from interactive forms and various document types (IDs, bank statements, resumes, asset lists).
2.  **Automating** data extraction using multimodal LLMs.
3.  **Validating** data consistency using schema checks and business rules.
4.  **Persisting** structured and unstructured data across specialized databases (Relational, NoSQL, Graph, Vector).
5.  **Predicting** eligibility using an explainable machine learning model.
6.  **Generating** recommendations for financial support and economic enablement opportunities (job matching, training) using Retrieval-Augmented Generation (RAG).
7.  **Orchestrating** the entire workflow using a stateful agentic framework (LangGraph).
8.  **Providing** observability through Langfuse Cloud.

## Architecture

The solution is designed as a set of containerized microservices orchestrated by LangGraph.

```mermaid
graph TD
    subgraph User Interface
        U(Applicant) -- Interacts/Uploads --> FE(Streamlit App @ :8501)
    end

    subgraph API & Orchestration Layer
        FE -- HTTP Requests --> API(FastAPI Backend @ :8000)
        API -- Triggers Workflow --> ORC(LangGraph Orchestrator)
        ORC -- Sends Traces --> OBS(Langfuse Cloud)
    end

    subgraph Agentic AI Core [LangGraph State Machine]
        ORC -- Manages State --> A2(Data Extraction Agent)
        A2 -- Data/Error --> ORC
        ORC --> A3(Data Validation Agent)
        A3 -- Validated/Error --> ORC
        ORC --> A4(Data Persistence Agent)
        A4 -- Success/Error --> ORC
        ORC --> A5(Eligibility Agent)
        A5 -- Prediction/Error --> ORC
        ORC --> A6(Recommendation Agent)
        A6 -- Recommendation/Error --> ORC
        ORC --> A7(Save Result Agent)
        A7 -- Final Status --> ORC
        ORC --> END((End))
    end

    subgraph Local Model Hosting
        A2 -- Parser LLM Call --> OLLAMA(Ollama @ :11434\n- qwen2-vl:7b\n- llama3.1:8b)
        A4 -- Embedding Call --> ST(SentenceTransformer\n'all-MiniLM-L6-v2')
        # A6 could also call Ollama or use ST embeddings
    end

    subgraph Data & Persistence Layer
        A4 -- (Raw JSON) --> DB_MONGO(MongoDB @ :27017)
        A4 -- (Golden Record) --> DB_PG(PostgreSQL @ :5432)
        A7 -- (Final Result) --> DB_PG
        A4 -- (Relationships) --> DB_NEO4J(Neo4j @ :7687)
        A4 -- (Embeddings) --> DB_QDRANT(Qdrant @ :6333)
        A6 -- (RAG Query) --> DB_QDRANT
    end

    subgraph Classical ML Service
        A5 -- Feature Vector --> ML_API(FastAPI @ :8001)
        ML_API -- Loads --> MODEL(SK-Learn Model\n*.joblib)
        ML_API -- Prediction --> A5
    end

    subgraph Caching
        API -- Session/Cache --> DB_REDIS(Redis @ :6379)
    end


    Data Flow:

The Applicant interacts with the Streamlit App, providing details and uploading documents.

Streamlit sends requests to the FastAPI Backend.

Upon receiving necessary data/files, FastAPI triggers the LangGraph Orchestrator, initializing the application state via a background task.

The Data Extraction Agent uses the ollama service (local LLMs like Qwen2-VL for images/PDFs, Llama 3.1 for text-based extraction from digital PDFs) and pandas (for Excel) to parse documents. Raw results are stored in MongoDB.

The Data Validation Agent uses Pydantic schemas and custom rules to check the extracted data for correctness and consistency.

If validation passes, the Data Persistence Agent saves:

Structured data (applicant profile, application status) to PostgreSQL.

Relationships (family, addresses) to Neo4j.

Embeddings (e.g., from resume text using SentenceTransformer) to Qdrant.

The Eligibility Agent fetches relevant features from PostgreSQL and sends them to the separate ML Service API.

The ML Service loads a pre-trained Scikit-learn model (HistGradientBoostingClassifier) and returns an eligibility prediction and score.

The Recommendation Agent uses the eligibility decision and performs a RAG query against Qdrant (using embeddings - placeholder V1) to find relevant economic enablement opportunities. It crafts a final recommendation message.

The Save Result Agent updates the application record in PostgreSQL with the final recommendation message and status (Completed/Error).

Throughout the process, agents send traces to Langfuse Cloud for observability. Redis is available for caching.

Setup Instructions
Prerequisites:

Docker and Docker Compose

Python >= 3.11

Git

Steps:

Clone the repository:

Bash

git clone <your-repo-url>
cd social-support-ai
Create .env file:
Copy the .env.example file (if you created one, otherwise create .env manually) to .env and fill in the required values:

Bash

# Example command if you have .env.example
# cp .env.example .env
# Edit .env using a text editor
You must provide:

PostgreSQL credentials (POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB)

Neo4j password (NEO4J_AUTH=neo4j/your_password)

Langfuse Cloud API Keys (LANGFUSE_HOST=https://cloud.langfuse.com, LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY) - Get these from your Langfuse Cloud project settings.

API Port (API_PORT=8000)

Build and Run Docker Containers:

Bash

docker-compose up -d --build
This will build the images for the API, Streamlit UI, and ML service, and start all required database/model containers. Wait a minute or two for all services to initialize.

Pull Ollama Models:
Download the required local LLMs into the running Ollama container:

Bash

docker-compose exec ollama ollama pull llama3.1:8b
docker-compose exec ollama ollama pull qwen2-vl:7b
# docker-compose exec ollama ollama pull nomic-embed-text # If using Ollama for embeddings
Run Database Migrations:
Apply the initial PostgreSQL schema:

Bash

docker-compose exec api alembic upgrade head
Running the Application
Once the containers are running:

Streamlit UI: Access the applicant interface at http://localhost:8501

FastAPI Backend Docs: Explore the API endpoints at http://localhost:8000/docs

Neo4j Browser: Visualize the graph database at http://localhost:7474 (Connect with user neo4j and the password set in your .env file).

Langfuse Cloud: Log in to https://cloud.langfuse.com to view traces for processed applications.

ML Service Health: Check the ML service status at http://localhost:8001/health

Technology Stack
Programming Language: Python 3.11+

Web Frameworks: FastAPI (Backend API, ML Service), Streamlit (Frontend UI)

Agent Orchestration: LangGraph

Local LLM Hosting: Ollama (serving Llama 3.1, Qwen2-VL)

Document Parsing: ollama client (VLM/LLM calls), pdfplumber, pandas

Embeddings: Sentence Transformers (all-MiniLM-L6-v2)

Machine Learning: Scikit-learn (HistGradientBoostingClassifier), Joblib

Databases:

Relational: PostgreSQL (with SQLAlchemy & Alembic)

NoSQL Document: MongoDB (with pymongo)

Graph: Neo4j (with neo4j driver)

Vector: Qdrant (with qdrant-client)

Cache: Redis (with redis)

Observability: Langfuse Cloud (langfuse client)

Containerization: Docker, Docker Compose

Schema/Validation: Pydantic

Directory Structure
social-support-ai/
├── README.md              # This file
├── solution_summary.pdf   # Solution summary document
├── .env.example           # Environment variable template
├── .env                   # Local environment variables (ignored by git)
├── .gitignore
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Docker service definitions
├── alembic.ini            # Alembic configuration
├── alembic/               # Alembic migration scripts
├── apps/                  # Application code
│   ├── api/               # FastAPI backend (main API)
│   └── streamlit_app/     # Streamlit frontend UI code
├── agents/                # LangGraph agent nodes and graph definition
│   ├── graph.py
│   ├── schemas.py         # Pydantic schemas for validation
│   └── state.py           # LangGraph state definition
├── docker/                # Dockerfiles for custom images
│   ├── Dockerfile.api
│   ├── Dockerfile.ml
│   └── Dockerfile.streamlit
├── ml/                    # Machine Learning model code
│   ├── models/            # Saved model files (e.g., .joblib)
│   ├── serve_model.py     # FastAPI service for ML model serving
│   └── train_dummy_model.py # Script to train placeholder model
├── parsers/               # Document parsing utilities
│   └── document_parser.py # Logic using Ollama/pdfplumber/pandas
└── storage/               # Database interaction code
    ├── database.py        # SQLAlchemy engine/session setup
    └── models.py          # SQLAlchemy ORM models (Postgres schema)