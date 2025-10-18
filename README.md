

````markdown
# Social Support AI Workflow Automation

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red?style=for-the-badge&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?style=for-the-badge&logo=docker)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?style=for-the-badge&logo=postgresql)
![Neo4j](https://img.shields.io/badge/Neo4j-5-008CC1?style=for-the-badge&logo=neo4j)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-orange?style=for-the-badge)

An AI-powered workflow to automate social support applications, reducing processing time from weeks to minutes using local LLMs and an agentic architecture.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Setup and Running Instructions](#setup-and-running-instructions)
- [Accessing Services](#accessing-services)
- [Project Structure](#project-structure)

## Project Overview

This project implements an AI-powered workflow to automate the application process for a government social security department. The goal is to drastically reduce the assessment and approval time from the current 5-20 working days to potentially minutes, leveraging multimodal AI, agentic orchestration, and local large language models (LLMs).

The solution addresses key pain points of the manual process, including data entry errors, information inconsistencies, review bottlenecks, and subjective decision-making.

## Key Features

- **Automated Data Extraction:** Uses local multimodal (Qwen2-VL) and text (Llama 3.1) models via Ollama to parse data from PDFs, images, and Excel files.
- **Agentic Orchestration:** A stateful, multi-step workflow managed by LangGraph ensures a robust and logical progression from extraction to decision.
- **Multi-Database Strategy:** Persists data across specialized databases for optimal use:
  - **PostgreSQL:** For structured "golden record" data.
  - **MongoDB:** For raw, unstructured extraction results.
  - **Neo4j:** For modeling and querying complex relationships to detect inconsistencies.
  - **Qdrant:** For vector embeddings to power semantic search for RAG.
- **Explainable Eligibility Scoring:** A Scikit-learn model served via a dedicated API provides objective, auditable eligibility predictions.
- **RAG-Powered Recommendations:** Generates economic enablement suggestions (job matching, training) using vector search.
- **Full Observability:** End-to-end tracing of the agentic workflow is handled via Langfuse Cloud.
- **Interactive UI:** A simple Streamlit application provides the interface for applicants.

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
    end

    subgraph Data & Persistence Layer
        A4 -- (Raw JSON) --> DB_MONGO(MongoDB @ :27017)
        A4 & A7 -- (Golden Record) --> DB_PG(PostgreSQL @ :5432)
        A4 -- (Relationships) --> DB_NEO4J(Neo4j @ :7687)
        A4 & A6 -- (Embeddings/RAG) --> DB_QDRANT(Qdrant @ :6333)
    end

    subgraph Classical ML Service
        A5 -- Feature Vector --> ML_API(FastAPI @ :8001)
        ML_API -- Loads --> MODEL(SK-Learn Model\n*.joblib)
        ML_API -- Prediction --> A5
    end

    subgraph Caching
        API -- Session/Cache --> DB_REDIS(Redis @ :6379)
    end
````

## Technology Stack

  - **Programming Language:** Python 3.11+
  - **Web Frameworks:** FastAPI, Streamlit
  - **Agent Orchestration:** LangGraph
  - **Local LLM Hosting:** Ollama
  - **Document Parsing:** `ollama` client, `pdfplumber`, `pandas`
  - **Embeddings:** Sentence Transformers
  - **Machine Learning:** Scikit-learn, Joblib
  - **Databases:** PostgreSQL (SQLAlchemy, Alembic), MongoDB, Neo4j, Qdrant, Redis
  - **Observability:** Langfuse Cloud
  - **Containerization:** Docker, Docker Compose
  - **Schema/Validation:** Pydantic

## Setup and Running Instructions

**Prerequisites:**

  * Docker and Docker Compose
  * Python \>= 3.11
  * Git

**Execution Steps:**

1.  **Clone the Repository:**

    ```bash
    git clone <your-repo-url>
    cd social-support-ai
    ```

2.  **Create `.env` File:**
    Create a file named `.env` in the project root. Copy the contents from `.env.example` (if provided) or create it manually with the following variables. **You must provide your own values.**

    ```ini
    # PostgreSQL Credentials
    POSTGRES_USER=social_admin
    POSTGRES_PASSWORD=strongpassword123
    POSTGRES_DB=social_support_db

    # Neo4j Password (format: neo4j/your_password)
    NEO4J_AUTH=neo4j/verystrongpassword

    # Langfuse Cloud Credentials (get from cloud.langfuse.com)
    LANGFUSE_HOST=[https://cloud.langfuse.com](https://cloud.langfuse.com)
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_PUBLIC_KEY=pk-lf-...

    # API Port
    API_PORT=8000
    ```

3.  **Build and Run Docker Containers:**
    This command will build the custom images and start all services.

    ```bash
    docker-compose up -d --build
    ```

    Please wait 1-2 minutes for all database services to initialize fully.

4.  **Pull Ollama Models:**
    Download the required local LLMs into the running Ollama container:

    ```bash
    docker-compose exec ollama ollama pull llama3.1:8b
    docker-compose exec ollama ollama pull qwen2-vl:7b
    ```

5.  **Run Database Migrations:**
    Apply the initial PostgreSQL schema using Alembic:

    ```bash
    docker-compose exec api alembic upgrade head
    ```

    The application is now ready.

## Accessing Services

Once the containers are running, the following services are available:

  - **Streamlit UI:** `http://localhost:8501`
  - **FastAPI Backend Docs:** `http://localhost:8000/docs`
  - **Neo4j Browser:** `http://localhost:7474`
  - **ML Service Health:** `http://localhost:8001/health`
  - **Langfuse Traces:** Log in to your account at `https://cloud.langfuse.com`

## Project Structure

```
social-support-ai/
├── README.md              # This file
├── solution_summary.pdf   # Solution summary document
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Docker service definitions
├── alembic/               # Alembic migration scripts
├── apps/                  # Application code (API & UI)
├── agents/                # LangGraph agent nodes and graph definition
├── docker/                # Dockerfiles for custom images
├── ml/                    # Machine Learning model code and serving API
├── parsers/               # Document parsing utilities
└── storage/               # Database interaction code and models
```

```
```