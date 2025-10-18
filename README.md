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