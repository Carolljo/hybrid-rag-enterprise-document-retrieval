# Hybrid RAG System for Enterprise Document Retrieval

A production-oriented Retrieval-Augmented Generation (RAG) system for grounded question answering over enterprise documents.

The system combines dense semantic retrieval, BM25 keyword search, Reciprocal Rank Fusion (RRF), CrossEncoder reranking, Gemini-based answer generation, citation verification, REST API access, a Streamlit frontend, automated evaluation, and Docker containerization.

---

## Project Overview

Enterprise information is often distributed across multiple documents such as security policies, employee handbooks, remote-work policies, incident-response procedures, leave policies, and operational guides.

Traditional keyword search can fail when a user's wording differs from the terminology used in the documents. Pure semantic search can also miss exact terms, policy names, acronyms, or security-related keywords.

This project implements a Hybrid RAG architecture that combines:

- Dense semantic retrieval
- BM25 keyword retrieval
- Reciprocal Rank Fusion
- CrossEncoder reranking
- Grounded LLM generation
- Citation extraction and verification
- FastAPI REST API
- Streamlit frontend
- Retrieval evaluation
- Golden-answer generation evaluation
- Hallucination-resistance evaluation
- Chunking strategy evaluation
- Automated API testing
- Docker containerization

---

## Problem Statement

The goal is to build a question-answering system that can:

1. Retrieve relevant information from multiple enterprise documents.
2. Handle both semantic and keyword-based queries.
3. Combine multiple retrieval strategies.
4. Rerank retrieved results for better relevance.
5. Generate answers using retrieved evidence.
6. Return source citations with generated answers.
7. Verify citation provenance against retrieved chunks.
8. Reduce hallucinations when requested information is not present.
9. Expose the pipeline through a REST API.
10. Provide a simple user-facing web interface.
11. Evaluate retrieval and generation quality systematically.
12. Run reproducibly using Docker containers.

---

## System Architecture

```text
Enterprise Documents
        |
        v
Document Loading
        |
        v
Text Chunking
        |
        +-----------------------+
        |                       |
        v                       v
Dense Retrieval           BM25 Retrieval
        |                       |
        v                       |
Vector Store                   |
        |                       |
        +-----------+-----------+
                    |
                    v
          Reciprocal Rank Fusion
                    |
                    v
          CrossEncoder Reranking
                    |
                    v
             Retrieved Context
                    |
                    v
              Prompt Builder
                    |
                    v
          Gemini LLM Generation
                    |
                    v
      Citation Extraction + Verification
                    |
                    v
               FastAPI API
                    |
                    v
           Streamlit Frontend
```

---

## How the Pipeline Works

### 1. Document Loading

Enterprise documents are loaded from the local document collection while preserving metadata such as the source filename.

### 2. Text Chunking

Documents are divided into overlapping character-based chunks suitable for retrieval and LLM context construction.

The production configuration currently uses:

```text
Chunk size: 800 characters
Chunk overlap: 150 characters
```

Alternative chunk configurations were benchmarked separately during evaluation.

### 3. Dense Retrieval

Sentence-transformer embeddings represent document chunks and user queries as dense vectors.

ChromaDB is used as the vector store for semantic retrieval.

### 4. BM25 Retrieval

BM25 provides lexical retrieval based on exact words and term frequency.

This is particularly useful for enterprise queries containing:

- Policy names
- Technical terminology
- Acronyms
- Security terms
- Exact phrases

### 5. Hybrid Retrieval

Dense and BM25 retrieval results are combined using Reciprocal Rank Fusion (RRF).

This allows the system to benefit from both semantic similarity and lexical matching.

### 6. CrossEncoder Reranking

The fused retrieval candidates are reranked using a CrossEncoder model.

Unlike the initial retrieval stage, the CrossEncoder evaluates the query and candidate text together, providing a stronger relevance signal.

### 7. Grounded Generation

The highest-ranked context is passed to Gemini with instructions to answer using the retrieved evidence.

The generation layer is designed to refuse unsupported questions rather than relying on outside knowledge.

### 8. Citation Verification

Retrieved source metadata is converted into structured citations.

Returned citations are checked against the retrieved document chunks to verify their provenance.

Citation confidence represents the proportion of returned citations that can be verified against the retrieved context.

It does **not** represent the probability that the generated answer itself is factually correct.

### 9. API Layer

FastAPI exposes the complete RAG pipeline through HTTP endpoints.

### 10. Frontend

A Streamlit frontend communicates with the FastAPI service and provides an interactive interface for asking questions and viewing:

- Generated answers
- Source documents
- Citation verification status
- Citation confidence

---

## Enterprise Document Collection

The sample knowledge base contains 10 enterprise documents covering:

- Acceptable use
- Data classification
- Employee policies
- Expense reimbursement
- Incident response
- IT security
- Leave policies
- Password and access management
- Product support
- Remote work

The documents intentionally contain overlapping information, allowing multi-document retrieval behavior to be evaluated.

---

## Project Structure

```text
hybrid-rag-enterprise-document-retrieval/
|
|-- api/
|   |-- config.py
|   |-- main.py
|   |-- schemas.py
|
|-- data/
|   |-- raw/
|   |-- processed/
|
|-- evaluation/
|   |-- evaluation_questions.json
|   |-- evaluate_retrieval.py
|   |-- evaluate_generation.py
|   |-- evaluate_chunking.py
|   |-- results.md
|
|-- frontend/
|   |-- app.py
|   |-- Dockerfile
|
|-- src/
|   |-- document_loader.py
|   |-- text_chunker.py
|   |-- embeddings.py
|   |-- vector_store.py
|   |-- bm25_retriever.py
|   |-- hybrid_retriever.py
|   |-- reranker.py
|   |-- prompt_builder.py
|   |-- llm.py
|   |-- citations.py
|   |-- rag_pipeline.py
|
|-- tests/
|   |-- test_api.py
|
|-- .dockerignore
|-- .env.example
|-- .gitignore
|-- docker-compose.yml
|-- Dockerfile
|-- requirements.txt
|-- README.md
```

---

## Technology Stack

- Python 3.11
- FastAPI
- Uvicorn
- Streamlit
- Google Gemini API
- Sentence Transformers
- CrossEncoder
- PyTorch
- ChromaDB
- BM25
- Pydantic Settings
- Pytest
- Docker
- Docker Compose

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd hybrid-rag-enterprise-document-retrieval
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Create a `.env` file in the project root.

Use `.env.example` as the template.

Example:

```env
GEMINI_API_KEY=your_gemini_api_key_here
RAG_MOCK_MODE=false
```

### Gemini API Key

`GEMINI_API_KEY` contains the Gemini API credential used for real answer generation.

The real `.env` file is excluded from Git.

### Mock Mode

The application supports a mock mode:

```env
RAG_MOCK_MODE=true
```

When mock mode is enabled, retrieval, reranking, citation extraction, and citation verification can run without making Gemini API requests.

For real end-to-end generation:

```env
RAG_MOCK_MODE=false
```

Mock mode is useful for development, API testing, and avoiding unnecessary external API usage.

---

## Running the API Locally

Start the FastAPI application:

```bash
uvicorn api.main:app --reload
```

During startup, the application initializes:

- Document loading
- Text chunking
- Embedding model
- Document embeddings
- ChromaDB collection
- BM25 index
- CrossEncoder reranker
- Gemini client when mock mode is disabled

FastAPI interactive documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### Health Check

```http
GET /health
```

Example successful response:

```json
{
  "status": "healthy",
  "rag_pipeline": "ready"
}
```

If the RAG pipeline has not initialized successfully, the endpoint returns HTTP `503`.

### Ask a Question

```http
POST /ask
```

Example request:

```json
{
  "question": "What is the VPN policy for remote access?"
}
```

The endpoint processes the question through the complete Hybrid RAG pipeline and returns a grounded answer together with supporting citations and citation-confidence information.

---

## Running the Streamlit Frontend

The Streamlit frontend provides an interactive interface for the RAG system.

When running outside Docker, start the API first and then launch the frontend according to its configured API URL.

The interface allows users to:

- Enter enterprise-policy questions
- View generated answers
- Inspect cited source documents
- See citation verification status
- View citation confidence

---

## Running with Docker

The project includes separate containers for:

- FastAPI backend
- Streamlit frontend

Docker Compose manages the two services and their internal networking.

Build the images:

```bash
docker compose build
```

Start the application:

```bash
docker compose up
```

The services are exposed at:

```text
FastAPI:   http://localhost:8000
Streamlit: http://localhost:8501
```

Inside the Compose network, the frontend communicates with the API using the service hostname rather than `localhost`.

Stop the containers with:

```bash
docker compose down
```

The Docker configuration uses CPU-only PyTorch packages to reduce container size compared with unnecessary CUDA-enabled dependencies.

---

## Automated Testing

API behavior is tested using Pytest.

Run:

```bash
pytest -v
```

Current result:

```text
7 tests passed
```

Coverage includes:

- Healthy pipeline status
- Pipeline-not-ready status
- Successful question answering
- Empty-question validation
- Missing-question validation
- Invalid question-type validation
- Unexpected pipeline failure handling

---

## Golden Evaluation Dataset

The project includes an expanded golden evaluation dataset containing:

```text
Total questions:        55
Answerable questions:   50
Unanswerable questions: 5
```

The dataset contains:

- Direct questions
- Multi-document questions
- Ambiguous questions
- Deliberately unanswerable questions

Each case contains expected source documents and, where applicable, a human-written reference answer.

The benchmark is designed specifically for the project's enterprise-document corpus.

---

## Retrieval Evaluation

Retrieval quality is evaluated against the 50 answerable golden questions.

Run:

```bash
python -m evaluation.evaluate_retrieval
```

### Current Results

| Metric | Result |
|---|---:|
| Answerable Questions | 50 |
| Questions With At Least One Relevant Source | 50/50 |
| Retrieval Hit Rate | 100.00% |
| Expected Source References | 82 |
| Expected Sources Retrieved | 66/82 |
| Overall Source Recall | 80.49% |

### Metric Interpretation

**Retrieval Hit Rate** measures whether at least one expected relevant source was retrieved for each question.

A 100% hit rate means every answerable evaluation question retrieved at least one expected source.

**Overall Source Recall** measures the proportion of all expected source-document references that appeared in the final retrieved results.

The system retrieved 66 of 82 expected sources, producing an overall source recall of 80.49%.

Therefore, a 100% hit rate does **not** mean retrieval is perfect. Some multi-document questions retrieve one expected source while missing additional relevant sources.

---

## Chunking Strategy Evaluation

Three chunk configurations were evaluated against the 50 answerable golden questions.

Run:

```bash
python -m evaluation.evaluate_chunking
```

### Results

| Configuration | Chunk Size | Overlap | Generated Chunks | Hit Rate | Source Recall |
|---|---:|---:|---:|---:|---:|
| Small | 400 | 75 | 87 | 100.00% | **85.37%** |
| Current | 800 | 150 | 45 | 100.00% | 80.49% |
| Large | 1200 | 200 | 30 | 100.00% | 84.15% |

The 400-character configuration achieved the highest source recall on this benchmark.

However, retrieval recall alone is not sufficient evidence to declare it the best production configuration. Smaller chunks can improve source retrieval while reducing surrounding context available to the generation model.

For that reason, the production configuration remains at 800 characters with 150-character overlap pending broader generation-quality evaluation.

---

## Generation Evaluation

The generation evaluator compares generated answers with human-written golden reference answers using semantic similarity.

The current pass threshold is:

```text
Semantic similarity >= 0.70
```

Semantic similarity is treated as an automated proxy for reference-answer agreement. It is **not** treated as a perfect factual-correctness metric.

The evaluator also measures:

- Answer pass rate
- Average semantic similarity
- Performance by question type
- Hallucination resistance
- Citation provenance confidence

### Batched Evaluation

Gemini API quotas can prevent all 55 questions from being evaluated in a single run.

The evaluator therefore supports inclusive question ranges:

```bash
python -m evaluation.evaluate_generation --start 1 --end 15
```

Additional batches can be run with:

```bash
python -m evaluation.evaluate_generation --start 16 --end 30
python -m evaluation.evaluate_generation --start 31 --end 45
python -m evaluation.evaluate_generation --start 46 --end 55
```

This prevents every evaluation attempt from having to restart at Question 1.

### Current Generation Status

A complete 55-question generation benchmark has **not yet been completed** because the external Gemini service encountered temporary availability and free-tier quota limits during evaluation.

One partial run successfully evaluated six answerable questions before quota exhaustion:

```text
Questions evaluated: 6
Passed: 5/6
Partial pass rate: 83.33%
Average semantic similarity: 0.788
```

These numbers are reported only as a **partial diagnostic run** and should not be interpreted as the final generation benchmark.

The full generation benchmark remains pending.

---

## Hallucination Resistance

The golden dataset contains five deliberately unsupported questions.

These test whether the generation layer refuses to invent information that does not exist in the enterprise corpus.

Examples include questions about information such as:

- Company annual revenue
- Company leadership
- Current stock price

The full five-question hallucination-resistance result will be recorded after the corresponding real-generation evaluation batch is completed.

---

## Citation Verification

The RAG pipeline extracts source citations from retrieved document metadata and verifies them against the retrieved chunks.

Citation confidence is calculated as the proportion of returned citations successfully verified against retrieved evidence.

A real end-to-end application test returned verified enterprise-document citations with:

```text
Citation Verification: 100%
```

This score represents **citation provenance verification only**.

It does not mean the generated answer has a 100% probability of being factually correct.

---

## Error Handling

The API includes handling for common failure conditions.

Examples include:

- Invalid questions
- Empty questions
- RAG pipeline initialization failures
- LLM or external-service failures
- Pipeline-not-ready conditions

Unexpected internal errors are converted to controlled API responses rather than exposing unnecessary implementation details to clients.

The generation evaluator also detects common Gemini availability and quota errors and stops cleanly instead of treating unavailable API calls as failed answers.

---

## Security and Configuration

Sensitive configuration is managed using environment variables.

The repository excludes items such as:

- `.env`
- Virtual environments
- Python cache files
- Pytest cache
- Local ChromaDB data

A safe `.env.example` documents required configuration without exposing real credentials.

API keys should never be committed to the repository or embedded directly in source code.

---

## Current Limitations

The current implementation has several limitations:

1. The enterprise corpus is intentionally small and project-specific.
2. ChromaDB is used locally rather than through a managed production vector service.
3. Retrieval does not recover every expected source; current overall source recall is 80.49%.
4. Generation depends on external Gemini API availability and quota.
5. The API does not currently implement authentication or authorization.
6. Document ingestion occurs during pipeline initialization rather than through a dynamic ingestion service.
7. The golden benchmark is manually constructed and specific to the included enterprise corpus.
8. Semantic similarity is only an automated proxy for generation quality.
9. The complete 55-question real-generation benchmark remains pending due to external API quota limitations.
10. Docker deployment has been validated locally but has not yet been deployed to a managed cloud container platform.

---

## Future Improvements

Potential extensions include:

- Larger enterprise document collections
- Automated document ingestion
- Incremental vector-index updates
- Managed vector databases
- Authentication and role-based access control
- Query and retrieval observability
- Response caching
- Retrieval parameter tuning
- Generation-aware chunk-size optimization
- Additional retrieval metrics such as Precision@K, Recall@K, and MRR
- Larger independent evaluation datasets
- Persistent evaluation-result storage
- Managed cloud deployment
- CI/CD pipelines
- Monitoring and production telemetry

---

## Key Engineering Concepts Demonstrated

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation
- Dense semantic search
- Sparse BM25 retrieval
- Hybrid information retrieval
- Reciprocal Rank Fusion
- CrossEncoder reranking
- Vector databases
- Prompt grounding
- Citation extraction
- Citation provenance verification
- Hallucination-resistance evaluation
- Golden-dataset evaluation
- Retrieval evaluation
- Chunking-strategy experimentation
- REST API development
- Streamlit application development
- Configuration and secret management
- Automated API testing
- Docker containerization
- Multi-container application orchestration

---

## Evaluation Status

| Component | Status |
|---|---|
| Hybrid Retrieval | Complete |
| CrossEncoder Reranking | Complete |
| Citation Pipeline | Complete |
| FastAPI Integration | Complete |
| Streamlit Frontend | Complete |
| Docker Containerization | Complete |
| Docker End-to-End Test | Passed |
| API Automated Tests | 7/7 Passed |
| Golden Evaluation Dataset | 55 Cases |
| Retrieval Hit Rate | 100.00% |
| Overall Source Recall | 80.49% |
| Chunking Comparison | Complete |
| Best Tested Chunk Source Recall | 85.37% |
| Full Generation Evaluation | Pending API quota availability |

---

## Evaluation Limitations

Evaluation results should be interpreted within the scope of this project.

The benchmark uses a small synthetic enterprise corpus and manually constructed questions. It is useful for comparing versions of this system, but it is not a general-purpose RAG benchmark.

Source-level retrieval evaluation determines whether expected documents were retrieved. It does not independently prove that every required fact or passage was present in the retrieved context.

Semantic similarity provides an automated comparison between generated and reference answers but cannot fully replace human factual evaluation.

External LLM availability and quota restrictions can also affect the ability to complete generation benchmarks.

Detailed measured results are maintained in:

```text
evaluation/results.md
```

---

## License

This project is intended for educational and portfolio purposes.