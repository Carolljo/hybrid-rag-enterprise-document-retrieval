# Hybrid RAG System for Enterprise Document Retrieval

A production-oriented Retrieval-Augmented Generation (RAG) system for grounded question answering over enterprise documents.

The system combines dense semantic retrieval, BM25 keyword search, Reciprocal Rank Fusion (RRF), CrossEncoder reranking, and Gemini-based answer generation to retrieve relevant enterprise information and generate answers grounded in source documents.

---

## Project Overview

Enterprise information is often distributed across multiple documents such as security policies, employee handbooks, remote-work policies, incident-response procedures, and operational guides.

Traditional keyword search can fail when a user's wording differs from the terminology used in the documents. Pure semantic search can also miss exact terms, policy names, or security-related keywords.

This project implements a Hybrid RAG architecture that combines:

- Dense semantic retrieval
- BM25 keyword retrieval
- Reciprocal Rank Fusion
- CrossEncoder reranking
- Grounded LLM generation
- Source citations
- FastAPI REST API
- Retrieval and hallucination-resistance evaluation

---

## Problem Statement

The goal is to build a question-answering system that can:

1. Retrieve relevant information from multiple enterprise documents.
2. Handle both semantic and keyword-based queries.
3. Rerank retrieved results for better relevance.
4. Generate answers using only retrieved evidence.
5. Return source citations with generated answers.
6. Reduce hallucinations when the requested information is not present.
7. Expose the complete pipeline through a REST API.

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
         Grounded Answer + Citations
                    |
                    v
                FastAPI
```

---

## How the Pipeline Works

### 1. Document Loading

Enterprise documents are loaded from the local document collection while preserving metadata such as the source filename.

### 2. Text Chunking

Documents are divided into smaller chunks suitable for retrieval and LLM context construction.

### 3. Dense Retrieval

Sentence-transformer embeddings represent document chunks and user queries as dense vectors.

A vector database is used to retrieve semantically similar chunks.

### 4. BM25 Retrieval

BM25 provides lexical retrieval based on exact words and term frequency.

This is useful for enterprise queries containing:

- policy names
- technical terminology
- acronyms
- security terms
- exact phrases

### 5. Hybrid Retrieval

Dense and BM25 retrieval results are combined using Reciprocal Rank Fusion (RRF).

This allows the system to benefit from both semantic similarity and keyword matching.

### 6. CrossEncoder Reranking

The fused candidates are reranked using a CrossEncoder model.

Unlike the initial retrieval stage, the CrossEncoder evaluates the query and candidate document together, providing a more precise relevance score.

### 7. Grounded Generation

The highest-ranked context is passed to Gemini with instructions to answer using the retrieved evidence.

The generation layer is designed to avoid inventing unsupported information when the documents do not contain an answer.

### 8. Citations

Generated responses include citations identifying the enterprise documents and chunks used as supporting evidence.

---

## Enterprise Document Collection

The sample knowledge base contains enterprise documents covering areas such as:

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

These documents provide overlapping information, making them useful for evaluating multi-document retrieval.

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
|-- .env.example
|-- .gitignore
|-- requirements.txt
|-- README.md
```

---

## Technology Stack

- Python 3.11
- FastAPI
- Uvicorn
- Google Gemini API
- Sentence Transformers
- CrossEncoder
- ChromaDB
- BM25
- Pydantic Settings
- Pytest

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd hybrid-rag-enterprise-document-retrieval
```

### 2. Create a virtual environment

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

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Create a `.env` file in the project root.

You can use `.env.example` as the template:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace the placeholder with a valid Gemini API key.

The real `.env` file is excluded from Git to prevent API keys from being committed.

---

## Running the API

Start the FastAPI application:

```bash
uvicorn api.main:app --reload
```

During startup, the application initializes the RAG pipeline, including document loading, chunking, embeddings, retrieval indexes, reranking, and the Gemini client.

Once the server is running, FastAPI's interactive API documentation is available at:

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

The endpoint processes the question through the complete Hybrid RAG pipeline and returns a grounded answer together with supporting citations.

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

The tests cover:

- Healthy API state
- Pipeline-not-ready state
- Successful question requests
- Empty questions
- Missing questions
- Invalid question types
- Pipeline failures

---

## Retrieval Evaluation

Retrieval quality is evaluated using a dedicated set of answerable enterprise questions with expected source documents.

Run:

```bash
python -m evaluation.evaluate_retrieval
```

### Current Results

| Metric | Result |
|---|---:|
| Evaluation Questions | 10 |
| Questions With At Least One Relevant Source | 10/10 |
| Retrieval Hit Rate | 100.00% |
| Expected Source Matches | 16/19 |
| Overall Source Recall | 84.21% |

### Metric Interpretation

**Retrieval Hit Rate** measures whether at least one expected relevant source was retrieved for each question.

A 100% hit rate means every evaluation question retrieved at least one expected source.

**Source Recall** measures the proportion of all expected source-document associations that were successfully retrieved.

The system retrieved 16 of 19 expected sources, resulting in an overall source recall of 84.21%.

This distinction is important because retrieving one relevant source is not the same as retrieving every relevant source.

---

## Hallucination-Resistance Evaluation

The evaluation dataset also contains questions whose answers are intentionally absent from the enterprise documents.

Examples include questions about:

- Company annual revenue
- Company CEO
- Current stock price

These questions are used to evaluate whether the generation layer refuses to invent unsupported information.

Run:

```bash
python -m evaluation.evaluate_generation
```

The evaluator includes handling for Gemini API quota exhaustion so that evaluation stops cleanly instead of repeatedly sending requests after the quota has been exhausted.

Final generation results should be recorded only after completing the evaluation with the configured Gemini model.

---

## Error Handling

The API includes handling for common failure conditions.

Examples include:

- Invalid or empty questions
- RAG pipeline initialization failures
- LLM or external-service failures
- Pipeline-not-ready conditions

Unexpected internal errors are converted to controlled API responses rather than exposing implementation details to clients.

---

## Security and Configuration

Sensitive configuration is managed using environment variables.

The repository excludes:

- `.env`
- Virtual environments
- Python cache files
- Pytest cache
- Local ChromaDB data
- Generated processed data

A safe `.env.example` file documents the required environment variables without exposing credentials.

---

## Current Limitations

The current implementation has several limitations:

1. The document collection is relatively small and stored locally.
2. The vector database is built locally rather than using a managed production vector service.
3. Retrieval does not recover every expected relevant source; current overall source recall is 84.21%.
4. LLM generation depends on external Gemini API availability and quota.
5. The current API does not implement authentication or authorization.
6. Document ingestion is performed during pipeline initialization rather than through a dynamic ingestion service.
7. Evaluation uses a relatively small manually defined question set.

These limitations provide clear areas for future development.

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
- Improved retrieval tuning
- Additional retrieval metrics such as Precision@K, Recall@K, and MRR
- Larger automated evaluation datasets
- Containerized deployment
- Cloud deployment
- CI/CD pipelines

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
- Hallucination-resistance evaluation
- Retrieval evaluation
- REST API development
- Configuration and secret management
- Automated API testing

---

## Evaluation Status

| Component | Status |
|---|---|
| Hybrid Retrieval | Complete |
| CrossEncoder Reranking | Complete |
| Citation Pipeline | Complete |
| FastAPI Integration | Complete |
| API Automated Tests | 7/7 Passed |
| Retrieval Hit Rate | 100.00% |
| Overall Source Recall | 84.21% |
| Hallucination-Resistance Evaluation | Final validation pending |

---

## License

This project is intended for educational and portfolio purposes.