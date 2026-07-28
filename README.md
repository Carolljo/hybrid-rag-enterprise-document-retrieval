# Hybrid RAG System for Enterprise Document Retrieval

A production-oriented Retrieval-Augmented Generation (RAG) system for
grounded question answering over enterprise documents.

The project combines dense semantic retrieval with BM25 keyword search
and Reciprocal Rank Fusion (RRF) to improve document retrieval quality.

## Planned Architecture

```text
Enterprise Documents
        ↓
Document Ingestion
        ↓
Text Chunking
        ↓
┌───────────────────┐
│                   │
↓                   ↓
Dense Retrieval    BM25 Retrieval
│                   │
↓                   │
Vector Database     │
│                   │
└─────────┬─────────┘
          ↓
Reciprocal Rank Fusion
          ↓
Retrieved Context
          ↓
LLM Generation
          ↓
Grounded Answer + Citations
          ↓
FastAPI
          ↓
Streamlit