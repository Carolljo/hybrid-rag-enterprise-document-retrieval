"""End-to-end Hybrid RAG question answering pipeline."""

from typing import Any

from src.bm25_retriever import build_bm25
from src.citations import extract_citations
from src.document_loader import load_documents
from src.embeddings import (
    embed_documents,
    load_embedding_model,
)
from src.hybrid_retriever import hybrid_search
from src.llm import (
    create_client,
    generate_response,
)
from src.prompt_builder import build_grounded_prompt
from src.reranker import (
    load_reranker,
    rerank_documents,
)
from src.text_chunker import chunk_documents
from src.vector_store import (
    add_documents,
    create_collection,
)


class RAGPipeline:
    """
    End-to-end Hybrid RAG pipeline.

    The pipeline loads enterprise documents, builds semantic
    and BM25 retrieval systems, reranks retrieved candidates,
    generates grounded answers using Gemini, and returns
    structured source citations.
    """

    def __init__(
        self,
        data_directory: str = "data/raw",
        retrieval_k: int = 5,
        rerank_k: int = 3,
    ) -> None:
        """
        Initialize the RAG pipeline.

        Args:
            data_directory:
                Directory containing enterprise documents.

            retrieval_k:
                Number of candidates retrieved before reranking.

            rerank_k:
                Number of chunks retained after reranking.
        """
        self.data_directory = data_directory
        self.retrieval_k = retrieval_k
        self.rerank_k = rerank_k

        self.embedding_model = None
        self.reranker = None
        self.collection = None
        self.bm25 = None
        self.documents: list[dict[str, Any]] = []
        self.client = None

    def initialize(self) -> None:
        """
        Load documents, models, retrieval indexes,
        vector database, reranker, and Gemini client.
        """
        print("Loading enterprise documents...")
        raw_documents = load_documents(
            self.data_directory
        )

        print("Chunking documents...")
        self.documents = chunk_documents(
            raw_documents
        )

        print("Loading embedding model...")
        self.embedding_model = load_embedding_model()

        print("Generating document embeddings...")
        embedded_documents = embed_documents(
            self.documents,
            self.embedding_model,
        )

        print("Creating vector database...")
        self.collection = create_collection(
            reset=True
        )

        add_documents(
            self.collection,
            embedded_documents,
        )

        print("Building BM25 index...")
        self.bm25, self.documents = build_bm25(
            self.documents
        )

        print("Loading CrossEncoder reranker...")
        self.reranker = load_reranker()

        print("Creating Gemini client...")
        self.client = create_client()

        print("RAG pipeline initialized successfully.")

    def retrieve(
        self,
        question: str,
    ) -> list[dict[str, Any]]:
        """
        Retrieve and rerank document chunks.

        Args:
            question:
                User question.

        Returns:
            Final reranked document chunks.
        """
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if (
            self.embedding_model is None
            or self.collection is None
            or self.bm25 is None
            or self.reranker is None
        ):
            raise RuntimeError(
                "RAG pipeline has not been initialized."
            )

        candidates = hybrid_search(
            question=question,
            model=self.embedding_model,
            collection=self.collection,
            bm25=self.bm25,
            documents=self.documents,
            top_k=self.retrieval_k,
        )

        return rerank_documents(
            query=question,
            documents=candidates,
            model=self.reranker,
            top_k=self.rerank_k,
        )

    def answer(
        self,
        question: str,
    ) -> dict[str, Any]:
        """
        Generate a grounded answer with structured citations.

        Args:
            question:
                User question.

        Returns:
            Dictionary containing the user question,
            generated answer, and structured citations.
        """
        if self.client is None:
            raise RuntimeError(
                "RAG pipeline has not been initialized."
            )

        retrieved_documents = self.retrieve(
            question
        )

        prompt = build_grounded_prompt(
            question=question,
            documents=retrieved_documents,
        )

        answer = generate_response(
            client=self.client,
            prompt=prompt,
        )

        citations = extract_citations(
            retrieved_documents
        )

        return {
            "question": question,
            "answer": answer,
            "citations": citations,
        }


if __name__ == "__main__":
    pipeline = RAGPipeline()

    pipeline.initialize()

    question = "What is the VPN policy?"

    print(f"\nQuestion: {question}")
    print("\nRunning complete RAG pipeline...")

    result = pipeline.answer(question)

    print("\nAnswer:")
    print(result["answer"])

    print("\nCitations")
    print("=" * 70)

    for index, citation in enumerate(
        result["citations"],
        start=1,
    ):
        print(
            f"{index}. "
            f"{citation['source']} "
            f"(Chunk {citation['chunk_id']})"
        )