"""
FAISS vector store for fast similarity search.

FAISS (Facebook AI Similarity Search) is a library for searching through
large collections of vectors quickly. We use it to find which document chunks
are most similar to a user's question.

The basic flow is:
1. Store all chunk embeddings in a FAISS index
2. When a user asks a question, embed the question
3. Search the index for the closest chunk embeddings
4. Return those chunks as context for the LLM

We also support filtering by company or section, so users can narrow
their search to specific parts of specific filings.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import Optional

from src.ingestion.chunker import DocumentChunk


class FAISSStore:
    """
    Wraps a FAISS index with metadata so we can filter and retrieve chunks.
    
    FAISS itself only stores vectors and returns indices. We keep a parallel
    list of chunk metadata so we can map those indices back to actual
    document chunks with company names, sections, page numbers, etc.
    """

    def __init__(self):
        self.index = None
        self.chunks: list[DocumentChunk] = []
        self.dimension: int = 0

    def build_index(self, embeddings: np.ndarray, chunks: list[DocumentChunk]):
        """
        Creates the FAISS index from embeddings and stores the chunk metadata.
        
        We use IndexFlatIP (Inner Product) because our embeddings are normalized,
        which means inner product gives us cosine similarity. This is the simplest
        index type but works fine for our data size (a few thousand chunks).
        """
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Mismatch: got {len(embeddings)} embeddings but {len(chunks)} chunks"
            )

        self.dimension = embeddings.shape[1]
        self.chunks = chunks

        # IndexFlatIP = exact search using inner product (cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        print(f"Built FAISS index with {self.index.ntotal} vectors (dim={self.dimension})")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        company: Optional[str] = None,
        section: Optional[str] = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Search for the most similar chunks to a query embedding.
        
        Args:
            query_embedding: the embedded user question
            top_k: how many results to return
            company: if set, only return chunks from this company (ticker like "AAPL")
            section: if set, only return chunks from this section (like "Item 1A - Risk Factors")
        
        Returns:
            list of (chunk, similarity_score) tuples, sorted by relevance
        """
        if self.index is None:
            raise RuntimeError("Index hasn't been built yet. Call build_index first.")

        # if we need to filter, we search for more results than needed
        # and then filter down, because FAISS doesn't support filtering natively
        search_k = top_k if (company is None and section is None) else top_k * 10

        scores, indices = self.index.search(query_embedding, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue

            chunk = self.chunks[idx]

            # apply filters
            if company and chunk.ticker != company:
                continue
            if section and chunk.section_name != section:
                continue

            results.append((chunk, float(score)))

            if len(results) >= top_k:
                break

        return results

    def search_multi_company(
        self,
        query_embedding: np.ndarray,
        companies: list[str],
        top_k_per_company: int = 3,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Search across multiple companies and get results from each one.
        
        This is used for comparison queries like "How do Apple and Microsoft
        differ on AI risk?" We want results from both companies, not just
        whichever one happens to match better.
        """
        all_results = []
        for company in companies:
            results = self.search(query_embedding, top_k=top_k_per_company, company=company)
            all_results.extend(results)

        # sort by score so the most relevant results come first
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results

    def get_available_companies(self) -> list[str]:
        """Returns a list of all company tickers in the index."""
        tickers = set(chunk.ticker for chunk in self.chunks)
        return sorted(tickers)

    def get_available_sections(self) -> list[str]:
        """Returns a list of all section names in the index."""
        sections = set(chunk.section_name for chunk in self.chunks)
        return sorted(sections)

    def save(self, output_dir: str = "data/indexes"):
        """
        Saves the FAISS index and chunk metadata to disk.
        
        We save two files:
        - faiss.index: the actual vector index
        - chunks_metadata.json: the chunk data that goes with each vector
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # save the FAISS index
        index_path = output_path / "faiss.index"
        faiss.write_index(self.index, str(index_path))

        # save chunk metadata
        meta_path = output_path / "chunks_metadata.json"
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False)

        print(f"Saved index and metadata to {output_path}")

    def load(self, input_dir: str = "data/indexes"):
        """Loads a previously saved FAISS index and chunk metadata."""
        input_path = Path(input_dir)

        index_path = input_path / "faiss.index"
        meta_path = input_path / "chunks_metadata.json"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Index files not found in {input_path}. Run the indexing step first."
            )

        self.index = faiss.read_index(str(index_path))
        self.dimension = self.index.d

        with open(meta_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        self.chunks = [DocumentChunk.from_dict(d) for d in chunks_data]

        print(f"Loaded index with {self.index.ntotal} vectors and {len(self.chunks)} chunks")

    @property
    def is_ready(self) -> bool:
        """Check if the index is loaded and has data."""
        return self.index is not None and self.index.ntotal > 0


if __name__ == "__main__":
    from src.ingestion.chunker import DocumentChunker
    from src.vectorstore.embedder import DocumentEmbedder

    # load chunks and embeddings
    chunks = DocumentChunker.load_chunks()
    embeddings = DocumentEmbedder.load_embeddings()

    # build and save the index
    store = FAISSStore()
    store.build_index(embeddings, chunks)
    store.save()

    # quick test search
    embedder = DocumentEmbedder()
    query_emb = embedder.embed_query("What are the main risk factors for Apple?")
    results = store.search(query_emb, top_k=3, company="AAPL")

    print("\nTest search results:")
    for chunk, score in results:
        print(f"  [{score:.3f}] {chunk.citation}")
        print(f"    {chunk.text[:150]}...")
