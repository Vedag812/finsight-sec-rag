"""
Generates embeddings for document chunks using sentence-transformers.

Embeddings are basically numerical representations of text. Two pieces of text
that mean similar things will have similar embeddings, which is how we can
search for relevant chunks when a user asks a question.

We're using the all-MiniLM-L6-v2 model which is small and fast. It's not the
most accurate model out there, but it's good enough for our use case and
runs fine on a laptop without a GPU.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.ingestion.chunker import DocumentChunk


class DocumentEmbedder:
    """
    Wraps the sentence-transformer model for generating embeddings.
    
    Takes a list of document chunks, embeds each one, and returns
    the embeddings as a numpy array. Also handles batching so we
    don't run out of memory on large document sets.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Embeds a list of text strings and returns a numpy array of embeddings.
        
        Uses batching to handle large numbers of texts without eating
        all the memory. Shows a progress bar so you know it's working.
        """
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i + batch_size]
            embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                normalize_embeddings=True,  # normalize so we can use dot product for similarity
            )
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings).astype("float32")

    def embed_chunks(self, chunks: list[DocumentChunk], batch_size: int = 64) -> np.ndarray:
        """
        Takes document chunks and returns their embeddings.
        
        We embed just the text content of each chunk. The metadata
        (company, section, etc.) is stored separately and matched
        by index position.
        """
        texts = [chunk.text for chunk in chunks]
        print(f"Embedding {len(texts)} chunks...")
        embeddings = self.embed_texts(texts, batch_size=batch_size)
        print(f"Done. Embeddings shape: {embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single search query. Used at query time."""
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        return embedding.astype("float32")

    def save_embeddings(self, embeddings: np.ndarray, output_dir: str = "data/processed"):
        """Saves embeddings to disk as a numpy file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / "embeddings.npy"
        np.save(save_path, embeddings)
        print(f"Saved embeddings to {save_path}")

    @staticmethod
    def load_embeddings(input_dir: str = "data/processed") -> np.ndarray:
        """Loads previously saved embeddings."""
        load_path = Path(input_dir) / "embeddings.npy"
        if not load_path.exists():
            raise FileNotFoundError(f"No embeddings found at {load_path}. Run the embedding step first.")
        embeddings = np.load(load_path)
        print(f"Loaded embeddings: {embeddings.shape}")
        return embeddings


if __name__ == "__main__":
    from src.ingestion.chunker import DocumentChunker

    # load chunks that were saved earlier
    chunks = DocumentChunker.load_chunks()

    # embed them
    embedder = DocumentEmbedder()
    embeddings = embedder.embed_chunks(chunks)
    embedder.save_embeddings(embeddings)
