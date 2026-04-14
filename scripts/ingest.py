"""
Ingestion pipeline script.

Run this once to download, parse, chunk, embed, and index all the
10-K filings. After this runs, the Streamlit app will have everything
it needs to answer questions.

Usage: python scripts/ingest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.sec_downloader import SECDownloader
from src.ingestion.document_parser import DocumentParser
from src.ingestion.chunker import DocumentChunker
from src.vectorstore.embedder import DocumentEmbedder
from src.vectorstore.faiss_store import FAISSStore


def run_ingestion():
    print("=" * 50)
    print("FinSight Ingestion Pipeline")
    print("=" * 50)

    # step 1: download filings from SEC EDGAR
    print("\n[Step 1/5] Downloading 10-K filings from SEC EDGAR...")
    downloader = SECDownloader()
    downloads = downloader.download_all(num_filings=1)
    downloader.save_metadata(downloads)

    # step 2: parse the HTML filings into sections
    print("\n[Step 2/5] Parsing filings into sections...")
    parser = DocumentParser()
    documents = parser.parse_all_filings()

    if not documents:
        print("No documents were parsed. Check if the download step worked.")
        return

    # step 3: chunk the sections
    print("\n[Step 3/5] Chunking sections...")
    chunker = DocumentChunker()
    chunks = chunker.chunk_all_documents(documents)
    chunker.save_chunks(chunks)

    # step 4: generate embeddings
    print("\n[Step 4/5] Generating embeddings (this might take a minute)...")
    embedder = DocumentEmbedder()
    embeddings = embedder.embed_chunks(chunks)
    embedder.save_embeddings(embeddings)

    # step 5: build FAISS index
    print("\n[Step 5/5] Building FAISS index...")
    store = FAISSStore()
    store.build_index(embeddings, chunks)
    store.save()

    print("\n" + "=" * 50)
    print("Done! The pipeline is ready.")
    print(f"  Companies: {len(store.get_available_companies())}")
    print(f"  Chunks indexed: {len(chunks)}")
    print(f"  Sections: {len(store.get_available_sections())}")
    print("\nYou can now run the app with: streamlit run src/app/main.py")
    print("=" * 50)


if __name__ == "__main__":
    run_ingestion()
