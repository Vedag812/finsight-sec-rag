"""Quick debug script to test each pipeline step."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import traceback

# Step 1: LLM
try:
    from src.pipeline.llm import GeminiClient
    print("1. Testing LLM client...")
    c = GeminiClient()
    res = c.classify("Say hello")
    print("   LLM OK:", res[:50])
except Exception as e:
    traceback.print_exc()

# Step 2: Retrieval
try:
    from src.vectorstore.faiss_store import FAISSStore
    from src.vectorstore.embedder import DocumentEmbedder
    print("2. Testing retrieval...")
    store = FAISSStore()
    store.load()
    embedder = DocumentEmbedder()
    emb = embedder.embed_query("What does Apple do?")
    results = store.search(emb, top_k=3)
    print(f"   Retrieved {len(results)} chunks")
    for chunk, score in results:
        cit = chunk.citation[:60]
        print(f"   Score: {score:.3f} | {cit}")
except Exception as e:
    traceback.print_exc()

# Step 3: Full pipeline
try:
    from src.pipeline.graph import RAGPipeline
    print("3. Testing full pipeline...")
    pipeline = RAGPipeline(faiss_store=store, embedder=embedder)
    result = pipeline.ask("What does Apple do?")
    ans = result["answer"]
    lat = result["latency_ms"]
    print(f"   Answer length: {len(ans)}")
    print(f"   Latency: {lat}ms")
    print(f"   First 200 chars: {ans[:200]}")
except Exception as e:
    traceback.print_exc()
