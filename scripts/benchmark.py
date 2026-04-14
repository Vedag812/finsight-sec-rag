"""
Simple benchmark script for measuring query performance.

Runs a set of test queries through the pipeline and records
how long each one takes. Outputs average latency, p95 latency,
and queries per minute.

Usage: python scripts/benchmark.py
"""

import time
import statistics
import sys
from pathlib import Path

# make imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorstore.embedder import DocumentEmbedder
from src.vectorstore.faiss_store import FAISSStore
from src.pipeline.graph import RAGPipeline


# test queries covering different types of questions
TEST_QUERIES = [
    "What are the main risk factors for Goldman Sachs?",
    "How does Apple describe its business?",
    "What is Microsoft's revenue growth?",
    "Compare risk factors between Tesla and Apple",
    "What does JPMorgan say about market risk?",
    "How does Goldman Sachs generate revenue?",
    "What are Tesla's manufacturing risks?",
    "Compare business models of Microsoft and Apple",
    "What regulatory risks does JPMorgan face?",
    "How does Goldman Sachs discuss technology in their filing?",
]


def run_benchmark():
    print("Loading pipeline...")
    store = FAISSStore()
    store.load()

    embedder = DocumentEmbedder()
    pipeline = RAGPipeline(faiss_store=store, embedder=embedder)

    print(f"Running {len(TEST_QUERIES)} test queries...\n")

    latencies = []
    results = []

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"[{i}/{len(TEST_QUERIES)}] {query}")

        start = time.time()
        result = pipeline.ask(query)
        elapsed_ms = (time.time() - start) * 1000

        latencies.append(elapsed_ms)
        results.append({
            "query": query,
            "latency_ms": elapsed_ms,
            "query_type": result["query_type"],
            "num_citations": len(result["citations"]),
            "answer_length": len(result["answer"]),
        })

        print(f"  {elapsed_ms:.0f}ms | {result['query_type']} | {len(result['citations'])} citations\n")

    # calculate stats
    avg_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    min_latency = min(latencies)
    max_latency = max(latencies)
    total_time = sum(latencies) / 1000  # in seconds
    queries_per_min = (len(TEST_QUERIES) / total_time) * 60

    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Queries run:          {len(TEST_QUERIES)}")
    print(f"Total time:           {total_time:.1f}s")
    print(f"")
    print(f"Average latency:      {avg_latency:.0f}ms")
    print(f"Median latency:       {median_latency:.0f}ms")
    print(f"P95 latency:          {p95_latency:.0f}ms")
    print(f"Min latency:          {min_latency:.0f}ms")
    print(f"Max latency:          {max_latency:.0f}ms")
    print(f"")
    print(f"Throughput:           {queries_per_min:.1f} queries/min")
    print("=" * 60)

    # save results to file
    output_path = Path("data/benchmark_results.txt")
    with open(output_path, "w") as f:
        f.write("FinSight Benchmark Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Queries: {len(TEST_QUERIES)}\n")
        f.write(f"Avg latency: {avg_latency:.0f}ms\n")
        f.write(f"Median latency: {median_latency:.0f}ms\n")
        f.write(f"P95 latency: {p95_latency:.0f}ms\n")
        f.write(f"Throughput: {queries_per_min:.1f} queries/min\n\n")
        f.write("Individual queries:\n")
        for r in results:
            f.write(f"  [{r['latency_ms']:.0f}ms] {r['query']}\n")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_benchmark()
