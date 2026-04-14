"""
The main RAG pipeline built with LangGraph.

Architecture: Classify → Retrieve → Generate  (2 LLM calls total)

The previous design had a dedicated "grade" node that made one API call
per retrieved chunk to check relevance — so a query touching 5 chunks cost
7 API calls (1 classify + 5 grade + 1 generate). That node is gone.

The generate prompts now instruct the model to silently skip irrelevant
chunks, so grading happens inside the single generate call for free.

The reformulate → retrieve fallback is preserved: it only fires when FAISS
returns zero chunks (empty retrieval), not on low relevance scores.

Flow:

    classify → retrieve → generate → END
                  |
                  └─ (empty results, first attempt only)
                         ↓
                    reformulate → retrieve → generate → END
"""

import time
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from src.pipeline.llm import GeminiClient
from src.pipeline.prompts import (
    CLASSIFY_QUERY_PROMPT,
    GENERATE_ANSWER_PROMPT,
    GENERATE_COMPARISON_PROMPT,
    REFORMULATE_QUERY_PROMPT,
    format_chunks_for_prompt,
    fill,
)
from src.vectorstore.embedder import DocumentEmbedder
from src.vectorstore.faiss_store import FAISSStore
from src.ingestion.chunker import DocumentChunk


class PipelineState(TypedDict):
    query: str                    # the user's original question
    query_type: str               # SINGLE_COMPANY | COMPARISON | GENERAL
    target_companies: list[str]   # tickers extracted by classify
    retrieved_chunks: list[tuple] # (chunk, score) pairs from FAISS
    reformulated_query: str       # rephrased query used on retry
    has_retried: bool             # guard: only retry once
    answer: str                   # final generated answer
    citations: list[str]          # unique citations from used chunks
    latency_ms: float             # end-to-end wall time
    error: str                    # non-empty if something went wrong


class RAGPipeline:
    """
    Retrieval-augmented generation pipeline over SEC 10-K filings.

    Pass it a plain-English question via .ask() and get back an answer
    with citations and timing info. LangGraph handles the step sequencing.
    """

    def __init__(
        self,
        faiss_store: FAISSStore,
        embedder: DocumentEmbedder,
        llm: Optional[GeminiClient] = None,
    ):
        self.store = faiss_store
        self.embedder = embedder
        self.llm = llm or GeminiClient()
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        """
        Builds the LangGraph state machine.

        Straight-line by default; the single conditional edge only
        branches to reformulate when FAISS returns nothing at all.
        """
        workflow = StateGraph(PipelineState)

        workflow.add_node("classify", self._classify_query)
        workflow.add_node("retrieve", self._retrieve_chunks)
        workflow.add_node("reformulate", self._reformulate_query)
        workflow.add_node("generate", self._generate_answer)

        workflow.set_entry_point("classify")
        workflow.add_edge("classify", "retrieve")

        # after retrieve: go straight to generate, or reformulate once if
        # FAISS returned nothing
        workflow.add_conditional_edges(
            "retrieve",
            self._should_reformulate,
            {
                "generate": "generate",
                "reformulate": "reformulate",
            },
        )

        workflow.add_edge("reformulate", "retrieve")
        workflow.add_edge("generate", END)

        return workflow.compile()

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _classify_query(self, state: PipelineState) -> dict:
        """
        LLM call 1 of 2.

        Determines query type (SINGLE_COMPANY / COMPARISON / GENERAL)
        and extracts any company tickers mentioned in the question.
        """
        prompt = fill(CLASSIFY_QUERY_PROMPT, query=state["query"])
        response = self.llm.classify(prompt)

        query_type = "GENERAL"
        companies: list[str] = []

        for line in response.strip().splitlines():
            line = line.strip()
            if line.startswith("TYPE:"):
                query_type = line.removeprefix("TYPE:").strip()
            elif line.startswith("COMPANIES:"):
                raw = line.removeprefix("COMPANIES:").strip()
                if raw != "NONE":
                    companies = [c.strip().upper() for c in raw.split(",")]

        # only keep tickers that actually exist in the FAISS store
        available = self.store.get_available_companies()
        companies = [c for c in companies if c in available]

        return {
            "query_type": query_type,
            "target_companies": companies,
        }

    def _retrieve_chunks(self, state: PipelineState) -> dict:
        """
        FAISS vector search — no LLM call.

        Uses the reformulated query on retry, original query otherwise.
        Search strategy depends on query type:
          - COMPARISON  → balanced results per company
          - SINGLE      → top-k from that company's index
          - GENERAL     → top-k across everything
        """
        search_query = state.get("reformulated_query") or state["query"]
        query_embedding = self.embedder.embed_query(search_query)

        companies = state.get("target_companies", [])
        query_type = state.get("query_type", "GENERAL")

        if query_type == "COMPARISON" and len(companies) >= 2:
            results = self.store.search_multi_company(
                query_embedding, companies, top_k_per_company=3
            )
        elif companies:
            results = self.store.search(
                query_embedding, top_k=5, company=companies[0]
            )
        else:
            results = self.store.search(query_embedding, top_k=5)

        return {"retrieved_chunks": results}

    def _should_reformulate(self, state: PipelineState) -> str:
        """
        Routing function called after every retrieve attempt.

        Reformulates only when:
          - FAISS returned zero chunks  (nothing to work with)
          - We haven't already retried  (prevent infinite loop)

        Low-relevance chunks are handled inside the generate prompt;
        we never branch here based on a relevance score.
        """
        chunks = state.get("retrieved_chunks", [])
        already_retried = state.get("has_retried", False)

        if not chunks and not already_retried:
            return "reformulate"

        return "generate"

    def _reformulate_query(self, state: PipelineState) -> dict:
        """
        Fallback: rephrases the question using formal SEC filing language.

        Fires at most once per question (has_retried guard). Useful when
        the user asks in casual language that doesn't match FAISS index terms.
        """
        prompt = fill(REFORMULATE_QUERY_PROMPT, query=state["query"])
        new_query = self.llm.classify(prompt).strip()

        print(f"  [reformulate] '{state['query']}' → '{new_query}'")

        return {
            "reformulated_query": new_query,
            "has_retried": True,
        }

    def _generate_answer(self, state: PipelineState) -> dict:
        """
        LLM call 2 of 2.

        Feeds all retrieved chunks into the appropriate prompt template.
        The prompt instructs the model to silently ignore irrelevant chunks,
        so no pre-filtering is needed here.
        """
        chunks = state.get("retrieved_chunks", [])
        context = format_chunks_for_prompt(chunks)
        query = state["query"]

        if state.get("query_type") == "COMPARISON":
            prompt = fill(GENERATE_COMPARISON_PROMPT, context=context, query=query)
        else:
            prompt = fill(GENERATE_ANSWER_PROMPT, context=context, query=query)

        answer = self.llm.analyze(prompt)
        citations = list(set(chunk.citation for chunk, _ in chunks))

        return {
            "answer": answer,
            "citations": citations,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, question: str) -> dict:
        """
        Ask a question about the ingested 10-K filings.

        Args:
            question: plain-English question about any filed company.

        Returns:
            dict with keys:
              answer          — formatted markdown string
              citations       — list of unique citation strings
              query_type      — SINGLE_COMPANY | COMPARISON | GENERAL
              target_companies — tickers identified in the question
              retrieved_chunks — raw (chunk, score) pairs for debugging
              latency_ms      — end-to-end wall time in milliseconds
        """
        start = time.time()

        initial_state: PipelineState = {
            "query": question,
            "query_type": "GENERAL",
            "target_companies": [],
            "retrieved_chunks": [],
            "reformulated_query": "",
            "has_retried": False,
            "answer": "",
            "citations": [],
            "latency_ms": 0.0,
            "error": "",
        }

        try:
            result = self.graph.invoke(initial_state)
            latency = (time.time() - start) * 1000
            return {
                "answer": result.get("answer", "Sorry, I couldn't generate an answer."),
                "citations": result.get("citations", []),
                "query_type": result.get("query_type", "GENERAL"),
                "target_companies": result.get("target_companies", []),
                "retrieved_chunks": result.get("retrieved_chunks", []),
                "latency_ms": round(latency, 1),
            }

        except Exception as exc:
            latency = (time.time() - start) * 1000
            return {
                "answer": f"Something went wrong: {exc}",
                "citations": [],
                "query_type": "ERROR",
                "target_companies": [],
                "retrieved_chunks": [],
                "latency_ms": round(latency, 1),
            }


if __name__ == "__main__":
    store = FAISSStore()
    store.load()

    embedder = DocumentEmbedder()
    pipeline = RAGPipeline(faiss_store=store, embedder=embedder)

    result = pipeline.ask("What are the main risk factors for Goldman Sachs?")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nCitations: {result['citations']}")
    print(f"Latency: {result['latency_ms']}ms")