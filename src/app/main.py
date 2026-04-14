"""
FinSight — SEC 10-K Filing Analysis
Run: python -m streamlit run src/app/main.py
"""

import streamlit as st
import sys
from pathlib import Path

# Fix Windows encoding - prevents charmap codec errors with Unicode
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.vectorstore.embedder import DocumentEmbedder
from src.vectorstore.faiss_store import FAISSStore
from src.pipeline.graph import RAGPipeline
from src.app.components import (
    render_custom_css,
    render_header,
    render_overview,
    render_answer,
    render_sources,
    render_sidebar,
)

st.set_page_config(
    page_title="FinSight — SEC 10-K Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_pipeline():
    store = FAISSStore()
    store.load()
    embedder = DocumentEmbedder()
    pipeline = RAGPipeline(faiss_store=store, embedder=embedder)
    return store, embedder, pipeline


def main():
    render_custom_css()

    try:
        store, embedder, pipeline = load_pipeline()
    except FileNotFoundError:
        st.error("Index not found. Run `python scripts/ingest.py` first.")
        return
    except Exception as e:
        st.error(f"Failed to load: {e}")
        return

    render_header()
    render_sidebar()

    # handle sidebar example clicks
    if "pending_query" in st.session_state:
        st.session_state["query_input"] = st.session_state.pop("pending_query")

    # search bar
    col1, col2 = st.columns([7, 1])
    with col1:
        query = st.text_input(
            "q", placeholder="Ask about any SEC 10-K filing...",
            label_visibility="collapsed", key="query_input",
        )
    with col2:
        go = st.button("Analyze", use_container_width=True, type="primary")

    st.divider()

    if go and query:
        with st.spinner("Searching filings..."):
            result = pipeline.ask(query)

        if result.get("query_type") == "ERROR":
            st.error(result["answer"])
            return

        # pre-extract source text with Gemini (cached, runs during spinner)
        chunks = result.get("retrieved_chunks", [])
        with st.spinner("Formatting sources..."):
            from src.app.components import extract_source_with_gemini
            for chunk, score in chunks:
                extract_source_with_gemini(chunk.text, query, chunk.company, chunk.section_name)

        render_answer(
            result["answer"],
            result["latency_ms"],
            result["query_type"],
            len(result["citations"]),
        )
        render_sources(chunks, query=query, embedder=embedder)
    else:
        # show the data-rich overview dashboard
        render_overview(store)


if __name__ == "__main__":
    main()
