"""
UI components for the FinSight dashboard.

What lives here:
  - CSS injection and page header
  - Source card rendering with deep-link generation to SEC.gov
  - Semantic highlighting fallback (embedding cosine similarity)
  - Gemini-powered source formatting (with model cascade fallback)
  - Sidebar with example queries
  - Overview tab: index stats, section breakdown, pipeline diagram

Everything is pure Streamlit + a little raw HTML/CSS injected via
st.markdown(unsafe_allow_html=True). No frontend build step needed.

Hot-reload friendly: save this file and Streamlit picks up changes
on the next interaction.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import quote

import google.generativeai as genai
import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from src.utils import get_secret


# ---------------------------------------------------------------------------
# Gemini setup
# We try models in speed order and fall back down the list if one is
# rate-limited or unavailable.
# ---------------------------------------------------------------------------
_GEMINI_KEY = get_secret("GEMINI_API_KEY")

_GEMINI_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-lite",
    "gemma-3-4b-it",
    "gemma-3-1b-it",
    "gemma-3n-e2b-it",
    "gemma-3n-e4b-it",
    "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.5-flash",
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemini-2.5-pro",
    "gemini-pro-latest",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
]

_gemini_model = None
if _GEMINI_KEY:
    genai.configure(api_key=_GEMINI_KEY)
    _gemini_model = genai.GenerativeModel(_GEMINI_MODELS[0])


# ---------------------------------------------------------------------------
# Static lookup tables
# Add new tickers here when you ingest more filings.
# ---------------------------------------------------------------------------
COMPANY_NAMES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "TSLA": "Tesla Inc.",
    "JPM":  "JPMorgan Chase",
    "GS":   "Goldman Sachs",
    "AMZN": "Amazon.com Inc.",
    "NVDA": "NVIDIA Corp.",
    "GOOGL":"Alphabet Inc.",
}

COMPANY_SECTORS = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "TSLA": "Automotive",
    "JPM":  "Banking",
    "GS":   "Inv. Banking",
    "AMZN": "Retail/Tech",
    "NVDA": "Semiconductors",
    "GOOGL":"Technology",
}

FILING_DATES = {
    "AAPL": "2025-10-31",
    "MSFT": "2025-07-30",
    "TSLA": "2026-01-29",
    "JPM":  "2026-02-13",
    "GS":   "2026-02-25",
    "AMZN": "2026-02-06",
    "NVDA": "2026-02-25",
    "GOOGL":"2026-02-05",
}

FILING_URLS = {
    "AAPL": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm",
    "MSFT": "https://www.sec.gov/Archives/edgar/data/789019/000095017025100235/msft-20250630.htm",
    "TSLA": "https://www.sec.gov/Archives/edgar/data/1318605/000162828026003952/tsla-20251231.htm",
    "JPM":  "https://www.sec.gov/Archives/edgar/data/19617/000162828026008131/jpm-20251231.htm",
    "GS":   "https://www.sec.gov/Archives/edgar/data/886982/000088698226000091/gs-20251231.htm",
    "AMZN": "https://www.sec.gov/Archives/edgar/data/1018724/000101872426000004/amzn-20251231.htm",
    "NVDA": "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm",
    "GOOGL":"https://www.sec.gov/Archives/edgar/data/1652044/000165204426000018/goog-20251231.htm",
}

LOCAL_FILES = {
    "AAPL": "data/raw/AAPL/10K_2025-10-31.htm",
    "MSFT": "data/raw/MSFT/10K_2025-07-30.htm",
    "TSLA": "data/raw/TSLA/10K_2026-01-29.htm",
    "JPM":  "data/raw/JPM/10K_2026-02-13.htm",
    "GS":   "data/raw/GS/10K_2026-02-25.htm",
    "AMZN": "data/raw/AMZN/10K_2026-02-06.htm",
    "NVDA": "data/raw/NVDA/10K_2026-02-25.htm",
    "GOOGL":"data/raw/GOOGL/10K_2026-02-05.htm",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_filing_url(ticker: str) -> str:
    return FILING_URLS.get(ticker, "https://www.sec.gov/edgar/searchedgar/companysearch")


def _ascii_only(text: str) -> str:
    """Strip non-ASCII bytes that can trip up URL encoding and HTML rendering."""
    return text.encode("ascii", errors="ignore").decode("ascii")


@st.cache_data(ttl=3600)
def _load_filing_text(ticker: str) -> str:
    """
    Read a local 10-K HTML file and return its plain text.

    Cached for an hour so repeated queries against the same filing
    don't re-parse the HTML on every call.
    """
    path = LOCAL_FILES.get(ticker)
    if not path:
        return ""

    p = Path(path)
    if not p.exists():
        return ""

    try:
        html = p.read_text(encoding="utf-8", errors="ignore")
        return BeautifulSoup(html, "html.parser").get_text()
    except Exception:
        return ""


def build_filing_deep_link(ticker: str, chunk_text: str) -> str:
    """
    Build a URL that opens the actual 10-K on SEC.gov and auto-scrolls
    to the passage that the chunk came from.

    Strategy (each step falls through to the next if it fails):
      1. Find a full sentence from the chunk that appears verbatim in
         the filing HTML and use it as the #:~:text= fragment.
      2. Try the first 50 characters of that sentence.
      3. Try any 7-word phrase from the chunk.
      4. Return the bare filing URL if nothing matched.

    The #:~:text= fragment is supported by Chrome, Edge, and other
    Chromium-based browsers. Firefox ignores it gracefully.
    """
    base_url = get_filing_url(ticker)
    filing_text = _load_filing_text(ticker)

    if not filing_text:
        return base_url

    # normalise the chunk into clean sentences
    clean = _ascii_only(re.sub(r"\s+", " ", chunk_text.strip()))
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", clean)
        if len(s.strip()) > 30
    ]

    filing_norm = _ascii_only(re.sub(r"\s+", " ", filing_text))

    for sent in sentences:
        sent = re.sub(r"\s+", " ", sent).strip()

        # try the full sentence (capped at 80 chars for a tidy URL)
        if sent in filing_norm:
            excerpt = sent[:80]
            cut = excerpt.rfind(" ")
            excerpt = excerpt[:cut] if cut > 30 else excerpt
            return f"{base_url}#:~:text={quote(excerpt.strip())}"

        # try just the opening 50 characters
        stub = sent[:50].strip()
        if stub and stub in filing_norm:
            return f"{base_url}#:~:text={quote(stub)}"

    # last resort: look for any 7-word run from the chunk
    words = clean.split()
    for i in range(len(words) - 6):
        phrase = " ".join(words[i : i + 7])
        if phrase in filing_norm:
            return f"{base_url}#:~:text={quote(phrase)}"

    return base_url


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, discarding very short fragments."""
    text = _ascii_only(re.sub(r"\s+", " ", text).strip())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]


def semantic_highlight(
    sentences: list[str],
    query: str,
    embedder,
    threshold: float = 0.35,
) -> list[tuple[str, bool]]:
    """
    Tag each sentence as relevant or not by comparing its embedding
    to the query embedding via cosine similarity.

    Falls back gracefully to all-False if either input is empty.
    Works with any embedder that exposes an embed_query(text) -> list[float] method.
    """
    if not sentences or not query:
        return [(s, False) for s in sentences]

    query_vec = np.array(embedder.embed_query(query)).flatten()

    results = []
    for sent in sentences:
        sent_vec = np.array(embedder.embed_query(sent)).flatten()
        norm = np.linalg.norm(query_vec) * np.linalg.norm(sent_vec) + 1e-8
        similarity = float(np.dot(query_vec, sent_vec) / norm)
        results.append((sent, similarity >= threshold))

    return results


@st.cache_data(ttl=3600, show_spinner=False)
def extract_source_with_gemini(
    chunk_text: str,
    query: str,
    company: str,
    section: str,
) -> str | None:
    """
    Ask Gemini to reformat a raw chunk into clean, readable HTML.

    Results are cached so the same (chunk, query) pair doesn't trigger
    a second API call. If the primary model is rate-limited, we work
    down the _GEMINI_MODELS list until one responds.

    Returns None if Gemini is not configured or all models fail.
    """
    if not _gemini_model:
        return None

    prompt = (
        f'Format this SEC filing excerpt into clean HTML.\n'
        f'Query context: "{query}"\n'
        f'Company: {company}, Section: {section}\n\n'
        f'TEXT:\n{chunk_text}\n\n'
        f'Rules: use <h4> for topics, <p> for paragraphs, <ul><li> for lists, '
        f'<strong> for key terms and numbers. Extract only what is relevant. '
        f'No preamble. Return HTML only, no markdown fences.'
    )

    try:
        return _gemini_model.generate_content(prompt).text
    except Exception:
        pass

    for model_name in _GEMINI_MODELS[1:]:
        try:
            return genai.GenerativeModel(model_name).generate_content(prompt).text
        except Exception:
            continue

    return None


# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------

def render_custom_css() -> None:
    """Inject the global stylesheet. Called once at app startup."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        .stApp { font-family: 'Inter', sans-serif; }
        .block-container { padding-top: 1.5rem; max-width: 1100px; }
        #MainMenu, footer, header { visibility: hidden; }

        /* ── Hero ── */
        .hero-logo { font-size: 2.6rem; font-weight: 800; color: #1a1a2e; letter-spacing: -1px; margin-bottom: 0.4rem; }
        .hero-logo span { color: #b8922f; }
        .hero-sub { font-size: 1.15rem; color: #4b5563; line-height: 1.75; }
        .hero-sub strong { color: #1a1a2e; }

        /* ── Section labels ── */
        .section-title {
            font-size: 0.85rem; color: #9ca3af;
            text-transform: uppercase; letter-spacing: 2px;
            font-weight: 700; margin: 1.8rem 0 0.8rem 0;
        }

        /* ── Answer card ── */
        .ans-meta {
            display: flex; gap: 2rem;
            margin-top: 1.2rem; padding-top: 1rem;
            border-top: 1px solid #f3f4f6;
        }
        .ans-meta span { font-size: 0.9rem; color: #9ca3af; }
        .ans-meta span b { color: #4b5563; font-weight: 600; }

        /* ── Source cards ── */
        .src-card {
            background: #fff; border: 1px solid #e5e7eb;
            border-left: 4px solid #b8922f; border-radius: 0 12px 12px 0;
            padding: 1.1rem 1.5rem; margin-bottom: 0.6rem;
            display: flex; align-items: center; gap: 1rem;
        }
        .src-num {
            background: #fef9ee; color: #b8922f;
            width: 32px; height: 32px; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.9rem; font-weight: 700; flex-shrink: 0;
        }
        .src-info { flex: 1; }
        .src-info .name  { font-size: 1.05rem; color: #1f2937; font-weight: 600; }
        .src-info .detail{ font-size: 0.88rem; color: #9ca3af; margin-top: 2px; }

        /* ── Source viewer (inline expander) ── */
        .source-viewer {
            background: #fafbfc; border: 1px solid #e5e7eb;
            border-radius: 10px; padding: 1.5rem; margin: 0.5rem 0 1rem 0;
            font-size: 1rem; color: #374151; line-height: 1.85;
        }
        .source-viewer .src-header {
            font-size: 0.75rem; color: #b8922f;
            text-transform: uppercase; letter-spacing: 1.5px;
            font-weight: 700; margin-bottom: 0.8rem;
            padding-bottom: 0.5rem; border-bottom: 1px solid #e5e7eb;
        }
        .sv-relevant {
            background: #fef9ee; border-left: 3px solid #b8922f;
            padding: 0.6rem 0.8rem; margin: 0.5rem 0;
            border-radius: 0 6px 6px 0; color: #1f2937; font-weight: 500;
        }
        .sv-normal  { padding: 0.3rem 0; margin: 0.3rem 0; color: #6b7280; }
        .sv-legend  {
            font-size: 0.75rem; color: #9ca3af;
            margin-top: 0.8rem; padding-top: 0.5rem;
            border-top: 1px solid #e5e7eb;
        }

        /* ── Input ── */
        .stTextInput > div > div > input {
            font-size: 1.1rem !important; padding: 0.85rem 1.1rem !important;
            border-radius: 10px !important; border: 2px solid #e5e7eb !important;
            font-family: 'Inter', sans-serif !important;
        }
        .stTextInput > div > div > input:focus { border-color: #b8922f !important; }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] { background: #fafbfc; }
        section[data-testid="stSidebar"] .stButton > button {
            background: #fff !important; color: #4b5563 !important;
            border: 1px solid #e5e7eb !important; font-size: 0.95rem !important;
            text-align: left !important; padding: 0.6rem 0.9rem !important;
            border-radius: 8px !important; font-weight: 500 !important;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            border-color: #b8922f !important;
            background: #fef9ee !important;
            color: #1a1a2e !important;
        }
        .sb-label {
            font-size: 0.75rem; color: #9ca3af;
            text-transform: uppercase; letter-spacing: 2px;
            font-weight: 700; margin-bottom: 0.6rem;
        }
        .streamlit-expanderHeader { font-size: 1rem !important; }
    </style>
    """, unsafe_allow_html=True)


def render_header() -> None:
    st.markdown("""
    <div class="hero-logo">Fin<span>Sight</span></div>
    <div class="hero-sub">
        AI-powered analysis of <strong>real SEC 10-K annual filings</strong>
        from 5 major companies. Ask any question — every answer is cited with
        the exact section and page from the original filing.
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Answer + source rendering
# ---------------------------------------------------------------------------

def render_answer(
    answer: str,
    latency_ms: float,
    query_type: str,
    num_citations: int,
) -> None:
    """Render the generated answer with its metadata footer."""
    st.markdown('<div class="section-title">Answer</div>', unsafe_allow_html=True)

    # strip <think>…</think> blocks emitted by some reasoning models
    clean = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    # escape $ so Streamlit doesn't treat them as LaTeX delimiters
    clean = clean.replace("$", r"\$")

    st.markdown(clean)
    st.markdown(f"""
        <div class="ans-meta">
            <span><b>{latency_ms:.0f}ms</b> latency</span>
            <span><b>{num_citations}</b> sources</span>
            <span><b>{query_type}</b></span>
        </div>
    """, unsafe_allow_html=True)


def render_sources(
    chunks_with_scores: list[tuple],
    query: str = "",
    embedder=None,
) -> None:
    """
    Render up to 5 unique source cards, each with an expander that shows
    either Gemini-formatted HTML or a semantic-highlighted fallback.

    Uniqueness is defined per (ticker, section_name) pair so the same
    section from the same company never appears twice in the list.
    """
    if not chunks_with_scores:
        return

    # deduplicate while preserving rank order
    seen: set[str] = set()
    unique: list[tuple] = []
    for chunk, score in chunks_with_scores:
        key = f"{chunk.ticker}_{chunk.section_name}"
        if key not in seen:
            seen.add(key)
            unique.append((chunk, score))

    if not unique:
        return

    st.markdown('<div class="section-title">Sources</div>', unsafe_allow_html=True)

    for i, (chunk, score) in enumerate(unique[:5], start=1):
        url = build_filing_deep_link(chunk.ticker, chunk.text)
        match_pct = score * 100

        col_info, col_link = st.columns([5, 1])
        with col_info:
            st.markdown(f"""
            <div class="src-card">
                <div class="src-num">{i}</div>
                <div class="src-info">
                    <div class="name">{chunk.company}</div>
                    <div class="detail">
                        {chunk.section_name} &middot;
                        Pages {chunk.page_start}–{chunk.page_end} &middot;
                        Filed {chunk.filing_date} &middot;
                        {match_pct:.0f}% match
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_link:
            st.link_button("Open in Filing", url, use_container_width=True)

        with st.expander(f"View extracted source — {chunk.section_name}", expanded=False):
            _render_source_body(chunk, query, embedder)


def _render_source_body(chunk, query: str, embedder) -> None:
    """
    Inner helper for render_sources.

    Tries Gemini first for clean HTML formatting. Falls back to
    embedding-based sentence highlighting if Gemini is unavailable
    or returns nothing.
    """
    header = (
        f'<div class="src-header">'
        f'From {chunk.company} — {chunk.section_name} — '
        f'Pages {chunk.page_start}–{chunk.page_end}'
        f'</div>'
    )

    # ── Gemini path ──────────────────────────────────────────────────────
    if query and _gemini_model:
        gemini_html = extract_source_with_gemini(
            chunk.text, query, chunk.company, chunk.section_name
        )
        if gemini_html:
            clean_html = gemini_html.replace("```html", "").replace("```", "").strip()
            st.markdown(
                f'<div class="source-viewer">{header}{clean_html}</div>',
                unsafe_allow_html=True,
            )
            return

    # ── Semantic-highlight fallback ───────────────────────────────────────
    sentences = split_into_sentences(chunk.text)
    scored = (
        semantic_highlight(sentences, query, embedder)
        if (embedder and query)
        else [(s, False) for s in sentences]
    )

    relevant_count = sum(1 for _, r in scored if r)

    body = "".join(
        f'<div class="sv-relevant">{sent}</div>'
        if is_relevant
        else f'<div class="sv-normal">{sent}</div>'
        for sent, is_relevant in scored
    )
    legend = (
        f'<div class="sv-legend">'
        f'&#9679; {relevant_count} of {len(scored)} sentences are semantically relevant'
        f'</div>'
    )

    st.markdown(
        f'<div class="source-viewer">{header}{body}{legend}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Overview tab
# ---------------------------------------------------------------------------

def get_chunk_stats(store) -> tuple[dict, dict]:
    """Count chunks per company and per section from the FAISS store."""
    company_chunks: dict[str, int] = {}
    section_chunks: dict[str, int] = {}
    for c in store.chunks:
        company_chunks[c.ticker] = company_chunks.get(c.ticker, 0) + 1
        section_chunks[c.section_name] = section_chunks.get(c.section_name, 0) + 1
    return company_chunks, section_chunks


def render_overview(store) -> None:
    """
    Render the Overview tab: top-level metrics, per-company and per-section
    breakdown tables/charts, filing links, and the pipeline diagram.
    """
    company_chunks, section_chunks = get_chunk_stats(store)

    # ── Top metrics ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Index Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Companies",    len(company_chunks))
    c2.metric("Total Chunks", f"{len(store.chunks):,}")
    c3.metric("Sections Parsed", len(section_chunks))
    c4.metric("Embedding Dim",   "384")

    # ── Data breakdown ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">Data Breakdown</div>', unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Chunks per Company**")
        rows = [
            {
                "Company": COMPANY_NAMES.get(t, t),
                "Ticker":  t,
                "Sector":  COMPANY_SECTORS.get(t, ""),
                "Chunks":  company_chunks[t],
                "Filed":   FILING_DATES.get(t, ""),
            }
            for t in sorted(company_chunks)
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("**View Original Filings**")
        for ticker in sorted(FILING_URLS):
            st.markdown(f"📄 [{COMPANY_NAMES[ticker]} 10-K]({FILING_URLS[ticker]})")

    with col_right:
        st.markdown("**Chunks per Section**")
        section_df = pd.DataFrame([
            {"Section": name, "Chunks": count}
            for name, count in sorted(section_chunks.items(), key=lambda x: -x[1])
        ])
        st.bar_chart(section_df.set_index("Section"), horizontal=True)

    # ── Company coverage bar chart ────────────────────────────────────────
    st.markdown('<div class="section-title">Company Coverage</div>', unsafe_allow_html=True)
    chart_df = pd.DataFrame({
        "Company": [COMPANY_NAMES.get(t, t) for t in sorted(company_chunks)],
        "Indexed Chunks": [company_chunks[t] for t in sorted(company_chunks)],
    })
    st.bar_chart(chart_df.set_index("Company"))

    # ── Pipeline diagram ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">RAG Pipeline</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    p1.info("🔍 **Classify**\n\nDetect query type & target companies")
    p2.info("📄 **Retrieve**\n\nFAISS similarity search across chunks")
    p3.info("💡 **Generate**\n\nCited answer from real filing text")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

_EXAMPLE_QUERIES = [
    "What are Goldman Sachs' main risk factors?",
    "How does Apple generate revenue?",
    "Compare Tesla and Microsoft's risk factors",
    "What does JPMorgan say about credit risk?",
    "Describe Goldman Sachs' business segments",
    "What are Apple's biggest competitive threats?",
    "How does Tesla describe its market opportunity?",
    "Compare performance of JPMorgan and Goldman Sachs",
]


def render_sidebar() -> None:
    """
    Render the sidebar with example queries and an About blurb.

    Clicking an example button stores the query in session state under
    'pending_query' — the main app loop picks this up and pre-fills the
    input box on the next rerun.
    """
    with st.sidebar:
        st.markdown('<div class="sb-label">Try These</div>', unsafe_allow_html=True)
        for query in _EXAMPLE_QUERIES:
            if st.button(query, key=f"ex_{hash(query)}", use_container_width=True):
                st.session_state["pending_query"] = query

        st.divider()
        st.markdown('<div class="sb-label">About</div>', unsafe_allow_html=True)
        st.caption(
            "**FinSight** downloads 10-K filings from SEC EDGAR, parses them "
            "by section, chunks the text, and indexes everything with FAISS. "
            "Questions are answered by retrieving the most relevant chunks and "
            "generating cited responses using an LLM."
        )