<div align="center">

# FinSight

**FinSight cuts 10-K research time from hours to seconds — with every answer cited back to the exact SEC filing sentence.**

An autonomous RAG pipeline that downloads real SEC 10-K filings, indexes them with FAISS, and answers complex financial questions using LLM inference — with verifiable, deep-linked citations to SEC.gov.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://finsight-sec-rag.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 🎯 The Problem

An analyst preparing for an earnings call at 11pm needs to cross-reference risk factors across three companies' 10-K filings. That's ~600 pages of dense legal prose. Ctrl+F doesn't understand context. ChatGPT hallucinates figures. Copy-pasting into prompts loses citation trails.

**FinSight solves this.** Ask a question in plain English, get a structured answer with every claim traced back to the exact section, page, and sentence in the original SEC filing — with a clickable link that opens the filing and auto-scrolls to the relevant passage.

---

## 📸 Screenshots

> **Add your own screenshots here after deployment!**

| Screenshot | What it shows |
|---|---|
| `![Dashboard](screenshots/overview.png)` | **Overview dashboard** — index stats, company coverage chart, section breakdown, pipeline diagram |
| `![Query Result](screenshots/query_result.png)` | **Single-company query** — structured answer with citations, latency metric |
| `![Comparison](screenshots/comparison.png)` | **Multi-company comparison** — side-by-side analysis of risk factors across companies |
| `![Sources](screenshots/sources.png)` | **Source cards** — expandable cards with Gemini-formatted HTML or semantic highlighting |
| `![Deep Link](screenshots/deeplink.png)` | **SEC.gov deep link** — clicking "Open in Filing" jumps to the exact passage on SEC.gov |

<!-- 
HOW TO ADD SCREENSHOTS:
1. Create a "screenshots" folder in the repo root
2. Take screenshots of the app running
3. Replace the placeholder text above with actual image paths
4. Commit and push
-->

---

## ⚡ Performance — By the Numbers

| Metric | Value |
|---|---|
| **API calls per query** | 2 (down from 7 in v1) |
| **Filings indexed** | 8 companies, 2,450 chunks |
| **Embedding dimensions** | 384 (MiniLM-L6-v2) |
| **Avg response time** | ~2-4 seconds |
| **Token footprint** | 40% smaller than v1 prompts |
| **Model fallback depth** | 14 LLM models (Groq) + 18 extraction models (Gemini) |
| **Companies supported** | Apple, Microsoft, Tesla, JPMorgan, Goldman Sachs, Amazon, NVIDIA, Alphabet |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│  CLASSIFY  (LLM call 1)                     │
│  Detect intent: SINGLE / COMPARISON / GENERAL│
│  Extract company tickers from natural lang   │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  RETRIEVE  (no LLM call — pure vector math) │
│  FAISS cosine similarity over 2,450 chunks  │
│  Company-filtered or balanced multi-company  │
└──────────────────┬──────────────────────────┘
                   │
            ┌──────┴──────┐
            │  Empty?     │
            ├─── No ──────┤
            │             ▼
            │     ┌───────────────────────────┐
            │     │  GENERATE  (LLM call 2)   │
            │     │  Self-filtering context    │
            │     │  Structured markdown out   │
            │     └───────────────────────────┘
            │
            └─── Yes (first time only)
                      │
                      ▼
              ┌───────────────────┐
              │  REFORMULATE      │
              │  Rephrase in SEC  │
              │  financial lexicon│
              └───────┬───────────┘
                      │
                      └──► back to RETRIEVE
```

### Why 2 calls instead of 7

The original pipeline made **1 classify + N grade + 1 generate** API calls per query, where N = number of retrieved chunks (typically 5). Each grading call asked the LLM "is this chunk relevant?" — a binary yes/no that burned tokens and added latency.

**The fix:** I folded grading into the generation prompt itself. The generate prompt now says *"Read ALL chunks below. Silently ignore any chunk that does not help answer the question."* The LLM self-filters irrelevant context as part of its normal reasoning — no extra API calls needed. This dropped the pipeline from **7 calls to 2 calls**, cutting latency by ~60%.

---

## 🛡️ The Fallback Cascade — Engineering Under Constraints

The Groq free tier rate-limits aggressively. Rather than queuing requests or showing users a "please wait" spinner, I built a **14-model cascade** that automatically falls through to the next available model on any rate limit or permission error:

```
openai/gpt-oss-120b (best quality)
    → llama-3.3-70b-versatile
        → kimi-k2-instruct
            → qwen3-32b
                → ... 10 more models ...
                    → llama-3.1-8b-instant (last resort, fastest)
```

**The user never sees a timeout.** If the 120B flagship model is rate-limited, the request silently falls to the next best model. The same approach applies to source extraction: **18 Gemini models** are chained so that the formatted source cards always render, even when individual models hit their free-tier quotas.

---

## 🔗 Citation Deep Links — The Detail That Matters

Every source citation includes a link that opens the actual filing on SEC.gov and **auto-scrolls to the exact passage** using Chromium's `#:~:text=` fragment identifier.

**How it works:**
1. Load the filing's HTML and extract its plain text
2. Find a sentence from the retrieved chunk that exists **verbatim** in the filing
3. Use that text as the URL fragment — the browser highlights and scrolls to it automatically

```
https://www.sec.gov/Archives/edgar/data/886982/.../gs-20251231.htm
    #:~:text=The%20firm%20is%20subject%20to%20credit%20risk
```

This works in Chrome, Edge, and all Chromium-based browsers. Firefox ignores the fragment gracefully (no errors, just no auto-scroll).

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Free API keys: [Groq](https://console.groq.com) + [Google AI Studio](https://aistudio.google.com/app/apikey)

### Setup

```bash
git clone https://github.com/Vedag812/finsight-sec-rag.git
cd finsight-sec-rag
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Ingest Filings

```bash
python scripts/ingest.py
```

This downloads 10-K filings from SEC EDGAR, parses them into sections, chunks the text, generates embeddings, and builds the FAISS index. Takes ~2 minutes.

### Run

```bash
streamlit run src/app/main.py
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Orchestration** | LangGraph | Deterministic state machine with conditional routing — no LangChain agent unpredictability |
| **Vector Store** | FAISS + SentenceTransformers | Sub-millisecond similarity search over 2,450 document chunks |
| **LLM Inference** | Groq API | 10x faster than OpenAI for the same model families — critical for interactive UX |
| **Source Formatting** | Google Gemini | Converts raw filing text into clean, readable HTML cards |
| **Data Ingestion** | sec-edgar-downloader + BeautifulSoup | Automated SEC EDGAR downloads with section-aware HTML parsing |
| **Frontend** | Streamlit | Rapid prototyping with hot-reload — edit prompts and see results instantly |
| **Deployment** | Streamlit Cloud / Docker | Zero-config cloud deploys with TOML-based secrets management |

---

## 📁 Project Structure

```
finsight-sec-rag/
├── src/
│   ├── app/
│   │   ├── main.py              # Streamlit entry point + secret injection
│   │   └── components.py        # UI rendering, CSS, source cards, Gemini extraction
│   ├── pipeline/
│   │   ├── graph.py             # LangGraph state machine (Classify → Retrieve → Generate)
│   │   ├── llm.py               # Groq client with 14-model fallback cascade
│   │   └── prompts.py           # XML-delimited prompt templates with late binding
│   ├── ingestion/
│   │   ├── sec_downloader.py    # SEC EDGAR API integration
│   │   ├── document_parser.py   # Section-aware 10-K HTML parser
│   │   └── chunker.py           # Semantic text chunking with overlap
│   ├── vectorstore/
│   │   ├── embedder.py          # SentenceTransformer embedding generation
│   │   └── faiss_store.py       # FAISS index build, save, load, search
│   └── utils.py                 # Cross-platform secret loading
├── scripts/
│   ├── ingest.py                # One-command full pipeline ingestion
│   ├── benchmark.py             # Automated latency benchmarking (10 queries)
│   └── debug_pipeline.py        # Step-by-step pipeline debugger
├── data/
│   ├── indexes/                 # FAISS index + chunk metadata (committed)
│   └── processed/               # Embeddings + chunk JSON (committed)
├── requirements.txt
├── Dockerfile
└── .env.example
```

---

## ⚠️ Limitations & Known Issues

**Being transparent about what this doesn't do:**

- **Free-tier dependent** — Both Groq and Gemini APIs are on free tiers. Under heavy concurrent usage, even the 14-model cascade can exhaust. The system degrades gracefully with a clear error message, but it's not production-SLA grade.
- **Static filings** — The system indexes a snapshot of 10-K filings. It doesn't auto-update when new filings are published. Running `ingest.py` again refreshes the index.
- **English only** — Prompt templates, section parsing, and chunk boundaries are all optimized for English-language SEC filings.
- **No multi-turn memory** — Each query is independent. The system doesn't remember previous questions within a session (yet).
- **Deep links require Chromium** — The `#:~:text=` fragment identifiers work in Chrome/Edge but not Firefox or Safari.- **Streamlit UI constraints** — Dollar signs in financial data required explicit escaping to prevent Streamlit's LaTeX renderer from misinterpreting `$1.2B` as math notation.

---

## 🧠 What I Learned Building This

1. **Prompt engineering is architecture** — Moving grading from a dedicated pipeline node into the generation prompt's instructions cut API calls by 70%. The "right" prompt isn't just about wording — it's about what infrastructure you can eliminate.

2. **Free-tier engineering is its own discipline** — Building reliable software on rate-limited APIs requires thinking in cascades, not retries. The 14-model fallback chain was born from hitting Groq's limits at 2am during testing.

3. **Citations are a trust mechanism** — An AI that says "revenue was $383B" is useless without proof. Fragment URL deep-links that jump to the exact sentence in the filing turn an AI answer into a verifiable research tool.

---

## 📄 License

MIT — use it, fork it, build on it.

---

<div align="center">

**Built by [Vedant Agarwal](https://github.com/Vedag812)**

*If this project helped you or sparked an idea, consider giving it a ⭐*

</div>
