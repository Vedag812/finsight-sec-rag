"""
Prompt templates for the RAG pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  QUICK CUSTOMISATION GUIDE  (no server restart needed!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Streamlit hot-reloads on save — every edit takes effect on the
very next question.

  _BASE        Global rules prepended to every prompt.
  CLASSIFY     Routing: detects query type + company tickers.
  GENERATE     Single-company answer with citations.
  COMPARISON   Multi-company side-by-side analysis.
  REFORMULATE  Retry: rephrases query in SEC language (rare).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DESIGN PRINCIPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. 2 LLM calls per query  (Classify + Generate). Grading is
     folded into the generate prompt — no per-chunk API calls.
  2. Late interpolation via fill() — {placeholders} survive
     until call-site so templates are defined once at import.
  3. XML delimiters (<chunks>, <question>) keep data separated
     from instructions across all model families.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Shared base — prepended to every prompt
# ---------------------------------------------------------------------------
_BASE = """\
You are FinSight, an expert SEC 10-K filing analyst.
Rules: no preamble, no filler ("Sure!", "Here is…"), no sign-offs.
Follow the output format exactly. Do not repeat the question.\
"""


def _prompt(body: str) -> str:
    """Prepend the shared base to a prompt body."""
    return f"{_BASE}\n\n{body.strip()}"


# ---------------------------------------------------------------------------
# 1. CLASSIFY  (LLM call 1 of 2)
# ---------------------------------------------------------------------------
CLASSIFY_QUERY_PROMPT = _prompt("""\
Classify this question about SEC 10-K filings and extract company tickers.

<question>
{query}
</question>

Output exactly two lines:
TYPE: <SINGLE_COMPANY | COMPARISON | GENERAL>
COMPANIES: <comma-separated tickers, or NONE>\
""")


# ---------------------------------------------------------------------------
# 2. GENERATE — single-company  (LLM call 2 of 2)
# ---------------------------------------------------------------------------
GENERATE_ANSWER_PROMPT = _prompt("""\
Answer the question using ONLY the chunks below. Silently skip irrelevant chunks.

Rules:
- Cite claims as **[Company, Section, Pages X-Y]**.
- Use **bold** for key terms and numbers.  Use ### headings.
- Use bullet points for 3+ items.  No walls of text.
- If data is missing, say so.  Never fabricate.
- End with ### Summary (2-3 sentences).

<chunks>
{context}
</chunks>

<question>
{query}
</question>

Answer:\
""")


# ---------------------------------------------------------------------------
# 3. GENERATE — comparison  (LLM call 2 of 2, alternate template)
# ---------------------------------------------------------------------------
GENERATE_COMPARISON_PROMPT = _prompt("""\
Compare the companies using ONLY the chunks below. Silently skip irrelevant chunks.

Rules:
- Cite claims as **[Company, Section, Pages X-Y]**.
- Use **bold** for company names, key terms, and numbers.
- Structure: ### Overview → ### [Company] per company → ### Key Differences → ### Key Similarities.
- Use bullet points for 3+ items.  No walls of text.
- If a company's data is absent: "No data available for [Company]."
- Never fabricate.

<chunks>
{context}
</chunks>

<question>
{query}
</question>

Comparison:\
""")


# ---------------------------------------------------------------------------
# 4. REFORMULATE  (optional — only on empty FAISS retrieval)
# ---------------------------------------------------------------------------
REFORMULATE_QUERY_PROMPT = _prompt("""\
The question below returned no search results from SEC 10-K filings.
Rephrase it using formal financial/business language found in SEC filings.

Example: "How does the company make money?" → "revenue sources and business segments"

<original_question>
{query}
</original_question>

Output the rephrased question only:\
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_chunks_for_prompt(chunks_with_scores: list) -> str:
    """
    Serialise retrieved chunks into an XML-delimited block for the
    GENERATE_* prompts.
    """
    parts: list[str] = []
    for i, (chunk, _score) in enumerate(chunks_with_scores, start=1):
        parts.append(
            f"<chunk id={i} citation='{chunk.citation}'>\n{chunk.text}\n</chunk>"
        )
    return "\n\n".join(parts)


def fill(template: str, **kwargs: str) -> str:
    """
    Late-bind {placeholders} in a prompt template.

    Raises KeyError if a required placeholder is missing — better than
    sending a broken prompt to the API.
    """
    return template.format_map(kwargs)