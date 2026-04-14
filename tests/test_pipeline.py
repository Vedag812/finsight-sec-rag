"""
Integration tests for the RAG pipeline.

These tests check that the pipeline steps work together properly.
We use mock data instead of actual API calls so tests run fast
and don't need a Gemini API key.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.ingestion.chunker import DocumentChunk
from src.vectorstore.faiss_store import FAISSStore


@pytest.fixture
def sample_chunks():
    """Creates a few fake chunks that look like real 10-K data."""
    return [
        DocumentChunk(
            chunk_id="abc123",
            text="The Company faces risks related to global economic conditions. "
                 "A downturn in the economy could reduce client activity and revenues.",
            company="The Goldman Sachs Group Inc.",
            ticker="GS",
            filing_date="2024-02-15",
            section_name="Item 1A - Risk Factors",
            page_start=25,
            page_end=27,
            chunk_index=0,
            total_chunks_in_section=5,
        ),
        DocumentChunk(
            chunk_id="def456",
            text="Apple designs, manufactures, and markets smartphones, personal computers, "
                 "tablets, wearables, and accessories. The Company also sells digital content "
                 "and services through the App Store and Apple Music.",
            company="Apple Inc.",
            ticker="AAPL",
            filing_date="2024-11-01",
            section_name="Item 1 - Business",
            page_start=3,
            page_end=5,
            chunk_index=0,
            total_chunks_in_section=8,
        ),
        DocumentChunk(
            chunk_id="ghi789",
            text="Microsoft revenue increased 16% driven by growth in Intelligent Cloud "
                 "and Productivity and Business Processes segments.",
            company="Microsoft Corporation",
            ticker="MSFT",
            filing_date="2024-07-30",
            section_name="Item 7 - MD&A",
            page_start=40,
            page_end=42,
            chunk_index=2,
            total_chunks_in_section=10,
        ),
    ]


class TestDocumentChunk:
    def test_citation_format(self, sample_chunks):
        """Citation should include company name, section, and page range."""
        chunk = sample_chunks[0]
        citation = chunk.citation
        assert "Goldman Sachs" in citation
        assert "Risk Factors" in citation
        assert "25" in citation

    def test_to_dict_and_back(self, sample_chunks):
        """Chunk should survive serialization round-trip."""
        original = sample_chunks[1]
        restored = DocumentChunk.from_dict(original.to_dict())
        assert restored.text == original.text
        assert restored.ticker == original.ticker
        assert restored.section_name == original.section_name


class TestFAISSStore:
    def test_available_companies(self, sample_chunks):
        """Should return all unique tickers from stored chunks."""
        store = FAISSStore()
        store.chunks = sample_chunks
        companies = store.get_available_companies()
        assert "GS" in companies
        assert "AAPL" in companies
        assert "MSFT" in companies

    def test_available_sections(self, sample_chunks):
        """Should return all unique section names."""
        store = FAISSStore()
        store.chunks = sample_chunks
        sections = store.get_available_sections()
        assert "Item 1A - Risk Factors" in sections
        assert "Item 1 - Business" in sections

    def test_is_ready_when_empty(self):
        """Store should report not ready when no index is loaded."""
        store = FAISSStore()
        assert store.is_ready is False


class TestPipelinePrompts:
    def test_prompt_formatting(self):
        """Prompts should format without errors."""
        from src.pipeline.prompts import (
            CLASSIFY_QUERY_PROMPT,
            GRADE_RELEVANCE_PROMPT,
            GENERATE_ANSWER_PROMPT,
            format_chunks_for_prompt,
        )

        # classify prompt
        result = CLASSIFY_QUERY_PROMPT.format(query="What does Apple do?")
        assert "What does Apple do?" in result

        # grade prompt
        result = GRADE_RELEVANCE_PROMPT.format(
            query="test question",
            chunk_text="test chunk content"
        )
        assert "test question" in result
        assert "test chunk content" in result

        # generate prompt
        result = GENERATE_ANSWER_PROMPT.format(
            context="some context here",
            query="test question"
        )
        assert "some context here" in result

    def test_format_chunks_for_prompt(self, sample_chunks):
        """Chunk formatter should include citations and text."""
        from src.pipeline.prompts import format_chunks_for_prompt

        chunks_with_scores = [(chunk, 0.85) for chunk in sample_chunks]
        formatted = format_chunks_for_prompt(chunks_with_scores)

        assert "Chunk 1" in formatted
        assert "Goldman Sachs" in formatted
        assert "global economic conditions" in formatted
