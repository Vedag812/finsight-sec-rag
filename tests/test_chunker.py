"""
Tests for the document chunker.

Makes sure chunks are created correctly with proper metadata,
sizes are within expected range, and section detection works.
"""

import pytest
from src.ingestion.chunker import DocumentChunker, DocumentChunk
from src.ingestion.document_parser import ParsedSection, ParsedDocument


@pytest.fixture
def chunker():
    return DocumentChunker(chunk_size=500, chunk_overlap=50)


@pytest.fixture
def sample_section():
    """Creates a fake section with enough text to produce multiple chunks."""
    text = (
        "This is a test section about risk factors. "
        "The company faces significant risks related to market conditions. "
        "Economic downturns could adversely affect our business operations. "
    ) * 20  # repeat to make it long enough

    return ParsedSection(
        company="Test Corp",
        ticker="TEST",
        filing_date="2024-01-15",
        section_name="Item 1A - Risk Factors",
        text=text,
        page_start=10,
        page_end=15,
    )


@pytest.fixture
def sample_document(sample_section):
    return ParsedDocument(
        company="Test Corp",
        ticker="TEST",
        filing_date="2024-01-15",
        sections=[sample_section],
        full_text=sample_section.text,
    )


class TestDocumentChunker:
    def test_short_text_stays_as_one_chunk(self, chunker):
        """Text shorter than chunk_size should not be split."""
        section = ParsedSection(
            company="Test Corp",
            ticker="TEST",
            filing_date="2024-01-15",
            section_name="Item 1 - Business",
            text="This is a short section.",
            page_start=1,
            page_end=1,
        )
        chunks = chunker.chunk_section(section)
        assert len(chunks) == 1
        assert chunks[0].text == "This is a short section."

    def test_long_text_gets_chunked(self, chunker, sample_section):
        """Longer text should be split into multiple chunks."""
        chunks = chunker.chunk_section(sample_section)
        assert len(chunks) > 1

    def test_chunks_preserve_metadata(self, chunker, sample_section):
        """Every chunk should carry the correct metadata from its source section."""
        chunks = chunker.chunk_section(sample_section)
        for chunk in chunks:
            assert chunk.company == "Test Corp"
            assert chunk.ticker == "TEST"
            assert chunk.section_name == "Item 1A - Risk Factors"
            assert chunk.filing_date == "2024-01-15"

    def test_chunk_ids_are_unique(self, chunker, sample_section):
        """Each chunk should have a unique ID."""
        chunks = chunker.chunk_section(sample_section)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs should be unique"

    def test_chunk_index_tracking(self, chunker, sample_section):
        """Chunks should know their position and the total count."""
        chunks = chunker.chunk_section(sample_section)
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.total_chunks_in_section == total

    def test_citation_format(self, chunker, sample_section):
        """Citations should follow the expected format."""
        chunks = chunker.chunk_section(sample_section)
        citation = chunks[0].citation
        assert "Test Corp" in citation
        assert "Item 1A - Risk Factors" in citation
        assert "Pages" in citation

    def test_chunk_document(self, chunker, sample_document):
        """Chunking a full document should work."""
        chunks = chunker.chunk_document(sample_document)
        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_chunk_serialization(self, chunker, sample_section):
        """Chunks should serialize to dict and back without losing data."""
        chunks = chunker.chunk_section(sample_section)
        original = chunks[0]

        # round-trip through dict
        data = original.to_dict()
        restored = DocumentChunk.from_dict(data)

        assert restored.chunk_id == original.chunk_id
        assert restored.text == original.text
        assert restored.company == original.company
        assert restored.ticker == original.ticker
        assert restored.section_name == original.section_name

    def test_no_empty_chunks(self, chunker, sample_section):
        """Chunker should never produce empty chunks."""
        chunks = chunker.chunk_section(sample_section)
        for chunk in chunks:
            assert len(chunk.text.strip()) > 0

    def test_page_numbers_are_reasonable(self, chunker, sample_section):
        """Page numbers should be positive and within the section's range."""
        chunks = chunker.chunk_section(sample_section)
        for chunk in chunks:
            assert chunk.page_start >= 1
            assert chunk.page_end >= chunk.page_start
