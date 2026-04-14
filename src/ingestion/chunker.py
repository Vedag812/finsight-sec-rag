"""
Splits parsed 10-K sections into smaller chunks for embedding.

Why do we chunk? Because you can't feed a 50-page section into an embedding
model all at once. Smaller chunks also help with retrieval - instead of getting
back a huge blob of text, you get the specific paragraph that answers your question.

The important thing here is that every chunk keeps its metadata (company, section,
page numbers). That's what lets us cite exactly where each piece of info came from
when we generate answers later. Without this, the citations wouldn't work.
"""

import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict

from src.ingestion.document_parser import ParsedSection, ParsedDocument


@dataclass
class DocumentChunk:
    """
    A single chunk of text from a 10-K filing.
    
    Carries all the metadata from the original document so we can
    always trace back where this text came from. The citation property
    formats this into a readable reference like [Apple Inc., Risk Factors, Pages 12-14].
    """
    chunk_id: str
    text: str
    company: str
    ticker: str
    filing_date: str
    section_name: str
    page_start: int
    page_end: int
    chunk_index: int
    total_chunks_in_section: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentChunk":
        return cls(**data)

    @property
    def citation(self) -> str:
        """Formats a readable citation for this chunk."""
        return f"[{self.company}, {self.section_name}, Pages {self.page_start}-{self.page_end}]"


class DocumentChunker:
    """
    Takes parsed sections and splits them into overlapping chunks.
    
    The overlap is there so we don't lose context at chunk boundaries.
    If a sentence gets cut in half at the end of one chunk, the overlap
    means the full sentence will appear at the start of the next chunk.
    
    Defaults:
    - 1500 characters per chunk (roughly 300-400 tokens)
    - 200 character overlap between chunks
    """

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _generate_chunk_id(self, ticker: str, section: str, index: int) -> str:
        """Makes a short unique ID for each chunk."""
        raw = f"{ticker}_{section}_{index}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _split_text(self, text: str) -> list[str]:
        """
        Splits text into chunks, trying to break at natural points.
        
        We prefer to break at paragraph boundaries (double newline).
        If that's not possible, we break at sentence endings (period + space).
        Last resort is breaking at any space so we don't cut words in half.
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                chunks.append(text[start:].strip())
                break

            chunk_text = text[start:end]

            # try paragraph break first
            last_para = chunk_text.rfind("\n\n")
            if last_para > self.chunk_size * 0.5:
                end = start + last_para + 2

            else:
                # try sentence break
                last_period = chunk_text.rfind(". ")
                if last_period > self.chunk_size * 0.3:
                    end = start + last_period + 2

                else:
                    # just break at a space
                    last_space = chunk_text.rfind(" ")
                    if last_space > 0:
                        end = start + last_space + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap

        return chunks

    def chunk_section(self, section: ParsedSection) -> list[DocumentChunk]:
        """Splits one section into chunks with metadata preserved."""
        text_pieces = self._split_text(section.text)
        total = len(text_pieces)

        chunks = []
        for i, text in enumerate(text_pieces):
            # figure out roughly which pages this chunk covers
            section_length = len(section.text)
            page_range = section.page_end - section.page_start
            if section_length > 0 and page_range > 0:
                chunk_start_ratio = sum(len(t) for t in text_pieces[:i]) / section_length
                chunk_end_ratio = sum(len(t) for t in text_pieces[:i+1]) / section_length
                page_start = section.page_start + int(chunk_start_ratio * page_range)
                page_end = section.page_start + int(chunk_end_ratio * page_range)
            else:
                page_start = section.page_start
                page_end = section.page_end

            chunk = DocumentChunk(
                chunk_id=self._generate_chunk_id(section.ticker, section.section_name, i),
                text=text,
                company=section.company,
                ticker=section.ticker,
                filing_date=section.filing_date,
                section_name=section.section_name,
                page_start=max(page_start, 1),
                page_end=max(page_end, 1),
                chunk_index=i,
                total_chunks_in_section=total,
            )
            chunks.append(chunk)

        return chunks

    def chunk_document(self, document: ParsedDocument) -> list[DocumentChunk]:
        """Chunks all sections of one document."""
        all_chunks = []
        for section in document.sections:
            section_chunks = self.chunk_section(section)
            all_chunks.extend(section_chunks)
        return all_chunks

    def chunk_all_documents(self, documents: list[ParsedDocument]) -> list[DocumentChunk]:
        """Chunks all documents and gives back one flat list."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            print(f"  {doc.ticker} ({doc.filing_date}): {len(chunks)} chunks")

        print(f"\nTotal: {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

    def save_chunks(self, chunks: list[DocumentChunk], output_dir: str = "data/processed"):
        """Saves chunks as JSON so we don't have to reprocess every time."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        chunks_data = [chunk.to_dict() for chunk in chunks]
        save_path = output_path / "chunks.json"

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(chunks)} chunks to {save_path}")

    @staticmethod
    def load_chunks(input_dir: str = "data/processed") -> list[DocumentChunk]:
        """Loads previously saved chunks from disk."""
        load_path = Path(input_dir) / "chunks.json"
        if not load_path.exists():
            raise FileNotFoundError(f"No chunks found at {load_path}. Run the ingestion pipeline first.")

        with open(load_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        chunks = [DocumentChunk.from_dict(d) for d in chunks_data]
        print(f"Loaded {len(chunks)} chunks from {load_path}")
        return chunks


if __name__ == "__main__":
    from src.ingestion.document_parser import DocumentParser

    parser = DocumentParser()
    docs = parser.parse_all_filings()

    chunker = DocumentChunker()
    chunks = chunker.chunk_all_documents(docs)
    chunker.save_chunks(chunks)

    if chunks:
        sample = chunks[0]
        print(f"\nSample chunk:")
        print(f"  Company: {sample.company}")
        print(f"  Section: {sample.section_name}")
        print(f"  Citation: {sample.citation}")
        print(f"  Text preview: {sample.text[:200]}...")
