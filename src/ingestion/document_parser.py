"""
Parses 10-K filings and splits them into sections.

10-K filings have a standard structure that the SEC requires. The sections
we care about most are:
- Item 1: What the company actually does
- Item 1A: Risk Factors (this is the goldmine, lists everything that could go wrong)
- Item 7: Management Discussion and Analysis (how leadership explains the year)
- Item 7A: Market risk disclosures
- Item 8: Financial statements

This parser reads the raw HTML filing, cleans it up, and figures out where
each section starts and ends. That way when we search for info later,
we know exactly which section it came from and can cite it properly.
"""

import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from bs4 import BeautifulSoup


# Patterns to find each section header in the document.
# 10-K filings aren't perfectly consistent in formatting across companies,
# so we have multiple patterns for each section to handle variations.
SECTION_PATTERNS = {
    "Item 1 - Business": [
        r"item\s*1[\.\s]*[-:]*\s*business",
        r"item\s*1\b(?!\s*[a-zA-Z])",
    ],
    "Item 1A - Risk Factors": [
        r"item\s*1a[\.\s]*[-:]*\s*risk\s*factors",
        r"item\s*1a\b",
    ],
    "Item 7 - MD&A": [
        r"item\s*7[\.\s]*[-:]*\s*management",
        r"item\s*7\b(?!\s*[a-zA-Z])",
    ],
    "Item 7A - Market Risk": [
        r"item\s*7a[\.\s]*[-:]*\s*quantitative",
        r"item\s*7a\b",
    ],
    "Item 8 - Financial Statements": [
        r"item\s*8[\.\s]*[-:]*\s*financial\s*statements",
        r"item\s*8\b",
    ],
}

# These mark where we should stop extracting (rest of the filing is
# less useful for our analysis)
END_PATTERNS = [
    r"item\s*9[\.\s]*[-:]*",
    r"item\s*9a[\.\s]*[-:]*",
    r"part\s*(iii|iv|3|4)",
]


@dataclass
class ParsedSection:
    """One section from a 10-K with metadata about where it came from."""
    company: str
    ticker: str
    filing_date: str
    section_name: str
    text: str
    page_start: int = 0
    page_end: int = 0


@dataclass
class ParsedDocument:
    """A full 10-K filing broken down into its sections."""
    company: str
    ticker: str
    filing_date: str
    sections: list[ParsedSection] = field(default_factory=list)
    full_text: str = ""

    @property
    def total_characters(self) -> int:
        return sum(len(s.text) for s in self.sections)


class DocumentParser:
    """
    Takes raw 10-K HTML files and extracts clean text organized by section.
    
    Two main steps:
    1. Strip all the HTML formatting and get readable text
    2. Use regex to find section headers and split the text accordingly
    """

    def __init__(self):
        # compile regex patterns once so we don't redo it for every document
        self._section_patterns = {}
        for section_name, patterns in SECTION_PATTERNS.items():
            self._section_patterns[section_name] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        self._end_patterns = [re.compile(p, re.IGNORECASE) for p in END_PATTERNS]

    def clean_html(self, html_content: str) -> str:
        """
        Strips HTML tags and returns clean readable text.
        
        10-K filings come with a ton of inline styles, table formatting,
        and other HTML noise. We just want the actual words.
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # get rid of scripts, styles, and other non-content elements
        for element in soup(["script", "style", "meta", "link"]):
            element.decompose()

        text = soup.get_text(separator="\n")

        # clean up whitespace while keeping paragraph breaks
        lines = []
        for line in text.splitlines():
            cleaned = line.strip()
            if cleaned:
                lines.append(cleaned)

        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    def _is_real_section_start(self, text: str, match_pos: int) -> bool:
        """
        Checks if a regex match is an actual section header, not a TOC entry.
        
        Table of contents entries look like "Item 1A. Risk Factors  5"
        (followed by a page number). Real section headers are followed by
        actual paragraphs of content. We check the 200 characters after
        the match to tell the difference.
        """
        # grab the text right after this match
        after_text = text[match_pos:match_pos + 300]
        lines_after = after_text.split("\n")

        # skip the header line itself
        content_lines = [l.strip() for l in lines_after[1:] if l.strip()]

        if not content_lines:
            return False

        # TOC entries are followed by short lines (page numbers, other items)
        # real sections are followed by actual sentences
        first_content = content_lines[0] if content_lines else ""

        # if the first content line is just a number (page number), it's TOC
        if first_content.isdigit():
            return False

        # if the first content line starts with "Item" it's still TOC
        if re.match(r"^item\s*\d", first_content, re.IGNORECASE):
            return False

        # if the first meaningful content line is at least 50 chars, it's real content
        for line in content_lines[:3]:
            if len(line) > 50:
                return True

        return False

    def find_section_boundaries(self, text: str) -> list[tuple[str, int]]:
        """
        Finds where each section starts in the document text.
        
        Returns a list of (section_name, position) tuples sorted by position.
        
        We look at ALL matches for each section pattern and pick the one
        that's actually a section header (followed by real content), not
        a table of contents entry (followed by a page number).
        """
        boundaries = []

        for section_name, patterns in self._section_patterns.items():
            found = False
            for pattern in patterns:
                matches = list(pattern.finditer(text))
                if matches:
                    for match in matches:
                        # skip very early matches (definitely TOC or header)
                        if match.start() < len(text) * 0.03:
                            continue

                        # check if this looks like a real section start
                        if self._is_real_section_start(text, match.start()):
                            boundaries.append((section_name, match.start()))
                            found = True
                            break

                if found:
                    break

        boundaries.sort(key=lambda x: x[1])
        return boundaries

    def extract_sections(
        self, text: str, ticker: str, company_name: str, filing_date: str
    ) -> list[ParsedSection]:
        """
        Splits the document into labeled sections based on where the headers are.
        
        If we can't find any section headers (some filings have weird formatting),
        we just return the whole document as one big section. Not ideal but better
        than returning nothing.
        """
        boundaries = self.find_section_boundaries(text)

        if not boundaries:
            print(f"  Heads up: couldn't find section headers in {ticker} filing. Using full text.")
            return [
                ParsedSection(
                    company=company_name,
                    ticker=ticker,
                    filing_date=filing_date,
                    section_name="Full Document",
                    text=text,
                )
            ]

        sections = []
        for i, (section_name, start_pos) in enumerate(boundaries):
            # section goes until the next section starts
            if i + 1 < len(boundaries):
                end_pos = boundaries[i + 1][1]
            else:
                # last section - look for an end marker or just take the rest
                end_pos = len(text)
                for end_pattern in self._end_patterns:
                    match = end_pattern.search(text, start_pos + 100)
                    if match:
                        end_pos = match.start()
                        break

            section_text = text[start_pos:end_pos].strip()

            # if a section is super short, it's probably a false match
            if len(section_text) < 200:
                continue

            # rough page number estimate (about 1250 chars per page)
            chars_per_page = 1250
            page_start = start_pos // chars_per_page + 1
            page_end = end_pos // chars_per_page + 1

            sections.append(
                ParsedSection(
                    company=company_name,
                    ticker=ticker,
                    filing_date=filing_date,
                    section_name=section_name,
                    text=section_text,
                    page_start=page_start,
                    page_end=page_end,
                )
            )

        return sections

    def parse_file(self, file_path: Path, ticker: str, company_name: str) -> ParsedDocument:
        """Parses a single 10-K HTML file into structured sections."""
        filing_date = file_path.stem.replace("10K_", "")

        print(f"  Parsing {ticker} filing from {filing_date}...")

        raw_html = file_path.read_text(encoding="utf-8", errors="ignore")
        clean_text = self.clean_html(raw_html)
        sections = self.extract_sections(clean_text, ticker, company_name, filing_date)

        doc = ParsedDocument(
            company=company_name,
            ticker=ticker,
            filing_date=filing_date,
            sections=sections,
            full_text=clean_text,
        )

        print(f"  Found {len(sections)} sections, {doc.total_characters:,} characters total")
        return doc

    def parse_all_filings(self, data_dir: str = "data/raw") -> list[ParsedDocument]:
        """
        Goes through all the downloaded filings and parses each one.
        Looks for .htm files organized by ticker in the data directory.
        """
        from src.ingestion.sec_downloader import DEFAULT_COMPANIES

        data_path = Path(data_dir)
        documents = []

        for ticker, info in DEFAULT_COMPANIES.items():
            company_dir = data_path / ticker
            if not company_dir.exists():
                print(f"  No filings found for {ticker}, skipping")
                continue

            for filing_path in sorted(company_dir.glob("10K_*.htm")):
                doc = self.parse_file(filing_path, ticker, info["name"])
                documents.append(doc)

        print(f"\nParsed {len(documents)} documents total")
        return documents


if __name__ == "__main__":
    parser = DocumentParser()
    docs = parser.parse_all_filings()
    for doc in docs:
        print(f"\n{doc.ticker} ({doc.filing_date}):")
        for section in doc.sections:
            print(f"  {section.section_name}: {len(section.text):,} chars (pages {section.page_start}-{section.page_end})")
