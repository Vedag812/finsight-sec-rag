"""
Downloads 10-K annual filings from the SEC EDGAR database.

Every public company in the US has to file a 10-K with the SEC every year.
It's basically a giant annual report covering their business, risks, finances, etc.
SEC keeps all of these in a public database called EDGAR, and you can download
them for free without any API key.

The only thing SEC asks is that you put your name and email in the request headers
so they can contact you if your script is causing issues on their servers.
"""

import os
import json
import time
import requests
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# Companies we're analyzing.
# I picked a mix of tech and finance companies so we can do cross-sector comparisons.
# Goldman Sachs is in here obviously since that's who I'm applying to.
# CIK is just the ID number SEC uses to identify each company.
DEFAULT_COMPANIES = {
    "AAPL": {"name": "Apple Inc.", "cik": "0000320193"},
    "MSFT": {"name": "Microsoft Corporation", "cik": "0000789019"},
    "TSLA": {"name": "Tesla Inc.", "cik": "0001318605"},
    "JPM": {"name": "JPMorgan Chase & Co.", "cik": "0000019617"},
    "GS": {"name": "The Goldman Sachs Group Inc.", "cik": "0000886982"},
    "AMZN": {"name": "Amazon.com Inc.", "cik": "0001018724"},
    "NVDA": {"name": "NVIDIA Corporation", "cik": "0001045810"},
    "GOOGL": {"name": "Alphabet Inc.", "cik": "0001652044"},
}


class SECDownloader:
    """
    Handles downloading 10-K filings from SEC EDGAR.
    
    Nothing complicated here. We hit the SEC API to find out what filings
    a company has, pick the most recent 10-K, and download it.
    The filings come as HTML files which we save locally.
    """

    BASE_URL = "https://data.sec.gov"
    ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # SEC wants you to identify yourself in the headers
        user_name = os.getenv("SEC_USER_NAME", "FinSight User")
        user_email = os.getenv("SEC_USER_EMAIL", "finsight@example.com")
        self.headers = {
            "User-Agent": f"{user_name} {user_email}",
            "Accept-Encoding": "gzip, deflate",
        }

        # SEC allows max 10 requests per second, we stay well under that
        self._last_request_time = 0
        self._min_interval = 0.15

    def _rate_limited_get(self, url: str) -> requests.Response:
        """Makes a GET request with rate limiting so we don't get blocked by SEC."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        response = requests.get(url, headers=self.headers, timeout=30)
        self._last_request_time = time.time()
        response.raise_for_status()
        return response

    def get_filing_metadata(self, cik: str) -> dict:
        """Fetches the list of all filings a company has made with the SEC."""
        url = f"{self.BASE_URL}/submissions/CIK{cik}.json"
        response = self._rate_limited_get(url)
        return response.json()

    def find_latest_10k(self, cik: str, count: int = 1) -> list[dict]:
        """
        Looks through a company's filing history and finds their most recent 10-K.
        
        We specifically look for form type "10-K" and skip things like 10-K/A
        (which are amendments to a previous filing, not the actual annual report).
        """
        metadata = self.get_filing_metadata(cik)
        recent_filings = metadata.get("filings", {}).get("recent", {})

        forms = recent_filings.get("form", [])
        accession_numbers = recent_filings.get("accessionNumber", [])
        filing_dates = recent_filings.get("filingDate", [])
        primary_docs = recent_filings.get("primaryDocument", [])

        results = []
        for i, form in enumerate(forms):
            if form == "10-K" and len(results) < count:
                results.append({
                    "accession_number": accession_numbers[i],
                    "filing_date": filing_dates[i],
                    "primary_document": primary_docs[i],
                })

        return results

    def download_filing(
        self, ticker: str, cik: str, accession_number: str, primary_doc: str, filing_date: str
    ) -> Optional[Path]:
        """
        Downloads one specific filing and saves it to disk.
        
        If we already have the file (checked by filename), we skip it.
        No point downloading the same 200-page filing twice.
        """
        safe_accession = accession_number.replace("-", "")
        company_dir = self.data_dir / ticker
        company_dir.mkdir(parents=True, exist_ok=True)

        # simple cache - if file exists, skip
        save_path = company_dir / f"10K_{filing_date}.htm"
        if save_path.exists():
            print(f"  Already have: {save_path.name}")
            return save_path

        # build the URL to the actual document on SEC's servers
        cik_clean = cik.lstrip("0")
        url = f"{self.ARCHIVES_URL}/{cik_clean}/{safe_accession}/{primary_doc}"

        try:
            response = self._rate_limited_get(url)
            save_path.write_bytes(response.content)
            size_kb = len(response.content) / 1024
            print(f"  Downloaded: {save_path.name} ({size_kb:.0f} KB)")
            return save_path
        except requests.RequestException as e:
            print(f"  Failed to download {ticker}: {e}")
            return None

    def download_company(self, ticker: str, num_filings: int = 1) -> list[Path]:
        """Downloads the latest 10-K for a company using its ticker symbol."""
        company_info = DEFAULT_COMPANIES.get(ticker)
        if not company_info:
            print(f"Unknown ticker: {ticker}. Add it to DEFAULT_COMPANIES first.")
            return []

        cik = company_info["cik"]
        name = company_info["name"]
        print(f"\nFetching 10-K for {name} ({ticker})...")

        filings = self.find_latest_10k(cik, count=num_filings)
        if not filings:
            print(f"  Couldn't find any 10-K filings for {ticker}")
            return []

        downloaded = []
        for filing in filings:
            path = self.download_filing(
                ticker=ticker,
                cik=cik,
                accession_number=filing["accession_number"],
                primary_doc=filing["primary_document"],
                filing_date=filing["filing_date"],
            )
            if path:
                downloaded.append(path)

        return downloaded

    def download_all(self, num_filings: int = 1) -> dict[str, list[Path]]:
        """Downloads 10-K filings for all 5 companies."""
        all_downloads = {}
        for ticker in DEFAULT_COMPANIES:
            paths = self.download_company(ticker, num_filings=num_filings)
            all_downloads[ticker] = paths

        total = sum(len(p) for p in all_downloads.values())
        print(f"\nDone. Got {total} filings.")
        return all_downloads

    def save_metadata(self, downloads: dict[str, list[Path]]):
        """Saves a record of what we downloaded so we can track it."""
        metadata = {}
        for ticker, paths in downloads.items():
            metadata[ticker] = {
                "company_name": DEFAULT_COMPANIES[ticker]["name"],
                "cik": DEFAULT_COMPANIES[ticker]["cik"],
                "filings": [str(p) for p in paths],
            }

        meta_path = self.data_dir / "download_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    downloader = SECDownloader()
    downloads = downloader.download_all(num_filings=1)
    downloader.save_metadata(downloads)
