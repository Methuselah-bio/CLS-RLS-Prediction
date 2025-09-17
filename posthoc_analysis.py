#!/usr/bin/env python3
"""
posthoc_analysis.py
-------------------

Perform simple post‑hoc analysis linking important features to the
scientific literature.  Given a list of feature names (for example,
gene identifiers) this script queries the PubMed API and prints the
titles of a few recent articles mentioning each feature.  The goal is
to provide starting points for biological interpretation of model
outputs.

Note: Access to the NCBI E‑utilities API is rate limited.  This script
inserts a short delay between requests.  If network access is
unavailable or the API returns errors, warnings are printed and empty
results are returned.

Example usage::

    python src/posthoc_analysis.py ATP1 SIR2
"""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from typing import List, Dict

try:
    import requests  # type: ignore
except ImportError:
    requests = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query PubMed for feature annotations")
    parser.add_argument("features", nargs="+", help="List of feature identifiers (e.g., gene names)")
    parser.add_argument("--max-articles", type=int, default=3, help="Maximum number of article titles to fetch per feature")
    return parser.parse_args()


def fetch_pubmed_titles(query: str, max_articles: int = 3) -> List[str]:
    """Query PubMed for a search term and return a list of article titles."""
    if requests is None:
        print("The requests library is required for PubMed queries; returning empty list.")
        return []
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = f"{base_url}esearch.fcgi?db=pubmed&term={query}&retmax={max_articles}"
    try:
        search_resp = requests.get(search_url, timeout=10)
        search_resp.raise_for_status()
        ids_xml = ET.fromstring(search_resp.text)
        ids = [elem.text for elem in ids_xml.findall(".//Id")] if ids_xml is not None else []
        titles: List[str] = []
        for pmid in ids:
            fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
            art_resp = requests.get(fetch_url, timeout=10)
            art_resp.raise_for_status()
            art_xml = ET.fromstring(art_resp.text)
            title_elem = art_xml.find(".//ArticleTitle")
            if title_elem is not None and title_elem.text:
                titles.append(title_elem.text.strip())
            # Sleep briefly to respect API rate limits
            time.sleep(0.34)
        return titles
    except Exception as exc:
        print(f"PubMed query failed for '{query}': {exc}")
        return []


def main() -> None:
    args = parse_args()
    for feature in args.features:
        print(f"\n=== {feature} ===")
        titles = fetch_pubmed_titles(feature, max_articles=args.max_articles)
        if not titles:
            print("  No articles found or an error occurred.")
            continue
        for idx, title in enumerate(titles, start=1):
            print(f"  {idx}. {title}")


if __name__ == "__main__":  # pragma: no cover
    main()