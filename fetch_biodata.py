"""
fetch_biodata.py
-----------------

This script automates the download and preprocessing of biological
datasets relevant to yeast ageing studies.  In particular it supports
fetching gene‑expression measurements from the Gene Expression Omnibus
(GEO) and formatting them into a tabular CSV file ready for ingestion
by the Methuselah pipeline.  It is designed to reduce manual steps
when replacing the legacy UCI yeast dataset with realistic
chronological lifespan (CLS) or nutrient perturbation data.

The primary use case is downloading a GEO series such as GSE100166,
which profiles gene expression under caloric restriction and
rapamycin treatment.  These conditions are commonly used in
chronological lifespan assays to probe how nutrient signalling affects
survival.  This script retrieves the series matrix using GEOparse,
extracts the numeric expression table and basic sample metadata (e.g.,
treatment group), and writes them to disk.  Users can then merge
these features with survival outcomes (e.g., colony forming unit
counts) from deletion screens or other phenotype tables.

Usage::

    python src/fetch_biodata.py --accession GSE100166 --output data/raw/gse100166.csv

Example configuration snippet::

    paths:
      raw_dir: data/raw
    task:
      data_type: geo
      geo_accession: GSE100166

Requirements
============
This script depends on the `GEOparse` package for downloading and
parsing GEO series matrices.  Install it via pip::

    pip install GEOparse

Note that GEOparse in turn requires internet access to fetch the
datasets.  If you are running this script in an environment without
network access, download the series matrix manually from the GEO
website and place it in the `paths.raw_dir` directory specified in
your configuration.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

try:
    import GEOparse  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "GEOparse is required for fetch_biodata.py. Install via `pip install GEOparse`"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and preprocess GEO datasets")
    parser.add_argument(
        "--accession",
        type=str,
        required=True,
        help="GEO accession (e.g., GSE100166) to fetch",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the processed CSV file",
    )
    parser.add_argument(
        "--destdir",
        type=str,
        default="data/raw",
        help="Directory where raw GEO files will be cached",
    )
    return parser.parse_args()


def fetch_geo_series(accession: str, destdir: Path) -> pd.DataFrame:
    """Download a GEO series and assemble its expression matrix.

    Parameters
    ----------
    accession : str
        GEO accession starting with 'GSE'.
    destdir : pathlib.Path
        Directory used by GEOparse to cache downloaded files.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing gene expression values for all samples.  The
        first column is the gene identifier followed by columns for each
        sample.  Only numeric columns from the GSM tables are retained.
    """
    logging.info("Fetching GEO accession %s", accession)
    gse = GEOparse.get_GEO(accession, destdir=str(destdir), annotate_gpl=True)
    expression_frames: List[pd.DataFrame] = []
    for gsm_name, gsm in gse.gsms.items():
        if hasattr(gsm, "table"):
            tbl = gsm.table
            numeric_cols = tbl.select_dtypes(include=[float, int]).columns
            if numeric_cols.empty:
                continue
            expr = tbl[numeric_cols].copy()
            expr.columns = [f"{gsm_name}_{col}" for col in numeric_cols]
            expression_frames.append(expr)
    if not expression_frames:
        raise ValueError(f"No numeric expression tables found for GEO accession {accession}")
    expr_df = pd.concat(expression_frames, axis=1, join="inner")
    expr_df.reset_index(drop=False, inplace=True)
    expr_df.rename(columns={expr_df.columns[0]: "gene"}, inplace=True)
    return expr_df


def extract_sample_metadata(gse) -> pd.DataFrame:
    """Extract basic sample metadata such as treatment conditions.

    GEO series often include sample characteristics like treatment
    conditions, time points or strains.  This helper attempts to parse
    these metadata into a tidy DataFrame indexed by GSM sample name.  It
    searches for 'characteristics' fields in each GSM and splits key–value
    pairs separated by ':' or '='.  Non‑informative fields are ignored.

    Parameters
    ----------
    gse : GEOparse.GSE
        Loaded GEO series object.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per sample and columns for each
        characteristic detected.  Missing values are filled with NaN.
    """
    metadata = {}
    for gsm_name, gsm in gse.gsms.items():
        char_dict = {}
        for ch in gsm.metadata.get("characteristics_ch1", []):
            # Split on common delimiters
            if ":" in ch:
                key, val = ch.split(":", 1)
            elif "=" in ch:
                key, val = ch.split("=", 1)
            else:
                continue
            key = key.strip().lower().replace(" ", "_")
            val = val.strip()
            char_dict[key] = val
        metadata[gsm_name] = char_dict
    meta_df = pd.DataFrame.from_dict(metadata, orient="index")
    meta_df.index.name = "gsm"
    return meta_df


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    destdir = Path(args.destdir)
    destdir.mkdir(parents=True, exist_ok=True)
    # Download and parse series
    gse = GEOparse.get_GEO(args.accession, destdir=str(destdir), annotate_gpl=True)
    expr_df = fetch_geo_series(args.accession, destdir)
    # Extract metadata
    meta_df = extract_sample_metadata(gse)
    # Combine expression and metadata by prefixing sample columns with metadata
    logging.info("Merging expression matrix with sample metadata")
    # The expression matrix has sample columns named GSMID_feature; we split sample ID
    sample_ids = [col.split("_")[0] for col in expr_df.columns[1:]]
    # Build repeated metadata rows for each gene row
    meta_rows: List[pd.DataFrame] = []
    for gsm_id in sample_ids:
        # For each sample column we create a DataFrame with gene identifier and metadata value
        # We'll join later after saving separate files; this is optional but demonstrates merging
        pass  # Metadata merging is dataset‑specific; left as placeholder
    # Save expression matrix to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expr_df.to_csv(output_path, index=False)
    logging.info("Saved expression matrix to %s", output_path)
    # Save sample metadata for reference
    meta_path = output_path.with_suffix(".metadata.csv")
    meta_df.to_csv(meta_path)
    logging.info("Saved sample metadata to %s", meta_path)


if __name__ == "__main__":  # pragma: no cover
    main()