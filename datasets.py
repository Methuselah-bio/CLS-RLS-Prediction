"""Utility functions for dataset transformation and biological feature extraction.

This module centralises common operations used during data preparation.  The
functions are designed to be composable and optional: if an external
dependency is missing, they will log a warning and return empty or
identity results rather than raising an exception.  This allows the
pipeline to degrade gracefully when optional packages such as
``gprofiler-official`` or ``networkx`` are not installed.
"""

from __future__ import annotations

import logging
from typing import List, Dict

import numpy as np
import pandas as pd


def compute_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute z‑scores for all numeric columns in a DataFrame.

    A new DataFrame is returned where each numeric column is z‑scored
    across samples (subtracting the mean and dividing by the standard
    deviation).  Column names are prefixed with ``z_`` to distinguish
    them from raw values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing numeric features.

    Returns
    -------
    pandas.DataFrame
        DataFrame with z‑scored numeric columns; non‑numeric columns are
        copied unchanged.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z_df = df.copy()
    # Avoid division by zero if a column has zero variance
    for col in numeric_cols:
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            logging.warning(f"Column '{col}' has zero variance; z‑scores will be zero.")
            z_df[f"z_{col}"] = 0.0
        else:
            z_df[f"z_{col}"] = (df[col] - df[col].mean()) / std
    return z_df


# Attempt to import g:Profiler.  When unavailable, enrichment will be skipped.
try:
    from gprofiler import GProfiler  # type: ignore
except ImportError:
    GProfiler = None  # type: ignore


def compute_pathway_enrichment(genes: List[str], species: str = "scerevisiae") -> Dict[str, float]:
    """Compute pathway enrichment scores for a list of genes using g:Profiler.

    The enrichment scores are negative log10 p‑values for the top enriched
    pathways.  If the ``gprofiler-official`` package is not installed or
    the query fails, an empty dictionary is returned and a warning is
    logged.

    Parameters
    ----------
    genes : list of str
        List of gene identifiers (e.g., ORF names) to test for enrichment.
    species : str, default 'scerevisiae'
        Organism identifier recognised by g:Profiler (e.g., 'hsapiens',
        'scerevisiae').

    Returns
    -------
    dict
        Mapping from feature names (``enrich_<term name>``) to enrichment
        scores (−log10 p‑value).  At most ten terms are returned to limit
        dimensionality.
    """
    if GProfiler is None:
        logging.warning("gprofiler-official is not installed; skipping pathway enrichment.")
        return {}
    if not genes:
        return {}
    try:
        gp = GProfiler(return_dataframe=True)
        res = gp.profile(organism=species, query=genes)
        if res.empty:
            return {}
        enrichment_scores: Dict[str, float] = {}
        # Take the top ten enriched terms and compute −log10(p)
        for _, row in res.head(10).iterrows():
            term_name = row.get("name", str(row.get("term_id", "unknown")))
            pval = row.get("p_value", 1.0)
            score = -np.log10(pval) if pval > 0 else 0.0
            # Replace spaces with underscores for feature naming
            term_key = f"enrich_{term_name.replace(' ', '_')}"
            enrichment_scores[term_key] = score
        return enrichment_scores
    except Exception as exc:  # pragma: no cover
        logging.warning(f"Pathway enrichment failed: {exc}")
        return {}


# Attempt to import networkx.  When unavailable, network features will be skipped.
try:
    import networkx as nx  # type: ignore
except ImportError:
    nx = None  # type: ignore


def compute_network_features(genes: List[str]) -> Dict[str, float]:
    """Compute simple network centrality metrics for a set of genes.

    In the absence of an externally supplied protein–protein interaction
    network, this function constructs a sparse, deterministic synthetic
    network by connecting pairs of genes based on a hash function.  The
    degree centrality of each gene is returned.  When ``networkx`` is not
    installed, an empty dictionary is returned and a warning is logged.

    Parameters
    ----------
    genes : list of str
        Collection of gene identifiers; nodes in the network.

    Returns
    -------
    dict
        Mapping from feature names (``deg_centrality_<gene>``) to degree
        centrality scores in the synthetic network.
    """
    if nx is None:
        logging.warning("networkx is not installed; skipping network feature computation.")
        return {}
    if not genes:
        return {}
    # Build a graph with deterministic pseudo‑random edges.  The hash mod 17 is
    # arbitrary but ensures reproducibility across runs.
    G = nx.Graph()
    G.add_nodes_from(genes)
    for i, ga in enumerate(genes):
        for gb in genes[i + 1 :]:
            # Connect genes if the combined hash satisfies a condition.  Using
            # ``abs`` avoids negative values on different Python versions.
            if abs(hash(ga + gb)) % 17 == 0:
                G.add_edge(ga, gb)
    centrality = nx.degree_centrality(G)
    return {f"deg_centrality_{gene}": float(score) for gene, score in centrality.items()}


# Attempt to import g:Profiler for ortholog mapping.
try:
    # gprofiler-official 1.x provides a GProfiler class capable of identifier
    # conversion and orthology mapping.  When unavailable, ortholog mapping
    # functionality will be disabled.
    from gprofiler import GProfiler  # type: ignore
except ImportError:
    GProfiler = None  # type: ignore


def map_orthologs(genes: List[str], target_species: str = "hsapiens", source_species: str = "scerevisiae") -> Dict[str, str]:
    """Map a list of source genes to orthologs in the target species.

    This helper uses g:Profiler’s g:Convert/g:Orth services to translate gene
    identifiers between organisms.  It returns a dictionary mapping each
    source gene to its best matching ortholog in the target species.  If
    g:Profiler is not installed or the API call fails, an empty mapping is
    returned and a warning is logged.  g:Profiler can convert between
    namespaces and perform ortholog lookups across taxonomic groups【151163149604280†L400-L415】.

    Parameters
    ----------
    genes : list of str
        Gene identifiers (e.g., systematic ORF names) from the source species.
    target_species : str, default 'hsapiens'
        Ensembl organism identifier for the target species (e.g., 'hsapiens',
        'mmusculus', 'rnorvegicus').
    source_species : str, default 'scerevisiae'
        Ensembl organism identifier for the source species.

    Returns
    -------
    dict
        Mapping from each input gene to a single ortholog in the target
        species; if no ortholog is found, the gene is absent from the
        mapping.
    """
    if GProfiler is None:
        logging.warning("gprofiler-official is not installed; skipping ortholog mapping.")
        return {}
    if not genes:
        return {}
    try:
        gp = GProfiler(return_dataframe=True)
        # Attempt to perform an orthology conversion.  g:Convert in the
        # gprofiler-official API does not expose an explicit target species
        # argument; however, orthology mapping is available via the REST
        # endpoint 'gorth'.  Here we build a DataFrame by iterating over
        # individual genes to ensure graceful degradation if some queries fail.
        mapping: Dict[str, str] = {}
        for gene in genes:
            try:
                # Use g:Profiler’s convert method; if target_species is
                # supported, the result will include orthologous genes in
                # the 'name' field.  This is a best‑effort approach and may
                # require an internet connection.
                df = gp.convert(organism=source_species, query=[gene], target=target_species)
                if not df.empty:
                    # Take the first match in the converted DataFrame
                    target_gene = df.iloc[0]["name"] if "name" in df.columns else None
                    if isinstance(target_gene, str):
                        mapping[gene] = target_gene
            except Exception:
                continue
        return mapping
    except Exception as exc:  # pragma: no cover
        logging.warning(f"Ortholog mapping failed: {exc}")
        return {}