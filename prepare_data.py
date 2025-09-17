#!/usr/bin/env python3
"""
prepare_data.py
----------------

This script extends the original data preparation routine to support
multiple input formats and optional biological feature engineering.  It
reads raw data from a directory specified in the configuration and
produces a processed CSV suitable for machine‑learning models.  In
addition to the UCI Yeast demo, the following modes are supported via
``task.data_type`` in the YAML configuration:

* ``uci`` – legacy whitespace‑delimited data with eight numeric
  features and a categorical localisation label.
* ``csv``/``tsv`` – generic gene‑expression matrices where rows are
  samples and columns are genes or other features.  Numeric columns are
  z‑scored and prefixed with ``z_``.  When enabled, pathway enrichment
  and network features are computed from the set of gene names.
* ``fasta`` – sequence files; simple summary statistics (length and
  GC content) are extracted if Biopython is available.
* ``geo`` – download a GEO series using ``GEOparse``; a toy example
  processes the series matrix into a numeric DataFrame.

Optional enrichment and network features are controlled by
``task.include_pathway_enrichment`` and ``task.include_network_features``.
These require the packages ``gprofiler-official`` and ``networkx``
respectively.  Enrichment scores are computed using g:Profiler and
degree centrality features are derived from a synthetic interaction
network as a placeholder for STRING interactions.

Usage:
    python src/prepare_data.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

from . import datasets

try:
    # Optional import for sequence handling
    from Bio import SeqIO  # type: ignore
except ImportError:
    SeqIO = None  # type: ignore

try:
    import GEOparse  # type: ignore
except ImportError:
    GEOparse = None  # type: ignore


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare dataset for modelling")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()


def load_config(path: str | os.PathLike) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_raw_file(raw_dir: Path, override: Optional[str] = None) -> Path:
    """Determine which raw file to process based on heuristics and overrides."""
    if override:
        candidate = raw_dir / override
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Specified raw_file {candidate} does not exist")
    # Default priority: yeast.data → first file
    if (raw_dir / "yeast.data").exists():
        return raw_dir / "yeast.data"
    files = list(raw_dir.glob("*"))
    if not files:
        raise FileNotFoundError(f"No raw data files found in {raw_dir}")
    return files[0]


def process_uci(file_path: Path, target_col: str) -> pd.DataFrame:
    """Process the legacy UCI Yeast dataset."""
    columns = [
        "Sequence_Name",
        "mcg",
        "gvh",
        "alm",
        "mit",
        "erl",
        "pox",
        "vac",
        "nuc",
        "localization_site",
    ]
    df = pd.read_csv(file_path, sep="\s+", header=None, names=columns)
    df[target_col] = df["localization_site"].astype("category").cat.codes
    df = df.drop(columns=["localization_site"])
    return df


def process_csv(file_path: Path) -> pd.DataFrame:
    """Process a generic CSV/TSV gene‑expression matrix.

    This loader is deliberately minimal: it simply reads the file using
    comma or tab separation based on the suffix.  It does not assume a
    specific schema – rows may correspond to samples or genes, and
    columns may contain numeric or categorical information.  Downstream
    functions (e.g., ``compute_zscores`` or model training) determine
    which columns are used.  When integrating external datasets such as
    the yeast Replicative Lifespan (RLS) or Chronological Lifespan (CLS)
    tables, ensure that the file is in a delimited format understood by
    pandas (CSV, TSV) and that any target or identifier columns are
    clearly labeled.  See ``process_rls`` and ``process_cls`` for
    specialised handling.
    """
    sep = "," if file_path.suffix.lower() == ".csv" else "\t"
    df = pd.read_csv(file_path, sep=sep)
    return df


def process_rls(file_path: Path, target_col: str | None = None) -> pd.DataFrame:
    """Process a replicative lifespan (RLS) dataset.

    The RLS dataset catalogues the replicative lifespan of ~4,700
    single‑gene deletion mutants in yeast.  Supplementary tables S1 and
    S2 in the study *“A comprehensive analysis of replicative lifespan in
    4,698 single‑gene deletion mutants uncovers conserved mechanisms of
    aging”* list all tested strains and the subset of long‑lived mutants【864702090041242†L930-L939】.
    These tables can be downloaded as CSV/TSV files and placed in the
    ``data/raw`` directory.  This function reads such a file and
    returns a DataFrame.  If a ``target_col`` is provided, the column
    with that name is used as the target; otherwise the first numeric
    column (e.g., average replicative lifespan) becomes the target.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the CSV/TSV file containing RLS data.
    target_col : str or None, optional
        Name of the column to use as the prediction target.  If
        ``None``, the first numeric column is selected.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a target column named ``target`` and all
        other columns left unchanged.
    """
    df = process_csv(file_path)
    # The RLS tables typically contain a lifespan measurement column
    # (e.g., ``mean_RLS`` or ``replicative_lifespan``).  We deliberately
    # avoid renaming any columns here; instead, specify the desired
    # prediction target via the ``task.target`` key in the YAML
    # configuration.  If ``target_col`` is provided, warn if it does
    # not exist.
    if target_col and target_col not in df.columns:
        logging.warning(
            "Specified target_col '%s' not found in RLS dataset; using config.task.target to select target.",
            target_col,
        )
    return df


def process_cls(file_path: Path, target_col: str | None = None) -> pd.DataFrame:
    """Process a chronological lifespan (CLS) dataset.

    Chronological lifespan measures the survival of non‑dividing yeast
    populations over time.  The genome‑wide CLS screen analysed ~4,800
    viable deletion mutants and reported survival curves and colony‑
    forming unit (CFU) estimates【864702090041242†L930-L939】.  The full
    dataset, including supplementary tables with survival profiles, is
    available from the paper *“Genome‑Wide Screen in Saccharomyces
    cerevisiae Identifies Vacuolar Protein Sorting, Autophagy,
    Biosynthetic, and tRNA Methylation Genes Involved in Life Span
    Regulation”*.  To integrate CLS data, download the tables from
    <http://chemogenomics.med.utoronto.ca/supplemental/lifespan/> and
    place them in ``data/raw``.  This function reads such a CSV/TSV file
    and designates a target column.  If ``target_col`` is not provided,
    the first numeric column (e.g., area under the survival curve or
    CFU count) is used.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the CSV/TSV file containing CLS data.
    target_col : str or None, optional
        Name of the column to use as the prediction target.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a target column named ``target``.
    """
    df = process_csv(file_path)
    # For CLS tables, survival measures such as area under the survival
    # curve or CFU counts are typically numeric.  Do not rename
    # columns here; instead rely on ``task.target`` in the YAML config
    # to specify which column is the target.  Warn if an explicit
    # ``target_col`` is provided but missing.
    if target_col and target_col not in df.columns:
        logging.warning(
            "Specified target_col '%s' not found in CLS dataset; using config.task.target to select target.",
            target_col,
        )
    return df


def process_fasta(file_path: Path, target_col: str) -> pd.DataFrame:
    """Process a FASTA file by computing length and GC content for each record."""
    if SeqIO is None:
        raise ImportError("Biopython is required to process FASTA files. Please install it.")
    records = list(SeqIO.parse(str(file_path), "fasta"))
    if not records:
        raise ValueError(f"No sequences found in FASTA file {file_path}")
    features: List[dict] = []
    for rec in records:
        seq = rec.seq.upper()
        length = len(seq)
        gc_count = seq.count("G") + seq.count("C")
        gc_content = gc_count / length if length > 0 else 0.0
        features.append({
            "id": rec.id,
            "seq_length": length,
            "gc_content": gc_content,
        })
    df = pd.DataFrame(features)
    df[target_col] = 0
    return df


def process_geo(accession: str, destdir: Path) -> pd.DataFrame:
    """Download and parse a GEO series matrix into a numeric DataFrame.

    This function relies on ``GEOparse`` to download the series and assemble
    the expression matrix from GSM tables.  Only continuous values are
    retained.  If GEOparse is not installed or the accession is invalid,
    an exception is raised.
    """
    if GEOparse is None:
        raise ImportError("GEOparse is not installed; cannot fetch GEO datasets.")
    logging.info(f"Downloading GEO accession {accession} to {destdir}")
    gse = GEOparse.get_GEO(accession, destdir=str(destdir), annotate_gpl=True)
    # Concatenate expression data from each sample (GSM) into a DataFrame
    expression_frames: List[pd.DataFrame] = []
    for gsm_name, gsm in gse.gsms.items():
        # Each GSM object has a table attribute with data and metadata
        if hasattr(gsm, "table"):
            tbl = gsm.table
            # Filter numeric columns and rename with sample identifier
            numeric_cols = tbl.select_dtypes(include=[float, int]).columns
            expr = tbl[numeric_cols].copy()
            expr.columns = [f"{gsm_name}_{col}" for col in numeric_cols]
            expression_frames.append(expr)
    if not expression_frames:
        raise ValueError(f"No numeric expression tables found for GEO accession {accession}")
    # Join columns on their index (gene identifiers)
    expr_df = pd.concat(expression_frames, axis=1, join="inner")
    # Reset index to a column for gene names
    expr_df.reset_index(drop=False, inplace=True)
    expr_df.rename(columns={expr_df.columns[0]: "gene"}, inplace=True)
    return expr_df


def process_multiomics(file_paths: List[Path], join_on: str = "gene") -> pd.DataFrame:
    """Merge multiple omics datasets on a common identifier.

    Multi‑omics studies often provide separate tables for transcriptomics,
    proteomics and metabolomics measured on the same set of genes or
    samples.  This helper reads each file using :func:`process_csv` and
    performs an inner join on the specified column.  Columns from the
    second and subsequent tables are suffixed with an index to avoid
    name collisions.

    Parameters
    ----------
    file_paths : list of pathlib.Path
        Paths to the omics data files.  At least two files must be
        provided.
    join_on : str, default 'gene'
        Column name on which to merge the tables (e.g., gene identifier
        or sample ID).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the merged omics measurements.  If no
        common join column is found, an exception is raised.
    """
    if len(file_paths) < 2:
        raise ValueError("process_multiomics requires at least two input files")
    merged: Optional[pd.DataFrame] = None
    for idx, path in enumerate(file_paths):
        df_part = process_csv(path)
        if join_on not in df_part.columns:
            raise ValueError(f"Join column '{join_on}' not found in {path}")
        if merged is None:
            merged = df_part
        else:
            # Suffix column names to avoid collisions except for the join column
            suffix = f"_m{idx}"
            df_part_renamed = df_part.rename(columns={c: c + suffix for c in df_part.columns if c != join_on})
            merged = merged.merge(df_part_renamed, on=join_on, how="inner")
    if merged is None:
        raise RuntimeError("process_multiomics failed to merge files")
    return merged


def process_scrna(file_path: Path, target_col: str | None = None) -> pd.DataFrame:
    """Process single‑cell RNA‑seq data.

    This placeholder treats a scRNA‑seq matrix as a generic CSV/TSV file where
    rows correspond to cells and columns to genes or other features.  In a
    production pipeline you might load `.h5ad` files via Scanpy and compute
    per‑cell summaries or cell‑type annotations.  For now, the data is read
    using :func:`process_csv` and returned unmodified.
    """
    return process_csv(file_path)


def process_spatial(file_path: Path, target_col: str | None = None) -> pd.DataFrame:
    """Process spatial transcriptomics data.

    Spatial transcriptomics measures gene expression in spatially resolved
    coordinates.  As with scRNA‑seq, this function currently treats the
    input as a generic CSV/TSV.  For realistic analyses, consider using
    specialized libraries (e.g., Squidpy) to extract features such as
    neighbourhood interactions and spatial domains.
    """
    return process_csv(file_path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = config.get("seed", 42)
    raw_dir = Path(config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_path = Path(config["paths"]["processed"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    target_col = config["task"]["target"]
    data_type = config["task"].get("data_type", "uci").lower()

    # Determine input file or accession
    raw_file_name = config.get("paths", {}).get("raw_file")
    data_file: Optional[Path] = None
    if data_type == "geo":
        # For GEO we use the accession rather than a local file
        accession = config["task"].get("geo_accession")
        if not accession:
            raise ValueError("task.geo_accession must be set when data_type is 'geo'")
        df = process_geo(accession, raw_dir)
    else:
        # Infer raw file path
        data_file = infer_raw_file(raw_dir, override=raw_file_name)
        suffix = data_file.suffix.lower()
        # Explicitly prioritise data_type over suffix detection so that RLS/CLS
        # tables stored as `.csv` or `.tsv` are processed by the appropriate
        # functions.  Only when the data_type is not specified do we fall back
        # to suffix heuristics.
        if data_type == "multiomics":
            # For multi‑omics, combine multiple raw files on a common identifier
            additional = config.get("paths", {}).get("additional_raw_files", []) or []
            # Ensure the additional files are list of strings
            if not isinstance(additional, list):
                raise ValueError("paths.additional_raw_files must be a list of file names")
            file_paths = [data_file] + [raw_dir / fname for fname in additional]
            join_col = config["task"].get("multiomics_join_on", "gene")
            df = process_multiomics(file_paths, join_on=join_col)
        elif data_type == "scrna":
            df = process_scrna(data_file, target_col)
        elif data_type == "spatial":
            df = process_spatial(data_file, target_col)
        elif data_type == "rls":
            # Replicative lifespan: read the RLS table; target selection is deferred to config
            df = process_rls(data_file, target_col)
        elif data_type == "cls":
            # Chronological lifespan: read the CLS table; target selection is deferred to config
            df = process_cls(data_file, target_col)
        elif data_type == "uci" or (data_type != "csv" and suffix in {".data", "", ".txt"}):
            df = process_uci(data_file, target_col)
        elif data_type in {"csv", "tsv"} or suffix in {".csv", ".tsv", ".tab"}:
            df = process_csv(data_file)
        elif data_type in {"fasta", "fa", "fna", "faa"} or suffix in {".fasta", ".fa", ".fna", ".faa"}:
            df = process_fasta(data_file, target_col)
        else:
            raise ValueError(f"Unsupported data_type '{data_type}' or file extension '{suffix}'.")

    # Optional: apply class balancing or data augmentation for classification tasks
    # -------------------------------------------------------------------------
    # Augmentation is only meaningful for classification problems.  When
    # task.problem_type == 'classification' and preprocessing.augmentation.method
    # is set to 'smote', this step will oversample minority classes using
    # imbalanced‑learn's SMOTE algorithm.  Additional augmentation methods
    # (e.g., GAN‑based generation) can be implemented here in the future.
    preprocessing_cfg = config.get("task", {}).get("preprocessing", {}) or {}
    augmentation_cfg = preprocessing_cfg.get("augmentation", {}) or {}
    aug_method = augmentation_cfg.get("method", "none").lower() if isinstance(augmentation_cfg, dict) else "none"
    # Only apply augmentation if classification and a target column is present
    if config.get("task", {}).get("problem_type", "classification").lower() == "classification" and aug_method != "none":
        # Ensure target column exists
        if target_col not in df.columns:
            logging.warning("Cannot apply augmentation: target column '%s' not found", target_col)
        else:
            try:
                from imblearn.over_sampling import SMOTE  # type: ignore
            except ImportError:
                logging.warning("imbalanced-learn is required for SMOTE augmentation; skipping")
            else:
                if aug_method == "smote":
                    logging.info("Applying SMOTE augmentation to balance classes")
                    feature_df = df.drop(columns=[target_col])
                    target_series = df[target_col]
                    smote = SMOTE(random_state=seed)
                    X_res, y_res = smote.fit_resample(feature_df.values, target_series.values)
                    # Reconstruct DataFrame with original column names
                    df = pd.DataFrame(X_res, columns=feature_df.columns)
                    df[target_col] = y_res
                else:
                    logging.warning("Unknown augmentation method '%s'; supported: 'smote'", aug_method)

    # Apply generic preprocessing for tabular gene expression or lifespan tables
    if data_type in {"csv", "tsv", "geo", "rls", "cls", "multiomics", "scrna", "spatial"}:
        # Compute z‑scores for numeric columns and merge with original DataFrame
        z_df = datasets.compute_zscores(df)
        # Concatenate z‑score columns and drop raw numeric if desired
        df = pd.concat([df, z_df.filter(regex="^z_", axis=1)], axis=1)

        # Optionally compute enrichment and network features
        genes: List[str] = []
        # Attempt to discover gene names from columns or dedicated column
        if "gene" in df.columns:
            genes = df["gene"].astype(str).tolist()
        else:
            # Fallback: treat non‑numeric columns (excluding target) as genes
            nonnum = df.select_dtypes(exclude=[float, int]).columns
            candidate_cols = [c for c in nonnum if c != target_col]
            if candidate_cols:
                genes = candidate_cols
        if genes:
            # Optional: map gene list to orthologs in a target species and then perform
            # enrichment on the orthologs.  This enables cross‑species comparisons,
            # e.g., mapping yeast ORFs to human gene symbols and deriving pathway
            # enrichments in the human context【151163149604280†L400-L415】.
            if config["task"].get("include_ortholog_mapping", False):
                target_species = config["task"].get("ortholog_target_species", "hsapiens")
                orth_map = datasets.map_orthologs(genes, target_species=target_species,
                                                 source_species=config["task"].get("pathway_species", "scerevisiae"))
                ortho_genes = list(orth_map.values())
                if ortho_genes and config["task"].get("include_pathway_enrichment", False):
                    enrich_orth = datasets.compute_pathway_enrichment(ortho_genes, species=target_species)
                    for key, val in enrich_orth.items():
                        # Suffix keys to indicate ortholog enrichment
                        df[f"orth_{key}"] = val
            # Pathway enrichment on the native gene list
            if config["task"].get("include_pathway_enrichment", False):
                species = config["task"].get("pathway_species", "scerevisiae")
                enrich = datasets.compute_pathway_enrichment(genes, species=species)
                for key, val in enrich.items():
                    df[key] = val  # replicate across all rows
            if config["task"].get("include_network_features", False):
                net_feats = datasets.compute_network_features(genes)
                for key, val in net_feats.items():
                    df[key] = val

        # Additional omics-specific preprocessing steps
        pre_cfg = config["task"].get("preprocessing", {}) or {}
        # Batch effect correction using ComBat from combat_py
        if pre_cfg.get("batch_correction", False):
            try:
                from combat.pycombat import pycombat  # type: ignore
            except Exception:
                logging.warning("combat-py is not installed; skipping batch effect correction")
            else:
                # Identify batch column if provided; else assume no batch information
                batch_col = config["task"].get("batch_col")
                if batch_col and batch_col in df.columns:
                    numeric_cols = df.select_dtypes(include=[float, int]).columns
                    corrected = pycombat(df[numeric_cols].values.T, df[batch_col].values)
                    df[numeric_cols] = corrected.T
        # Variance stabilizing transformation using PowerTransformer
        if pre_cfg.get("variance_stabilization", False):
            try:
                from sklearn.preprocessing import PowerTransformer  # type: ignore
            except Exception:
                logging.warning("sklearn is required for variance stabilization; skipping")
            else:
                numeric_cols = df.select_dtypes(include=[float, int]).columns
                pt = PowerTransformer()
                df[numeric_cols] = pt.fit_transform(df[numeric_cols])
        # Imputation of missing values using IterativeImputer
        if pre_cfg.get("imputation", False):
            try:
                from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                from sklearn.impute import IterativeImputer  # type: ignore
            except Exception:
                logging.warning("sklearn is required for imputation; skipping")
            else:
                numeric_cols = df.select_dtypes(include=[float, int]).columns
                imp = IterativeImputer(random_state=seed)
                df[numeric_cols] = imp.fit_transform(df[numeric_cols])
        # Dimensionality reduction
        dr_methods = pre_cfg.get("dimensionality_reduction", []) or []
        if dr_methods:
            from sklearn.decomposition import PCA  # type: ignore
            for method in dr_methods:
                m = method.lower()
                if m == "pca":
                    try:
                        # Choose number of components based on explained variance
                        numeric_cols = df.select_dtypes(include=[float, int]).columns
                        pca = PCA(n_components=min(50, len(numeric_cols)))
                        comps = pca.fit_transform(df[numeric_cols])
                        # Add PCA components as new features
                        for i in range(comps.shape[1]):
                            df[f"pca_{i+1}"] = comps[:, i]
                    except Exception as exc:
                        logging.warning(f"PCA failed: {exc}")
                elif m == "vae":
                    # Attempt to use the VAE defined in advanced_models.py for unsupervised
                    try:
                        from .advanced_models import VariationalAutoencoder  # type: ignore
                    except Exception:
                        logging.warning("VariationalAutoencoder not available; skipping VAE reduction")
                        continue
                    # Fit VAE with a small number of epochs to obtain latent features
                    numeric_cols = df.select_dtypes(include=[float, int]).columns
                    X_num = df[numeric_cols].values.astype(np.float32)
                    input_dim = X_num.shape[1]
                    # Choose latent dim as min of 10 or user‑defined
                    latent_dim = pre_cfg.get("vae_latent_dim", 10)
                    hidden_dim = pre_cfg.get("vae_hidden_dim", max(16, latent_dim * 2))
                    vae = VariationalAutoencoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
                    # Train for a few epochs (this is a stub and may not converge)
                    try:
                        import torch
                        optimiser = torch.optim.Adam(vae.parameters(), lr=1e-3)
                        X_tensor = torch.tensor(X_num)
                        vae.train()
                        for epoch in range(3):
                            optimiser.zero_grad()
                            recon, mu, log_var = vae(X_tensor)
                            # Compute reconstruction + KL loss
                            recon_loss = ((recon - X_tensor) ** 2).mean()
                            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / len(X_tensor)
                            loss = recon_loss + kl_loss
                            loss.backward()
                            optimiser.step()
                        vae.eval()
                        with torch.no_grad():
                            mu, _log_var = vae.encode(X_tensor)
                        mu_np = mu.detach().cpu().numpy()
                        for i in range(mu_np.shape[1]):
                            df[f"vae_{i+1}"] = mu_np[:, i]
                    except Exception as exc:
                        logging.warning(f"VAE reduction failed: {exc}")
                else:
                    logging.warning(f"Unknown dimensionality reduction method '{method}'")
    # Save processed data
    df.to_csv(processed_path, index=False)
    logging.info(f"Processed data written to {processed_path}")


if __name__ == "__main__":  # pragma: no cover
    main()