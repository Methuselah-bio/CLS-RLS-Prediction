# Project Roadmap

This document outlines planned milestones and features for the enhanced
Methuselah‑prediction pipeline.  The roadmap provides a high‑level
overview rather than strict deadlines, recognising that biological and
computational research evolves rapidly.

## v1.0 – Chronological Lifespan Integration

 - [x] Add support for real CLS datasets (e.g., Yeast Aging Database and GEO
  series) alongside the UCI demo.
 - [x] Integrate replicative lifespan (RLS) datasets (tables S1/S2 from the
  genome‑wide RLS screen) and associated expression data (e.g., GEO
  accession GSE37241) into the pipeline.
- [x] Implement gene‑expression z‑scoring, pathway enrichment and network
  feature extraction.
- [x] Introduce balanced random forests and cost‑sensitive learning.
- [x] Extend interpretability scripts to support grouped SHAP and post‑hoc
  literature queries.
- [x] Provide Snakemake workflow and configuration knobs for reproducible
  pipelines.
- [ ] Validate CLS predictions against published findings; collaborate
  with aging laboratories for benchmarking.

## v1.1 – Survival Analysis and Graph Models

- [ ] Integrate survival analysis models (Cox proportional hazards and
  accelerated failure time) for time‑to‑event CLS data.
- [x] Provide a working graph neural network implementation leveraging
  protein–protein interaction networks.
- [ ] Curate high‑confidence yeast interaction networks (e.g., from
  STRING) and incorporate them into the feature engineering.
- [ ] Benchmark models on larger datasets and report runtime/memory
  profiles.

## v1.2 – AutoML and Hyperparameter Tuning

- [x] Add Optuna-based Bayesian optimisation for hyperparameter tuning.
- [ ] Integrate FLAML or other AutoML frameworks for automated model
  selection.
- [ ] Support nested cross‑validation for unbiased evaluation of tuned
  models.

## v1.3 – Documentation and Community

- [ ] Convert notebooks and scripts into a Jupyter Book for
  interactive tutorials.
- [ ] Publish a preprint describing the pipeline and solicit feedback.
- [ ] Engage with the BioStars community to gather use cases and
  improvements.
- [ ] Expand tests to cover edge cases such as missing values,
  multicollinearity and extreme class imbalance.
- [ ] Strengthen ethical discussion regarding translational relevance
  and limitations of yeast models.