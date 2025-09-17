"""Snakemake workflow orchestrating the Methuselah‑prediction pipeline.

This workflow defines three rules:

* ``prepare`` – convert raw data into a processed CSV.  See ``prepare_data.py``.
* ``train`` – train the classifier specified in the YAML configuration.  See ``train.py``.
* ``interpret`` – compute SHAP values and save a summary plot.  See ``interpret_shap.py``.

Paths are read from ``configs/base.yaml`` via the ``config`` dictionary.  Run
``snakemake -c N`` to execute the pipeline using N cores.
"""

import os

configfile: "configs/base.yaml"

processed_csv = config["paths"]["processed"]
results_dir = config["paths"]["results"]
metrics_json = os.path.join(results_dir, "metrics.json")
model_file = os.path.join(results_dir, "model.joblib")
shap_png = os.path.join(results_dir, "shap_summary.png")

rule all:
    input:
        processed_csv,
        metrics_json,
        shap_png

rule prepare:
    output:
        processed_csv
    shell:
        "python src/prepare_data.py --config {configfile}"

rule train:
    input:
        processed_csv
    output:
        metrics_json,
        model_file
    shell:
        "python src/train.py --config {configfile}"

rule interpret:
    input:
        model_file,
        processed_csv
    output:
        shap_png
    shell:
        "python src/interpret_shap.py --config {configfile}"