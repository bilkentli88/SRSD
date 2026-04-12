# On the Trade-off Between Stability and Responsiveness in Sequential Detection

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research-orange)
![Reproducibility](https://img.shields.io/badge/focus-reproducibility-informational)

## Overview

This repository contains the code accompanying the paper:

**On the Trade-off Between Stability and Responsiveness in Sequential Detection**

Sequential detection systems must balance two competing objectives: they should remain stable under noisy and uncertain observations, yet respond quickly once an event begins to emerge. This repository supports the study of that trade-off through the concept of **response inertia**, defined as policy-induced post-onset delay caused by stability-oriented operating logic under noisy and gradually accumulating evidence.

The repository reproduces the main experimental components of the paper, including:

- controlled synthetic experiments used to illustrate the stability-responsiveness frontier,
- NAB-based real-data experiments,
- response-inertia diagnostics used to compare conservative and adaptive policies, and
- supporting artifacts associated with both the main paper and the supplementary material.

## Key Features

- Reproducible synthetic experiments for studying the stability-responsiveness frontier
- NAB-based real-data evaluation across heterogeneous time-series streams
- Adaptive and static policy comparisons under common evidence signals
- Response-inertia diagnostics for identifying policy-induced post-onset delay
- Support for reproducing both main-paper and supplementary experimental artifacts

## Repository Structure

    repo/
    ├── README.md
    ├── LICENSE
    ├── requirements.txt
    └── src/
        ├── run_synthetic_experiment.py
        ├── run_nab_experiment.py
        ├── evidence.py
        ├── nab_loader.py
        ├── policies.py
        ├── evaluation.py
        ├── metrics.py
        └── response_inertia_analysis.py

## Requirements

The code has been organized as a lightweight research repository rather than a packaged software library. The main dependencies are listed in `requirements.txt`.

Typical dependencies include:

- numpy
- pandas
- matplotlib
- scipy

## Installation

Clone the repository and install the dependencies:

    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
    cd YOUR_REPOSITORY
    pip install -r requirements.txt

## Data

### Synthetic experiment

The synthetic experiment does not require any external dataset. Running the synthetic script will generate outputs directly.

### NAB real-data experiment

The real-data experiment uses the **Numenta Anomaly Benchmark (NAB)** dataset, which must be obtained separately and placed in a local directory.

The NAB root directory is expected to contain the standard dataset files, including `combined_labels.json` and the corresponding CSV series files.

A typical local layout is:

    data/
    └── NAB/
        ├── combined_labels.json
        ├── realKnownCause/
        ├── realTraffic/
        ├── realTweets/
        └── realAWSCloudwatch/

## Usage

### 1. Run the synthetic experiment

This script reproduces the synthetic study, including frontier summaries, ablation tables, and figures.

    python src/run_synthetic_experiment.py --output_dir outputs/synthetic

### 2. Run the NAB real-data experiment

This script reproduces the real-data evaluation, aggregate tables, response-inertia diagnostics, and case-study figures.

    python src/run_nab_experiment.py --nab_root data/NAB --output_dir outputs/nab

### Optional arguments

The scripts also support optional arguments for adjusting output locations and selected experiment settings. For example:

    python src/run_synthetic_experiment.py --output_dir outputs/synthetic --n_seeds 100
    python src/run_nab_experiment.py --nab_root data/NAB --output_dir outputs/nab --half_window_steps 24

## Output

### Synthetic experiment outputs

The synthetic script writes outputs under the specified synthetic output directory, typically with subfolders such as:

    outputs/synthetic/
    ├── raw_csv/
    ├── summary_csv/
    └── figures/

These outputs include frontier summaries, Wilcoxon comparison tables, ablation summaries, and representative figures.

### NAB experiment outputs

The NAB script writes outputs under the specified NAB output directory, including summary tables and figures such as:

    outputs/nab/
    ├── figures/
    ├── nab_per_series_summary.csv
    ├── nab_policy_aggregate.csv
    ├── nab_category_aggregate.csv
    ├── nab_response_inertia_windows.csv
    ├── nab_response_inertia_summary.csv
    ├── nab_response_inertia_by_category.csv
    ├── nab_response_inertia_lead_vs_nonlead.csv
    └── nab_response_inertia_threshold_sensitivity.csv

The repository is intended to reproduce both the main-paper outputs and supplementary experimental artifacts.

## Notes

- The top-level executable scripts are `run_synthetic_experiment.py` and `run_nab_experiment.py`.
- Helper modules such as `response_inertia_analysis.py` are imported by the experiment scripts and are not intended to be run as standalone entry points.
- Paths to datasets and output folders can be provided through command-line arguments rather than hard-coded local paths.

## Citation

If you use this repository, please cite the associated paper.

    @article{yourname2026responseinertia,
      title   = {On the Trade-off Between Stability and Responsiveness in Sequential Detection},
      author  = {Your Name},
      journal = {Under review},
      year    = {2026}
    }

## License

This repository is released under the MIT License. See the `LICENSE` file for details.
