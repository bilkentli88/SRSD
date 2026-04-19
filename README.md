# Response Inertia in Sequential Detection: A Policy-Level Analysis of Stability–Responsiveness Trade-offs

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research-orange)
![Reproducibility](https://img.shields.io/badge/focus-reproducibility-informational)

## Overview

This repository contains the code accompanying the paper:

**Response Inertia in Sequential Detection: A Policy-Level Analysis of Stability–Responsiveness Trade-offs**

Sequential detection systems must balance two competing objectives: they should remain stable under noisy and uncertain observations, yet respond quickly once an event begins to emerge. This repository supports the study of that trade-off through the concept of **response inertia**, defined as policy-induced post-onset delay caused by stability-oriented operating logic under noisy and gradually accumulating evidence.

The repository reproduces the main experimental components of the paper, including:

- controlled synthetic experiments used to illustrate the stability–responsiveness frontier,
- real-data experiments on **NAB**,
- real-data experiments on **Yahoo**,
- response-inertia diagnostics used to compare conservative and adaptive policies, and
- supporting artifacts associated with both the main paper and the supplementary material.

---

## Main Components

The repository includes code for:

- **Synthetic experiments**  
  Used to study the stability–responsiveness frontier, trigger overlap, and trigger placement under switching regularization.

- **NAB real-data experiments**  
  Used to evaluate static and adaptive policies on heterogeneous real-world anomaly streams.

- **Yahoo real-data experiments**  
  Used as a second real benchmark to test whether the same response-inertia pattern persists beyond NAB.

- **Response-inertia diagnostics**  
  Used to compare conservative and adaptive policies on the same evidence trajectories and identify windows in which earlier adaptive detection coincides with already elevated evidence.

---

## Repository Structure

```text
repo/
├── README.md
├── LICENSE
├── requirements.txt
└── src/
    ├── run_synthetic_experiment.py
    ├── run_nab_experiment.py
    ├── run_yahoo_experiment.py
    ├── evidence.py
    ├── nab_loader.py
    ├── yahoo_loader.py
    ├── policies.py
    ├── evaluation.py
    ├── metrics.py
    └── response_inertia_analysis.py
```

---

## Requirements

The code is organized as a lightweight research repository rather than a packaged software library. The main dependencies are listed in `requirements.txt`.

Typical dependencies include:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`

You may also need any additional libraries referenced in the experiment scripts.

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
pip install -r requirements.txt
```

---

## Data

### 1. Synthetic experiment

The synthetic experiment does **not** require any external dataset. Running the synthetic script will generate outputs directly.

### 2. NAB real-data experiment

The NAB experiment uses the **Numenta Anomaly Benchmark (NAB)** dataset, which must be obtained separately and placed in a local directory.

The NAB root directory is expected to contain the standard dataset files, including `combined_labels.json` and the corresponding CSV series files.

A typical local layout is:

```text
data/
└── NAB/
    ├── combined_labels.json
    ├── realKnownCause/
    ├── realTraffic/
    ├── realTweets/
    └── realAWSCloudwatch/
```

### 3. Yahoo real-data experiment

The Yahoo experiment expects locally downloaded Yahoo anomaly CSV files stored in a single folder.

Each CSV file should contain the columns:

```text
timestamp,value,is_anomaly
```

A typical local layout is:

```text
data/
└── Yahoo/
    ├── real_1.csv
    ├── real_2.csv
    ├── real_3.csv
    └── ...
```

In the current paper setting, the Yahoo experiment uses the **real subset** of the benchmark.

---

## Usage

### 1. Run the synthetic experiment

This script reproduces the synthetic study, including frontier summaries, trigger-placement analyses, tables, and figures.

```bash
python src/run_synthetic_experiment.py --output_dir outputs/synthetic
```

### 2. Run the NAB real-data experiment

This script reproduces the NAB evaluation, including aggregate tables, response-inertia diagnostics, and case-study outputs.

```bash
python src/run_nab_experiment.py --nab_root data/NAB --output_dir outputs/nab
```

### 3. Run the Yahoo real-data experiment

This script reproduces the Yahoo evaluation, including policy-level aggregates and response-inertia summaries.

```bash
python src/run_yahoo_experiment.py --yahoo_root data/Yahoo --output_dir outputs/yahoo
```

### Final Yahoo setting used in the paper

The Yahoo experiment script supports command-line tuning. The following command corresponds to the benchmark-level setting used for the final paper results:

```bash
python src/run_yahoo_experiment.py \
  --yahoo_root data/Yahoo \
  --output_dir outputs/yahoo \
  --conservative_threshold 3.8 \
  --conservative_enter_count 2 \
  --conservative_exit_count 2 \
  --adaptive_base_threshold 3.25 \
  --adaptive_exit_margin 0.35 \
  --adaptive_switch_penalty 0.18
```

---

## Output

### Synthetic experiment outputs

The synthetic script writes outputs under the specified synthetic output directory, typically with subfolders such as:

```text
outputs/synthetic/
├── raw_csv/
├── summary_csv/
└── figures/
```

These outputs include frontier summaries, Wilcoxon comparison tables, ablation summaries, and representative figures.

### NAB experiment outputs

The NAB script writes outputs under the specified NAB output directory, including summary tables and figures such as:

```text
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
```

### Yahoo experiment outputs

The Yahoo script writes outputs under the specified Yahoo output directory, including:

```text
outputs/yahoo/
├── yahoo_per_series_summary.csv
├── yahoo_policy_aggregate.csv
├── yahoo_category_aggregate.csv
├── yahoo_response_inertia_windows.csv
├── yahoo_response_inertia_summary.csv
├── yahoo_response_inertia_lead_vs_nonlead.csv
└── yahoo_response_inertia_threshold_sensitivity.csv
```

The repository is intended to reproduce both the main-paper outputs and supplementary experimental artifacts.

---

## Notes

- The main executable scripts are:
  - `run_synthetic_experiment.py`
  - `run_nab_experiment.py`
  - `run_yahoo_experiment.py`

- Helper modules such as `response_inertia_analysis.py` are imported by the experiment scripts and are not intended to be run as standalone entry points.

- Dataset paths and output folders are provided through command-line arguments rather than hard-coded local paths.

- The Yahoo timestamps are converted internally into equally spaced datetimes so that the same timestamp-based evaluation logic can be used consistently across benchmarks.

---



---

## License

This repository is released under the MIT License. See the `LICENSE` file for details.
