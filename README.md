# Response Inertia in Sequential Detection: A Policy-Level Framework for Selective Temporal Adaptation

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research-orange)
![Reproducibility](https://img.shields.io/badge/focus-reproducibility-informational)

## Overview

This repository contains the code accompanying the paper:

**Response Inertia in Sequential Detection: A Policy-Level Framework for Selective Temporal Adaptation**

Sequential detection systems must balance two competing objectives. They should remain stable under noisy and uncertain observations, but they should also respond quickly once an event begins to emerge. This repository supports the study of this stability-responsiveness trade-off through the concept of **response inertia**, defined as policy-induced post-onset delay caused by stability-oriented operating logic under noisy and gradually accumulating evidence.

The repository reproduces the main experimental components of the paper:

- controlled synthetic experiments illustrating the stability-responsiveness frontier;
- real-data policy-level experiments on the **Numenta Anomaly Benchmark (NAB)**;
- real-data policy-level experiments on the **Yahoo Webscope anomaly benchmark**;
- response-inertia diagnostics comparing conservative and adaptive policies on the same evidence trajectories;
- supporting outputs for the main paper and supplementary material.

The code is intended for research reproducibility. It is not a packaged software library.

---

## Main Idea

The experiments compare different **operating policies** on the same causal evidence stream. This is important because the paper studies whether delay comes from weak evidence or from the detector's own conservative policy logic.

The repository therefore focuses on policy-level quantities such as:

- event-window hit rate;
- detection delay;
- false-alarm onsets;
- switching count;
- adaptive-lead windows;
- elevated-evidence diagnostics.

The goal is not to maximize a single benchmark score, but to decompose the stability-responsiveness trade-off.

---

## Repository Structure

A typical repository layout is:

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

The main executable scripts are:

```text
src/run_synthetic_experiment.py
src/run_nab_experiment.py
src/run_yahoo_experiment.py
```

Helper modules such as `evidence.py`, `policies.py`, `metrics.py`, and `response_inertia_analysis.py` are imported by the experiment scripts and are not intended to be run as standalone entry points.

---

## Requirements

The experiments require Python 3.10 or later.

The main dependencies are listed in `requirements.txt`. Typical dependencies include:

```text
numpy
pandas
matplotlib
scipy
```

The experiments are lightweight and can be run on a standard laptop. No GPU is required.

---

## Installation

### Option 1: Using `venv`

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Create and activate a virtual environment.

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Option 2: Using Conda

```bash
conda create -n response-inertia python=3.10
conda activate response-inertia
pip install -r requirements.txt
```

---

## Data Setup

### 1. Synthetic Experiment

The synthetic experiment does **not** require any external dataset. Running the synthetic script generates the data internally and writes the outputs to the selected output directory.

---

### 2. NAB Real-Data Experiment

The NAB experiment uses the **Numenta Anomaly Benchmark (NAB)** dataset. The dataset must be downloaded separately.

Place the NAB files in a local folder such as:

```text
data/
└── NAB/
    ├── combined_labels.json
    ├── realKnownCause/
    ├── realTraffic/
    ├── realTweets/
    └── realAWSCloudwatch/
```

The important point is that the folder passed to `--nab_root` must contain `combined_labels.json` and the NAB CSV subfolders.

Example:

```bash
python src/run_nab_experiment.py --nab_root data/NAB --output_dir outputs/nab
```

---

### 3. Yahoo Real-Data Experiment

The Yahoo experiment expects locally downloaded Yahoo anomaly CSV files stored in a single folder.

Each CSV file should contain the following columns:

```text
timestamp,value,is_anomaly
```

A typical layout is:

```text
data/
└── Yahoo/
    ├── real_1.csv
    ├── real_2.csv
    ├── real_3.csv
    └── ...
```

In the paper setting, the Yahoo experiment uses the **real subset** of the benchmark.

Example:

```bash
python src/run_yahoo_experiment.py --yahoo_root data/Yahoo --output_dir outputs/yahoo
```

---

## Running the Experiments

### 1. Synthetic Experiment

Run:

```bash
python src/run_synthetic_experiment.py --output_dir outputs/synthetic
```

This reproduces the controlled synthetic study, including:

- the stability-responsiveness frontier;
- switching-regularization comparisons;
- trigger-overlap analysis;
- trigger-placement analysis;
- representative figures;
- summary CSV files.

Typical output directory:

```text
outputs/synthetic/
├── raw_csv/
├── summary_csv/
└── figures/
```

---

### 2. NAB Experiment

Run:

```bash
python src/run_nab_experiment.py \
  --nab_root data/NAB \
  --output_dir outputs/nab
```

This reproduces the NAB policy-level evaluation, including:

- static aggressive policy;
- static mid policy;
- static conservative policy;
- adaptive policy;
- adaptive policy without switching penalty;
- response-inertia diagnostics.

Typical output files include:

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

---

### 3. Yahoo Experiment

Run:

```bash
python src/run_yahoo_experiment.py \
  --yahoo_root data/Yahoo \
  --output_dir outputs/yahoo
```

This reproduces the Yahoo policy-level evaluation and response-inertia diagnostics.

Typical output files include:

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

---

## Final Yahoo Setting Used in the Paper

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

If these values are changed, the output values may differ from those reported in the paper.

---

## What Each Experiment Produces

### Synthetic Experiment

The synthetic experiment produces the tables and figures used to analyze the stability-responsiveness frontier and trigger placement. It is the easiest experiment to run because it does not require external data.

Recommended first command for new users:

```bash
python src/run_synthetic_experiment.py --output_dir outputs/synthetic
```

---

### NAB Experiment

The NAB experiment produces aggregate policy-level tables and response-inertia diagnostics. It requires the NAB dataset to be available locally.

Recommended command:

```bash
python src/run_nab_experiment.py --nab_root data/NAB --output_dir outputs/nab
```

---

### Yahoo Experiment

The Yahoo experiment produces aggregate policy-level tables and response-inertia diagnostics on the real Yahoo anomaly benchmark subset.

Recommended command:

```bash
python src/run_yahoo_experiment.py --yahoo_root data/Yahoo --output_dir outputs/yahoo
```

For exact reproduction of the final paper setting, use the full Yahoo command given above.

---









## Expected Workflow for a New User

A new user can follow this minimal workflow:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/run_synthetic_experiment.py --output_dir outputs/synthetic
```

After confirming that the synthetic experiment runs successfully, continue with NAB and Yahoo after downloading the external datasets.

---

## License

This repository is released under the MIT License. See the `LICENSE` file for details.

---

## Citation

If you use this code, please cite the accompanying paper:

```text
Altay, A. T. Response Inertia in Sequential Detection: A Policy-Level Framework for Selective Temporal Adaptation. Knowledge-Based Systems, revised manuscript.
```
