# On the Trade-off Between Stability and Responsiveness in Sequential Detection

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research-orange)
![Reproducibility](https://img.shields.io/badge/focus-reproducibility-informational)

## Overview

This repository contains the code accompanying the paper:

**On the Trade-off Between Stability and Responsiveness in Sequential Detection**

Sequential detection systems must remain stable under noisy and uncertain observations, yet respond quickly when an event begins to emerge. This repository supports the study of that trade-off through the concept of **response inertia**, defined as policy-induced post-onset delay caused by stability-oriented operating logic under noisy and gradually accumulating evidence.

The codebase supports the main empirical components of the paper, including controlled synthetic experiments and NAB-based real-data evaluation.

## Key Features

- Synthetic experiments illustrating the **stability--responsiveness frontier**
- NAB-based empirical evaluation on heterogeneous real streams
- Response-inertia analysis for comparing adaptive and conservative policies
- Support for studying how conservative thresholds and switching penalties affect detection delay
- Reproducible experimental pipeline aligned with the manuscript

## Installation

Clone the repository and install the required dependencies:

    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
    cd YOUR_REPOSITORY
    pip install -r requirements.txt

## Usage Example

A typical workflow is:

1. Run the synthetic experiments
2. Run the NAB evaluation
3. Run the response-inertia analysis
4. Regenerate summary outputs and figures as needed

Example commands:

    python src/run_synthetic_experiments.py
    python src/nab_master_evaluation_refined.py
    python src/response_inertia_analysis.py

Please ensure that the NAB dataset is downloaded separately and placed in the expected path used by the scripts.

## Citation

If you use this repository, please cite the associated paper.

    @article{yourname2026responseinertia,
      title   = {On the Trade-off Between Stability and Responsiveness in Sequential Detection},
      author  = {Your Name},
      journal = {Under review},
      year    = {2026}
    }
