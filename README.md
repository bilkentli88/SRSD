# On the Trade-off Between Stability and Responsiveness in Sequential Detection

This repository contains the code accompanying the paper:

**On the Trade-off Between Stability and Responsiveness in Sequential Detection**

## Overview

Sequential detection systems must remain stable under noisy and uncertain observations, yet respond quickly when an event begins to emerge. This repository supports the study of this trade-off through the concept of **response inertia**, defined as policy-induced post-onset delay caused by stability-oriented operating logic under noisy and gradually accumulating evidence.

The repository includes code for:

- synthetic experiments illustrating the stability–responsiveness frontier,
- NAB-based real-data evaluation,
- response-inertia analysis,
- supporting empirical results reported in the paper.

## Main idea

The paper studies how mechanisms designed to improve stability—such as conservative thresholds, switching penalties, hysteresis, and smoothing—can delay detection during critical periods. It introduces **temporally adaptive policies** that selectively relax conservativeness in delay-sensitive regimes.

## Repository structure

- `src/` — source code for the synthetic experiments, NAB evaluation, and analysis scripts

## Requirements

Install the required Python packages with:

```bash
pip install -r requirements.txt
