Official GitHub repository for the paper: **"Lightweight Machine Learning Architecture for Real-Time DDoS Detection in Resource-Constrained IoT Edge Networks"**.

## Overview
This repository contains the complete implementation pipeline, dataset benchmarking tools, and machine learning scripts designed to evaluate lightweight DDoS anomaly detection on IoT edge gateways using the **CICIoT2023** dataset.

## Key Features
- **Feature Selection:** Gini-Importance Random Forest pipeline reducing 80+ telemetry parameters down to 15 core features.
- **Classifier:** LightGBM with leaf-wise growth algorithm for ultra-low latency inference.
- **Performance Highlights:**
  - Classification Accuracy: **99.12%**
  - Inference Latency: **0.04 ms/flow**
  - Average Power Footprint: **3.1 W** on ARM Cortex-A72 hardware setup.

## Requirements & Installation
Install the necessary Python modules using pip:

```bash
pip install pandas numpy scikit-learn lightgbm
