# Beyond Centralized Diagnosis: A Privacy-Preserving Federated Learning Framework for Robust Pneumonia Detection

A federated learning framework for pneumonia detection from chest X-ray images, designed to investigate **data heterogeneity, class imbalance, and privacy-preserving collaborative learning** without requiring centralized access to client-level medical images.

## Overview

Centralized medical AI requires collecting patient data in a common location, which can create privacy, governance, and data-sharing challenges. This project investigates **Federated Learning (FL)** as an alternative, where multiple simulated clients collaboratively train a global model while keeping their local image data decentralized.

The study compares centralized training with **IID and Non-IID Federated Averaging (FedAvg)** and evaluates how client-level data heterogeneity affects pneumonia detection performance.

## Research Objectives

* Investigate privacy-preserving collaborative learning for pneumonia detection.
* Evaluate the impact of **IID vs. Non-IID data distributions** on federated performance.
* Address class imbalance through class-aware training strategies.
* Compare centralized and federated learning performance using clinically relevant metrics.
* Analyze model errors using confusion matrices and false-positive/false-negative analysis.

## Dataset

The experiments use the **Kermany chest X-ray dataset** containing NORMAL and PNEUMONIA chest X-ray images.

> The raw X-ray images are intentionally **not included in this repository**. Users should obtain the dataset separately and configure the dataset path locally.

## Federated Learning Setup

Four simulated clients were used to investigate collaborative training under different data distributions.

### Experimental Settings

| Setting        | Description                                          |
| -------------- | ---------------------------------------------------- |
| Centralized    | Training using centrally available data              |
| IID FedAvg     | Clients receive approximately IID data distributions |
| Non-IID FedAvg | Clients contain heterogeneous class distributions    |
| Aggregation    | Federated Averaging (FedAvg)                         |

## Final Non-IID FedAvg Results

The final Non-IID FedAvg experiment achieved:

| Metric             |      Score |
| ------------------ | ---------: |
| Accuracy           | **91.54%** |
| Precision          | **90.66%** |
| Recall             | **93.01%** |
| Specificity        | **83.63%** |
| F1 Score           | **91.82%** |
| ROC-AUC            | **95.04%** |
| PR-AUC             | **96.82%** |
| Decision Threshold |   **0.20** |

The results demonstrate strong pneumonia sensitivity while maintaining substantially improved specificity compared with the corresponding IID federated setting.

## Model Comparison

The repository contains the final comparison results for:

* Centralized weighted training
* IID FedAvg
* Non-IID FedAvg

The comparison includes accuracy, precision, recall, specificity, F1 score, ROC-AUC, and confusion-matrix statistics.

## Visual Analysis

The repository includes:

* Federated training accuracy and loss curves
* Centralized, IID, and Non-IID confusion matrices
* Model performance comparison
* False-positive / false-negative comparison
* Non-IID vs. IID improvement analysis

## Repository Structure

```text
Pneumonia-Federated-Learning/
│
├── README.md
├── Pneumonia_GitHub.ipynb
│
├── FINAL_MODEL_COMPARISON.csv
├── error_analysis.csv
├── noniid_final_robustness_result.csv
├── noniid_robustness_results.csv
│
└── FINAL/
    └── figures/
        ├── confusion_matrix_centralized.png
        ├── confusion_matrix_iid.png
        ├── confusion_matrix_noniid.png
        ├── federated_accuracy.png
        ├── federated_loss.png
        ├── fp_fn_comparison.png
        ├── model_performance_comparison.png
        └── noniid_vs_iid_improvement.png
```

## Reproducibility

1. Obtain the Kermany chest X-ray dataset.
2. Organize the dataset into the required `train` and `test` directories.
3. Configure the dataset path in the notebook.
4. Install the required Python dependencies.
5. Run `Pneumonia_GitHub.ipynb`.

Raw datasets and trained `.keras` model artifacts are excluded from this repository to keep the repository lightweight.

## Key Research Contribution

This work focuses on the intersection of **medical AI, federated learning, data heterogeneity, and class imbalance**. Rather than evaluating pneumonia detection only under centralized assumptions, the study examines how decentralized and heterogeneous client distributions influence model behavior and diagnostic performance.

The analysis provides a practical foundation for extending pneumonia detection toward more realistic multi-institutional federated environments.

## Technologies

**Python · TensorFlow · Keras · NumPy · Pandas · Matplotlib · Seaborn · Scikit-learn · Federated Learning · FedAvg**

## Future Work

* Evaluate the framework across larger numbers of clients.
* Investigate additional non-IID partitioning strategies.
* Explore personalized federated learning.
* Evaluate communication efficiency and client participation.
* Investigate stronger privacy mechanisms such as differential privacy and secure aggregation.
* Validate the framework on external clinical datasets.

## Author

**Aleeba Ahmad**
Computer Engineering, UET Lahore

[GitHub](https://github.com/Aleebaahmad)
