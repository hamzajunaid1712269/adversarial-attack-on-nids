This phase builds an end‑to‑end pipeline for Network Intrusion Detection (NIDS) on the UNSW‑NB15 dataset and evaluates how classical/MLP models behave under adversarial attacks. It covers data prep, model training, adversarial example generation, and robustness analysis with clear visuals.

## What we built
Dataset: Merge official train/test CSVs, drop id, split into normal vs anomaly.

## EDA & prep:
Missingness matrix, Pearson correlation heatmap, class‑wise count plots for key categoricals, and numeric distributions.

Feature engineering: Label‑encode proto/service/state; correlation‑based pruning (|r| ≥ 0.95); Min‑Max scaling.

## Models
Keras MLP 
Random Forest
Decision Tree 


## Attack & robustness
JSMA (Jacobian‑based Saliency Map Attack) via CleverHans (TF v1‑compat) to craft adversarial samples from the clean test set and measure degradation.

Metrics & plots: Accuracy, F1, confusion matrices (clean vs adversarial)
