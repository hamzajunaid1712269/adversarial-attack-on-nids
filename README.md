This phase builds an end‑to‑end pipeline for Network Intrusion Detection (NIDS) on the UNSW‑NB15 dataset and evaluates how classical/MLP models behave under adversarial attacks. It covers data prep, model training, adversarial example generation, and robustness analysis with clear visuals.

## What we built
Dataset: Merge official train/test CSVs, drop id, split into normal vs anomaly.

## EDA & prep:
Missingness matrix, Pearson correlation heatmap, class‑wise count plots for key categoricals, and numeric distributions.

Feature engineering: Label‑encode proto/service/state; correlation‑based pruning (|r| ≥ 0.95); Min‑Max scaling.

## Models
Keras MLP (2 hidden layers, ReLU, sigmoid, Adam, EarlyStopping).

Random Forest (e.g., 300 trees, tuned depth/splitting).

Decision Tree (tuned max_depth, min_samples_split, min_samples_leaf).

Linear SVM via SGD (hinge loss, L2).

## Attack & robustness
JSMA (Jacobian‑based Saliency Map Attack) via CleverHans (TF v1‑compat) to craft adversarial samples from the clean test set and measure degradation.

Metrics & plots: Accuracy, F1, confusion matrices (clean vs adversarial), and a feature‑susceptibility bar chart (which features were perturbed most often).
