# Emotion-Guided Cross-Attention for Multi-Task Abusive Language Detection

A multi-task learning framework that jointly detects fine-grained emotional abuse behaviors and binary relationship abuse in text.

## Overview

This project addresses the challenge of detecting subtle emotional abuse in online text, particularly in relationship contexts. Using a shared transformer encoder with a task-guided cross-attention mechanism, the model learns emotional behavior signals (Task 1) and uses them to improve binary abuse detection (Task 2).

## Tasks

- **Task 1 (Multi-label):** Detects fine-grained emotional behaviors using the Unhealthy Comments Corpus (UCC) — 5 emotional labels
- **Task 2 (Binary):** Classifies abusive vs. non-abusive relationship text using the Reddit relationship abuse dataset

## Model Architecture

- Shared transformer encoder (e.g., BERT-based)
- Masked mean pooling for fixed-size representation
- Task-guided cross-attention: Task 1 representations act as queries over the encoder output
- Separate classification heads per task

## Key Features

- **Focal Loss** for Task 1 to handle severe class imbalance
- **Uncertainty-based loss weighting** (Kendall et al.) for joint training
- **Dynamic threshold tuning** per label on validation set to maximize macro-F1

## Results

| Model Variant | F1 Score | Accuracy |
|---|---|---|
| Full Model (with T1 guidance) | 0.9052 | 0.9087 |
| Without T1 Guidance (ablation) | 0.8727 | 0.8838 |

Task 1 (multi-label emotional behaviors) achieves ~0.54 macro-F1, a substantial improvement over near-zero baselines in prior work.

## Datasets

- [Unhealthy Comments Corpus (UCC)](https://arxiv.org/abs/2010.07410)
- Reddit relationship abuse dataset (#WhyIStayed / #WhyILeft)

