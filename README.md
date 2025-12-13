# rec-movielens

![License](https://img.shields.io/badge/License-MIT-blue)

---

### Introduction

A simple implementation of Recommender System based on the [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1M/).

---

### Quick Start

#### Training

```bash
python train.py
```

#### Evaluation

```bash
python evaluate.py
```

---

### Roadmap

**Step 0: ID-based Matching** _(Collaborative Filtering)_

> ![MF](https://img.shields.io/badge/MF-Done-success) ![GMF](https://img.shields.io/badge/GMF-Done-success) ![NCF](https://img.shields.io/badge/NCF-Done-success)

**Step 1: Feature Interaction** _(Ranking Classics)_

> ![FM](https://img.shields.io/badge/FM-Building-yellow) ![Wide&Deep](https://img.shields.io/badge/Wide&Deep-Planned-lightgrey) ![DeepFM](https://img.shields.io/badge/DeepFM-Planned-lightgrey)

**Step 2: Sequential Interest** _(User Behavior Modeling)_

> ![DIN](https://img.shields.io/badge/DIN-Planned-lightgrey) ![DIEN](https://img.shields.io/badge/DIEN-Planned-lightgrey) ![SASRec](https://img.shields.io/badge/SASRec-Planned-lightgrey)

**Step 3: Graph Connectivity** _(Graph Neural Networks)_

> ![GraphSAGE](https://img.shields.io/badge/GraphSAGE-Planned-lightgrey) ![LightGCN](https://img.shields.io/badge/LightGCN-Planned-lightgrey)

**Step 4: Multi-Task Learning** _(CTR & CVR Joint Training)_

> ![MMoE](https://img.shields.io/badge/MMoE-Optional-lightgrey) ![ESMM](https://img.shields.io/badge/ESMM-Optional-lightgrey)

---

### Requirements

- `Python` >= 3.10
- `PyTorch` >= 2.0.0 (CUDA recommended)
- Other dependencies: `pandas`, `numpy`, `tqdm`, `matplotlib`

---


