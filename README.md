# rec-movielens

![License](https://img.shields.io/badge/License-MIT-blue)

---

### Introduction

A simple implementation of Recommender System based on the [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1M/).

---

### Model Zoo

![MF](https://img.shields.io/badge/MF-Done-success)
![GMF](https://img.shields.io/badge/GMF-Done-success)
![NCF](https://img.shields.io/badge/NCF-Buliding-yellow)
![DeepFM](https://img.shields.io/badge/DeepFM-Planned-lightgrey)
![DIN](https://img.shields.io/badge/DIN-Planned-lightgrey)
![SASRec](https://img.shields.io/badge/SASRec-Planned-lightgrey)
![LightGCN](https://img.shields.io/badge/LightGCN-Planned-lightgrey)
![LLM4Rec](https://img.shields.io/badge/LLM4Rec-Planned-lightgrey)

---

### Requirements

- `Python` >= 3.10
- `PyTorch` >= 2.0.0 (CUDA recommended)
- Other dependencies: `pandas`, `numpy`, `tqdm`, `matplotlib`

---

### Quick Start

#### Basic Usage

```bash
python train.py
```
#### Hyperparameter Tuning

```bash
python train.py --num_features 32 --num_epochs 50 --lr 0.005
```

#### Experiment Management

```bash
python train.py --model_type 'ncf' --seed 24 --save_dir ./log/exp_ncf_24
```

---

### Project Structure

```text
.        
├── data/                  # Data directory
├── log/                   # Log and saved models
├── src/                   # Source code package
│   ├── __init__.py        # Package initialization
│   ├── dataset.py         # Dataset loading logic
│   ├── model.py           # Model architecture definition
│   └── utils.py           # Argument parsing & utility functions
├── .gitignore             # Git ignore rules
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
└── train.py               # Main training script
```
