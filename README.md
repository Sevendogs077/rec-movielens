# rec-movielens-100k

Based on [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/)

---
### Introduction

This is my simple test of Recommender System. 

---

### Requirements

```text
numpy
pandas
torch>=2.0.0
```
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
python train.py --seed 24  --train_ratio 0.9 --save_dir ./saved_models/exp_24_ratio_0.9
```

---

### Project Structure

```text
.        
├── data/                  # Data directory (contains u.data)
├── saved_models/          # Model checkpoints
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
