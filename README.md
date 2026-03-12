# Neural Networks from Scratch

> **Note:** This is a 2019 personal project built for learning purposes. It demonstrates a ground-up implementation of feedforward neural networks using only NumPy — no deep learning frameworks.

A from-scratch implementation of multilayer feedforward neural networks for both classification and regression, including an autoencoder variant, benchmarked across six public datasets.

## Overview

This project implements the following model types without any ML frameworks (NumPy only):

- **Multilayer feedforward network** — classification and regression
- **Autoencoder** — for unsupervised feature learning and reconstruction

All models are evaluated using **5-fold cross-validation** and compared against a simple baseline:
- Classification baseline: majority class predictor
- Regression baseline: mean response across training data

Model comparisons are made using a **statistical t-test** (comparison of two means) to assess whether differences in performance are significant.

## Results

Full results and discussion are in [Neural_Networks_Writeup.pdf](./Neural_Networks_Writeup.pdf). The writeup covers hyperparameter tuning, performance across datasets, and analysis of where the from-scratch network does and doesn't hold up.

## Datasets

Six public datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php), located in the `datasets/` folder. A worked example using the **Abalone** dataset is provided as a Jupyter notebook (`Abalone experiment.ipynb`).

## How to Run

### Requirements

```bash
pip install numpy pandas jupyter
```

### Run a model

Each model type has its own script:

| Script | Purpose |
|---|---|
| `MLPCls.py` | MLP classifier |
| `MLPReg.py` | MLP regressor |
| `autoencoder.py` | Autoencoder |
| `logistic_regression.py` | Logistic regression baseline |
| `linear_regression.py` | Linear regression baseline |
| `toolkit.py` | Shared utilities (data loading, cross-validation, evaluation) |

Run a script directly:

```bash
python MLPCls.py
```

Or explore the Abalone worked example:

```bash
jupyter notebook "Abalone experiment.ipynb"
```

## Tech Stack

Python · NumPy · pandas · Jupyter# neural-networks


Here I've implemented a (somewhat inefficient) multilayer feedforward network from scratch for both classification and regression, along with an auto-encoder to compare performance between different model types.

Five fold cross validation is utilized to configure hyperparameters and evaluate model performance. For the classification tasks, model performance is evaluated through a simple accuracy measure, and for regression tasks, model performance is evaluated through mean squared error. Performance between models is compared through a statistical *t*-test for comparison of two means. 

All models are also compared against a "baseline", simple majority model. For regression tasks, this means that we use the mean response across all training data. For classification tasks, it simply outputs the most common class. 

Full writeup and discussion of results is in the [Neural_Networks_Writeup.pdf](/Neural_Networks_Writeup.pdf) file.

The datasets used are in the datasets folder. These are public datasets available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).

One ipynb is provided as to how the results were calculated (through various Jupyter Notebooks), which is of the Abalone dataset.
