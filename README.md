# Predicting Student Academic Outcomes

Supporting implementation for the Machine Learning and Predictive Analytics assessment.

## Aim

To investigate whether student outcomes (Dropout, Enrolled or Graduate) can be predicted using information available by the end of the first semester.

## Models

- Logistic Regression — baseline and selected final model
- Random Forest — comparison model
- Artificial Neural Network — exploratory extension

## Dataset

Predict Students' Dropout and Academic Success  
UCI Machine Learning Repository, Dataset ID 697.

The six second-semester curricular-unit variables are removed before modelling so that the prediction point remains at the end of the first semester.

## Code structure

The code has been condensed into three scripts to reduce repetition while keeping the analysis stages clear:

1. `src/explore_data.py` — dataset checks, target distribution and bivariate analysis
2. `src/classification_models.py` — preprocessing, 5-fold stratified cross-validation, Logistic Regression and Random Forest comparison, grid search, final test evaluation and coefficient interpretation
3. `src/neural_network.py` — exploratory ANN using a validation subset of the development data

The classical modelling script follows the module workflow: an 80/20 stratified train-test split, cross-validation on the training set only, preprocessing inside scikit-learn pipelines, grid search for hyperparameter tuning, and one final evaluation on the held-out test set.

## Run the classical analysis

Create and activate the base environment:

```bash
conda env create -f environment.yml
conda activate student-ml
```

Run:

```bash
python scripts/explore_data.py
python scripts/classification_models.py
```

## Run the ANN extension

Create and activate the ANN environment:

```bash
conda env create -f environment-ann.yml
conda activate student-ml-ann
```

Then run:

```bash
python scripts/neural_network.py
```

## Outputs

Generated tables and figures are saved in `outputs/` and `outputs/figures/`.

The main evaluation measures are accuracy, macro precision, macro recall, macro F1, class-specific performance and confusion matrices.
