# ML Fundamentals

Six notebooks working through the core machine learning workflow, each
built around one question and ending with a decision.

## Notebooks

**01 — Regression** ([notebook](notebooks/01_regression.ipynb))
Linear, polynomial, Ridge, Lasso and ElasticNet on California Housing.
A degree-3 polynomial model collapsed to a validation RMSE of 1469, and
no amount of regularization fixed it. Swapping StandardScaler for
QuantileTransformer dropped it to 0.61 — a bigger gain than any model or
hyperparameter change achieved.

**02 — Classification** ([notebook](notebooks/02_classification.ipynb))
Five classifiers on Adult Income (24% positive class). A majority-class
baseline reaches 76% accuracy while missing every high earner. Includes
threshold tuning showing precision and recall traded from 0.61/0.79 to
0.84/0.45 on the same model.

**03 — Preprocessing and leakage** ([notebook](notebooks/03_preprocessing.ipynb))
Measures leakage rather than describing it. Imputation and scaling leaks
cost 0.00001 RMSE; feature-selection leakage cost 0.0096 — and made the
model look *better*, which is what makes it easy to ship by accident.

**04 — Model selection** ([notebook](notebooks/04_model_selection.ipynb))
Cross-validation, GridSearchCV and RandomizedSearchCV, plus the gap
between a search's reported best score and honest held-out performance.

**05 — Ensembles** ([notebook](notebooks/05_ensembles.ipynb))
Bagging, boosting, voting and stacking, scored against training cost.
Bagging 100 trees matched a single logistic regression. Boosting gained
0.022 AUC. Stacking took 12× longer than the winner for no gain.

**06 — Unsupervised** ([notebook](notebooks/06_unsupervised.ipynb))
PCA and clustering on Wine. K-Means recovered the true classes at
ARI 0.90 — while the silhouette score, the only metric available without
labels, called the same clustering mediocre at 0.285.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

All datasets download automatically. No data files required.