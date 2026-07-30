# Breast Cancer Wisconsin Classification Workflow

A deliberately bounded, end-to-end machine-learning project built to consolidate the fundamental supervised-learning workflow in scikit-learn.

This repository is an educational project. It is **not** a clinical diagnostic system and must not be interpreted or used as medical advice, a screening tool or evidence of clinical validity.

---

## Project overview

The purpose of this project is to move from isolated Python, pandas and machine-learning exercises into one complete predictive-modelling workflow.

The project uses the Breast Cancer Wisconsin Diagnostic dataset supplied through scikit-learn. The task is binary classification: use the dataset's numerical features to classify samples as malignant or benign.

The first version intentionally uses a small and controlled scope:

- one built-in dataset;
- one simple baseline;
- one real model;
- one preprocessing pipeline;
- one train/test split;
- a small set of classification metrics;
- concise error and limitation analysis.

The objective is not to maximise model performance. The objective is to understand why each stage of a machine-learning workflow exists and how the stages connect.

---

## Learning objectives

By completing this project, I aim to understand and be able to explain:

- the difference between features and a target;
- why supervised classification requires labelled examples;
- how training and test data serve different purposes;
- why stratification is useful for classification splits;
- why a baseline model is necessary;
- why numerical feature scaling is included in the logistic-regression pipeline;
- how a scikit-learn `Pipeline` combines preprocessing and modelling;
- what `fit()`, `predict()` and `predict_proba()` represent;
- how accuracy, precision, recall and F1 answer different questions;
- how to interpret a confusion matrix;
- the meaning of false positives and false negatives;
- why apparently strong test performance does not establish clinical usefulness;
- how to communicate model limitations honestly.

---

## Dataset

The project uses scikit-learn's built-in Breast Cancer Wisconsin Diagnostic dataset.

At the time this project was designed, the scikit-learn dataset contained:

- 569 samples;
- 30 real-valued numerical features;
- two target classes: malignant and benign.

The dataset is loaded directly through `sklearn.datasets.load_breast_cancer`, so no separate CSV download is required.

Example loading pattern:

```python
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(
    return_X_y=True,
    as_frame=True,
)
```

The target labels and feature descriptions should be inspected from the dataset object rather than assumed.

Official reference: [scikit-learn `load_breast_cancer`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

---

## Problem definition

### Input

Thirty numerical measurements supplied for each sample.

### Target

A binary class label indicating malignant or benign.

### Task type

Supervised binary classification.

### Learning question

Can a simple logistic-regression pipeline outperform a trivial baseline on held-out examples from this educational dataset?

This is intentionally narrower than asking whether the model is medically useful.

---

## Repository structure

```text
breast-cancer-ml-workflow/
├── README.md
├── breast_cancer_workflow.ipynb
├── requirements.txt
└── .gitignore
```

### `README.md`

Explains the problem, learning objectives, workflow, evaluation approach, scope and limitations.

### `breast_cancer_workflow.ipynb`

Contains the complete analysis and modelling workflow.

### `requirements.txt`

Lists the Python packages required to run the notebook.

### `.gitignore`

Excludes local environments, notebook checkpoints, operating-system files and other generated artefacts.

---

## Version 1 workflow

The notebook should be organised into the following sections.

### 1. Imports and reproducibility

Import only the libraries required for the workflow.

Use a fixed `random_state` for reproducible splitting and model behaviour where applicable.

### 2. Load the dataset

Load the dataset as pandas objects.

Inspect:

- feature names;
- target names;
- shape of `X`;
- shape of `y`.

### 3. Basic data inspection

Keep exploratory analysis deliberately limited.

Include:

```python
X.shape
X.head()
X.info()
X.describe()
y.value_counts()
X.isna().sum()
```

Questions to answer:

- How many rows and features are present?
- Are all features numerical?
- Are values missing?
- Is the target balanced or imbalanced?
- Do feature scales differ substantially?

### 4. Define features and target

Explain:

- `X` contains the input features;
- `y` contains the class label to predict.

### 5. Create a stratified train/test split

Use an 80/20 split with stratification and a fixed random state.

Example:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)
```

Explain:

- the training set is used to fit the model;
- the test set is held back for final evaluation;
- stratification approximately preserves class proportions.

### 6. Establish a baseline

Use `DummyClassifier` as a trivial reference point.

A baseline answers:

> Does the real model provide useful predictive improvement over a simple strategy?

Evaluate the baseline with the same test data and core metrics used for logistic regression.

### 7. Build the logistic-regression pipeline

Create a pipeline containing:

```text
StandardScaler
    ->
LogisticRegression
```

The scaler and model must remain inside one scikit-learn `Pipeline`.

This ensures that scaling is learned from the training data during fitting and then applied consistently during prediction.

### 8. Fit the model

Use:

```python
pipeline.fit(X_train, y_train)
```

Explain that fitting estimates the model parameters using the training examples.

### 9. Generate predictions

Use:

```python
y_pred = pipeline.predict(X_test)
```

Optionally inspect predicted probabilities using `predict_proba`, but do not tune the classification threshold in Version 1.

### 10. Evaluate performance

Calculate:

- accuracy;
- precision;
- recall;
- F1 score;
- confusion matrix.

The positive class must be stated explicitly before interpreting precision and recall. Do not assume the library's numeric encoding without checking the target names.

### 11. Compare with the baseline

Use a concise table comparing baseline and logistic-regression performance.

Discuss:

- whether logistic regression improved meaningfully over the baseline;
- which metrics changed;
- whether one metric is more important than another for the stated educational question.

### 12. Interpret errors

Explain:

- true positives;
- true negatives;
- false positives;
- false negatives.

Because the dataset concerns malignant and benign samples, take care to map numeric labels to class names correctly before discussing errors.

Do not claim that the test errors predict real clinical consequences directly. The dataset and project are educational and do not reproduce a clinical deployment environment.

### 13. Limitations

The notebook must include a clear limitations section covering:

- small dataset size;
- a single historical dataset;
- possible sampling and measurement bias;
- no external validation;
- no assessment across hospitals, devices or populations;
- no prospective evaluation;
- no workflow or human-factors evaluation;
- no calibration or threshold analysis in Version 1;
- no evidence of clinical safety or effectiveness;
- possible instability from one train/test split;
- educational rather than diagnostic purpose.

### 14. Conclusion

Summarise:

- whether the pipeline outperformed the baseline;
- what was learned about the ML workflow;
- what remains deliberately excluded until the later checkpoint.

---

## Evaluation metrics

### Accuracy

The proportion of all test examples classified correctly.

Accuracy can be misleading when classes are highly imbalanced, so it should not be interpreted alone.

### Precision

Among examples predicted as the positive class, the proportion that were actually positive.

### Recall

Among all examples belonging to the positive class, the proportion identified by the model.

### F1 score

The harmonic mean of precision and recall.

F1 is useful when both false positives and false negatives matter and a single summary is required.

### Confusion matrix

A table showing the counts of correct and incorrect predictions for each class.

The class order and numeric label mapping must be printed and documented before interpreting the matrix.

---

## Scope restrictions for Version 1

Version 1 must not include:

- principal component analysis;
- feature selection;
- SHAP;
- permutation importance;
- GridSearchCV;
- RandomizedSearchCV;
- threshold optimisation;
- neural networks;
- decision trees or ensemble comparisons;
- deployment;
- Streamlit;
- an API;
- Docker;
- cloud infrastructure;
- a large exploratory-data-analysis section;
- claims of diagnostic performance or clinical utility.

These restrictions are intentional. The project is designed to consolidate the basic workflow rather than demonstrate every available technique.

---

## Planned checkpoint extension

This extension should only be completed after:

- Kaggle Intermediate Machine Learning;
- Kaggle Data Cleaning;
- Kaggle Data Visualization;
- scikit-learn MOOC Module 2: Selecting the Best Model;
- scikit-learn MOOC Module 3: Hyperparameter Tuning;
- scikit-learn MOOC Module 4: Linear Models.

At that point, add only:

1. cross-validation for the default pipeline;
2. one learning curve;
3. a small logistic-regression hyperparameter search;
4. comparison of default and tuned validation performance;
5. a short explanation of whether tuning materially improved the model.

Do not turn the checkpoint into a separate project or broad model benchmark.

---

## Running the project

### 1. Clone the repository

```bash
git clone <repository-url>
cd breast-cancer-ml-workflow
```

### 2. Create and activate a virtual environment

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Open the notebook

```bash
jupyter notebook breast_cancer_workflow.ipynb
```

Run the notebook from the first cell to the final cell.

---

## Reproducibility

The project should:

- use a fixed random state;
- keep preprocessing inside the pipeline;
- record package versions in `requirements.txt` once confirmed;
- run from top to bottom in a fresh environment;
- avoid hidden notebook state;
- clearly separate training and test data.

---

## Ethical and clinical statement

This repository is a machine-learning learning exercise.

It must not be used for:

- patient diagnosis;
- screening;
- treatment decisions;
- clinical triage;
- risk communication;
- medical-device development without appropriate validation and governance.

A clinically deployable system would require substantially more evidence, including representative data, external and prospective validation, calibration, subgroup analysis, workflow testing, human-factors assessment, monitoring, regulatory consideration and accountable clinical oversight.

---

## Project status

### Version 1

- [ ] Repository created
- [ ] Dataset loaded and inspected
- [ ] Train/test split completed
- [ ] Dummy baseline trained
- [ ] Logistic-regression pipeline trained
- [ ] Core metrics calculated
- [ ] Confusion matrix interpreted
- [ ] Baseline comparison completed
- [ ] Limitations documented
- [ ] Notebook runs from top to bottom
- [ ] README finalised

### Later checkpoint

- [ ] Cross-validation added
- [ ] Learning curve added
- [ ] Small hyperparameter search added
- [ ] Default and tuned performance compared
- [ ] Conclusions updated without expanding scope

---

## References

- [Scikit-learn Breast Cancer Wisconsin dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [Scikit-learn MOOC](https://inria.github.io/scikit-learn-mooc/)
- [Scikit-learn Pipeline documentation](https://scikit-learn.org/stable/modules/compose.html#pipeline)

