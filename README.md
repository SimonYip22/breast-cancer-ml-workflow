# Breast Cancer Wisconsin Classification

An end-to-end binary classification project using scikit-learn and the Breast Cancer Wisconsin Diagnostic dataset.

The project compares a simple baseline with a logistic regression pipeline and evaluates performance using standard classification metrics and error analysis.

## Project overview

The objective is to predict whether a tumour sample is classified as malignant or benign from numerical measurements derived from digitised images of breast mass cell nuclei.

The workflow covers:

- dataset inspection;
- feature and target definition;
- stratified train/test splitting;
- baseline modelling;
- preprocessing with `StandardScaler`;
- logistic regression using a scikit-learn `Pipeline`;
- classification metrics;
- confusion-matrix interpretation;
- model limitations.

## Dataset

The project uses the Breast Cancer Wisconsin Diagnostic dataset included with scikit-learn.

The dataset contains:

- 569 samples;
- 30 numerical features;
- two target classes: malignant and benign.

The data is loaded directly with `sklearn.datasets.load_breast_cancer`, so no external download is required.

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
```

Dataset documentation: [scikit-learn Breast Cancer Wisconsin dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

## Machine-learning task

- **Task:** supervised binary classification
- **Features:** numerical measurements for each sample
- **Target:** malignant or benign classification
- **Baseline:** `DummyClassifier`
- **Primary model:** logistic regression
- **Preprocessing:** standardisation with `StandardScaler`

The preprocessing and classifier are combined in a scikit-learn `Pipeline` so that scaling is fitted only on the training data and applied consistently during evaluation.

## Workflow

The notebook follows this sequence:

1. Load and inspect the dataset.
2. Confirm feature types, target labels and class distribution.
3. Split the data into stratified training and test sets.
4. Train and evaluate a `DummyClassifier` baseline.
5. Build a `StandardScaler` and `LogisticRegression` pipeline.
6. Train the pipeline on the training set.
7. Generate predictions for the held-out test set.
8. Compare the model against the baseline.
9. Evaluate errors and summarise limitations.

## Evaluation

Model performance is assessed using:

- accuracy;
- precision;
- recall;
- F1 score;
- confusion matrix.

The target-label mapping is checked explicitly before interpreting the positive class, false positives and false negatives.

## Repository structure

```text
breast-cancer-ml-workflow/
├── README.md
├── PROJECT_PLAN.md
├── breast_cancer_workflow.ipynb
├── requirements.txt
└── .gitignore
```

- `README.md` — project overview, methodology, setup and results summary.
- `PROJECT_PLAN.md` — implementation plan, learning objectives, scope and completion criteria.
- `breast_cancer_workflow.ipynb` — complete analysis and modelling workflow.
- `requirements.txt` — Python dependencies.
- `.gitignore` — excluded local and generated files.

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd breast-cancer-ml-workflow
```

Create and activate a virtual environment.

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

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Open the notebook:

```bash
jupyter notebook breast_cancer_workflow.ipynb
```

Run the notebook from top to bottom to reproduce the analysis and model evaluation.

## Results

The completed project will report:

- baseline performance;
- logistic regression performance;
- the difference between baseline and model results;
- confusion-matrix findings;
- representative error patterns;
- the main modelling limitations.

The final values will be added after the notebook has been completed and validated.

## Limitations

This project uses a small, historical dataset supplied for machine-learning education and benchmarking. Performance on its held-out test split does not establish clinical validity or generalisation across hospitals, populations, acquisition systems or real diagnostic workflows.

The model is not intended for diagnosis, screening, treatment decisions or other clinical use.

## References

- [Scikit-learn Breast Cancer Wisconsin dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [Scikit-learn documentation](https://scikit-learn.org/stable/)
- [Scikit-learn MOOC](https://inria.github.io/scikit-learn-mooc/)
