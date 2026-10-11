# Diabetes risk estimation prototype

A local Flask application that compares Logistic Regression, Random Forest, K-Nearest Neighbors, and a small artificial neural network (MLP). It reports holdout accuracy, precision, recall, and F1, then uses the model with the highest holdout F1 for a probability estimate. Permutation importance summarizes global model reliance; per-patient sensitivity compares the estimate with one feature at a time replaced by its training median.

## Run locally (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
python app.py
```

Open http://127.0.0.1:5000. Training downloads the Pima Indians Diabetes dataset from the UCI repository, so the first training run requires internet access. The model artifact is written to `artifacts/diabetes_model.joblib` and is excluded from version control.

Run the focused tests with `python -m pytest -q`.

## Data and method

The training script fetches the 768-row Pima table from the public CSV mirror at `jbrownlee/Datasets` and validates its eight-feature schema and binary target before training. Zero values for glucose, blood pressure, skin thickness, insulin, and BMI are treated as missing, then median-imputed inside each model pipeline. Data is split once with stratification (80% train, 20% holdout; seed 42). The model is selected by mean F1 across five folds within the training partition; the untouched holdout reports accuracy, precision, recall, and F1 for all four algorithms and supplies permutation importance.

The web form accepts male or female as patient context and allows blank insulin, skin thickness, and diabetes-pedigree measurements, which are imputed from the training partition. Gender is validated and returned with the prediction but is not a model feature. The selected model's output is a positive-class probability. The default 0.50 threshold is an engineering default, not a clinically selected operating point. Sensitivity values are model comparisons, not causal effects.

## Important limitations

This is an educational research prototype, not a medical device, diagnostic system, or substitute for professional care. The source dataset's original cohort was women of Pima heritage. Its size, age, population, and collection context do not support generalizing the model to other populations or clinical settings. The model has not undergone external validation, prospective evaluation, or probability calibration. Do not use it to make care decisions.