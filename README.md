# Diabetes Risk Estimator

An end-to-end machine-learning project: a model trained on about 100,000 patient records, served through an API, with a web page where anyone can enter eight health details and see whether the model flags a **higher** or **lower** diabetes risk.

**Live demo:** [diabetes-risk-predictor-web.vercel.app](https://diabetes-risk-predictor-web.vercel.app/)

> **Not medical advice.** This is a student project and a demo, not a diagnosis. Only a doctor and a lab test can tell you if you have diabetes.

---

## What it does

You enter age, gender, BMI, smoking history, hypertension, heart disease, HbA1c and blood glucose. The model places you in a higher-risk or lower-risk group. The page also shows a reference scale for HbA1c (typical, prediabetes, diabetes) with a marker that moves as you type.

The model is tuned to flag generously, so false alarms are more common than misses. It can be wrong in both directions.

## Results at a glance

| | Logistic regression (baseline) | Gradient boosting (final) |
| --- | --- | --- |
| Cross-validated ROC-AUC (5-fold) | 0.962 ± 0.0004 | **0.979 ± 0.0005** |
| Test ROC-AUC | 0.960 | **0.977** |
| Recall, diabetes class | 0.88 | **0.91** |
| Precision, diabetes class | 0.43 | **0.48** |
| Missed cases (of 1,696) | 210 | **156** |
| False alarms (of 17,530) | 1,998 | **1,692** |

Gradient boosting was better on every measure. Details are in the [Model](#model) section.

## Repository layout

```
backend/   FastAPI service (main.py), requirements.txt, trained model (model.joblib)
web/       index.html, the single-page frontend
src/       clean.py, model.py, model2.py, train.py, stats.py
```

The dataset itself is not stored here. Download it from Kaggle (link below) and place the CSV next to the scripts.

## Data

The [Diabetes prediction dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) from Kaggle: medical and demographic records with a diabetes label.

| Check | Result |
| --- | --- |
| Shape | 100,000 rows, 9 columns |
| Missing values | None |
| Duplicate rows | 3,854 (96,146 unique rows) |
| Diabetes = 1 | 8,500 rows (8.5%), so the classes are heavily imbalanced |
| Age range | 0.08 to 80 |
| BMI range | 10.01 to 95.69 |
| Blood glucose range | 80 to 300 |

## Data cleaning decisions

| Issue | Decision | Why |
| --- | --- | --- |
| 3,854 duplicate rows | Dropped | Identical rows could land in both the train and test sets and inflate scores |
| `smoking_history` has overlapping labels | Merged `not current` and `ever` into `former`; kept `never` and `current` | Those labels describe past smoking and overlap with `former` |
| `smoking_history = No Info` (about 36%) | Kept as its own `unknown` category | It is a hidden missing value, and dropping a third of the data would lose too much |
| 18 rows with gender `Other` | Dropped | Too few rows for the model to learn anything reliable about that group (see limitations) |
| BMI and age outliers | Checked, not removed | Extreme values are physically possible, and gradient boosting is not sensitive to them |
| Categorical columns | One-hot encoded (`pd.get_dummies`, `drop_first=True`) | Models need numeric input |

After cleaning, 96,128 rows remain.

## Model

- **Split:** 80% train and 20% test (19,226 test rows), stratified on the diabetes label, `random_state=42`.
- **Baseline:** logistic regression with feature scaling. Predicting "no diabetes" for everyone scores 91.2% accuracy, so accuracy is not a useful measure here.
- **Final model:** gradient boosting (`HistGradientBoostingClassifier`).
- **Imbalance handling:** `class_weight="balanced"` for both models, which makes a missed diabetic case cost more than a false alarm.
- **Validation:** stratified 5-fold cross-validation on the training set. The cross-validated ROC-AUC (0.979) and the test ROC-AUC (0.977) agree closely, so the result is stable and not an artefact of one split.

**Confusion matrix, gradient boosting, test set:**

| | Predicted no | Predicted yes |
| --- | --- | --- |
| **Actually no** | 15,838 | 1,692 |
| **Actually yes** | 156 | 1,540 |

Precision for the diabetes class is 0.48 and recall is 0.91 (F1 0.62). Precision is under 0.5 because the class weighting deliberately favours catching real cases. In a screening setting a missed case usually costs more than an extra test, but the trade-off is a choice, not a free win.

**What drives the predictions** (permutation importance, drop in ROC-AUC):

| Feature | Importance |
| --- | --- |
| HbA1c level | 0.244 |
| Blood glucose level | 0.126 |
| Age | 0.022 |
| BMI | 0.006 |
| All other features | below 0.002 |

HbA1c and blood glucose are close to how diabetes is diagnosed in practice, so the model is partly rediscovering the diagnostic criteria. That makes this task easier than predicting from lifestyle factors alone.

The deployed model is refitted on all the cleaned data. The results above come from the earlier held-out evaluation.

## Statistical analysis

`src/stats.py` tests the main relationships in the cleaned data. With about 96,000 rows nearly every difference is "significant", so each test reports an effect size next to its p-value.

| Question | Test | Result |
| --- | --- | --- |
| Does HbA1c differ between diabetic and non-diabetic people? | Mann-Whitney U | Median 6.6 vs 5.8 (p < 0.001). A random diabetic person has the higher HbA1c 84% of the time. |
| Does blood glucose differ? | Welch t-test | Mean 194 vs 133 mg/dL (p < 0.001), Cohen's d = 1.27, a large effect. |
| Is smoking history linked to diabetes? | Chi-square | Diabetes rate: former 14.0%, current 10.3%, never 9.7%, unknown 4.4% (p < 0.001). Cramér's V = 0.13, a small effect. |
| Does BMI differ by smoking group? | One-way ANOVA | F = 1,858 (p < 0.001), eta squared = 0.055. Smoking history explains about 5% of BMI variation. |

**Caveat:** the `unknown` smoking group is much younger on average (about 33, against 44 to 52 for the others), so its lower diabetes rate probably reflects age rather than smoking. This was not tested formally.

## How it is built

- **Model and API:** trained in Python with scikit-learn, served with FastAPI on Render.
- **Frontend:** one static page in `web/index.html` with no build step or framework, hosted on Vercel.
- **Design:** single-screen layout, a live HbA1c scale, and an always-visible disclaimer.
- **CORS:** the API only accepts browser requests from the deployed Vercel site.

## API

`POST /predict`

```json
{
  "age": 45,
  "gender": "Female",
  "hypertension": 0,
  "heart_disease": 0,
  "smoking_history": "never",
  "bmi": 27.5,
  "hba1c": 6.1,
  "glucose": 140
}
```

Response:

```json
{ "risk": "higher", "note": "Educational estimate only. Not a diagnosis." }
```

| Field | Accepted values |
| --- | --- |
| `age` | 1 to 80 |
| `gender` | Female or Male |
| `hypertension`, `heart_disease` | 0 or 1 |
| `smoking_history` | never, former, current, unknown |
| `bmi` | 10 to 60 |
| `hba1c` | 3.5 to 9.0 |
| `glucose` | 80 to 300 |

Values outside these ranges return a `422` error. `GET /health` returns `{"status": "ok"}`. The backend runs on a free plan, so the first request after a quiet period can take up to a minute.

The API returns only "higher" or "lower", not a probability. The models are trained with class weights, which push predicted probabilities upward, so showing them as percentages would be misleading.

## Reproduce the results

1. Download `diabetes_prediction_dataset.csv` from Kaggle and place it in the folder with the scripts.
2. Install the libraries: `pip install pandas scikit-learn scipy joblib`
3. Run in order:

```bash
python src/clean.py     # dedupe, merge smoking labels, encode, write diabetes_clean.csv
python src/model.py     # train/test split and logistic regression baseline
python src/model2.py    # cross-validation, gradient boosting, feature importance
python src/stats.py     # statistical tests
python src/train.py     # fit on all data and save model.joblib (copy it into backend/)
```

The scripts read their CSV files from the current folder, so run them from the folder that contains the data.

## Run locally

Backend:

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` to try the API.

Frontend:

```bash
cd web
python -m http.server 5500
```

Open `http://localhost:5500`. To use a local backend, change `API_URL` near the bottom of `index.html`, and add the page's address to `allow_origins` in `backend/main.py`.

## Deploy

- **Backend (Render):** web service with root directory `backend`, build command `pip install -r requirements.txt`, start command `uvicorn main:app --host 0.0.0.0 --port $PORT`. The `scikit-learn` version in `requirements.txt` must match the version used to save the model.
- **Frontend (Vercel):** import the repo, set **Framework Preset** to "Other" and **Root Directory** to `web`, leave the build command empty, then deploy. Add the Vercel URL to the backend's `allow_origins`.

## Limitations

- Trained on one public dataset, so results may not hold for every population. I could not verify how the data was collected.
- Uses only eight inputs and ignores family history, symptoms and other factors.
- HbA1c and glucose dominate the predictions, and both need a lab test, so the model adds little for someone who already has those results.
- Gender is limited to Female and Male because there were too few `Other` rows to learn from.
- Flags generously: expect more false alarms than misses. About half of the people flagged as higher risk are not diabetic in the test set.
- About a third of the smoking data is `unknown`, and that group is much younger than the rest, so smoking-related findings should be read carefully.
- Fairness was not assessed across gender or age groups.
- A learning project, not a screening tool.

## Author

**Aryan Khanna**

- [LinkedIn](https://www.linkedin.com/in/arnn/)
- [GitHub](https://github.com/Arnnnnnn)
