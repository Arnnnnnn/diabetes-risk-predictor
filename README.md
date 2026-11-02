# Diabetes Risk Estimator

An end-to-end machine-learning project: a model trained on about 100,000 patient records, served through an API, with a web page where anyone can enter eight health details and see whether the model flags a **higher** or **lower** diabetes risk.

**Live demo:** [diabetes-risk-predictor-web.vercel.app](https://diabetes-risk-predictor-web.vercel.app/)

> **Not medical advice.** This is a student project and a demo, not a diagnosis. Only a doctor and a lab test can tell you if you have diabetes.

---

## What it does

You enter age, gender, BMI, smoking history, hypertension, heart disease, HbA1c and blood glucose. The model places you in a higher-risk or lower-risk group. The page also shows a reference scale for HbA1c (typical, prediabetes, diabetes) with a marker that moves as you type.

The model is tuned to flag generously, so false alarms are more common than misses. It can be wrong in both directions.

## Data

The [Diabetes prediction dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) from Kaggle: medical and demographic records with a diabetes label.

| Check | Result |
| --- | --- |
| Shape | 100,000 rows, 9 columns |
| Missing values | None |
| Duplicate rows | 3,854 (96,146 unique rows) |
| Diabetes = 1 | 8,500 rows (8.5%), so the classes are heavily imbalanced |
| Age range | 0.08 to 80 |
| Blood glucose range | 80 to 300 |

## Data cleaning decisions

| Issue | Decision | Why |
| --- | --- | --- |
| 3,854 duplicate rows | Dropped | Identical rows would leak between the train and test sets and inflate scores |
| `smoking_history` has overlapping labels | Merged `not current` and `ever` into `former`; kept `never` and `current` | Those labels describe past smoking and overlap with `former` |
| `smoking_history = No Info` (about 36%) | Kept as its own `unknown` category | It is a hidden missing value, and dropping a third of the data would lose too much |
| 18 rows with gender `Other` | Dropped | Too few rows for the model to learn anything reliable about that group (listed as a limitation) |
| BMI and age outliers | Checked, not removed | Looked at min and max values to see whether they were errors or just unusual people |
| Categorical columns | One-hot encoded (`pd.get_dummies`, `drop_first=True`) | Models need numeric input |

## Model

<!-- TODO: fill in what you actually trained and measured -->
- **Baseline:** logistic regression
- **Final model:** <!-- e.g. gradient boosting -->
- **Imbalance handling:** <!-- e.g. class weights or a lowered decision threshold -->
- **Results on the held-out test set:** <!-- accuracy, precision, recall, ROC-AUC -->

Because only 8.5% of records are positive, accuracy alone is misleading, so recall on the diabetes class matters most here.

## How it is built

- **Model and API:** trained in Python, served as an API on Render
- **Frontend:** one static page in `web/index.html` with no build step and no dependencies, hosted on Vercel
- **Design:** single-screen layout, a live HbA1c scale, and an always-visible disclaimer

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
{ "risk": "higher" }
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

Values outside these ranges return a `422` error. The backend runs on a free plan, so the first request after a quiet period can take up to a minute.

## Run the frontend locally

```bash
cd web
python -m http.server 8000
```

Open `http://localhost:8000`. To use your own backend, change `API_URL` near the bottom of `index.html`, and allow the page's address in the backend's CORS settings.

## Deploy

- **Frontend (Vercel):** import the repo, set **Framework Preset** to "Other" and **Root Directory** to `web`, leave the build command empty, then deploy.
- **Backend (Render):** deploy the API as a web service and add your Vercel URL to its allowed CORS origins.

## Limitations

- Trained on one public dataset, so results may not hold for every population.
- Uses only eight inputs and ignores family history, symptoms and other factors.
- Gender is limited to Female and Male because there were too few `Other` rows to learn from.
- Flags generously: expect more false alarms than misses.
- A learning project, not a screening tool.

## Author

**Aryan Khanna**

- [LinkedIn](https://www.linkedin.com/in/arnn/)
- [GitHub](https://github.com/Arnnnnnn)
