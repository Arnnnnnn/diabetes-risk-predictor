import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

df = pd.read_csv("diabetes_clean.csv")
X = df.drop(columns=["diabetes"]).astype(float)
y = df["diabetes"]

model = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
model.fit(X, y)

joblib.dump({"model": model, "columns": list(X.columns)}, "model.joblib")
print("Saved model.joblib with columns:", list(X.columns))