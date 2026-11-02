import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.inspection import permutation_importance

df = pd.read_csv("diabetes_clean.csv")
X = df.drop(columns=["diabetes"]).astype(float)
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

logreg = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)

gb = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, m in [("Logistic regression", logreg), ("Gradient boosting", gb)]:
    scores = cross_val_score(m, X_train, y_train, cv=cv, scoring="roc_auc")
    print(name, "CV ROC-AUC:", round(scores.mean(), 4), "+/-", round(scores.std(), 4))

gb.fit(X_train, y_train)
pred = gb.predict(X_test)
proba = gb.predict_proba(X_test)[:, 1]
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print("Test ROC-AUC:", round(roc_auc_score(y_test, proba), 4))

imp = permutation_importance(gb, X_test, y_test, scoring="roc_auc",
                             n_repeats=5, random_state=42)
ranking = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False)
print(ranking)