import pandas as pd

df = pd.read_csv('diabetes_prediction_dataset.csv')
print("Before:", df.shape)

df = df.drop_duplicates().reset_index(drop=True)
print("After dedupe:", df.shape)

smoking_map = {"never": "never", "current": "current", "former": "former",
               "not current": "former", "ever": "former", "No Info": "unknown"}
df["smoking_history"] = df["smoking_history"].map(smoking_map)
print(df["smoking_history"].value_counts())
print("Unmapped smoking values:", df["smoking_history"].isna().sum())

df = df[df["gender"] != "Other"].reset_index(drop=True)
print("After dropping Other:", df.shape)

print(df[["age", "bmi", "HbA1c_level", "blood_glucose_level"]].describe())

df_model = pd.get_dummies(df, columns=["gender", "smoking_history"], drop_first=True)
print(df_model.columns.tolist())
print(df_model.shape)

df_model.to_csv("diabetes_clean.csv", index=False)
print("Saved diabetes_clean.csv")