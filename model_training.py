# -*- coding: utf-8 -*-
"""
Online Advertising Performance - EDA & Machine Learning Model Training
"""

# ==========================================================
# ONLINE ADVERTISING PERFORMANCE - COMPLETE ML PROJECT
# ==========================================================

# ==============================
# 1️⃣ IMPORT LIBRARIES
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ==============================
# 2️⃣ LOAD DATASET
# ==============================

df = pd.read_csv("online_advertising_performance_data.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ==============================
# 3️⃣ DATA CLEANING
# ==============================

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna("Unknown")
    else:
        df[col] = df[col].fillna(0)

# ==============================
# 4️⃣ FEATURE ENGINEERING
# ==============================

df['CTR'] = df['clicks'] / df['displays'].replace(0, np.nan)
df['CPC'] = df['cost'] / df['clicks'].replace(0, np.nan)
df['ROI'] = (df['revenue'] - df['cost']) / df['cost'].replace(0, np.nan)

df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# ==============================
# 5️⃣ EXTENDED EDA
# ==============================

print("\nStatistical Summary:")
print(df.describe())

# Distribution plots
df.hist(figsize=(15,10))
plt.suptitle("Feature Distributions")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes(include=['int64','float64']).corr(),
            annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ==============================
# 6️⃣ SPLIT FEATURES & TARGET
# ==============================

X = df.drop(['revenue'], axis=1)
y = df['revenue']

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

# ==============================
# 7️⃣ PREPROCESSING (OneHot + Scaling)
# ==============================

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ]
)

# ==============================
# 8️⃣ TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 9️⃣ FEATURE SELECTION
# ==============================

# Auto adjust k value safely
k_value = min(20, X.shape[1])
feature_selector = SelectKBest(score_func=f_regression, k=k_value)

# ==============================
# 🔟 DEFINE MODELS
# ==============================

models = {
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05,
                            max_depth=6, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=300, random_state=42),
    "SVR": SVR()
}

results = []
best_model = None
best_score = -np.inf

# ==============================
# 1️⃣1️⃣ TRAIN & EVALUATE
# ==============================

for name, model in models.items():

    print("\n===============================")
    print("Training:", name)
    print("===============================")

    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('feature_selection', feature_selector),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    cv_score = cross_val_score(pipeline, X, y, cv=5, scoring='r2').mean()

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)
    print("Cross Validation R2:", cv_score)

    results.append([name, mae, rmse, r2, cv_score])

    if r2 > best_score:
        best_score = r2
        best_model = pipeline

    # Residual Plot
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred, y_test - y_pred)
    plt.axhline(y=0)
    plt.title(f"Residual Plot - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.show()

# ==============================
# 1️⃣2️⃣ MODEL COMPARISON
# ==============================

results_df = pd.DataFrame(results,
                          columns=["Model","MAE","RMSE","R2","CV_R2"])

print("\nMODEL COMPARISON:")
print(results_df.sort_values(by="R2", ascending=False))

plt.figure(figsize=(8,5))
sns.barplot(x="Model", y="R2", data=results_df)
plt.title("Model Comparison (R2 Score)")
plt.xticks(rotation=45)
plt.show()

print("\nBest Model Selected:",
      results_df.sort_values(by="R2", ascending=False).iloc[0]["Model"])

# ==============================
# 1️⃣3️⃣ SHAP EXPLAINABILITY
# ==============================

print("\nGenerating SHAP Explainability...")

# Fit preprocessing only
X_processed = preprocessor.fit_transform(X)
feature_names = preprocessor.get_feature_names_out()

xgb_model = XGBRegressor(n_estimators=300, random_state=42)
xgb_model.fit(X_processed, y)

explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_processed)

shap.summary_plot(shap_values, X_processed, feature_names=feature_names)

# ==============================
# 1️⃣4️⃣ SAVE BEST MODEL
# ==============================

joblib.dump(best_model, "best_advertising_model.pkl", compress=3)
print("\nBest model saved successfully (compressed < 100MB)!")

# ==============================
# PROJECT COMPLETE
# ==============================
# python model_training.py