# -*- coding: utf-8 -*-
"""
================================================================================
  DIGITAL MARKETING CAMPAIGN ANALYSIS — ML TRAINING PIPELINE
  Author : Samaa Shaikh
  Purpose: Train, evaluate and serialize the best regression model for
           campaign revenue forecasting.
================================================================================

LEAKAGE AUDIT SUMMARY
---------------------
Target variable : revenue

Features EXCLUDED from ML (with reason):
  - ROI                     : (revenue - cost) / cost — mathematically contains
                              the target. Cannot be known before revenue is known.
  - post_click_sales_amount : Post-campaign sales figure that is a direct proxy
                              of revenue. Recorded alongside revenue; not
                              available before the target is realized.
  - Unnamed: 12/13          : Empty artifact columns (all NaN).

Features KEPT for ML (prediction-time availability):
  - month, day              : Campaign schedule — known before launch.
  - campaign_number         : Campaign identity — selected before launch.
  - user_engagement         : Audience targeting tier — set before launch.
  - banner                  : Creative asset dimension — chosen before launch.
  - placement               : Publisher slot — booked before launch.
  - displays                : Planned impression volume (campaign input).
  - cost                    : Planned budget spend (campaign input).
  - clicks                  : Anticipated click volume (user-supplied forecast).
  - post_click_conversions  : Expected conversions (user-supplied forecast).
  - CTR                     : clicks / displays — derivable from forecast inputs.
  - CPC                     : cost / clicks   — derivable from forecast inputs.

BUSINESS ANALYTICS NOTE
-----------------------
Metrics such as ROAS, ROI, Net Profit, and Profit Margin are computed during
EDA for business insight. They are intentionally excluded from the ML feature
matrix because they require revenue to be known (leakage). They remain valid
for retrospective dashboard analysis.
================================================================================
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "online_advertising_performance_data.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
MODEL_PATH  = os.path.join(BASE_DIR, "best_advertising_model.pkl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# SECTION 1 — LOAD DATASET
# ==============================================================================
print("=" * 70)
print("SECTION 1: Loading Dataset")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().lower() for c in df.columns]

# Drop empty artifact columns produced by Excel/CSV export
df = df.drop(columns=[c for c in df.columns if 'unnamed' in c], errors='ignore')

print(f"Raw shape   : {df.shape}")
print(f"Columns     : {df.columns.tolist()}")
print(f"Missing vals:\n{df.isnull().sum()}\n")

# ==============================================================================
# SECTION 2 — DATA CLEANING
# ==============================================================================
print("=" * 70)
print("SECTION 2: Data Cleaning")
print("=" * 70)

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna("Unknown")
    else:
        df[col] = df[col].fillna(0)

df.replace([np.inf, -np.inf], 0, inplace=True)
print("Missing values after cleaning:", df.isnull().sum().sum())
print(f"Clean shape : {df.shape}\n")

# ==============================================================================
# SECTION 3 — FEATURE ENGINEERING (ML-safe only)
# ==============================================================================
print("=" * 70)
print("SECTION 3: Feature Engineering")
print("=" * 70)

# --- VALID ML FEATURES (derived from legitimate forecast inputs) ---
# CTR: click-through rate — computable from planned displays and forecasted clicks
df['ctr'] = np.where(df['displays'] > 0,
                     df['clicks'] / df['displays'], 0.0)

# CPC: cost per click — computable from planned budget and forecasted clicks
df['cpc'] = np.where(df['clicks'] > 0,
                     df['cost'] / df['clicks'], 0.0)

print("Engineered ML features: CTR, CPC")

# --- BUSINESS ANALYTICS METRICS (computed for EDA only — NOT used in ML) ---
# These require revenue to be known; they are target-derived and must not
# appear in the ML feature matrix.
df['roi']          = np.where(df['cost'] > 0,
                              (df['revenue'] - df['cost']) / df['cost'], 0.0)
df['roas']         = np.where(df['cost'] > 0,
                              df['revenue'] / df['cost'], 0.0)
df['net_profit']   = df['revenue'] - df['cost']
df['profit_margin'] = np.where(df['revenue'] > 0,
                                (df['net_profit'] / df['revenue']) * 100, 0.0)

print("Business analytics metrics: ROI, ROAS, Net Profit, Profit Margin")
print("  >> These are computed for EDA/dashboard only. EXCLUDED from ML.\n")
df.replace([np.inf, -np.inf], 0, inplace=True)

# ==============================================================================
# SECTION 4 — EXPLORATORY DATA ANALYSIS (save to outputs/)
# ==============================================================================
print("=" * 70)
print("SECTION 4: Exploratory Data Analysis")
print("=" * 70)

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# --- 4a. Feature Distributions ---
fig, axes = plt.subplots(4, 4, figsize=(18, 14))
axes = axes.flatten()
plot_cols = [c for c in numeric_cols if c not in ['roi', 'roas', 'net_profit', 'profit_margin']][:16]
for i, col in enumerate(plot_cols):
    axes[i].hist(df[col].dropna(), bins=40, color='#3b82f6', edgecolor='white', alpha=0.8)
    axes[i].set_title(col, fontsize=10, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].tick_params(labelsize=8)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Feature Distributions — Online Advertising Dataset",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
path_dist = os.path.join(OUTPUT_DIR, "feature_distributions.png")
plt.savefig(path_dist, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path_dist}")

# --- 4b. Correlation Heatmap (ML-safe numeric features only) ---
corr_cols = ['displays', 'cost', 'clicks', 'post_click_conversions',
             'ctr', 'cpc', 'revenue']
corr_df = df[corr_cols].copy()
plt.figure(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_df.corr(), dtype=bool))
sns.heatmap(corr_df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, linewidths=0.5, square=True, vmin=-1, vmax=1,
            cbar_kws={'shrink': 0.8})
plt.title("Pearson Correlation Matrix — ML Feature Space vs. Revenue",
          fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
path_corr = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
plt.savefig(path_corr, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path_corr}")

# --- 4c. Campaign Performance Benchmark ---
camp_agg = df.groupby('campaign_number').agg(
    Revenue=('revenue', 'sum'),
    Cost=('cost', 'sum'),
    Net_Profit=('net_profit', 'sum')
).reset_index()
x = np.arange(len(camp_agg))
w = 0.28
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w, camp_agg['Revenue'], w, label='Revenue', color='#3b82f6')
ax.bar(x,     camp_agg['Cost'],    w, label='Cost',    color='#ef4444')
ax.bar(x + w, camp_agg['Net_Profit'], w, label='Net Profit', color='#10b981')
ax.set_xticks(x)
ax.set_xticklabels(camp_agg['campaign_number'])
ax.set_ylabel('Amount ($)')
ax.set_title('Campaign Performance Benchmark', fontsize=13, fontweight='bold')
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
plt.tight_layout()
path_camp = os.path.join(OUTPUT_DIR, "campaign_performance_benchmark.png")
plt.savefig(path_camp, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path_camp}")

# --- 4d. Creative Dimension Revenue ---
banner_agg = df.groupby('banner')['revenue'].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(banner_agg.index, banner_agg.values, color=sns.color_palette('tab10', len(banner_agg)))
ax.set_xlabel('Banner Size')
ax.set_ylabel('Total Revenue ($)')
ax.set_title('Revenue by Creative Dimension (Banner Size)', fontsize=13, fontweight='bold')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
            f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
path_banner = os.path.join(OUTPUT_DIR, "creative_dimension_analysis.png")
plt.savefig(path_banner, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path_banner}\n")

# ==============================================================================
# SECTION 5 — FEATURE / TARGET SEPARATION
# ==============================================================================
print("=" * 70)
print("SECTION 5: Feature / Target Separation")
print("=" * 70)

# ── ML FEATURE SET (leakage-free) ──────────────────────────────────────────
ML_FEATURES = [
    # Campaign configuration — available before campaign launch
    'month', 'day', 'campaign_number', 'user_engagement', 'banner', 'placement',
    # Campaign operating inputs — planned/forecasted values
    'displays', 'cost', 'clicks', 'post_click_conversions',
    # Engineered from forecast inputs — valid at prediction time
    'ctr', 'cpc',
]

# ── EXCLUDED FEATURES ──────────────────────────────────────────────────────
# roi                    → derived from revenue (target leakage)
# post_click_sales_amount→ post-campaign sales proxy of revenue (leakage)
# roas, net_profit, etc. → business analytics only; all derived from revenue

TARGET = 'revenue'

X = df[ML_FEATURES].copy()
y = df[TARGET].copy()

print(f"Target        : {TARGET}")
print(f"ML features   : {ML_FEATURES}")
print(f"Excluded (leakage): ['roi', 'post_click_sales_amount', 'roas', "
      f"'net_profit', 'profit_margin']")
print(f"\nX shape: {X.shape}  |  y shape: {y.shape}\n")

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols   = X.select_dtypes(exclude=['object']).columns.tolist()
print(f"Categorical : {categorical_cols}")
print(f"Numerical   : {numerical_cols}\n")

# ==============================================================================
# SECTION 6 — TRAIN / TEST SPLIT
# ==============================================================================
print("=" * 70)
print("SECTION 6: Train / Test Split (80 / 20, random_state=42)")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set : {X_train.shape}  |  Test set : {X_test.shape}\n")

# ==============================================================================
# SECTION 7 — PREPROCESSING PIPELINE
# ==============================================================================
print("=" * 70)
print("SECTION 7: Preprocessing — OneHotEncoder + StandardScaler")
print("=" * 70)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('num', StandardScaler(), numerical_cols),
    ],
    remainder='drop'
)
print("Preprocessor configured.\n")

# ==============================================================================
# SECTION 8 — MODEL DEFINITIONS
# ==============================================================================
print("=" * 70)
print("SECTION 8: Model Definitions")
print("=" * 70)

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        random_state=42, verbosity=0),
    "LightGBM": LGBMRegressor(
        n_estimators=300, random_state=42, verbose=-1),
    "SVR": SVR(kernel='rbf', C=10, epsilon=0.1, max_iter=3000),
}
print(f"Models to evaluate: {list(models.keys())}\n", flush=True)

# ==============================================================================
# SECTION 9 — TRAINING, EVALUATION & CROSS-VALIDATION
# ==============================================================================
print("=" * 70, flush=True)
print("SECTION 9: Training, Evaluation & 5-Fold Cross-Validation", flush=True)
print("=" * 70, flush=True)

results      = []
best_model   = None
best_score   = -np.inf
best_name    = ""
best_y_pred  = None

for name, estimator in models.items():
    print(f"\n--- {name} ---", flush=True)

    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model',         estimator),
    ])

    # Fit
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Hold-out metrics
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    # 5-fold cross-validation on full dataset
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2', n_jobs=-1)
    cv_r2     = cv_scores.mean()

    print(f"  MAE      : {mae:.4f}", flush=True)
    print(f"  RMSE     : {rmse:.4f}", flush=True)
    print(f"  R²       : {r2:.4f}", flush=True)
    print(f"  CV R² (5-fold): {cv_r2:.4f}  ± {cv_scores.std():.4f}", flush=True)

    results.append({
        'Model':  name,
        'MAE':    round(mae,  4),
        'RMSE':   round(rmse, 4),
        'R2':     round(r2,   4),
        'CV_R2':  round(cv_r2, 4),
    })

    if r2 > best_score:
        best_score  = r2
        best_model  = pipeline
        best_name   = name
        best_y_pred = y_pred

# ==============================================================================
# SECTION 10 — MODEL COMPARISON TABLE & CHART
# ==============================================================================
print("\n" + "=" * 70, flush=True)
print("SECTION 10: Model Comparison", flush=True)
print("=" * 70, flush=True)

results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False), flush=True)
print(f"\nBest model (by hold-out R²): {best_name}  (R² = {best_score:.4f})", flush=True)

# Save CSV
csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
results_df.to_csv(csv_path, index=False)
print(f"\n  Saved: {csv_path}", flush=True)

# Save comparison chart
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#2563eb', '#7c3aed', '#059669', '#dc2626']
bars   = ax.bar(results_df['Model'], results_df['R2'], color=colors, width=0.55,
                edgecolor='white', linewidth=1.2)
ax.bar(results_df['Model'], results_df['CV_R2'],
       color=[c + '60' for c in ['#2563eb', '#7c3aed', '#059669', '#dc2626']],
       width=0.55, bottom=0, alpha=0.5, label='CV R² (5-fold)')

for bar, row in zip(bars, results_df.itertuples()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"R²={row.R2:.3f}\nCV={row.CV_R2:.3f}", ha='center', va='bottom',
            fontsize=9, fontweight='bold')

ax.set_ylabel('R² Score', fontsize=11)
ax.set_title('Model Comparison — Hold-out R² vs. 5-Fold CV R²\n(Leakage-free feature set)',
             fontsize=12, fontweight='bold')
ax.set_ylim(0, min(results_df['R2'].max() * 1.22, 1.05))
ax.axhline(0, color='gray', linewidth=0.8)
ax.legend(['Hold-out R²', 'CV R² overlay'], loc='lower right')
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
chart_path_alt = os.path.join(OUTPUT_DIR, "model_benchmark_comparison.png")
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.savefig(chart_path_alt, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {chart_path}", flush=True)

# ==============================================================================
# SECTION 11 — REGRESSION DIAGNOSTICS (best model)
# ==============================================================================
print("\n" + "=" * 70)
print(f"SECTION 11: Regression Diagnostics — {best_name}")
print("=" * 70)

residuals = y_test.values - best_y_pred

# --- 11a. Predicted vs. Actual ---
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, best_y_pred, alpha=0.35, s=12, color='#3b82f6', edgecolors='none')
lim = [min(y_test.min(), best_y_pred.min()) * 0.95,
       max(y_test.max(), best_y_pred.max()) * 1.05]
ax.plot(lim, lim, 'r--', linewidth=1.5, label='Perfect prediction line')
ax.set_xlabel('Actual Revenue ($)', fontsize=11)
ax.set_ylabel('Predicted Revenue ($)', fontsize=11)
ax.set_title(f'Predicted vs. Actual Revenue\n{best_name}  (R² = {best_score:.4f})',
             fontsize=12, fontweight='bold')
ax.legend()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
plt.tight_layout()
pva_path = os.path.join(OUTPUT_DIR, "predicted_vs_actual.png")
plt.savefig(pva_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {pva_path}")

# --- 11b. Residual Plot ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(best_y_pred, residuals, alpha=0.35, s=12, color='#7c3aed', edgecolors='none')
ax.axhline(0, color='red', linewidth=1.5, linestyle='--')
ax.set_xlabel('Predicted Revenue ($)', fontsize=11)
ax.set_ylabel('Residual (Actual − Predicted) ($)', fontsize=11)
ax.set_title(f'Residual Plot — {best_name}', fontsize=12, fontweight='bold')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
plt.tight_layout()
res_path = os.path.join(OUTPUT_DIR, "residual_plot.png")
plt.savefig(res_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {res_path}")

# ==============================================================================
# SECTION 12 — FEATURE IMPORTANCE (SHAP — XGBoost model)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 12: Feature Importance via SHAP (XGBoost, trained on full data)")
print("=" * 70)
print("Note: SHAP shows the contribution of each input feature to model output.")
print("This is feature importance analysis, not causal attribution.\n")

# Fit a standalone XGBoost on preprocessed features (full dataset) for SHAP
_preprocessor_shap = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('num', StandardScaler(), numerical_cols),
    ],
    remainder='drop'
)
X_shap = _preprocessor_shap.fit_transform(X)
feat_names = _preprocessor_shap.get_feature_names_out()

xgb_shap = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
xgb_shap.fit(X_shap, y)

# Use a sample for speed
sample_size = min(2000, X_shap.shape[0])
rng = np.random.default_rng(42)
idx = rng.choice(X_shap.shape[0], size=sample_size, replace=False)
X_sample = X_shap[idx]

explainer   = shap.Explainer(xgb_shap, X_sample)
shap_values = explainer(X_sample)

fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_sample, feature_names=feat_names,
                  max_display=15, show=False)
plt.title("SHAP Feature Importance — XGBoost (leakage-free feature set)",
          fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
shap_path = os.path.join(OUTPUT_DIR, "shap_feature_importance.png")
plt.savefig(shap_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {shap_path}")

# ==============================================================================
# SECTION 13 — SAVE BEST PIPELINE
# ==============================================================================
print("\n" + "=" * 70)
print(f"SECTION 13: Saving Best Model — {best_name}")
print("=" * 70)

joblib.dump(best_model, MODEL_PATH, compress=3)
print(f"  Saved: {MODEL_PATH}")
print(f"  Model     : {best_name}")
print(f"  R²        : {best_score:.4f}")
print(f"  Features  : {ML_FEATURES}")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("PIPELINE COMPLETE — Summary")
print("=" * 70)
print(f"\nModel Comparison:\n{results_df.to_string(index=False)}")
print(f"\nBest Model : {best_name}")
print(f"Hold-out R²: {best_score:.4f}")
print(f"\nOutputs written to: {OUTPUT_DIR}")
print("  model_comparison.csv")
print("  model_comparison.png")
print("  predicted_vs_actual.png")
print("  residual_plot.png")
print("  shap_feature_importance.png")
print("  feature_distributions.png")
print("  correlation_heatmap.png")
print("  campaign_performance_benchmark.png")
print("  creative_dimension_analysis.png")
print(f"\nModel saved: {MODEL_PATH}")
print("=" * 70)

# Run with: python model_training.py