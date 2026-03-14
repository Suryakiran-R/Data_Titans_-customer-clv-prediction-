import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Phase 3: XGBoost Model Training and Saving")
print("=" * 45)

# 1. LOAD & PREPARE DATA
print("\n1. Loading the feature-engineered data...")
df = pd.read_csv('/content/retail_features_30day_ready.csv')

# Target
y = df['target_30d_spend']

# Define features (matching the previous selection in cell -HXAC0xjkgdq)
drop_cols = [
    'CustomerID', 'Customer ID', 'Invoice', 'InvoiceNo', 'StockCode',
    'Description', 'Country', 'InvoiceDate',
    'next_invoice', 'days_to_next', 'target_30d_spend'
]
feature_cols = [
    c for c in df.columns
    if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)
]
X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"Selected {len(feature_cols)} features for XGBoost training.")

# 2. Time-aware split (first 80% train, last 20% test)
print("\n2. Performing time-aware train-test split...")
df_sorted = df.sort_values(['CustomerID', 'InvoiceDate'])
idx = df_sorted.index
split_idx = int(0.8 * len(df_sorted))

train_idx = idx[:split_idx]
test_idx = idx[split_idx:]

X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]

print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
print(f"Positive class ratio in train: {y_train.mean():.3%}, test: {y_test.mean():.3%}")

# 3. Initialize and Train XGBoost Classifier (tuned for imbalance)
print("\n3. Training XGBoost Classifier...")
xgb_model = XGBClassifier(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(1 - y_train.mean()) / y_train.mean(),  # handle 1.7% pos
    objective='binary:logistic',
    eval_metric='auc',
    tree_method='hist',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

print("XGBoost training complete!")

# 4. Evaluate the model
print("\n4. Evaluating XGBoost model performance...")
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
y_pred = xgb_model.predict(X_test)

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC on test set: {roc_auc:.4f}")

print("\nClassification Report on test set:")
print(classification_report(y_test, y_pred))

# 5. Save the trained model and feature columns to pickle files
print("\n5. Saving the trained XGBoost model and feature list...")
joblib.dump(xgb_model, '/content/xgboost_30day_model.pkl')
joblib.dump(feature_cols, '/content/xgboost_30day_features.pkl')

print("✅ XGBoost model saved to '/content/xgboost_30day_model.pkl'")
print("✅ Feature list saved to '/content/xgboost_30day_features.pkl'")
print("Phase 3 COMPLETE!")
