import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# 1. Load feature-engineered data
df = pd.read_csv('/content/retail_features_30day_ready.csv')

# 2. Define target
y = df['target_30d_spend'].astype(int)

# 3. Select numeric, non‑leaky features
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

print(f"Using {len(feature_cols)} features:")
print(feature_cols)

# 4. Train–test split (stratified, to keep 1.7% rate in both)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Define Random Forest (reasonable starting config)
rf = RandomForestClassifier(
    n_estimators=400,          # number of trees
    max_depth=10,             # limit depth to reduce overfitting
    min_samples_split=100,
    min_samples_leaf=50,
    max_features='sqrt',
    class_weight='balanced',  # handle 1.7% positives
    n_jobs=-1,
    random_state=42
)

# 6. Train
rf.fit(X_train, y_train)

# 7. Evaluate
y_proba = rf.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_proba)
print(f"\nRandom Forest ROC AUC: {auc:.4f}\n")
print("Classification report:")
print(classification_report(y_test, y_pred, digits=4))

# 8. (Optional) feature importance
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 important features:")
print(importances.head(10))

# 9. Save model for later use
import joblib
joblib.dump(rf, '/content/random_forest_30day.pkl')
joblib.dump(feature_cols, '/content/rf_feature_cols.pkl')
print("\nSaved model to /content/random_forest_30day.pkl")
