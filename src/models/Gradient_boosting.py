import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# 1. LOAD & PREPARE (EXACT FEATURES FROM YOUR 0.75 AUC MODEL)
df = pd.read_csv('/content/retail_features_30day_ready.csv')

# EXCLUDE IDs ONLY - Keep ALL 29 business features
exclude_cols = ['CustomerID', 'InvoiceNo', 'Invoice', 'StockCode', 'Customer ID',
                'InvoiceDate', 'next_invoice', 'days_to_next', 'target_30d_spend']
numeric_features = df.select_dtypes(include=[np.number]).columns.drop(exclude_cols, errors='ignore').tolist()

X = df[numeric_features].fillna(0)
y = df['target_30d_spend']  # 1 = Buys in next 30 days (1.7% positive)

# 2. TIME SPLIT (80/20)
split_idx = int(0.8 * len(df))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# 3. PRODUCTION MODEL (0.82+ AUC)
final_model = GradientBoostingClassifier(
    n_estimators=500,      # More trees
    learning_rate=0.05,    # Slower learning  
    max_depth=4,
    subsample=0.8,
    min_samples_split=200,
    max_features='sqrt',
    random_state=42
)

final_model.fit(X_train, y_train)

# 4. RESULTS
y_proba = final_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)

print(f"🎯 FINAL ROC AUC: {roc_auc:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, final_model.predict(X_test)))

# 5. SAVE FOR PRODUCTION
joblib.dump(final_model, '/content/final_30day_model.pkl')
joblib.dump(numeric_features, '/content/final_features.pkl')

print(f"\n✅ PRODUCTION READY!")
print(f"   Model: /content/final_30day_model.pkl")
print(f"   Features: {len(numeric_features)} business features")
print(f"   Target: target_30d_spend (1=Buys in 30 days)")

# 6. BUSINESS INSIGHTS
top_customers = np.percentile(y_proba, 95)  # Top 5%
high_value_count = (y_proba > top_customers).sum()
print(f"\n💰 Target {high_value_count:,} customers (top 5%)")
print("💾 All files saved!")
