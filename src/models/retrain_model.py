import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np
import os

# Load original data
df = pd.read_csv('data/processed/customer_features.csv')

# Load new customer entries if they exist
if os.path.exists('data/new_entries/new_customers.csv'):
    new_df = pd.read_csv('data/new_entries/new_customers.csv')
    df = pd.concat([df, new_df], ignore_index=True)

# Prepare data
y = df['target_30d_spend']
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

# Retrain model
rf = RandomForestRegressor(random_state=42)
rf.fit(X, y)

# Save updated model and features
joblib.dump(rf, 'models_saved/best_model.pkl')
joblib.dump(feature_cols, 'models_saved/feature_columns.pkl')

print("Model retrained and saved")