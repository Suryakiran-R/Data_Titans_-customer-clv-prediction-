import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

# 1. Load data
csv_path = "data/processed/customer_features.csv"
df = pd.read_csv(csv_path)

# 2. Downcast numeric columns
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[num_cols] = df[num_cols].apply(pd.to_numeric, downcast="float")

# 3. Target and features
target_col = "target_30d_spend"

cols_to_drop = [
    target_col,
    "InvoiceNo",
    "StockCode",
    "Description",
    "InvoiceDate",
    "Country",
    "CustomerID",
    "Customer ID",
    "Invoice"
]
cols_to_drop = [c for c in cols_to_drop if c in df.columns]

X = df.drop(columns=cols_to_drop)
y = df[target_col]

# Keep only numeric features to avoid 'Wednesday' issue
X = X.select_dtypes(include=["number"])

print("Shape of X:", X.shape)

# 4. Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train Random Forest
rf = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

# 6. Evaluate (manual RMSE, no 'squared' argument)
y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)   # returns MSE
rmse = np.sqrt(mse)                        # convert to RMSE manually
r2 = r2_score(y_test, y_pred)

print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

# 7. Save model and features
import joblib
joblib.dump(rf, 'models_saved/best_model.pkl')
joblib.dump(list(X.columns), 'models_saved/feature_columns.pkl')
print("Model and features saved")
