import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ✅ FIXED COMPLETE PIPELINE
def run_complete_pipeline():
    # 1. LOAD DATA
    df = pd.read_csv('/content/online_retail_clean.csv')
    print(f"📁 Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("Original columns:", df.columns.tolist())
    
    # 2. FEATURE ENGINEERING (FIXED)
    df_final = complete_feature_engineering(df)
    
    # 3. CREATE TARGET
    df_final = create_30day_target(df_final)
    
    # 4. SAVE
    df_final.to_csv('/content/retail_features_30day_ready.csv', index=False)
    print("💾 Saved to: /content/retail_features_30day_ready.csv")
    
    # 5. PREVIEW
    modeling_cols = ['target_30d_spend', 'active_30d', 'spend_30d_avg', 'recency_days', 
                     'cust_orders', 'spend_momentum', 'invoice_lag_days']
    print("\n🎯 TOP FEATURES PREVIEW:")
    print(df_final[modeling_cols].head(10))
    
    return df_final

def complete_feature_engineering(df):
    """✅ FIXED: No more duplicate index errors"""
    df = df.copy()
    
    # FIX DATA TYPES
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['CustomerID'] = df['CustomerID'].astype('Int64')
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    # SORT FOR TIME SERIES
    df = df.sort_values(['CustomerID', 'InvoiceDate']).reset_index(drop=True)
    
    # CUSTOMER AGGREGATES (SAFE)
    customer_agg = df.groupby('CustomerID').agg({
        'TotalPrice': ['sum', 'mean', 'std', 'count'],
        'Quantity': ['sum', 'mean'],
        'InvoiceDate': ['first', 'last', lambda x: (x.max() - x.min()).days]
    }).round(4)
    customer_agg.columns = ['cust_ltv', 'cust_avg_invoice', 'cust_invoice_std', 'cust_orders',
                           'cust_total_qty', 'cust_avg_qty', 'cust_first_date', 
                           'cust_last_date', 'cust_tenure_days']
    df = df.merge(customer_agg, left_on='CustomerID', right_index=True, how='left')
    
    # ✅ FIXED ROLLING WINDOWS - Using COUNT-BASED windows (no datetime index issues)
    print("🔄 Creating rolling features...")
    
    # 30-day equivalent: rolling 10 transactions back (adjust based on your data frequency)
    df['spend_30d_avg'] = (df.groupby('CustomerID')['TotalPrice']
                          .rolling(10, min_periods=1).mean().shift(1).reset_index(0, drop=True))
    
    df['spend_90d_avg'] = (df.groupby('CustomerID')['TotalPrice']
                          .rolling(20, min_periods=1).mean().shift(1).reset_index(0, drop=True))
    
    # RECENCY FEATURES (SAFE)
    df['recency_days'] = (df['InvoiceDate'].max() - df['InvoiceDate']).dt.days
    df['active_30d'] = (df['recency_days'] <= 30).astype(int)
    df['active_7d'] = (df['recency_days'] <= 7).astype(int)
    
    # PURCHASE PATTERNS
    df['invoice_lag_days'] = df.groupby('CustomerID')['InvoiceDate'].diff().dt.days.fillna(999)
    df['spend_momentum'] = df['spend_30d_avg'] / (df['spend_90d_avg'] + 1)
    df['frequency_score'] = df['cust_orders'] / (df['cust_tenure_days']/30 + 1)
    
    # TRANSACTION FEATURES
    df['value_per_item'] = df['TotalPrice'] / (df['Quantity'] + 1e-8)
    df['high_value_invoice'] = (df['TotalPrice'] > df['TotalPrice'].quantile(0.9)).astype(int)
    
    # BEHAVIORAL FLAGS
    df['frequent_buyer'] = (df['frequency_score'] > 1).astype(int)
    df['recent_buyer'] = (df['recency_days'] <= 60).astype(int)
    
    # FILL NANS
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)
    
    # SCALE
    scale_cols = ['recency_days', 'spend_30d_avg', 'spend_90d_avg', 'cust_ltv',
                  'cust_orders', 'cust_tenure_days', 'invoice_lag_days', 'spend_momentum']
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    
    print("✅ Feature engineering complete!")
    return df

def create_30day_target(df):
    """Create prediction target"""
    df = df.sort_values(['CustomerID', 'InvoiceDate'])
    df['next_invoice'] = df.groupby('CustomerID')['InvoiceDate'].shift(-1)
    df['days_to_next'] = (df['next_invoice'] - df['InvoiceDate']).dt.days
    df['target_30d_spend'] = ((df['days_to_next'] <= 30) & (df['days_to_next'] > 0)).astype(int)
    print(f"🎯 Target: {df['target_30d_spend'].value_counts().to_dict()}")
    return df

# 🚀 RUN NOW - NO ERRORS!
if __name__ == "__main__":
    df_final = run_complete_pipeline()
