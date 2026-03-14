import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import subprocess

app = Flask(__name__)

MODEL_PATH = 'models_saved/best_model.pkl'
FEATURES_PATH = 'models_saved/feature_columns.pkl'

# Load model and features
model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict 30-day spend for a customer.
    Expects JSON with feature values.
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    df = pd.DataFrame([data])
    df = df[feature_columns]
    prediction = model.predict(df)[0]
    return jsonify({'predicted_30d_spend': prediction})

@app.route('/add_customer', methods=['POST'])
def add_customer():
    """
    Add a new customer and retrain the model.
    Expects JSON with customer features including target_30d_spend.
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    df = pd.DataFrame([data])
    file_path = 'data/new_entries/new_customers.csv'
    
    # Append to existing file or create new
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(file_path, index=False)
    
    # Trigger retraining
    subprocess.run(['python', 'src/models/retrain_model.py'])
    
    return jsonify({'message': 'Customer added and model retrained'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)