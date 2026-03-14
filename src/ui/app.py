import streamlit as st
import requests
import joblib

st.title('Customer Lifetime Value Prediction')

# Load feature columns
feature_columns = joblib.load('models_saved/feature_columns.pkl')

# Create input fields dynamically
inputs = {}
st.header('Enter Customer Features')
for feature in feature_columns:
    inputs[feature] = st.number_input(f'{feature}', value=0.0, step=0.01)

# Predict button
if st.button('Get Prediction'):
    try:
        response = requests.post('http://localhost:5000/predict', json=inputs)
        if response.status_code == 200:
            result = response.json()
            st.success(f'Predicted 30-day spend: ${result["predicted_30d_spend"]:.2f}')
        else:
            st.error(f'Prediction failed: {response.text}')
    except Exception as e:
        st.error(f'Error: {str(e)}')

# Add customer button
if st.button('Add Customer and Retrain Model'):
    # Note: For adding, we need the target_30d_spend as well
    target = st.number_input('Actual 30-day spend (for training)', value=0.0, step=0.01)
    inputs['target_30d_spend'] = target
    
    try:
        response = requests.post('http://localhost:5000/add_customer', json=inputs)
        if response.status_code == 200:
            st.success('Customer added and model retrained successfully!')
        else:
            st.error(f'Failed to add customer: {response.text}')
    except Exception as e:
        st.error(f'Error: {str(e)}')