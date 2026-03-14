# Data_Titans_(customer-clv-prediction)
# Customer Lifetime Value (CLV) Prediction System

A complete machine learning system for predicting customer lifetime value (CLV) based on transaction data. The system allows users to input raw transaction details and get predictions for 30-day spend amounts, with automatic model retraining.

## Features

- **Raw Transaction Input**: Users enter transaction details (Invoice, StockCode, Quantity, etc.) instead of pre-processed features
- **Real-time Predictions**: Get 30-day spend predictions in dollars
- **Automatic Retraining**: Add new customer data to retrain the model
- **Web Interface**: Streamlit UI for easy interaction
- **REST API**: Flask API for programmatic access
- **Preprocessing**: Automatic feature engineering from raw data

## Project Structure

```
Data_Titans_-customer-clv-prediction/
├── src/
│   ├── api/
│   │   └── app.py              # Flask REST API
│   ├── ui/
│   │   └── app.py              # Streamlit web interface
│   ├── models/
│   │   ├── RF_regresser.py     # Model training script
│   │   └── retrain_model.py    # Automatic retraining script
│   └── preprocessing/
│       └── preprocess.py       # Feature preprocessing utilities
├── models_saved/
│   ├── best_model.pkl          # Trained RandomForestRegressor
│   └── feature_columns.pkl     # Feature column names
├── data/
│   ├── processed/
│   │   └── customer_features.csv  # Processed dataset
│   └── new_entries/
│       └── new_customers.csv   # New customer data for retraining
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Data_Titans_-customer-clv-prediction-
   ```

2. **Install dependencies**:
   ```bash
   python -m pip install pandas scikit-learn joblib flask streamlit
   ```

3. **Ensure data is in place**:
   - Place your processed customer features CSV at `data/processed/customer_features.csv`
   - The system expects columns including transaction details and target `target_30d_spend`

## Usage

### Training the Model

Run the training script to create/update the model:
```bash
python src/models/RF_regresser.py
```
This trains a RandomForestRegressor on the processed data and saves the model.

### Starting the System

1. **Start the Flask API** (in one terminal):
   ```bash
   python src/api/app.py
   ```
   The API runs on `http://localhost:5000`

2. **Start the Streamlit UI** (in another terminal):
   ```bash
   python -m streamlit run src/ui/app.py
   ```
   Opens a web browser with the interface.

### Using the Web Interface

1. Enter transaction details in the form fields
2. Click "Get Prediction" to see the predicted 30-day spend
3. To retrain: Enter data + actual spend, then click "Add Customer and Retrain Model"

### Using the API

**Predict spend:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [{
      "Invoice": "489434",
      "StockCode": "85048",
      "Description": "15CM CHRISTMAS GLASS BALL 20 LIGHTS",
      "Quantity": 12,
      "InvoiceDate": "12/1/09 7:45",
      "Price": 6.95,
      "Customer ID": "13085",
      "Country": "United Kingdom"
    }]
  }'
```

**Add customer and retrain:**
```bash
curl -X POST http://localhost:5000/add_customer \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [...],
    "target_30d_spend": 150.00
  }'
```

## How It Works

### Data Flow
1. **Input**: Raw transaction data (8 fields)
2. **Preprocessing**: `compute_features()` extracts 28 engineered features
3. **Prediction**: RandomForestRegressor predicts 30-day spend
4. **Retraining**: New data is added and model is retrained automatically

### Feature Engineering
The system computes customer behavior features from single transactions:
- Transaction metrics (quantity, price, total)
- Temporal features (recency, date components)
- Customer history estimates (defaults for new customers)
- Behavioral scores (activity, value metrics)

### Model
- **Type**: RandomForestRegressor
- **Features**: 28 engineered features
- **Target**: `target_30d_spend` (30-day future spend)
- **Performance**: R² ~0.92 on test data

## API Reference

### POST /predict
Predict 30-day spend from transaction data.

**Request Body:**
```json
{
  "transactions": [
    {
      "Invoice": "string",
      "StockCode": "string", 
      "Description": "string",
      "Quantity": number,
      "InvoiceDate": "MM/D/YY H:MM",
      "Price": number,
      "Customer ID": "string",
      "Country": "string"
    }
  ]
}
```

**Response:**
```json
{
  "predicted_30d_spend": 45.67
}
```

### POST /add_customer
Add customer data and retrain model.

**Request Body:**
```json
{
  "transactions": [...],
  "target_30d_spend": 150.00
}
```

**Response:**
```json
{
  "message": "Customer added and model retrained"
}
```

## Development

### Adding New Features
1. Update `compute_features()` in `src/api/app.py`
2. Retrain the model with `RF_regresser.py`
3. Update `feature_columns.pkl`

### Customizing the Model
Modify `src/models/RF_regresser.py` to use different algorithms or hyperparameters.

### Testing
- Unit tests for preprocessing functions
- API endpoint testing with Postman/curl
- UI testing with different input scenarios

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

[Add license information]

## Support

For issues or questions, please open a GitHub issue.
