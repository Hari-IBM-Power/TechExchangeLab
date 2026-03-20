# Time Series Savings Prediction System

A time series analysis system using Exponential Smoothing to predict user savings for the next month based on transaction history.

## Project Structure

```
TCC Timeseries/
├── main.py                                    # Main orchestration script
├── transactions_2025_salary_filtered.csv      # Transaction data
├── requirements.txt                           # Python dependencies
├── src/
│   ├── __init__.py                           # Package initialization
│   ├── data_engineering.py                   # Data processing functions
│   ├── ml_model.py                           # ML model implementation
│   └── model_user_*.pkl                      # Saved trained models
└── README.md                                  # This file
```

## Features

- **Separate Data Engineering & ML Modules**: Clean separation of concerns
- **User-Specific Time Series**: Individual models for each user
- **Exponential Smoothing**: Statistical forecasting method for time series
- **Model Persistence**: Trained models saved as pickle files in src/ folder
- **Automatic Training**: Trains model if user exists but model doesn't
- **Smart Error Handling**: Returns error only if user doesn't exist in dataset

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Prediction System

The main.py script takes a user_id as input parameter:

```python
from main import main

# Predict savings for any user
result = main(user_id)
```

### Behavior

The system intelligently handles three scenarios:

1. **Model exists**: Loads saved model and predicts
2. **Model doesn't exist but user has data**: Trains new model and predicts
3. **User doesn't exist in dataset**: Returns error message

### Example 1: Existing Model (User 0)

```python
result = main(0)

# Output:
============================================================
USER SAVINGS PREDICTION SYSTEM
Using Exponential Smoothing Time Series Analysis
============================================================

============================================================
SAVINGS PREDICTION FOR USER 0
============================================================

✓ User 0 found in dataset
✓ Model found: src/model_user_0.pkl
✓ Model loaded successfully

============================================================
PREDICTION RESULTS
============================================================

Last Month (2025-12):
  Savings: ₹7,252.00
  End Balance: ₹83,792.69

Historical Statistics:
  Average Monthly Savings: ₹6,982.72
  Std Dev: ₹749.01

────────────────────────────────────────────────────────────
PREDICTED NEXT MONTH SAVINGS: ₹7,061.18
────────────────────────────────────────────────────────────

✓ Prediction completed successfully!

# Returned dictionary:
{
    'user_id': 0,
    'predicted_next_month_savings': 7061.18,
    'last_month_savings': 7252.0,
    'last_month_balance': 83792.69,
    'last_month_date': '2025-12',
    'average_savings': 6982.72,
    'std_savings': 749.01,
    'n_months': 12,
    'model_filepath': 'src/model_user_0.pkl'
}
```

### Example 2: Auto-Training (User 3, No Model)

```python
result = main(3)

# Output:
============================================================
USER SAVINGS PREDICTION SYSTEM
Using Exponential Smoothing Time Series Analysis
============================================================

============================================================
SAVINGS PREDICTION FOR USER 3
============================================================

✓ User 3 found in dataset
✗ Model not found for user 3
→ Training new model...
✓ Loaded 12 months of data
  Date range: 2025-01 to 2025-12
Warning: Only 12 observations. Using simple exponential smoothing.
Model saved to src/model_user_3.pkl
✓ Model trained and saved to src/model_user_3.pkl
✓ Model AIC: 182.43

============================================================
PREDICTION RESULTS
============================================================

Last Month (2025-12):
  Savings: ₹11,669.91
  End Balance: ₹147,093.01

Historical Statistics:
  Average Monthly Savings: ₹12,257.75
  Std Dev: ₹1,630.84

────────────────────────────────────────────────────────────
PREDICTED NEXT MONTH SAVINGS: ₹13,424.01
────────────────────────────────────────────────────────────

✓ Prediction completed successfully!

# Returned dictionary:
{
    'user_id': 3,
    'predicted_next_month_savings': 13424.01,
    'last_month_savings': 11669.91,
    'last_month_balance': 147093.01,
    'last_month_date': '2025-12',
    'average_savings': 12257.75,
    'std_savings': 1630.84,
    'n_months': 12,
    'model_filepath': 'src/model_user_3.pkl'
}
```

### Example 3: User Not in Dataset (User 99)

```python
result = main(99)

# Output:
============================================================
USER SAVINGS PREDICTION SYSTEM
Using Exponential Smoothing Time Series Analysis
============================================================

============================================================
SAVINGS PREDICTION FOR USER 99
============================================================

✗ Error: User 99 is not trained. User not found in dataset. Available users: [0, 1, 2, 3, 4, 5]

# Returned dictionary:
{
    'error': 'User 99 is not trained. User not found in dataset. Available users: [0, 1, 2, 3, 4, 5]',
    'user_id': 99,
    'success': False
}
```

## Module Details

### 1. Data Engineering (`src/data_engineering.py`)

Handles all data processing tasks:
- **`load_transaction_data()`**: Load CSV data
- **`preprocess_data()`**: Clean and prepare data
- **`calculate_monthly_savings()`**: Calculate savings per month per user
- **`prepare_timeseries_data()`**: Format data for time series analysis
- **`get_user_savings_timeseries()`**: Complete pipeline for user data
- **`validate_user_exists()`**: Check if user exists
- **`get_available_users()`**: List all available users

### 2. Machine Learning (`src/ml_model.py`)

Implements Exponential Smoothing forecasting:
- **`SavingsPredictor`**: Main model class
  - `train()`: Train the model on time series data
  - `predict()`: Forecast future savings
  - `save_model()`: Save trained model to pickle file
  - `load_model()`: Load saved model from pickle file
  - `get_model_summary()`: Get model statistics (AIC, BIC, etc.)
- **`train_and_save_model()`**: Helper function to train and save
- **`load_and_predict()`**: Helper function to load and predict

### 3. Main Orchestration (`main.py`)

Coordinates the entire pipeline:
- **`predict_user_savings(user_id)`**: Complete prediction pipeline
  - Validates user exists in dataset
  - Checks if model file exists
  - If model exists: loads and predicts
  - If model doesn't exist: trains, saves, and predicts
  - Returns results dictionary
- **`main(user_id)`**: Main entry point with error handling

## How It Works

1. **User Input**: Function receives user_id as parameter
2. **Validation**: Checks if user exists in dataset
3. **Model Check**: Verifies if pre-trained model exists
4. **Smart Loading/Training**:
   - If model exists → Load and predict
   - If model doesn't exist → Train, save, and predict
5. **Data Loading**: Loads historical transaction data for context
6. **Prediction**: Model forecasts next month's savings
7. **Results**: Returns comprehensive dictionary with prediction and statistics

## Savings Calculation

Savings for each month is calculated as:
```
Savings = End of Month Balance - Start of Month Balance
```

- **Positive savings**: User saved money (balance increased)
- **Negative savings**: User spent more than earned (balance decreased)

## Model Details

- **Algorithm**: Exponential Smoothing (Holt-Winters)
- **Components**: 
  - Trend: Additive
  - Seasonal: Additive (if sufficient data, ≥24 observations)
  - Simple: If insufficient data (<24 observations)
- **Seasonal Period**: 12 months
- **Optimization**: Automatic parameter optimization
- **Evaluation Metrics**: AIC, BIC, AICc

## Data Format

The input CSV should have the following columns:
- `User`: User ID (integer)
- `Card`: Card number
- `Year`: Transaction year
- `Month`: Transaction month
- `Day`: Transaction day
- `Time`: Transaction time
- `Amount`: Transaction amount
- `Cr_Dr`: Credit or Debit indicator
- `Balance`: Account balance after transaction

## Available Users

The dataset contains users 0-5. Models are created automatically on first use:
- User 0: ✓ Model available
- User 1-5: Models created on first prediction request

## Model Persistence

Trained models are saved in the `src/` directory as:
- `model_user_0.pkl`
- `model_user_1.pkl`
- etc.

Each model file contains:
- Fitted Exponential Smoothing model
- Training data
- Model parameters

Models are automatically created and saved when a user is predicted for the first time.

## Return Values

### Success Response
```python
{
    'user_id': int,
    'predicted_next_month_savings': float,
    'last_month_savings': float,
    'last_month_balance': float,
    'last_month_date': str,
    'average_savings': float,
    'std_savings': float,
    'n_months': int,
    'model_filepath': str
}
```

### Error Response (User Not in Dataset)
```python
{
    'error': str,
    'user_id': int,
    'success': False
}
```

## Requirements

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- statsmodels >= 0.14.0
- scipy >= 1.10.0

## Notes

- Models are automatically trained on first use if they don't exist
- Only returns error if user doesn't exist in the dataset
- Models are saved as pickle files for fast loading on subsequent predictions
- All monetary values are displayed in Indian Rupees (₹)
- The system handles insufficient data gracefully by using simpler models