# TCC Timeseries - User Savings Prediction System

A machine learning system for predicting user savings using time series analysis,
with an additional savings plan classification feature.

## 🌟 Features

### 1. **Savings Prediction API**

- Predict next month's savings for users using Exponential Smoothing
- Automatic model training for new users
- RESTful API with Flask
- Historical savings analysis and statistics

### 2. **Savings Plan Classifier**

- Machine learning classifier for retirement savings plans
- Handles demographic and financial features
- Logistic regression with preprocessing pipeline

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Swagger UI](#swagger-ui)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Testing](#testing)
- [Data Requirements](#data-requirements)
- [Model Details](#model-details)
- [Contributing](#contributing)

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd "TCC Timeseries"
```

2. **Create virtual environment**

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a `.env` file in the project root with your watsonx.data Iceberg credentials:

```bash
LH_USER=your_iceberg_username
LH_PW=your_iceberg_password
```

> **Note:** Never commit `.env` to version control — it is already listed in `.gitignore`.

1. **Install test dependencies (optional)**

```bash
pip install -r requirements-test.txt
```

## Quick Start

### Running the API Server

```bash
python app.py
```

The API will be available at `http://localhost:9476`

### Making a Prediction

```bash
curl -X POST http://localhost:9476/savings \
  -H "Content-Type: application/json" \
  -d '{"user_id": 6}'
```

### Command Line Usage

```bash
python main.py
```

Edit the `user_id` in `main.py` to predict for different users.

## API Documentation

### Base URL

```text
http://localhost:9476
```

### Endpoints

#### 1. **GET /** - API Information

Returns service information and a list of available endpoints.

**Response:**

```json
{
  "service": "User Savings Prediction API",
  "version": "1.0.0",
  "endpoints": { "..." }
}
```

#### 2. **GET /health** - Health Check

Check if the API is running.

**Response:**

```json
{
  "status": "healthy",
  "service": "User Savings Prediction API"
}
```

#### 3. **POST /savings** - Predict User Savings

Predict next month's savings for a specific user.

**Request:**

```json
{
  "user_id": 6
}
```

**Response:**

```json
{
  "success": true,
  "user_id": 6,
  "predicted_next_month_savings": 15000.50,
  "last_month_savings": 14500.00,
  "last_month_balance": 125000.00,
  "last_month_date": "2025-12",
  "average_savings": 14000.00,
  "std_savings": 2500.00,
  "n_months": 12,
  "model_filepath": "src/models/model_user_6.pkl",
  "user_details": { "..." }
}
```

#### 4. **POST /predict** - Savings Plan Classification

Predict the savings plan for a user based on their demographic and financial features.

**Request:**

```json
{
  "features": [
    45, 65, 1979, 6, "Male", "New York", "NY",
    "10001", 75000, 120000, 50000, 720, "Plan A"
  ]
}
```

**Response:**

```json
{
  "success": true,
  "prediction": ["Plan A"]
}
```

## Swagger UI

The API ships with an interactive [Swagger UI](https://swagger.io/tools/swagger-ui/) powered by
[`flask-swagger-ui`](https://github.com/sveint/flask-swagger-ui) and an
[OpenAPI 3.0](https://swagger.io/specification/) specification file.

### Accessing the Swagger UI

Start the server and open the following URL in your browser:

```text
http://localhost:9476/docs
```

The Swagger UI lets you:

- Browse all available endpoints with their full request/response schemas
- Execute live API calls directly from the browser (no extra tooling needed)
- Inspect example request bodies and expected responses

### Raw OpenAPI Specification

The machine-readable OpenAPI 3.0 YAML spec is served at:

```text
http://localhost:9476/openapi.yaml
```

You can import this file into tools like Postman, Insomnia, or any OpenAPI-compatible client.

### Swagger UI Screenshot

| URL | Purpose |
|-----|---------|
| `http://localhost:9476/docs` | Interactive Swagger UI |
| `http://localhost:9476/openapi.yaml` | Raw OpenAPI 3.0 spec (YAML) |

## Project Structure

```text
TCC Timeseries/
├── app.py                          # Flask API server
├── main.py                         # Main orchestration script
├── openapi.yaml                    # OpenAPI 3.0 specification
├── requirements.txt                # Production dependencies
├── requirements-test.txt           # Test dependencies
├── pytest.ini                      # Pytest configuration
├── README.md                       # This file
│
├── src/                            # Source code
│   ├── __init__.py
│   ├── data_engineering.py         # Data loading and preprocessing
│   ├── ml_model.py                 # Time series ML model
│   ├── train_savings_plan_classifier.py  # Classifier training
│   │
│   ├── data/                       # Data files
│   │   ├── sd254_users.csv         # User profile data
│   │   ├── synthetic_retirement_data.csv
│   │   └── transactions_2025_salary_filtered.csv
│   │
│   └── models/                     # Trained models
│       ├── model_user_*.pkl        # User-specific models
│       └── savings_plan_classifier.pkl
│
└── tests/                          # Test suite
    ├── __init__.py
    ├── conftest.py                 # Pytest fixtures
    ├── test_app.py                 # API tests
    ├── test_main.py                # Orchestration tests
    ├── test_data_engineering.py    # Data pipeline tests
    ├── test_ml_model.py            # ML model tests
    ├── test_train_savings_plan_classifier.py
    ├── README.md                   # Test documentation
    └── TEST_COVERAGE_SUMMARY.md    # Coverage report
```

## Usage Examples

### Example 1: Predict Savings for User

```python
from main import predict_user_savings

# Predict for user ID 6
result = predict_user_savings(user_id=6)

print(f"Predicted savings: ₹{result['predicted_next_month_savings']:,.2f}")
print(f"Average savings: ₹{result['average_savings']:,.2f}")
```

### Example 2: Train and Save Model

```python
from src.data_engineering import get_user_savings_timeseries
from src.ml_model import train_and_save_model

# Get time series data
ts_data, monthly_df = get_user_savings_timeseries(
    'src/data/transactions_2025_salary_filtered.csv',
    user_id=6
)

# Train and save model
predictor, filepath = train_and_save_model(ts_data, user_id=6)
print(f"Model saved to: {filepath}")
```

### Example 3: Load Model and Predict

```python
from src.ml_model import load_and_predict

# Load existing model and predict
next_month, forecast = load_and_predict(
    'src/models/model_user_6.pkl',
    steps=3  # Predict 3 months ahead
)

print(f"Next month: ₹{next_month:,.2f}")
print(f"3-month forecast: {forecast.tolist()}")
```

## Testing

The project includes comprehensive unit tests with **119 tests** and **100% coverage**.

### Install Test Dependencies

```bash
pip install -r requirements-test.txt
```

### Run All Tests

```bash
pytest
```

### Run with Verbose Output

```bash
pytest -v
```

### Run with Coverage Report (terminal)

```bash
pytest --cov=app --cov=main --cov=src --cov-report=term-missing
```

### Run with Coverage Report (HTML)

```bash
pytest --cov=app --cov=main --cov=src --cov-report=html
# Report is generated at htmlcov/index.html
```

### Run a Specific Test File

```bash
pytest tests/test_app.py -v
pytest tests/test_main.py -v
pytest tests/test_data_engineering.py -v
pytest tests/test_ml_model.py -v
pytest tests/test_train_savings_plan_classifier.py -v
```

### Run a Specific Test Class or Case

```bash
pytest tests/test_app.py::TestSavingsEndpoint -v
pytest tests/test_app.py::TestSavingsEndpoint::test_savings_endpoint_success -v
```

### Run Tests Matching a Keyword

```bash
pytest -k "savings" -v
```

### Test Coverage Summary

| File | Tests | Coverage |
| ---- | ----- | -------- |
| `test_app.py` | 33 | `app.py` — 100% |
| `test_main.py` | 12 | `main.py` — 100% |
| `test_data_engineering.py` | 25 | `src/data_engineering.py` — 100% |
| `test_ml_model.py` | 25 | `src/ml_model.py` — 100% |
| `test_train_savings_plan_classifier.py` | 24 | `src/train_savings_plan_classifier.py` — 100% |

See [tests/README.md](tests/README.md) for detailed test documentation.

## Data Requirements

### Transaction Data Format

CSV file with columns:

- `User`: User ID (integer)
- `Year`, `Month`, `Day`: Date components
- `Time`: Transaction time
- `Amount`: Transaction amount
- `Balance`: Account balance after transaction

### User Data Format

CSV file with columns:

- `User ID`: User identifier
- `Current Age`, `Retirement Age`: Age information
- `Birth Year`, `Birth Month`: Birth information
- `Gender`, `City`, `State`, `Zipcode`: Demographics
- `Per Capita Income - Zipcode`: Income data
- `Yearly Income - Person`: Personal income
- `Total Debt`, `FICO Score`: Financial metrics
- `Savings Plan`: Savings plan category

## Model Details

### Exponential Smoothing Model

- **Algorithm**: Holt-Winters Exponential Smoothing
- **Components**: Trend (additive) + Seasonal (additive)
- **Seasonal Period**: 12 months
- **Optimization**: Automatic parameter optimization
- **Metrics**: AIC, BIC, AICC for model selection

### Savings Plan Classifier

- **Algorithm**: Logistic Regression
- **Preprocessing**:
  - Numeric features: Median imputation + Standard scaling
  - Categorical features: Constant imputation + One-hot encoding
- **Class Balancing**: Balanced class weights
- **Validation**: Stratified train-test split (80/20)

## 🔧 Configuration

### 🌐 Deployment Configuration (IP Address)

When deploying to a **remote server or Linux LPAR**, two files must be updated with the server's IP address:

#### 1. `openapi.yaml` — Swagger UI server target

Open [`openapi.yaml`](openapi.yaml) and find the `servers` block near the top (around line 15).
Replace `<YOUR-SERVER-IP>` with your server's actual IP and move it to be the **first entry** so Swagger UI uses it by default:

```yaml
servers:
  - url: http://192.168.1.100:9476      # ← your server IP here (first = default)
    description: Remote server / Linux LPAR
  - url: http://127.0.0.1:9476
    description: Local development server
```

#### 2. `app.py` — Flask bind address

Open [`app.py`](app.py) and find the last line (around line 318).
Replace `127.0.0.1` with the same server IP so Flask accepts external connections:

```python
app.run(debug=True, host='192.168.1.100', port=9476)  # ← your server IP here
```

> **Note:** Do not use `0.0.0.0` as the host — bind to the specific server IP only.
> Also ensure port `9476` is open in the server's firewall.

---

### Environment Variables

Create a `.env` file in the project root. The following variables are supported:

```bash
# watsonx.data Iceberg credentials (required)
LH_USER=your_iceberg_username   # Username for the Presto/Iceberg connection
LH_PW=your_iceberg_password     # Password for the Presto/Iceberg connection

# Flask configuration (optional)
FLASK_ENV=development
FLASK_DEBUG=True

# Model paths (optional)
MODEL_DIR=src/models
DATA_DIR=src/data
```

`LH_USER` and `LH_PW` are read by `src/data_engineering_iceberg.py` to authenticate against the watsonx.data Lakehouse. The API will fail to serve `/savings` requests if either variable is missing.

### Model Parameters

Edit in respective modules:

- `src/ml_model.py`: Exponential smoothing parameters
- `src/train_savings_plan_classifier.py`: Classifier parameters

## 🐛 Troubleshooting

### Common Issues

#### 1. Model not found error

- Solution: The system will automatically train a model for the user on first prediction

#### 2. User not found error

- Solution: Check that the user exists in the transaction data
- Use `get_available_users()` to see valid user IDs

#### 3. Import errors

- Solution: Ensure all dependencies are installed: `pip install -r requirements.txt`

## 📝 API Response Codes

- `200 OK`: Successful request
- `400 Bad Request`: Invalid input or user not found
- `500 Internal Server Error`: Server-side error

## 🔐 Security Considerations

- Input validation on all API endpoints
- No sensitive data in error messages

## 🚀 Performance

- **Model Training**: ~1-2 seconds per user
- **Prediction**: <100ms per request

## 📈 Future Enhancements

- [ ] Add authentication and authorization
- [ ] Implement caching for frequently accessed predictions
- [ ] Add support for multiple time series models
- [ ] Real-time model retraining
- [ ] Dashboard for visualization
- [ ] Batch prediction endpoints
- [ ] Model versioning and A/B testing

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is proprietary software. All rights reserved.

## 👥 Authors

- Development Team - TCC Timeseries Project

## 🙏 Acknowledgments

- Built with Flask, scikit-learn, and statsmodels
- Time series analysis with Holt-Winters Exponential Smoothing
- Interactive API docs via Swagger UI (flask-swagger-ui)

---

**Made with Bob** 🤖

For questions or support, please open an issue in the repository.
