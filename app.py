"""
Flask API for User Savings Prediction
Provides REST API endpoint to predict next month's savings for users
"""
# pylint: disable=duplicate-code

import logging
import os

import joblib
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_swagger_ui import get_swaggerui_blueprint
from pydantic import ValidationError

from config import SAVINGS_PLAN_MODEL_PATH
from main import main as predict_savings
from src.predict_request import PredictRequest

app = Flask(__name__)
logger = logging.getLogger(__name__)

# Swagger UI
SWAGGER_URL = '/docs'
API_SPEC_URL = '/openapi.yaml'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_SPEC_URL,
    config={'app_name': 'User Savings Prediction API'}
)
app.register_blueprint(swaggerui_blueprint)


@app.route('/openapi.yaml')
def serve_openapi_spec():
    """Serve the OpenAPI specification file."""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'openapi.yaml')


# Load the savings plan classifier model
SAVINGS_PLAN_MODEL = None

try:
    SAVINGS_PLAN_MODEL = joblib.load(SAVINGS_PLAN_MODEL_PATH)
    print(f"Loaded savings plan classifier from {SAVINGS_PLAN_MODEL_PATH}")  # pragma: no cover
except Exception as e:  # pylint: disable=broad-except
    print(f"Warning: Could not load savings plan classifier: {e}")



@app.route('/savings', methods=['POST'])
def predict_user_savings():
    """
    POST /savings endpoint

    Request Body (JSON):
    {
        "user_id": int  # Required: User ID to predict savings for
    }

    Response (JSON):
    Success (200):
    {
        "success": true,
        "user_id": int,
        "predicted_next_month_savings": float,
        "last_month_savings": float,
        "last_month_balance": float,
        "last_month_date": str,
        "average_savings": float,
        "std_savings": float,
        "n_months": int,
        "model_filepath": str
    }

    Error (400/500):
    {
        "success": false,
        "error": str,
        "user_id": int (if provided)
    }
    """
    data = None

    try:
        # Get JSON data from request
        data = request.get_json()

        # Validate request body
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body must be JSON'
            }), 400

        # Validate user_id parameter
        if 'user_id' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required parameter: user_id'
            }), 400

        user_id = data['user_id']

        # Validate user_id is an integer
        try:
            user_id = int(user_id)
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'error': 'user_id must be an integer',
                'user_id': user_id
            }), 400

        # Call the main prediction function
        result = predict_savings(user_id)

        # Check if prediction was successful
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error'],
                'user_id': user_id
            }), 400

        # Add success flag to result
        result['success'] = True

        return jsonify(result), 200

    except Exception as e:  # pylint: disable=broad-except
        # Handle unexpected errors
        try:
            user_id = data.get('user_id') if 'data' in locals() and data else None
        except Exception:  # pylint: disable=broad-except
            user_id = None

        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'user_id': user_id
        }), 500


@app.route('/predict', methods=['POST'])
def predict_savings_plan():
    """
    POST /predict endpoint

    Predict savings plan classification based on user features.

    Request Body (JSON), each field strictly typed:
    {
        "current_age": int, "retirement_age": int,
        "birth_year": int, "birth_month": int,
        "gender": str, "city": str, "state": str, "zipcode": str,
        "per_capita_income_zipcode": number,
        "yearly_income_person": number,
        "total_debt": number,
        "fico_score": int,
        "savings_plan": str  # Optional
    }

    Response (JSON):
    Success (200):
    {
        "success": true,
        "prediction": [predicted_class]
    }

    Error (400/500):
    {
        "success": false,
        "error": str
    }
    """
    # Check if model is loaded
    if SAVINGS_PLAN_MODEL is None:
        return jsonify({
            'success': False,
            'error': 'Savings plan classifier model not loaded'
        }), 500

    try:
        # Get JSON data from request
        data = request.get_json(force=True)

        # Validate request body
        if not data or not isinstance(data, dict):
            return jsonify({
                'success': False,
                'error': 'Request body must be a JSON object'
            }), 400

        # Validate field names and datatypes
        try:
            payload = PredictRequest.model_validate(data)
        except ValidationError as e:
            invalid_fields = sorted({
                '.'.join(str(part) for part in err['loc']) for err in e.errors()
            })
            return jsonify({
                'success': False,
                'error': 'Missing, unknown or wrongly typed fields: '
                         + ', '.join(invalid_fields)
            }), 400

        # Convert to DataFrame with the model's column names
        df = pd.DataFrame([payload.to_model_row()])

        # Predict
        prediction = SAVINGS_PLAN_MODEL.predict(df)

        return jsonify({
            'success': True,
            'prediction': prediction.tolist()
        }), 200

    except (ValueError, TypeError, AttributeError) as e:
        logger.error("Savings plan prediction failed: %s", e)
        return jsonify({
            'success': False,
            'error': 'Prediction failed due to invalid input data'
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    GET /health endpoint
    Simple health check to verify the API is running
    """
    return jsonify({
        'status': 'healthy',
        'service': 'User Savings Prediction API'
    }), 200


@app.route('/', methods=['GET'])
def index():
    """
    GET / endpoint
    API documentation and welcome message
    """
    return jsonify({
        'service': 'User Savings Prediction API',
        'version': '1.0.0',
        'endpoints': {
            'POST /savings': {
                'description': 'Predict next month savings for a user',
                'parameters': {
                    'user_id': 'integer (required)'
                },
                'example': {
                    'request': {'user_id': 1},
                    'response': {
                        'success': True,
                        'user_id': 1,
                        'predicted_next_month_savings': 15000.50,
                        'last_month_savings': 14500.00,
                        'average_savings': 14000.00
                    }
                }
            },
            'POST /predict': {
                'description': 'Predict savings plan classification based on user features',
                'parameters': {
                    'current_age': 'integer (required)',
                    'retirement_age': 'integer (required)',
                    'birth_year': 'integer (required)',
                    'birth_month': 'integer (required)',
                    'gender': 'string (required)',
                    'city': 'string (required)',
                    'state': 'string (required)',
                    'zipcode': 'string (required)',
                    'per_capita_income_zipcode': 'number (required)',
                    'yearly_income_person': 'number (required)',
                    'total_debt': 'number (required)',
                    'fico_score': 'integer (required)',
                    'savings_plan': 'string (optional)'
                },
                'example': {
                    'request': {
                        'current_age': 45,
                        'retirement_age': 65,
                        'birth_year': 1979,
                        'birth_month': 6,
                        'gender': 'Male',
                        'city': 'New York',
                        'state': 'NY',
                        'zipcode': '10001',
                        'per_capita_income_zipcode': 75000,
                        'yearly_income_person': 120000,
                        'total_debt': 50000,
                        'fico_score': 720
                    },
                    'response': {
                        'success': True,
                        'prediction': ['Plan A']
                    }
                }
            },
            'GET /health': {
                'description': 'Health check endpoint'
            },
            'GET /docs': {
                'description': 'Swagger UI — interactive API documentation'
            },
            'GET /openapi.yaml': {
                'description': 'Raw OpenAPI 3.0 specification'
            }
        }
    }), 200


if __name__ == '__main__':  # pragma: no cover
    app.run(debug=True, host='0.0.0.0', port=9476)

# Made with Bob
