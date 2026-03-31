"""
Flask API for User Savings Prediction
Provides REST API endpoint to predict next month's savings for users
"""

from flask import Flask, request, jsonify
from main import main as predict_savings

app = Flask(__name__)


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
        
    except Exception as e:
        # Handle unexpected errors
        #error_trace = traceback.format_exc()
        #print(f"Unexpected error in /savings endpoint: {error_trace}")
        
        try:
            user_id = data.get('user_id') if 'data' in locals() and data else None
        except:
            user_id = None
        
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'user_id': user_id
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
            'GET /health': {
                'description': 'Health check endpoint'
            }
        }
    }), 200


if __name__ == '__main__':
    # Run the Flask app
    # Debug mode is enabled for development
    # In production, use a proper WSGI server like gunicorn
    app.run(debug=True, host='0.0.0.0', port=9476)

# Made with Bob
