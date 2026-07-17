"""
Unit tests for Flask API (app.py)
"""
import json
from unittest.mock import patch, MagicMock

import numpy as np
import pytest  # pylint: disable=unused-import


class TestSavingsEndpoint:
    """Tests for /savings POST endpoint"""

    def test_savings_endpoint_success(self, flask_client):
        """Test successful prediction request"""
        with patch('app.predict_savings') as mock_predict:
            mock_predict.return_value = {
                'user_id': 1,
                'user_details': {'User ID': 1, 'Person': 'Sasha Sadr'},
                'predicted_next_month_savings': 15000.50,
                'last_month_savings': 14500.00,
                'last_month_balance': 50000.00,
                'last_month_date': '2025-12',
                'average_savings': 14000.00,
                'std_savings': 2000.00,
                'n_months': 12,
                'model_filepath': 'src/models/model_user_1.pkl'
            }

            response = flask_client.post(
                '/savings',
                data=json.dumps({'user_id': 1}),
                content_type='application/json'
            )

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True
            assert data['user_id'] == 1
            assert 'predicted_next_month_savings' in data
            assert data['user_details']['Person'] == 'Sasha Sadr'

    def test_savings_endpoint_missing_user_id(self, flask_client):
        """Test request without user_id"""
        response = flask_client.post(
            '/savings',
            data=json.dumps({'other_field': 'value'}),
            content_type='application/json'
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        # The actual error message from app.py line 59
        assert 'missing required parameter' in data['error'].lower()

    def test_savings_endpoint_invalid_user_id_type(self, flask_client):
        """Test request with invalid user_id type"""
        response = flask_client.post(
            '/savings',
            data=json.dumps({'user_id': 'invalid'}),
            content_type='application/json'
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'must be an integer' in data['error']

    def test_savings_endpoint_no_json_body(self, flask_client):
        """Test request without JSON body"""
        response = flask_client.post(
            '/savings',
            data='not json',
            content_type='text/plain'
        )

        # Flask will return 415 for wrong content type or 400 for invalid JSON
        assert response.status_code in [400, 415, 500]

    def test_savings_endpoint_user_not_found(self, flask_client):
        """Test request for non-existent user"""
        with patch('app.predict_savings') as mock_predict:
            mock_predict.return_value = {
                'error': 'User 999 is not trained. User not found in dataset.',
                'user_id': 999,
                'success': False
            }

            response = flask_client.post(
                '/savings',
                data=json.dumps({'user_id': 999}),
                content_type='application/json'
            )

            assert response.status_code == 400
            data = json.loads(response.data)
            assert data['success'] is False
            assert 'error' in data

    def test_savings_endpoint_internal_error(self, flask_client):
        """Test handling of internal server errors"""
        with patch('app.predict_savings') as mock_predict:
            mock_predict.side_effect = Exception("Internal error")

            response = flask_client.post(
                '/savings',
                data=json.dumps({'user_id': 1}),
                content_type='application/json'
            )

            assert response.status_code == 500
            data = json.loads(response.data)
            assert data['success'] is False
            assert 'Internal server error' in data['error']

    def test_savings_endpoint_string_user_id_conversion(self, flask_client):
        """Test that string user_id is converted to integer"""
        with patch('app.predict_savings') as mock_predict:
            mock_predict.return_value = {
                'user_id': 1,
                'user_details': {'User ID': 1, 'Person': 'Sasha Sadr'},
                'predicted_next_month_savings': 15000.50
            }

            response = flask_client.post(
                '/savings',
                data=json.dumps({'user_id': '1'}),
                content_type='application/json'
            )

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['user_id'] == 1


class TestHealthEndpoint:
    """Tests for /health GET endpoint"""

    def test_health_check(self, flask_client):
        """Test health check endpoint"""
        response = flask_client.get('/health')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'service' in data

    def test_health_check_returns_json(self, flask_client):
        """Test that health check returns JSON"""
        response = flask_client.get('/health')

        assert response.content_type == 'application/json'


class TestIndexEndpoint:
    """Tests for / GET endpoint"""

    def test_index_endpoint(self, flask_client):
        """Test index/documentation endpoint"""
        response = flask_client.get('/')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'service' in data
        assert 'version' in data
        assert 'endpoints' in data

    def test_index_has_documentation(self, flask_client):
        """Test that index provides API documentation"""
        response = flask_client.get('/')

        data = json.loads(response.data)
        assert 'POST /savings' in data['endpoints']
        assert 'GET /health' in data['endpoints']

    def test_index_has_example(self, flask_client):
        """Test that index includes usage example"""
        response = flask_client.get('/')

        data = json.loads(response.data)
        assert 'example' in data['endpoints']['POST /savings']


class TestAPIValidation:
    """Tests for API input validation"""

    def test_negative_user_id(self, flask_client):
        """Test that negative user_id is handled"""
        with patch('app.predict_savings') as mock_predict:
            mock_predict.return_value = {
                'error': 'User -1 is not trained.',
                'user_id': -1,
                'success': False
            }

            response = flask_client.post(
                '/savings',
                data=json.dumps({'user_id': -1}),
                content_type='application/json'
            )

            # Should accept but return error from prediction
            assert response.status_code == 400

    def test_zero_user_id(self, flask_client):
        """Test that user_id 0 is valid"""
        with patch('app.predict_savings') as mock_predict:
            mock_predict.return_value = {
                'user_id': 0,
                'user_details': {'User ID': 0, 'Person': 'Hazel Robinson'},
                'predicted_next_month_savings': 7000.00
            }

            response = flask_client.post(
                '/savings',
                data=json.dumps({'user_id': 0}),
                content_type='application/json'
            )

            assert response.status_code == 200

    def test_large_user_id(self, flask_client):
        """Test handling of very large user_id"""
        with patch('app.predict_savings') as mock_predict:
            mock_predict.return_value = {
                'error': 'User not found',
                'user_id': 999999,
                'success': False
            }

            response = flask_client.post(
                '/savings',
                data=json.dumps({'user_id': 999999}),
                content_type='application/json'
            )

            assert response.status_code == 400


class TestResponseFormat:
    """Tests for API response format"""

    def test_success_response_format(self, flask_client):
        """Test that success response has correct format"""
        with patch('app.predict_savings') as mock_predict:
            mock_predict.return_value = {
                'user_id': 1,
                'user_details': {'User ID': 1, 'Person': 'Sasha Sadr'},
                'predicted_next_month_savings': 15000.50,
                'last_month_savings': 14500.00,
                'last_month_balance': 50000.00,
                'last_month_date': '2025-12',
                'average_savings': 14000.00,
                'std_savings': 2000.00,
                'n_months': 12,
                'model_filepath': 'src/models/model_user_1.pkl'
            }

            response = flask_client.post(
                '/savings',
                data=json.dumps({'user_id': 1}),
                content_type='application/json'
            )

            data = json.loads(response.data)

            # Check all expected fields are present
            assert 'success' in data
            assert 'user_id' in data
            assert 'predicted_next_month_savings' in data
            assert 'last_month_savings' in data
            assert 'average_savings' in data
            assert 'user_details' in data

    def test_error_response_format(self, flask_client):
        """Test that error response has correct format"""
        response = flask_client.post(
            '/savings',
            data=json.dumps({}),
            content_type='application/json'
        )

        data = json.loads(response.data)

        # Check error response format
        assert 'success' in data
        assert data['success'] is False
        assert 'error' in data

    def test_response_is_json(self, flask_client):
        """Test that all responses are JSON"""
        with patch('app.predict_savings') as mock_predict:
            mock_predict.return_value = {
                'user_id': 1,
                'user_details': {'User ID': 1, 'Person': 'Sasha Sadr'}
            }

            response = flask_client.post(
                '/savings',
                data=json.dumps({'user_id': 1}),
                content_type='application/json'
            )

            assert response.content_type == 'application/json'


class TestPredictEndpoint:
    """Tests for /predict POST endpoint"""

    def test_predict_endpoint_success(self, flask_client):
        """Test successful prediction request"""
        with patch('app.savings_plan_model') as mock_model:
            mock_model.predict.return_value = np.array(['Plan A'])

            response = flask_client.post(
                '/predict',
                data=json.dumps({
                    'features': [45, 65, 1979, 6, 'Male', 'New York', 'NY', '10001',
                                75000, 120000, 50000, 720, 'Plan A']
                }),
                content_type='application/json'
            )

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True
            assert 'prediction' in data
            assert data['prediction'] == ['Plan A']

    def test_predict_endpoint_missing_features(self, flask_client):
        """Test request without features parameter"""
        with patch('app.savings_plan_model') as mock_model:
            mock_model.return_value = MagicMock()

            response = flask_client.post(
                '/predict',
                data=json.dumps({'other_field': 'value'}),
                content_type='application/json'
            )

            assert response.status_code == 400
            data = json.loads(response.data)
            assert data['success'] is False
            assert 'Missing required parameter: features' in data['error']

    def test_predict_endpoint_features_not_list(self, flask_client):
        """Test request with features not as a list"""
        with patch('app.savings_plan_model') as mock_model:
            mock_model.return_value = MagicMock()

            response = flask_client.post(
                '/predict',
                data=json.dumps({'features': 'not a list'}),
                content_type='application/json'
            )

            assert response.status_code == 400
            data = json.loads(response.data)
            assert data['success'] is False
            assert 'features must be a list' in data['error']

    def test_predict_endpoint_model_not_loaded(self, flask_client):
        """Test request when model is not loaded"""
        with patch('app.savings_plan_model', None):
            response = flask_client.post(
                '/predict',
                data=json.dumps({
                    'features': [45, 65, 1979, 6, 'Male', 'New York', 'NY', '10001',
                                75000, 120000, 50000, 720, 'Plan A']
                }),
                content_type='application/json'
            )

            assert response.status_code == 500
            data = json.loads(response.data)
            assert data['success'] is False
            assert 'model not loaded' in data['error'].lower()

    def test_predict_endpoint_prediction_error(self, flask_client):
        """Test handling of prediction errors"""
        with patch('app.savings_plan_model') as mock_model:
            mock_model.predict.side_effect = Exception("Prediction failed")

            response = flask_client.post(
                '/predict',
                data=json.dumps({
                    'features': [45, 65, 1979, 6, 'Male', 'New York', 'NY', '10001',
                                75000, 120000, 50000, 720, 'Plan A']
                }),
                content_type='application/json'
            )

            assert response.status_code == 500
            data = json.loads(response.data)
            assert data['success'] is False
            assert 'error' in data

    def test_predict_endpoint_multiple_predictions(self, flask_client):
        """Test prediction with multiple classes"""
        with patch('app.savings_plan_model') as mock_model:
            mock_model.predict.return_value = np.array(['Plan B'])

            response = flask_client.post(
                '/predict',
                data=json.dumps({
                    'features': [50, 67, 1974, 3, 'Female', 'Los Angeles', 'CA', '90001',
                                85000, 150000, 30000, 750, 'Plan B']
                }),
                content_type='application/json'
            )

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True
            assert len(data['prediction']) == 1

    def test_predict_endpoint_empty_features(self, flask_client):
        """Test request with empty features list"""
        with patch('app.savings_plan_model') as mock_model:
            mock_model.predict.side_effect = ValueError("Empty features")

            response = flask_client.post(
                '/predict',
                data=json.dumps({'features': []}),
                content_type='application/json'
            )

            assert response.status_code == 500
            data = json.loads(response.data)
            assert data['success'] is False

    def test_predict_endpoint_invalid_json(self, flask_client):
        """Test request with invalid JSON"""
        response = flask_client.post(
            '/predict',
            data='not valid json',
            content_type='application/json'
        )

        assert response.status_code in [400, 500]

    def test_predict_endpoint_response_format(self, flask_client):
        """Test that response has correct format"""
        with patch('app.savings_plan_model') as mock_model:
            mock_model.predict.return_value = np.array(['Plan A'])

            response = flask_client.post(
                '/predict',
                data=json.dumps({
                    'features': [45, 65, 1979, 6, 'Male', 'New York', 'NY', '10001',
                                75000, 120000, 50000, 720, 'Plan A']
                }),
                content_type='application/json'
            )

            data = json.loads(response.data)
            assert 'success' in data
            assert 'prediction' in data
            assert isinstance(data['prediction'], list)

    def test_predict_endpoint_numeric_features(self, flask_client):
        """Test prediction with all numeric features"""
        with patch('app.savings_plan_model') as mock_model:
            mock_model.predict.return_value = np.array(['Plan C'])

            response = flask_client.post(
                '/predict',
                data=json.dumps({
                    'features': [30, 60, 1994, 12, 'Other', 'Chicago', 'IL', '60601',
                                60000, 90000, 20000, 680, 'Plan C']
                }),
                content_type='application/json'
            )

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True


class TestOpenApiEndpoint:  # pylint: disable=too-few-public-methods
    """Tests for /openapi.yaml GET endpoint"""

    def test_serve_openapi_spec(self, flask_client):
        """Test that /openapi.yaml serves the spec file"""
        response = flask_client.get('/openapi.yaml')
        # File should be served (200) or not found (404) depending on environment;
        # either way the route executes — we just ensure it doesn't 500.
        assert response.status_code in [200, 404]


class TestSavingsEndpointInnerExcept:  # pylint: disable=too-few-public-methods
    """Tests for the inner bare-except branch in /savings error handler (lines 147-148)"""

    def test_savings_inner_except_fires_when_data_get_raises(self, flask_client):
        """Trigger the bare-except: data.get() raises while inside the outer except block."""
        class _BadData:  # pylint: disable=too-few-public-methods
            """Truthy dict-like whose .get() raises, so the bare-except on line 147 fires."""
            def __contains__(self, item):
                return True  # pass the 'user_id' in data check

            def __getitem__(self, key):
                raise RuntimeError("getitem failed")

            def get(self, key, default=None):  # pylint: disable=missing-function-docstring
                raise RuntimeError("get failed")

            def __bool__(self):
                return True

        with patch('flask.Request.get_json', return_value=_BadData()), \
             patch('app.predict_savings') as mock_predict:
            # predict_savings raises so we reach the outer except
            mock_predict.side_effect = Exception("boom")

            response = flask_client.post(
                '/savings',
                data='{}',
                content_type='application/json'
            )
            # The outer except ran; user_id is null because inner except fired
            assert response.status_code == 500
            data = json.loads(response.data)
            assert data['success'] is False
            assert data['user_id'] is None


class TestPredictEndpointNullBody:  # pylint: disable=too-few-public-methods
    """Tests for /predict endpoint when parsed JSON is None/falsy"""

    def test_predict_null_body_returns_400(self, flask_client):
        """POST /predict with JSON 'null' body parses to None, triggering the 'not data' branch"""
        with patch('app.savings_plan_model') as mock_model:
            mock_model.predict = MagicMock()
            # JSON 'null' → Python None (falsy), hitting the `if not data` branch
            response = flask_client.post(
                '/predict',
                data=b'null',
                content_type='application/json'
            )
            assert response.status_code == 400
            data = json.loads(response.data)
            assert data['success'] is False
            assert 'Request body must be JSON' in data['error']


# Made with Bob


class TestModelLoadFailure:  # pylint: disable=too-few-public-methods
    """Covers the except branch when joblib.load raises at module import time."""

    def test_model_load_exception_is_caught(self):
        """Reload app with joblib.load patched to raise, covering the except print branch."""
        import importlib  # pylint: disable=import-outside-toplevel
        import app as app_module  # pylint: disable=import-outside-toplevel

        with patch('joblib.load', side_effect=OSError("no model")):
            importlib.reload(app_module)
            assert app_module.savings_plan_model is None
