"""
Unit tests for Flask API (app.py)
"""
import pytest
import json
from unittest.mock import patch, MagicMock


class TestSavingsEndpoint:
    """Tests for /savings POST endpoint"""
    
    def test_savings_endpoint_success(self, flask_client):
        """Test successful prediction request"""
        with patch('app.predict_savings') as mock_predict:
            mock_predict.return_value = {
                'user_id': 1,
                'predicted_next_month_savings': 15000.50,
                'last_month_savings': 14500.00,
                'last_month_balance': 50000.00,
                'last_month_date': '2025-12',
                'average_savings': 14000.00,
                'std_savings': 2000.00,
                'n_months': 12,
                'model_filepath': 'src/model_user_1.pkl'
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
                'predicted_next_month_savings': 15000.50,
                'last_month_savings': 14500.00,
                'last_month_balance': 50000.00,
                'last_month_date': '2025-12',
                'average_savings': 14000.00,
                'std_savings': 2000.00,
                'n_months': 12,
                'model_filepath': 'src/model_user_1.pkl'
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
            mock_predict.return_value = {'user_id': 1}
            
            response = flask_client.post(
                '/savings',
                data=json.dumps({'user_id': 1}),
                content_type='application/json'
            )
            
            assert response.content_type == 'application/json'


