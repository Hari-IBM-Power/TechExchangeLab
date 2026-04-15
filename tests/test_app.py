"""
Unit tests for Flask API (app.py)
"""
import io
import pytest
import json
import numpy as np
from unittest.mock import patch, MagicMock
from src.pdf_embedding_pipeline import PDFEmbeddingPipelineError


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


class TestPDFEmbeddingsEndpoint:
    """Tests for /pdf/embeddings POST endpoint"""

    def test_pdf_embeddings_success(self, flask_client):
        """Test successful PDF embedding request"""
        with patch('app.process_pdf_file') as mock_process:
            mock_process.return_value = {
                'document_name': 'sample.pdf',
                'markdown_path': '/tmp/sample.md',
                'collection_name': 'docs',
                'collection_status': 'created',
                'chunk_count': 5,
                'average_tokens_per_chunk': 320.4,
                'inserted_count': 5,
                'db_path': './db',
                'embedding_model': 'sentence-transformers/all-mpnet-base-v2'
            }

            response = flask_client.post(
                '/pdf/embeddings',
                data={
                    'file': (io.BytesIO(b'%PDF-1.4 sample'), 'sample.pdf'),
                    'collection_name': 'docs',
                    'db_path': './db'
                },
                content_type='multipart/form-data'
            )

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True
            assert data['document_name'] == 'sample.pdf'
            assert data['chunk_count'] == 5

    def test_pdf_embeddings_missing_file(self, flask_client):
        """Test request without file"""
        response = flask_client.post(
            '/pdf/embeddings',
            data={'collection_name': 'docs'},
            content_type='multipart/form-data'
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'Missing required file field' in data['error']

    def test_pdf_embeddings_invalid_extension(self, flask_client):
        """Test request with non-PDF file"""
        response = flask_client.post(
            '/pdf/embeddings',
            data={'file': (io.BytesIO(b'hello'), 'notes.txt')},
            content_type='multipart/form-data'
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'must be a PDF' in data['error']

    def test_pdf_embeddings_pipeline_error(self, flask_client):
        """Test handled pipeline failure"""
        with patch('app.process_pdf_file') as mock_process:
            mock_process.side_effect = PDFEmbeddingPipelineError('Conversion failed')

            response = flask_client.post(
                '/pdf/embeddings',
                data={'file': (io.BytesIO(b'%PDF-1.4 sample'), 'sample.pdf')},
                content_type='multipart/form-data'
            )

            assert response.status_code == 400
            data = json.loads(response.data)
            assert data['success'] is False
            assert 'Conversion failed' in data['error']

    def test_pdf_embeddings_internal_error(self, flask_client):
        """Test unexpected server error"""
        with patch('app.process_pdf_file') as mock_process:
            mock_process.side_effect = Exception('Unexpected failure')

            response = flask_client.post(
                '/pdf/embeddings',
                data={'file': (io.BytesIO(b'%PDF-1.4 sample'), 'sample.pdf')},
                content_type='multipart/form-data'
            )

            assert response.status_code == 500
            data = json.loads(response.data)
            assert data['success'] is False
            assert 'Internal server error' in data['error']




class TestPredictEndpoint:
    """Tests for /predict POST endpoint"""
    
    def test_predict_endpoint_success(self, flask_client):
        """Test successful prediction request"""
        import numpy as np
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
        import numpy as np
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
        import numpy as np
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
        import numpy as np
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


# Made with Bob


class TestRetrieveEndpoint:
    """Tests for /retrieve POST endpoint"""
    
    def test_retrieve_endpoint_success(self, flask_client):
        """Test successful document retrieval and reranking"""
        mock_documents = [
            {
                'content': 'Document content 1',
                'metadata': {'filename': 'doc1', 'chunk_index': 0},
                'distance': 0.1,
                'rank': 1,
            },
            {
                'content': 'Document content 2',
                'metadata': {'filename': 'doc2', 'chunk_index': 1},
                'distance': 0.2,
                'rank': 2,
            },
        ]
        
        with patch('src.document_retrieval.retrieve_documents') as mock_retrieve, \
             patch('src.document_retrieval.rerank_documents') as mock_rerank:
            
            mock_retrieve.return_value = mock_documents
            mock_rerank.return_value = mock_documents[:1]  # Return top 1 after reranking
            
            response = flask_client.post(
                '/retrieve',
                data=json.dumps({
                    'query': 'test query',
                    'collection_name': 'test_collection',
                    'top_k': 20,
                    'rerank_top_k': 5,
                    'include_scores': True,
                    'include_metadata': True
                }),
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True
            assert data['query'] == 'test query'
            assert len(data['chunks']) == 1
            assert data['total_retrieved'] == 2
            assert data['total_reranked'] == 1
            assert 'timestamp' in data
            
            # Verify retrieve_documents was called with correct parameters
            mock_retrieve.assert_called_once_with(
                query='test query',
                collection_name='test_collection',
                top_k=20,
            )
            
            # Verify rerank_documents was called
            mock_rerank.assert_called_once()
    
    def test_retrieve_endpoint_missing_query(self, flask_client):
        """Test request without query parameter"""
        response = flask_client.post(
            '/retrieve',
            data=json.dumps({'collection_name': 'test'}),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'query' in data['error'].lower()
    
    def test_retrieve_endpoint_empty_query(self, flask_client):
        """Test request with empty query string"""
        response = flask_client.post(
            '/retrieve',
            data=json.dumps({'query': '   '}),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'non-empty' in data['error'].lower()
    
    def test_retrieve_endpoint_invalid_top_k(self, flask_client):
        """Test request with invalid top_k parameter"""
        response = flask_client.post(
            '/retrieve',
            data=json.dumps({'query': 'test', 'top_k': -5}),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'top_k' in data['error'].lower()
    
    def test_retrieve_endpoint_invalid_rerank_top_k(self, flask_client):
        """Test request with invalid rerank_top_k parameter"""
        response = flask_client.post(
            '/retrieve',
            data=json.dumps({'query': 'test', 'rerank_top_k': 0}),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'rerank_top_k' in data['error'].lower()
    
    def test_retrieve_endpoint_rerank_greater_than_top_k(self, flask_client):
        """Test request where rerank_top_k > top_k"""
        response = flask_client.post(
            '/retrieve',
            data=json.dumps({
                'query': 'test',
                'top_k': 5,
                'rerank_top_k': 10
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'cannot be greater than' in data['error'].lower()
    
    def test_retrieve_endpoint_no_json_body(self, flask_client):
        """Test request without JSON body"""
        response = flask_client.post(
            '/retrieve',
            data='not json',
            content_type='text/plain'
        )
        
        assert response.status_code in [400, 415, 500]
    
    def test_retrieve_endpoint_retrieval_error(self, flask_client):
        """Test handling of document retrieval errors"""
        with patch('src.document_retrieval.retrieve_documents') as mock_retrieve:
            mock_retrieve.side_effect = Exception("Collection not found")
            
            response = flask_client.post(
                '/retrieve',
                data=json.dumps({'query': 'test query'}),
                content_type='application/json'
            )
            
            assert response.status_code == 400
            data = json.loads(response.data)
            assert data['success'] is False
            assert 'retrieval failed' in data['error'].lower()
    
    def test_retrieve_endpoint_reranking_error(self, flask_client):
        """Test handling of reranking errors"""
        mock_documents = [
            {'content': 'doc1', 'metadata': {}, 'distance': 0.1, 'rank': 1}
        ]
        
        with patch('src.document_retrieval.retrieve_documents') as mock_retrieve, \
             patch('src.document_retrieval.rerank_documents') as mock_rerank:
            
            mock_retrieve.return_value = mock_documents
            mock_rerank.side_effect = Exception("Reranker model error")
            
            response = flask_client.post(
                '/retrieve',
                data=json.dumps({'query': 'test query'}),
                content_type='application/json'
            )
            
            assert response.status_code == 500
            data = json.loads(response.data)
            assert data['success'] is False
            assert 'reranking failed' in data['error'].lower()
    
    def test_retrieve_endpoint_default_parameters(self, flask_client):
        """Test request with default parameters"""
        with patch('src.document_retrieval.retrieve_documents') as mock_retrieve, \
             patch('src.document_retrieval.rerank_documents') as mock_rerank:
            
            mock_retrieve.return_value = []
            mock_rerank.return_value = []
            
            response = flask_client.post(
                '/retrieve',
                data=json.dumps({'query': 'test query'}),
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True
            
            # Verify default parameters were used
            mock_retrieve.assert_called_once()
            call_kwargs = mock_retrieve.call_args[1]
            assert call_kwargs['top_k'] == 20
            assert call_kwargs['collection_name'] == 'pdf_documents'
    
    def test_retrieve_endpoint_without_scores_and_metadata(self, flask_client):
        """Test response format without scores and metadata"""
        mock_documents = [
            {
                'content': 'Document content',
                'metadata': {'filename': 'doc1'},
                'distance': 0.1,
                'rank': 1,
                'rerank_score': 0.95,
            }
        ]
        
        with patch('src.document_retrieval.retrieve_documents') as mock_retrieve, \
             patch('src.document_retrieval.rerank_documents') as mock_rerank:
            
            mock_retrieve.return_value = mock_documents
            mock_rerank.return_value = mock_documents
            
            response = flask_client.post(
                '/retrieve',
                data=json.dumps({
                    'query': 'test query',
                    'include_scores': False,
                    'include_metadata': False
                }),
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True
            
            # Verify chunks don't include scores or metadata
            chunk = data['chunks'][0]
            assert 'content' in chunk
            assert 'rank' in chunk
            assert 'score' not in chunk
            assert 'metadata' not in chunk
    
    def test_retrieve_endpoint_with_scores_and_metadata(self, flask_client):
        """Test response format with scores and metadata"""
        mock_documents = [
            {
                'content': 'Document content',
                'metadata': {'filename': 'doc1', 'chunk_index': 0},
                'distance': 0.1,
                'rank': 1,
                'rerank_score': 0.95,
            }
        ]
        
        with patch('src.document_retrieval.retrieve_documents') as mock_retrieve, \
             patch('src.document_retrieval.rerank_documents') as mock_rerank:
            
            mock_retrieve.return_value = mock_documents
            mock_rerank.return_value = mock_documents
            
            response = flask_client.post(
                '/retrieve',
                data=json.dumps({
                    'query': 'test query',
                    'include_scores': True,
                    'include_metadata': True
                }),
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True
            
            # Verify chunks include scores and metadata
            chunk = data['chunks'][0]
            assert 'content' in chunk
            assert 'rank' in chunk
            assert 'score' in chunk
            assert chunk['score'] == 0.95
            assert 'metadata' in chunk
            assert chunk['metadata']['filename'] == 'doc1'
    
    def test_retrieve_endpoint_empty_results(self, flask_client):
        """Test handling of empty retrieval results"""
        with patch('src.document_retrieval.retrieve_documents') as mock_retrieve, \
             patch('src.document_retrieval.rerank_documents') as mock_rerank:
            
            mock_retrieve.return_value = []
            mock_rerank.return_value = []
            
            response = flask_client.post(
                '/retrieve',
                data=json.dumps({'query': 'nonexistent query'}),
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True
            assert data['chunks'] == []
            assert data['total_retrieved'] == 0
            assert data['total_reranked'] == 0
    
    def test_retrieve_endpoint_response_format(self, flask_client):
        """Test that response has correct format"""
        mock_documents = [
            {
                'content': 'Test content',
                'metadata': {},
                'distance': 0.1,
                'rank': 1,
                'rerank_score': 0.9,
            }
        ]
        
        with patch('src.document_retrieval.retrieve_documents') as mock_retrieve, \
             patch('src.document_retrieval.rerank_documents') as mock_rerank:
            
            mock_retrieve.return_value = mock_documents
            mock_rerank.return_value = mock_documents
            
            response = flask_client.post(
                '/retrieve',
                data=json.dumps({'query': 'test'}),
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            # Verify all required fields are present
            assert 'success' in data
            assert 'query' in data
            assert 'chunks' in data
            assert 'total_retrieved' in data
            assert 'total_reranked' in data
            assert 'timestamp' in data
            
            # Verify timestamp format (ISO 8601)
            assert data['timestamp'].endswith('Z')
            assert 'T' in data['timestamp']


class TestRetrieveDocumentsFunction:
    """Tests for retrieve_documents() helper function"""
    
    def test_retrieve_documents_success(self):
        """Test successful document retrieval"""
        from src.document_retrieval import retrieve_documents
        
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'documents': [['Document 1', 'Document 2']],
            'metadatas': [[{'filename': 'doc1'}, {'filename': 'doc2'}]],
            'distances': [[0.1, 0.2]],
        }
        
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        
        with patch('src.document_retrieval.chromadb.PersistentClient', return_value=mock_client), \
             patch('src.document_retrieval.embedding_functions.SentenceTransformerEmbeddingFunction') as mock_emb:
            
            mock_emb.return_value = MagicMock()
            
            results = retrieve_documents(
                query='test query',
                collection_name='test_collection',
                top_k=10
            )
            
            assert len(results) == 2
            assert results[0]['content'] == 'Document 1'
            assert results[0]['rank'] == 1
            assert results[0]['distance'] == 0.1
            assert results[0]['metadata']['filename'] == 'doc1'
            assert results[1]['rank'] == 2
    
    def test_retrieve_documents_collection_not_found(self):
        """Test error when collection doesn't exist"""
        from src.document_retrieval import retrieve_documents
        
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Collection not found")
        
        with patch('src.document_retrieval.chromadb.PersistentClient', return_value=mock_client), \
             patch('src.document_retrieval.embedding_functions.SentenceTransformerEmbeddingFunction'):
            
            with pytest.raises(Exception) as exc_info:
                retrieve_documents(
                    query='test query',
                    collection_name='nonexistent'
                )
            
            assert "not found" in str(exc_info.value).lower()
    
    def test_retrieve_documents_empty_results(self):
        """Test handling of empty query results"""
        from src.document_retrieval import retrieve_documents
        
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]],
        }
        
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        
        with patch('src.document_retrieval.chromadb.PersistentClient', return_value=mock_client), \
             patch('src.document_retrieval.embedding_functions.SentenceTransformerEmbeddingFunction'):
            
            results = retrieve_documents(
                query='nonexistent query',
                collection_name='test_collection'
            )
            
            assert results == []
    
    def test_retrieve_documents_chromadb_unavailable(self):
        """Test error when ChromaDB is not available"""
        from src.document_retrieval import retrieve_documents
        
        with patch('src.document_retrieval.chromadb', None):
            with pytest.raises(Exception) as exc_info:
                retrieve_documents(query='test')
            
            assert "chromadb" in str(exc_info.value).lower()
    
    def test_retrieve_documents_custom_parameters(self):
        """Test retrieve_documents with custom parameters"""
        from src.document_retrieval import retrieve_documents
        
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'documents': [['Doc 1']],
            'metadatas': [[{'key': 'value'}]],
            'distances': [[0.05]],
        }
        
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        
        with patch('src.document_retrieval.chromadb.PersistentClient', return_value=mock_client) as mock_pc, \
             patch('src.document_retrieval.embedding_functions.SentenceTransformerEmbeddingFunction') as mock_emb:
            
            mock_emb_instance = MagicMock()
            mock_emb.return_value = mock_emb_instance
            
            results = retrieve_documents(
                query='custom query',
                collection_name='custom_collection',
                db_path='./custom_db',
                top_k=15,
                model_name='custom-model'
            )
            
            # Verify ChromaDB client was initialized with custom path
            mock_pc.assert_called_once_with(path='./custom_db')
            
            # Verify embedding function was created with custom model
            mock_emb.assert_called_once_with(model_name='custom-model')
            
            # Verify collection query was called with correct parameters
            mock_collection.query.assert_called_once_with(
                query_texts=['custom query'],
                n_results=15
            )
            
            assert len(results) == 1


class TestRerankDocumentsFunction:
    """Tests for rerank_documents() helper function"""
    
    def test_rerank_documents_success(self):
        """Test successful document reranking"""
        from src.document_retrieval import rerank_documents
        
        documents = [
            {'content': 'Document 1', 'rank': 1, 'metadata': {}},
            {'content': 'Document 2', 'rank': 2, 'metadata': {}},
            {'content': 'Document 3', 'rank': 3, 'metadata': {}},
        ]
        
        mock_reranker = MagicMock()
        # Simulate reranker scores (higher is better)
        mock_reranker.predict.return_value = [0.5, 0.9, 0.3]
        
        with patch('src.document_retrieval.reranker_model', mock_reranker):
            results = rerank_documents(
                query='test query',
                documents=documents,
                top_k=2
            )
            
            # Verify reranker was called with correct pairs
            mock_reranker.predict.assert_called_once()
            pairs = mock_reranker.predict.call_args[0][0]
            assert len(pairs) == 3
            assert pairs[0] == ['test query', 'Document 1']
            
            # Verify results are sorted by score (descending)
            assert len(results) == 2
            assert results[0]['content'] == 'Document 2'  # Highest score (0.9)
            assert results[0]['rank'] == 1
            assert results[0]['rerank_score'] == 0.9
            assert results[1]['content'] == 'Document 1'  # Second highest (0.5)
            assert results[1]['rank'] == 2
    
    def test_rerank_documents_empty_list(self):
        """Test reranking with empty document list"""
        from src.document_retrieval import rerank_documents
        
        with patch('src.document_retrieval.reranker_model', MagicMock()):
            results = rerank_documents(
                query='test query',
                documents=[],
                top_k=5
            )
            
            assert results == []
    
    def test_rerank_documents_model_unavailable(self):
        """Test error when reranker model is not available"""
        from src.document_retrieval import rerank_documents
        
        with patch('src.document_retrieval.reranker_model', None):
            with pytest.raises(Exception) as exc_info:
                rerank_documents(
                    query='test',
                    documents=[{'content': 'doc'}]
                )
            
            assert "reranker model" in str(exc_info.value).lower()
    
    def test_rerank_documents_top_k_greater_than_documents(self):
        """Test reranking when top_k > number of documents"""
        from src.document_retrieval import rerank_documents
        
        documents = [
            {'content': 'Document 1', 'rank': 1},
            {'content': 'Document 2', 'rank': 2},
        ]
        
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.8, 0.6]
        
        with patch('src.document_retrieval.reranker_model', mock_reranker):
            results = rerank_documents(
                query='test query',
                documents=documents,
                top_k=10  # More than available documents
            )
            
            # Should return all documents, sorted by score
            assert len(results) == 2
            assert results[0]['rerank_score'] == 0.8
            assert results[1]['rerank_score'] == 0.6
    
    def test_rerank_documents_preserves_metadata(self):
        """Test that reranking preserves document metadata"""
        from src.document_retrieval import rerank_documents
        
        documents = [
            {
                'content': 'Doc 1',
                'rank': 1,
                'metadata': {'filename': 'file1.pdf', 'page': 1},
                'distance': 0.1
            },
            {
                'content': 'Doc 2',
                'rank': 2,
                'metadata': {'filename': 'file2.pdf', 'page': 5},
                'distance': 0.2
            },
        ]
        
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.7, 0.9]
        
        with patch('src.document_retrieval.reranker_model', mock_reranker):
            results = rerank_documents(
                query='test query',
                documents=documents,
                top_k=2
            )
            
            # Verify metadata is preserved
            assert results[0]['metadata']['filename'] == 'file2.pdf'
            assert results[0]['metadata']['page'] == 5
            assert results[0]['distance'] == 0.2
            
            assert results[1]['metadata']['filename'] == 'file1.pdf'
            assert results[1]['metadata']['page'] == 1
    
    def test_rerank_documents_score_type(self):
        """Test that rerank scores are converted to float"""
        from src.document_retrieval import rerank_documents
        import numpy as np
        
        documents = [{'content': 'Doc 1', 'rank': 1}]
        
        mock_reranker = MagicMock()
        # Simulate numpy array return (common with ML models)
        mock_reranker.predict.return_value = np.array([0.85])
        
        with patch('src.document_retrieval.reranker_model', mock_reranker):
            results = rerank_documents(
                query='test',
                documents=documents,
                top_k=1
            )
            
            # Verify score is a Python float, not numpy type
            assert isinstance(results[0]['rerank_score'], float)
            assert results[0]['rerank_score'] == 0.85
