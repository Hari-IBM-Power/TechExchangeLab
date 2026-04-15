"""
Flask API for User Savings Prediction
Provides REST API endpoint to predict next month's savings for users
"""

import os
import tempfile
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, request, jsonify
from main import main as predict_savings
from src.pdf_embedding_pipeline import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DB_PATH,
    PDFEmbeddingPipelineError,
    process_pdf_file,
)
from src.document_retrieval import retrieve_documents, rerank_documents

app = Flask(__name__)

# Load the savings plan classifier model
SAVINGS_PLAN_MODEL_PATH = "./src/models/savings_plan_classifier.pkl"
savings_plan_model = None

try:
    savings_plan_model = joblib.load(SAVINGS_PLAN_MODEL_PATH)
    print(f"Loaded savings plan classifier from {SAVINGS_PLAN_MODEL_PATH}")
except Exception as e:
    print(f"Warning: Could not load savings plan classifier: {e}")

# Feature names for savings plan classifier
SAVINGS_PLAN_FEATURE_NAMES = [
    "Current Age",
    "Retirement Age",
    "Birth Year",
    "Birth Month",
    "Gender",
    "City",
    "State",
    "Zipcode",
    "Per Capita Income - Zipcode",
    "Yearly Income - Person",
    "Total Debt",
    "FICO Score",
    "Savings Plan",
]


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


@app.route('/pdf/embeddings', methods=['POST'])
def create_pdf_embeddings():
    """
    POST /pdf/embeddings endpoint

    Multipart Form Data:
    - file: PDF file (required)
    - collection_name: target Chroma collection (optional)
    - db_path: Chroma persistence path (optional, default ./db)
    - output_dir: markdown output directory (optional)
    """
    temp_pdf_path = None

    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Missing required file field: file'
            }), 400

        uploaded_file = request.files['file']
        if uploaded_file.filename is None or uploaded_file.filename.strip() == '':
            return jsonify({
                'success': False,
                'error': 'Uploaded file must have a filename'
            }), 400

        if not uploaded_file.filename.lower().endswith('.pdf'):
            return jsonify({
                'success': False,
                'error': 'Uploaded file must be a PDF'
            }), 400

        collection_name = request.form.get('collection_name', DEFAULT_COLLECTION_NAME)
        db_path = request.form.get('db_path', DEFAULT_DB_PATH)
        output_dir = request.form.get('output_dir')

        temp_dir = tempfile.mkdtemp(prefix='pdf_upload_')
        temp_pdf_path = os.path.join(temp_dir, uploaded_file.filename)
        uploaded_file.save(temp_pdf_path)

        result = process_pdf_file(
            pdf_path=temp_pdf_path,
            collection_name=collection_name,
            db_path=db_path,
            output_dir=output_dir,
        )
        result['success'] = True

        return jsonify(result), 200

    except PDFEmbeddingPipelineError as exc:
        return jsonify({
            'success': False,
            'error': str(exc)
        }), 400
    except Exception as exc:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(exc)}'
        }), 500
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except OSError:
                pass

@app.route('/retrieve', methods=['POST'])
def retrieve_endpoint():
    """
    POST /retrieve endpoint
    
    Retrieve document chunks without LLM generation.
    Supports configurable retrieval and reranking parameters.
    
    Request Body (JSON):
    {
        "query": str (required),
        "collection_name": str (optional, default: "pdf_documents"),
        "top_k": int (optional, default: 20),
        "rerank_top_k": int (optional, default: 5),
        "include_scores": bool (optional, default: false),
        "include_metadata": bool (optional, default: false)
    }
    
    Response (JSON):
    Success (200):
    {
        "success": true,
        "query": str,
        "chunks": [
            {
                "content": str,
                "rank": int,
                "score": float (if include_scores=true),
                "metadata": dict (if include_metadata=true)
            }
        ],
        "total_retrieved": int,
        "total_reranked": int,
        "timestamp": str
    }
    
    Error (400/500):
    {
        "success": false,
        "error": str
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
        
        # Validate required parameters
        if 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required parameter: query'
            }), 400
        
        query = data['query']
        if not isinstance(query, str) or not query.strip():
            return jsonify({
                'success': False,
                'error': 'query must be a non-empty string'
            }), 400
        
        # Get optional parameters with defaults
        collection_name = data.get('collection_name', DEFAULT_COLLECTION_NAME)
        top_k = data.get('top_k', 20)
        rerank_top_k = data.get('rerank_top_k', 5)
        include_scores = data.get('include_scores', False)
        include_metadata = data.get('include_metadata', False)
        
        # Validate parameter types
        if not isinstance(top_k, int) or top_k <= 0:
            return jsonify({
                'success': False,
                'error': 'top_k must be a positive integer'
            }), 400
        
        if not isinstance(rerank_top_k, int) or rerank_top_k <= 0:
            return jsonify({
                'success': False,
                'error': 'rerank_top_k must be a positive integer'
            }), 400
        
        if rerank_top_k > top_k:
            return jsonify({
                'success': False,
                'error': 'rerank_top_k cannot be greater than top_k'
            }), 400
        
        # Retrieve documents
        try:
            documents = retrieve_documents(
                query=query,
                collection_name=collection_name,
                top_k=top_k,
            )
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Document retrieval failed: {str(e)}'
            }), 400
        
        total_retrieved = len(documents)
        
        # Rerank documents if we have results
        if documents:
            try:
                documents = rerank_documents(
                    query=query,
                    documents=documents,
                    top_k=rerank_top_k,
                )
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Document reranking failed: {str(e)}'
                }), 500
        
        total_reranked = len(documents)
        
        # Format chunks for response
        chunks = []
        for doc in documents:
            chunk = {
                'content': doc['content'],
                'rank': doc['rank'],
            }
            
            if include_scores and 'rerank_score' in doc:
                chunk['score'] = doc['rerank_score']
            
            if include_metadata and 'metadata' in doc:
                chunk['metadata'] = doc['metadata']
            
            chunks.append(chunk)
        
        # Build response
        response = {
            'success': True,
            'query': query,
            'chunks': chunks,
            'total_retrieved': total_retrieved,
            'total_reranked': total_reranked,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500



@app.route('/predict', methods=['POST'])
def predict_savings_plan():
    """
    POST /predict endpoint
    
    Predict savings plan classification based on user features.
    
    Request Body (JSON):
    {
        "features": [
            Current Age, Retirement Age, Birth Year, Birth Month,
            Gender, City, State, Zipcode, Per Capita Income - Zipcode,
            Yearly Income - Person, Total Debt, FICO Score, Savings Plan
        ]
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
    if savings_plan_model is None:
        return jsonify({
            'success': False,
            'error': 'Savings plan classifier model not loaded'
        }), 500
    
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        
        # Validate request body
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body must be JSON'
            }), 400
        
        # Validate features parameter
        if 'features' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required parameter: features'
            }), 400
        
        features = data['features']
        
        # Validate features is a list
        if not isinstance(features, list):
            return jsonify({
                'success': False,
                'error': 'features must be a list'
            }), 400
        
        # Convert to DataFrame with column names
        df = pd.DataFrame([features], columns=SAVINGS_PLAN_FEATURE_NAMES)
        
        # Predict
        prediction = savings_plan_model.predict(df)
        
        return jsonify({
            'success': True,
            'prediction': prediction.tolist()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
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
                    'features': 'array of 13 values (required): [Current Age, Retirement Age, Birth Year, Birth Month, Gender, City, State, Zipcode, Per Capita Income - Zipcode, Yearly Income - Person, Total Debt, FICO Score, Savings Plan]'
                },
                'example': {
                    'request': {
                        'features': [45, 65, 1979, 6, 'Male', 'New York', 'NY', '10001', 75000, 120000, 50000, 720, 'Plan A']
                    },
                    'response': {
                        'success': True,
                        'prediction': ['Plan A']
                    }
                }
            },
            'POST /pdf/embeddings': {
                'description': 'Upload a PDF, convert to markdown, chunk it, generate embeddings, and store in ChromaDB',
                'parameters': {
                    'file': 'PDF file (required, multipart/form-data)',
                    'collection_name': 'string (optional)',
                    'db_path': 'string (optional, default ./db)',
                    'output_dir': 'string (optional)'
                }
            },
            'POST /retrieve': {
                'description': 'Retrieve document chunks without LLM generation, with configurable retrieval and reranking',
                'parameters': {
                    'query': 'string (required) - search query',
                    'collection_name': 'string (optional, default: pdf_documents)',
                    'top_k': 'integer (optional, default: 20) - number of documents to retrieve',
                    'rerank_top_k': 'integer (optional, default: 5) - number of documents after reranking',
                    'include_scores': 'boolean (optional, default: false) - include relevance scores',
                    'include_metadata': 'boolean (optional, default: false) - include document metadata'
                },
                'example': {
                    'request': {
                        'query': 'retirement planning strategies',
                        'collection_name': 'pdf_documents',
                        'top_k': 20,
                        'rerank_top_k': 5,
                        'include_scores': True,
                        'include_metadata': True
                    },
                    'response': {
                        'success': True,
                        'query': 'retirement planning strategies',
                        'chunks': [
                            {
                                'content': 'Document content...',
                                'rank': 1,
                                'score': 0.95,
                                'metadata': {'filename': 'document', 'chunk_index': 0}
                            }
                        ],
                        'total_retrieved': 20,
                        'total_reranked': 5,
                        'timestamp': '2026-04-14T08:45:00.000Z'
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
