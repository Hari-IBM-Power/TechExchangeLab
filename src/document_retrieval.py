"""
Document retrieval and reranking functionality for RAG pipeline.
Provides functions to retrieve documents from ChromaDB and rerank them using cross-encoder models.
"""

from typing import Any, Dict, List

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None
    embedding_functions = None

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

from src.pdf_embedding_pipeline import DEFAULT_COLLECTION_NAME, DEFAULT_DB_PATH, DEFAULT_EMBEDDING_MODEL


# Reranker model configuration
RERANKER_MODEL = "ibm-granite/granite-embedding-reranker-english-r2"
reranker_model = None

try:
    if CrossEncoder is not None:
        reranker_model = CrossEncoder(RERANKER_MODEL)
        print(f"Loaded reranker model: {RERANKER_MODEL}")
except Exception as e:
    print(f"Warning: Could not load reranker model: {e}")


def retrieve_documents(
    query: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    db_path: str = DEFAULT_DB_PATH,
    top_k: int = 20,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> List[Dict[str, Any]]:
    """
    Retrieve top_k documents from ChromaDB collection based on query.
    
    Args:
        query: Search query string
        collection_name: Name of ChromaDB collection
        db_path: Path to ChromaDB persistence directory
        top_k: Number of documents to retrieve
        model_name: Embedding model name
        
    Returns:
        List of dictionaries containing document content and metadata
        
    Raises:
        Exception: If ChromaDB or dependencies are not available
    """
    if chromadb is None or embedding_functions is None:
        raise Exception("ChromaDB is not available. Install chromadb package.")
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=db_path)
    
    # Create embedding function
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )
    
    # Get collection
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )
    except Exception as e:
        raise Exception(f"Collection '{collection_name}' not found: {str(e)}")
    
    # Query collection
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )
    
    # Format results
    documents = []
    if results and results['documents'] and len(results['documents']) > 0:
        for idx, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][idx] if results['metadatas'] else {}
            distance = results['distances'][0][idx] if results['distances'] else None
            
            documents.append({
                'content': doc,
                'metadata': metadata,
                'distance': distance,
                'rank': idx + 1,
            })
    
    return documents


def rerank_documents(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using cross-encoder reranker model.
    
    Args:
        query: Search query string
        documents: List of document dictionaries from retrieve_documents
        top_k: Number of top documents to return after reranking
        
    Returns:
        List of reranked documents with relevance scores
        
    Raises:
        Exception: If reranker model is not available
    """
    if reranker_model is None:
        raise Exception("Reranker model is not available. Install sentence-transformers package.")
    
    if not documents:
        return []
    
    # Prepare query-document pairs for reranking
    pairs = [[query, doc['content']] for doc in documents]
    
    # Get relevance scores
    scores = reranker_model.predict(pairs)
    
    # Add scores to documents and sort by score (descending)
    for idx, doc in enumerate(documents):
        doc['rerank_score'] = float(scores[idx])
    
    # Sort by rerank_score descending
    reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
    
    # Return top_k documents with updated ranks
    top_reranked = reranked[:top_k]
    for idx, doc in enumerate(top_reranked):
        doc['rank'] = idx + 1
    
    return top_reranked

# Made with Bob
