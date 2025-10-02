# Minimal RAG API with basic embedding fallback
# Use this if you have package compatibility issues

import os
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from contextlib import asynccontextmanager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
vectorizer = None
tfidf_matrix = None
index = None
documents = None

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    include_scores: Optional[bool] = True

class SearchResult(BaseModel):
    text: str
    score: float
    similarity: float
    keyword_match: float
    rank: int

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    total_documents: int
    embedding_method: str

def clean_text(text):
    """Basic text cleaning."""
    # Remove extra whitespace and special characters
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.strip().lower()

async def load_rag_system():
    """Load the RAG system with fallback to TF-IDF if SentenceTransformer fails."""
    global vectorizer, tfidf_matrix, index, documents
    
    try:
        logger.info("Loading RAG system...")
        
        # Check if files exist
        if not os.path.exists("faiss_index.bin"):
            raise FileNotFoundError("faiss_index.bin not found!")
        if not os.path.exists("documents.pkl"):
            raise FileNotFoundError("documents.pkl not found!")
        
        # Load documents first
        logger.info("Loading document chunks...")
        with open("documents.pkl", "rb") as f:
            documents = pickle.load(f)
        
        # Try to use the existing FAISS index with SentenceTransformer
        try:
            logger.info("Trying to load SentenceTransformer...")
            from sentence_transformers import SentenceTransformer
            global embedding_model
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Load FAISS index
            logger.info("Loading FAISS index...")
            index = faiss.read_index("faiss_index.bin")
            
            logger.info("✅ Using SentenceTransformer + FAISS")
            
        except Exception as e:
            logger.warning(f"SentenceTransformer failed: {e}")
            logger.info("Falling back to TF-IDF vectorization...")
            
            # Fallback to TF-IDF
            cleaned_docs = [clean_text(doc) for doc in documents]
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(cleaned_docs)
            
            logger.info("✅ Using TF-IDF vectorization")
        
        logger.info(f"📄 Total documents: {len(documents)}")
        
    except Exception as e:
        logger.error(f"❌ Error loading RAG system: {str(e)}")
        raise e

def search_with_tfidf(query: str, top_k: int = 3) -> List[tuple]:
    """Search using TF-IDF vectorization."""
    query_cleaned = clean_text(query)
    query_vector = vectorizer.transform([query_cleaned])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get top results
    top_indices = similarities.argsort()[::-1][:top_k * 2]  # Get more for filtering
    
    results = []
    query_words = query.lower().split()
    
    for idx in top_indices:
        if similarities[idx] > 0.01:  # Minimum similarity threshold
            text = documents[idx]
            similarity = similarities[idx]
            
            # Skip very short texts
            if len(text.strip()) < 100:
                continue
                
            # Calculate keyword match
            text_lower = text.lower()
            keyword_matches = sum(1 for word in query_words if word in text_lower)
            keyword_ratio = keyword_matches / len(query_words) if query_words else 0
            
            # Combined score
            combined_score = similarity + (keyword_ratio * 0.2)
            
            results.append((text, combined_score, similarity, 0.0, keyword_ratio))
            
            if len(results) >= top_k:
                break
    
    return results

def search_with_faiss(query: str, top_k: int = 3) -> List[tuple]:
    """Search using FAISS (if available)."""
    query_embedding = embedding_model.encode([query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    
    search_k = min(top_k * 3, len(documents))
    distances, indices = index.search(query_embedding.astype('float32'), search_k)
    
    results = []
    query_words = query.lower().split()
    
    for i in range(search_k):
        idx = indices[0][i]
        text = documents[idx]
        score = distances[0][i]
        
        # Skip short texts and noise
        if len(text.strip()) < 100:
            continue
            
        similarity_score = 1.0 / (1.0 + score)
        
        # Calculate keyword match
        text_lower = text.lower()
        keyword_matches = sum(1 for word in query_words if word in text_lower)
        keyword_ratio = keyword_matches / len(query_words) if query_words else 0
        
        combined_score = similarity_score + (keyword_ratio * 0.3)
        
        results.append((text, combined_score, similarity_score, score, keyword_ratio))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await load_rag_system()
    yield
    # Shutdown
    logger.info("Shutting down RAG system...")

# Initialize FastAPI app
app = FastAPI(
    title="RAG API with Vector Database (Compatible Version)",
    description="Fast API backend with fallback compatibility for different embedding methods",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "RAG API with Vector Database (Compatible Version)",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    if documents is None:
        raise HTTPException(status_code=503, detail="RAG system not loaded")
    
    method = "SentenceTransformer + FAISS" if index is not None else "TF-IDF"
    
    return HealthResponse(
        status="healthy",
        total_documents=len(documents),
        embedding_method=method
    )

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    if documents is None:
        raise HTTPException(status_code=503, detail="RAG system not loaded")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        import time
        start_time = time.time()
        
        # Use appropriate search method
        if index is not None:
            results = search_with_faiss(request.query, request.top_k)
        else:
            results = search_with_tfidf(request.query, request.top_k)
        
        search_results = []
        for i, (text, combined_score, sim_score, l2_dist, keyword_ratio) in enumerate(results):
            search_results.append(SearchResult(
                text=text,
                score=round(combined_score, 4),
                similarity=round(sim_score, 4),
                keyword_match=round(keyword_ratio, 2),
                rank=i + 1
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/search", response_model=SearchResponse)
async def search_documents_get(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(3, ge=1, le=10, description="Number of results to return")
):
    request = SearchRequest(query=q, top_k=top_k)
    return await search_documents(request)

@app.post("/ask")
async def ask_question(request: SearchRequest):
    search_response = await search_documents(request)
    
    if not search_response.results:
        return {
            "question": request.query,
            "answer": "I couldn't find relevant information to answer your question.",
            "sources": [],
            "confidence": 0.0
        }
    
    # Combine results for answer with better formatting
    context_texts = []
    for i, result in enumerate(search_response.results, 1):
        # Clean up the text and add section markers
        clean_text = result.text.strip()
        # Remove excessive whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text)
        context_texts.append(f"[Source {i}] {clean_text}")
    
    # Join with proper separators
    combined_context = "\n\n".join(context_texts)
    
    # Increase character limit significantly
    max_chars = 3000  # Increased from 1000 to 3000
    if len(combined_context) > max_chars:
        # Find a good cutoff point (end of sentence)
        cutoff = combined_context[:max_chars].rfind('.')
        if cutoff == -1:  # No sentence end found
            cutoff = max_chars
        combined_context = combined_context[:cutoff + 1] + "\n\n[Response truncated - see sources for complete information]"
    
    sources = [{
        "text_preview": result.text[:300] + "...",  # Increased preview length
        "score": result.score,
        "rank": result.rank
    } for result in search_response.results]
    
    confidence = search_response.results[0].score
    
    return {
        "question": request.query,
        "answer": combined_context,
        "sources": sources,
        "confidence": confidence,
        "processing_time_ms": search_response.processing_time_ms,
        "total_sources": len(search_response.results)
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Compatible RAG API Server...")
    uvicorn.run(
        "minimal_main:app",  # Change this to match your filename
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )