from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import pickle
import time
from sentence_transformers import SentenceTransformer
import openai
import os
from dotenv import load_dotenv



# --------------------------
# Load .env and set OpenAI API key
# --------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")

# --------------------------
# FastAPI app with startup
# --------------------------
app = FastAPI(title="RAG Chatbot API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000", "http://127.0.0.1:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global variables
embedding_model = None
index = None
documents = None

# --------------------------
# Request model
# --------------------------
class AskRequest(BaseModel):
    query: str
    top_k: int = 3

# --------------------------
# Startup event
# --------------------------
@app.on_event("startup")
async def startup_event():
    global embedding_model, index, documents
    print("Loading SentenceTransformer model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading documents.pkl...")
    with open("documents.pkl", "rb") as f:
        documents = pickle.load(f)

    print("Loading FAISS index...")
    index = faiss.read_index("faiss_index.bin")
    print(f"✅ Loaded {len(documents)} documents and FAISS index.")

# --------------------------
# Retrieval function
# --------------------------
def retrieve_from_faiss(query: str, top_k: int):
    query_embedding = embedding_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in range(top_k):
        idx = indices[0][i]
        results.append({
            "text": documents[idx],
            "score": float(distances[0][i])
        })
    return results

# --------------------------
# /ask endpoint
# --------------------------
@app.post("/ask")
async def ask_question(request: AskRequest):
    start_time = time.time()
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Step 1: Retrieve relevant chunks
        results = retrieve_from_faiss(query, request.top_k)

        if not results:
            return {
                "question": query,
                "answer": "Sorry, I couldn’t find relevant information.",
                "sources": [],
                "confidence": 0.0,
                "processing_time_ms": f"{(time.time()-start_time)*1000:.2f}",
                "total_sources": 0
            }

        # Step 2: Build context with numbered sources
        context_with_citations = ""
        sources = []
        for i, r in enumerate(results):
            snippet = r["text"][:500]
            context_with_citations += f"[Source {i+1}] {snippet}\n"
            sources.append({
                "text_preview": snippet,
                "score": r["score"],
                "rank": i+1
            })

        # Step 3: Build prompt for OpenAI
        prompt = f"""
You are a helpful assistant. Use ONLY the sources below to answer the question.
Add inline citations like [Source 1], [Source 2].
If the answer is not in the sources, say "I don’t know."

Question: {query}

Sources:
{context_with_citations}
"""

        # Step 4: Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        answer = response["choices"][0]["message"]["content"].strip()

        return {
            "question": query,
            "answer": answer,
            #"sources": sources,
            #"confidence": results[0]["score"],
            #"processing_time_ms": f"{(time.time()-start_time)*1000:.2f}",
            #"total_sources": len(results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------
# Health check
# --------------------------
@app.get("/health")
async def health():
    return {
        "status": "healthy" if documents and index else "not ready",
        "total_documents": len(documents) if documents else 0,
        "embedding_method": "SentenceTransformer + FAISS"
    }

# --------------------------
# Root
# --------------------------
@app.get("/")
async def root():
    return {"message": "RAG Chatbot API running"}
