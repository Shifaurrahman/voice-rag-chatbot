from fastapi import FastAPI, HTTPException, File, UploadFile, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import faiss
import pickle
import time
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import io
from typing import Optional

# --------------------------
# FastAPI app with startup
# --------------------------
app = FastAPI(title="RAG Chatbot API with Voice")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Transcribed-Query", "X-Answer-Text"]
)

# Global variables
embedding_model = None
index = None
documents = None

# --------------------------
# Request models
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
# Core RAG function with API key
# --------------------------
def get_rag_answer(query: str, api_key: str, top_k: int = 3):
    """Core RAG logic with user-provided API key"""
    # Initialize OpenAI client with user's API key
    client = OpenAI(api_key=api_key)
    
    results = retrieve_from_faiss(query, top_k)
    
    if not results:
        return "Sorry, I couldn't find relevant information."
    
    # Build context without numbered sources
    context_text = "\n".join([r["text"][:500] for r in results])
    
    # Build prompt for OpenAI
    prompt = f"""
You are a helpful assistant. Use ONLY the sources below to answer the question.
If the answer is not in the sources, say "I don't know."

Question: {query}

Information:
{context_text}
"""
    
    # Call OpenAI API with user's key
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    
    return response.choices[0].message.content.strip()

# --------------------------
# Text-based /ask endpoint
# --------------------------
@app.post("/ask")
async def ask_question(
    request: AskRequest,
    x_api_key: Optional[str] = Header(None)
):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="OpenAI API key required in X-API-Key header")
    
    start_time = time.time()
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        answer = get_rag_answer(query, x_api_key, request.top_k)
        
        return {
            "question": query,
            "answer": answer,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------
# Text-to-Voice endpoint
# --------------------------
@app.post("/text-to-speech")
async def text_to_speech(
    text: str,
    voice: str = "alloy",
    x_api_key: Optional[str] = Header(None)
):
    """
    Convert text to speech.
    
    Parameters:
    - text: Text to convert
    - voice: TTS voice (alloy, echo, fable, onyx, nova, shimmer)
    - x_api_key: OpenAI API key in header
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="OpenAI API key required in X-API-Key header")
    
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
        client = OpenAI(api_key=x_api_key)
        
        tts_response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        
        audio_stream = io.BytesIO(tts_response.content)
        
        return StreamingResponse(
            audio_stream,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
        
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
        "embedding_method": "SentenceTransformer + FAISS",
        "voice_enabled": True
    }

# --------------------------
# Root
# --------------------------
@app.get("/")
async def root():
    return {
        "message": "RAG Chatbot API with Voice Support",
        "endpoints": {
            "text": "/ask",
            "voice": "/ask-voice",
            "transcribe": "/transcribe",
            "tts": "/text-to-speech"
        },
        "note": "All endpoints require X-API-Key header with OpenAI API key"
    }
        

# --------------------------
# Voice-to-Voice endpoint
# --------------------------
@app.post("/ask-voice")
async def ask_voice(
    audio: UploadFile = File(...),
    top_k: int = 3,
    voice: str = "alloy",
    x_api_key: Optional[str] = Header(None)
):
    """
    Accept voice input, transcribe it, get RAG answer, and return voice response.
    
    Parameters:
    - audio: Audio file (mp3, mp4, mpeg, mpga, m4a, wav, webm)
    - top_k: Number of relevant documents to retrieve
    - voice: TTS voice (alloy, echo, fable, onyx, nova, shimmer)
    - x_api_key: OpenAI API key in header
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="OpenAI API key required in X-API-Key header")
    
    try:
        # Initialize OpenAI client with user's API key
        client = OpenAI(api_key=x_api_key)
        
        # Step 1: Read audio file
        audio_data = await audio.read()
        
        # Step 2: Transcribe using OpenAI Whisper
        audio_file = io.BytesIO(audio_data)
        audio_file.name = audio.filename or "audio.ogg"
        
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        
        query = transcript.text.strip()
        
        if not query:
            raise HTTPException(status_code=400, detail="Could not transcribe audio.")
        
        # Step 3: Get RAG answer
        answer = get_rag_answer(query, x_api_key, top_k)
        
        # Step 4: Convert answer to speech
        tts_response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=answer
        )
        
        # Step 5: Return audio response
        audio_stream = io.BytesIO(tts_response.content)
        
        # Encode headers to handle special characters
        import base64
        query_encoded = base64.b64encode(query.encode('utf-8')).decode('ascii')
        answer_encoded = base64.b64encode(answer[:500].encode('utf-8')).decode('ascii')
        
        return StreamingResponse(
            audio_stream,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=response.mp3",
                "X-Transcribed-Query": query,
                "X-Answer-Text": answer
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice processing error: {str(e)}")

# --------------------------
# Voice-to-Text endpoint (transcription only)
# --------------------------
@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None)
):
    """
    Transcribe audio to text only.
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="OpenAI API key required in X-API-Key header")
    
    try:
        client = OpenAI(api_key=x_api_key)
        
        audio_data = await audio.read()
        audio_file = io.BytesIO(audio_data)
        audio_file.name = audio.filename or "audio.ogg"
        
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        
        return {
            "transcription": transcript.text,
            "language": getattr(transcript, "language", "unknown")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))