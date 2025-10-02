from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import faiss
import pickle
import time
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv
import io
from pathlib import Path

# --------------------------
# Load .env and set OpenAI API key
# --------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

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

class VoiceAskRequest(BaseModel):
    top_k: int = 3
    voice: str = "alloy"  # Options: alloy, echo, fable, onyx, nova, shimmer

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
# Core RAG function
# --------------------------
def get_rag_answer(query: str, top_k: int = 3):
    """Core RAG logic extracted for reuse"""
    results = retrieve_from_faiss(query, top_k)
    
    if not results:
        return "Sorry, I couldn't find relevant information."
    
    # Build context with numbered sources
    context_with_citations = ""
    for i, r in enumerate(results):
        snippet = r["text"][:500]
        context_with_citations += f"[Source {i+1}] {snippet}\n"
    
    # Build prompt for OpenAI
    prompt = f"""
You are a helpful assistant. Use ONLY the sources below to answer the question.
Add inline citations like [Source 1], [Source 2].
If the answer is not in the sources, say "I don't know."

Question: {query}

Sources:
{context_with_citations}
"""
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    
    return response.choices[0].message.content.strip()

# --------------------------
# Text-based /ask endpoint (original)
# --------------------------
@app.post("/ask")
async def ask_question(request: AskRequest):
    start_time = time.time()
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        answer = get_rag_answer(query, request.top_k)
        
        return {
            "question": query,
            "answer": answer,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------
# NEW: Voice-to-Voice endpoint
# --------------------------
@app.post("/ask-voice")
async def ask_voice(
    audio: UploadFile = File(...),
    top_k: int = 3,
    voice: str = "alloy"
):
    """
    Accept voice input, transcribe it, get RAG answer, and return voice response.
    
    Parameters:
    - audio: Audio file (mp3, mp4, mpeg, mpga, m4a, wav, webm)
    - top_k: Number of relevant documents to retrieve
    - voice: TTS voice (alloy, echo, fable, onyx, nova, shimmer)
    """
    try:
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
        answer = get_rag_answer(query, top_k)
        
        # Step 4: Convert answer to speech
        tts_response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=answer
        )
        
        # Step 5: Return audio response
        audio_stream = io.BytesIO(tts_response.content)
        
        return StreamingResponse(
            audio_stream,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=response.mp3",
                "X-Transcribed-Query": query,
                "X-Answer-Text": answer[:500]  # First 500 chars in header
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice processing error: {str(e)}")

# --------------------------
# NEW: Voice-to-Text endpoint (transcription only)
# --------------------------
@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe audio to text only.
    """
    try:
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

# --------------------------
# NEW: Text-to-Voice endpoint
# --------------------------
@app.post("/text-to-speech")
async def text_to_speech(text: str, voice: str = "alloy"):
    """
    Convert text to speech.
    
    Parameters:
    - text: Text to convert
    - voice: TTS voice (alloy, echo, fable, onyx, nova, shimmer)
    """
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
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
        }
    }