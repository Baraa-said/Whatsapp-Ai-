"""
WhatsApp AI RAG Chatbot - FastAPI Backend
Serves the WhatsApp simulation frontend and handles chat requests
"""

import os
import sys
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag_chain import RAGChain
from src.vector_store import VectorStore
from src.document_loader import DocumentLoader
from src.config import Config

app = FastAPI(title="WhatsApp AI Simulation")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
class AppState:
    rag_chain: Optional[RAGChain] = None
    vector_store: Optional[VectorStore] = None

state = AppState()

# Startup Event
@app.on_event("startup")
async def startup_event():
    """Initialize application state on startup"""
    try:
        # Initialize vector store
        state.vector_store = VectorStore()
        
        # Check if vector store already exists on disk
        if state.vector_store.exists():
            print("📂 Found existing vector store. Loading...")
            state.vector_store.load()
            state.rag_chain = RAGChain(state.vector_store)
            print("✅ Vector store loaded from disk!")
            return

        # If not, try to load default data from the default_documents folder
        default_folder = "default_documents"
        if os.path.exists(default_folder) and os.path.isdir(default_folder):
            print(f"🚀 Found default data folder: {default_folder}. Loading all documents...")
            
            # Load all documents from the folder
            loader = DocumentLoader()
            documents = loader.load_directory(default_folder)
            
            if documents:
                # Create vector store
                state.vector_store.create_from_documents(documents)
                
                # Save it to disk to avoid re-embedding next time
                state.vector_store.save()
                print("💾 Default data embedded and saved to disk!")
                
                # Initialize RAG chain
                state.rag_chain = RAGChain(state.vector_store)
                print("✅ RAG chain initialized!")
            else:
                print(f"ℹ️ No supported documents found in {default_folder}")
        else:
            print(f"ℹ️ No default data folder found at {default_folder}")
            
    except Exception as e:
        print(f"⚠️ Error initializing application: {e}")

# Models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    timestamp: str
    sources: Optional[List[Dict[str, Any]]] = None

# Routes
@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Handle chat messages"""
    if not state.rag_chain:
        # Try to initialize if vector store exists
        try:
            if not state.vector_store:
                state.vector_store = VectorStore()
            
            if state.vector_store.exists():
                state.vector_store.load()
                state.rag_chain = RAGChain(state.vector_store)
            else:
                return {
                    "answer": "⏳ I'm still loading the knowledge base. Please try again in a moment!",
                    "sources": []
                }
        except Exception as e:
            # Graceful fallback for missing API key or other init errors
            print(f"Initialization error: {e}")
            return {
                "answer": "⚠️ I'm ready to chat, but I need a Google API key to generate responses. Please add `GOOGLE_API_KEY` to your `.env` file!",
                "sources": []
            }
    
    try:
        result = state.rag_chain.query(request.message, include_sources=True)
        return {
            "answer": result["answer"],
            "sources": result.get("sources", [])
        }
    except Exception as e:
        print(f"Query error: {e}")
        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }

@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process documents"""
    try:
        temp_paths = []
        os.makedirs("temp_uploads", exist_ok=True)
        
        for file in files:
            path = f"temp_uploads/{file.filename}"
            with open(path, "wb") as f:
                content = await file.read()
                f.write(content)
            temp_paths.append(path)
        
        # Process documents
        loader = DocumentLoader()
        documents = loader.load_and_split(temp_paths)
        
        state.vector_store = VectorStore()
        state.vector_store.create_from_documents(documents)
        state.vector_store.save()
        
        # Initialize chain
        state.rag_chain = RAGChain(state.vector_store)
        
        # Cleanup
        for path in temp_paths:
            os.remove(path)
        os.rmdir("temp_uploads")
        
        return {"message": f"Successfully processed {len(files)} files", "count": len(files)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset")
async def reset_history():
    """Clear chat history"""
    if state.rag_chain:
        state.rag_chain.clear_history()
    return {"message": "History cleared"}

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
