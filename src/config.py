"""
RAG Chatbot Configuration Module
Contains all configuration settings for the WhatsApp AI RAG Chatbot
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the RAG chatbot application"""
    
    # Google Gemini Settings
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
    
    # Vector Store Settings
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vector_store_gemini")
    
    # Document Processing Settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    
    # RAG Settings
    TOP_K_RESULTS = 4  # Number of relevant chunks to retrieve
    TEMPERATURE = 0.7  # LLM temperature for response generation
    MAX_TOKENS = 1000  # Maximum tokens in response
    
    # Supported file types
    SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx", ".md"]
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required. Please set it in .env file")
        return True

# System prompt for the chatbot
SYSTEM_PROMPT = """You are a helpful WhatsApp AI assistant powered by RAG (Retrieval-Augmented Generation).

Your role is to:
1. Answer questions based on the provided context from uploaded documents
2. Be conversational and friendly, like a WhatsApp chat
3. If the context doesn't contain relevant information, say so honestly
4. Keep responses concise but informative
5. Use emojis occasionally to be more engaging 😊

When answering:
- Always base your answers on the provided context
- If you're not sure, say "I don't have enough information about that"
- Be helpful and suggest what information might be useful

Context from documents:
{context}

Remember: You're simulating a WhatsApp chat interface, so keep responses conversational!"""
