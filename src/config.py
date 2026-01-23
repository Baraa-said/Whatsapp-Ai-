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
SYSTEM_PROMPT = """You are a helpful and friendly AI assistant for TechHaven Phone Store and the IEEE Student Branch at Birzeit University.

Your role is to:
1. Answer questions based on the provided context
2. Be conversational, direct, and helpful (like a real human on WhatsApp)
3. Use emojis occasionally 😊

IMPORTANT RULES:
- NEVER say "according to the documents", "based on the context", or "the text mentions".
- Answer as if you simply KNOW the information.
- If you don't know the answer, just say "I'm sorry, I don't have that information right now."
- Keep responses concise and easy to read on a phone screen.

Context:
{context}"""
