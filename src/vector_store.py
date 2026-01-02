"""
Vector Store Module
Handles FAISS vector database operations for document embeddings
"""

import os
from typing import List, Optional, Tuple
from pydantic import SecretStr
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import Config


class VectorStore:
    """Manages FAISS vector store for document embeddings"""
    
    def __init__(self):
        """Initialize the vector store with OpenAI embeddings"""
        Config.validate()
        
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            api_key=SecretStr(Config.OPENAI_API_KEY)
        )
        self.vector_store: Optional[FAISS] = None
        self.store_path = Config.VECTOR_STORE_PATH
    
    def create_from_documents(self, documents: List[Document]) -> FAISS:
        """
        Create a new vector store from documents
        
        Args:
            documents: List of Document objects to embed
            
        Returns:
            FAISS vector store instance
        """
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        print(f"🔄 Creating embeddings for {len(documents)} document chunks...")
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        print("✅ Vector store created successfully!")
        return self.vector_store
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to existing vector store
        
        Args:
            documents: List of Document objects to add
        """
        if self.vector_store is None:
            self.create_from_documents(documents)
        else:
            print(f"🔄 Adding {len(documents)} document chunks to vector store...")
            self.vector_store.add_documents(documents)
            print("✅ Documents added successfully!")
    
    def save(self, path: Optional[str] = None):
        """
        Save vector store to disk
        
        Args:
            path: Optional custom path to save to
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save. Create one first.")
        
        save_path = path or self.store_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        self.vector_store.save_local(save_path)
        print(f"💾 Vector store saved to {save_path}")
    
    def load(self, path: Optional[str] = None) -> FAISS:
        """
        Load vector store from disk
        
        Args:
            path: Optional custom path to load from
            
        Returns:
            FAISS vector store instance
        """
        load_path = path or self.store_path
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No vector store found at {load_path}")
        
        self.vector_store = FAISS.load_local(
            load_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"📂 Vector store loaded from {load_path}")
        return self.vector_store
    
    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Search for similar documents
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of relevant Document objects
        """
        if self.vector_store is None:
            raise ValueError("No vector store available. Create or load one first.")
        
        k = k or Config.TOP_K_RESULTS
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def similarity_search_with_score(self, query: str, k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with relevance scores
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        if self.vector_store is None:
            raise ValueError("No vector store available. Create or load one first.")
        
        k = k or Config.TOP_K_RESULTS
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results
    
    def exists(self, path: Optional[str] = None) -> bool:
        """Check if a vector store exists at the given path"""
        check_path = path or self.store_path
        return os.path.exists(check_path)
    
    def get_retriever(self, k: Optional[int] = None):
        """
        Get a retriever interface for the vector store
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            Retriever instance
        """
        if self.vector_store is None:
            raise ValueError("No vector store available. Create or load one first.")
        
        k = k or Config.TOP_K_RESULTS
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
