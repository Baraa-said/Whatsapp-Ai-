"""
RAG Chain Module
Implements the Retrieval-Augmented Generation chain using LangChain
"""

import os
from typing import List, Optional, Dict, Any, Union
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import Config, SYSTEM_PROMPT
from src.vector_store import VectorStore


class RAGChain:
    """
    RAG (Retrieval-Augmented Generation) Chain
    Combines document retrieval with LLM generation for context-aware responses
    """
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize the RAG chain
        
        Args:
            vector_store: Initialized VectorStore instance
        """
        self.vector_store = vector_store
        
        # Initialize the LLM with Groq (fast and free tier available)
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=Config.TEMPERATURE,
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Conversation history (content can be str or list from LLM)
        self.conversation_history: List[Dict[str, Any]] = []
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string"""
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(f"[Document {i} - {source}]\n{doc.page_content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _get_conversation_context(self) -> str:
        """Get recent conversation history as context"""
        if not self.conversation_history:
            return ""
        
        history_str = "\nRecent conversation:\n"
        for msg in self.conversation_history[-6:]:  # Last 3 exchanges
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"
        
        return history_str
    
    def query(self, question: str, include_sources: bool = False) -> Dict[str, Any]:
        """
        Process a user question through the RAG pipeline
        
        Args:
            question: User's question
            include_sources: Whether to include source documents in response
            
        Returns:
            Dictionary with 'answer' and optionally 'sources'
        """
        # Retrieve relevant documents
        relevant_docs = self.vector_store.similarity_search(question)
        
        # Format context
        context = self._format_context(relevant_docs)
        conversation_context = self._get_conversation_context()
        
        # Build the prompt
        system_message = SYSTEM_PROMPT.format(context=context)
        if conversation_context:
            system_message += conversation_context
        
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=question)
        ]
        
        # Generate response
        response = self.llm.invoke(messages)
        answer = response.content
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})
        
        # Build result
        result = {"answer": answer}
        
        if include_sources:
            result["sources"] = [
                {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", "Unknown")
                }
                for doc in relevant_docs
            ]
        
        return result
    
    def query_with_scores(self, question: str) -> Dict[str, Any]:
        """
        Query with relevance scores for debugging
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and scored sources
        """
        # Retrieve with scores
        results_with_scores = self.vector_store.similarity_search_with_score(question)
        
        # Format context from results
        documents = [doc for doc, _ in results_with_scores]
        context = self._format_context(documents)
        
        # Build and send prompt
        system_message = SYSTEM_PROMPT.format(context=context)
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=question)
        ]
        
        response = self.llm.invoke(messages)
        answer = response.content
        
        # Update history
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})
        
        return {
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", "Unknown"),
                    "relevance_score": float(score)
                }
                for doc, score in results_with_scores
            ]
        }
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🗑️ Conversation history cleared")
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history.copy()


class SimpleChatbot:
    """
    Simple chatbot interface that wraps the RAG chain
    Provides an easy-to-use interface for the Streamlit app
    """
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.rag_chain: Optional[RAGChain] = None
        self.is_initialized = False
    
    def initialize(self, documents: Optional[List[Document]] = None):
        """
        Initialize the chatbot with documents
        
        Args:
            documents: Optional list of documents. If not provided,
                      tries to load existing vector store.
        """
        if documents:
            self.vector_store.create_from_documents(documents)
        elif self.vector_store.exists():
            self.vector_store.load()
        else:
            raise ValueError("No documents provided and no existing vector store found")
        
        self.rag_chain = RAGChain(self.vector_store)
        self.is_initialized = True
        print("🤖 Chatbot initialized and ready!")
    
    def chat(self, message: str) -> str:
        """
        Send a message and get a response
        
        Args:
            message: User message
            
        Returns:
            Chatbot response string
        """
        if not self.is_initialized or self.rag_chain is None:
            return "⚠️ Please initialize the chatbot with documents first!"
        
        result = self.rag_chain.query(message)
        answer = result["answer"]
        # Ensure we return a string
        if isinstance(answer, list):
            return str(answer)
        return str(answer)
    
    def chat_with_sources(self, message: str) -> Dict[str, Any]:
        """
        Send a message and get response with sources
        
        Args:
            message: User message
            
        Returns:
            Dictionary with answer and sources
        """
        if not self.is_initialized or self.rag_chain is None:
            return {
                "answer": "⚠️ Please initialize the chatbot with documents first!",
                "sources": []
            }
        
        return self.rag_chain.query(message, include_sources=True)
    
    def clear_chat(self):
        """Clear chat history"""
        if self.rag_chain:
            self.rag_chain.clear_history()
    
    def save_knowledge_base(self):
        """Save the current knowledge base"""
        self.vector_store.save()
    
    def load_knowledge_base(self):
        """Load existing knowledge base"""
        self.vector_store.load()
        self.rag_chain = RAGChain(self.vector_store)
        self.is_initialized = True
