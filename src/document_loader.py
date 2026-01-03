"""
Document Loader Module
Handles loading and processing of various document types (PDF, TXT, DOCX, MD)
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Config


class DocumentLoader:
    """Handles loading and chunking of documents for RAG pipeline"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a single document based on its file extension
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata["source"] = os.path.basename(file_path)
            doc.metadata["file_type"] = ext
        
        return documents
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Load multiple documents
        
        Args:
            file_paths: List of paths to document files
            
        Returns:
            List of all Document objects
        """
        all_documents = []
        
        for file_path in file_paths:
            try:
                docs = self.load_document(file_path)
                all_documents.extend(docs)
                print(f"✅ Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {str(e)}")
        
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        chunks = self.text_splitter.split_documents(documents)
        print(f"📄 Split into {len(chunks)} chunks")
        return chunks
    
    def load_and_split(self, file_paths: List[str]) -> List[Document]:
        """
        Load and split documents in one step
        
        Args:
            file_paths: List of paths to document files
            
        Returns:
            List of chunked Document objects
        """
        documents = self.load_documents(file_paths)
        return self.split_documents(documents)
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """
        Load all supported documents from a directory
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            List of chunked Document objects
        """
        file_paths = []
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in Config.SUPPORTED_EXTENSIONS:
                    file_paths.append(os.path.join(root, file))
        
        if not file_paths:
            print(f"⚠️ No supported documents found in {directory_path}")
            return []
        
        print(f"📁 Found {len(file_paths)} documents in {directory_path}")
        return self.load_and_split(file_paths)


# Convenience function
def load_documents(file_paths: List[str]) -> List[Document]:
    """Quick function to load and split documents"""
    loader = DocumentLoader()
    return loader.load_and_split(file_paths)
