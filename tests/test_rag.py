"""
Unit Tests for RAG Chatbot Components
"""

import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfig:
    """Tests for configuration module"""
    
    def test_config_defaults(self):
        """Test default configuration values"""
        from src.config import Config
        
        assert Config.CHUNK_SIZE == 1000
        assert Config.CHUNK_OVERLAP == 200
        assert Config.TOP_K_RESULTS == 4
        assert Config.TEMPERATURE == 0.7
    
    def test_supported_extensions(self):
        """Test supported file extensions"""
        from src.config import Config
        
        assert ".pdf" in Config.SUPPORTED_EXTENSIONS
        assert ".txt" in Config.SUPPORTED_EXTENSIONS
        assert ".docx" in Config.SUPPORTED_EXTENSIONS
        assert ".md" in Config.SUPPORTED_EXTENSIONS


class TestDocumentLoader:
    """Tests for document loading functionality"""
    
    @patch('src.document_loader.TextLoader')
    def test_load_txt_document(self, mock_loader):
        """Test loading a text file"""
        from src.document_loader import DocumentLoader
        
        # Setup mock
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {}
        mock_loader.return_value.load.return_value = [mock_doc]
        
        loader = DocumentLoader()
        # Note: This would need actual file for integration test
    
    def test_unsupported_file_type(self):
        """Test handling of unsupported file types"""
        from src.document_loader import DocumentLoader
        
        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load_document("test.xyz")


class TestVectorStore:
    """Tests for vector store functionality"""
    
    @patch('src.vector_store.OpenAIEmbeddings')
    @patch('src.vector_store.Config')
    def test_vector_store_initialization(self, mock_config, mock_embeddings):
        """Test vector store initialization"""
        mock_config.OPENAI_API_KEY = "test-key"
        mock_config.EMBEDDING_MODEL = "text-embedding-ada-002"
        mock_config.VECTOR_STORE_PATH = "./test_store"
        mock_config.validate.return_value = True
        
        from src.vector_store import VectorStore
        
        store = VectorStore()
        assert store.vector_store is None
    
    def test_similarity_search_without_store(self):
        """Test that similarity search fails without initialized store"""
        with patch('src.vector_store.Config') as mock_config:
            mock_config.OPENAI_API_KEY = "test-key"
            mock_config.validate.return_value = True
            
            from src.vector_store import VectorStore
            
            store = VectorStore()
            with pytest.raises(ValueError, match="No vector store available"):
                store.similarity_search("test query")


class TestRAGChain:
    """Tests for RAG chain functionality"""
    
    def test_format_context_empty(self):
        """Test context formatting with no documents"""
        with patch('src.rag_chain.Config') as mock_config:
            mock_config.OPENAI_API_KEY = "test-key"
            mock_config.OPENAI_MODEL = "gpt-3.5-turbo"
            mock_config.TEMPERATURE = 0.7
            mock_config.MAX_TOKENS = 1000
            mock_config.validate.return_value = True
            
            with patch('src.rag_chain.ChatOpenAI'):
                from src.rag_chain import RAGChain
                
                mock_vector_store = Mock()
                chain = RAGChain(mock_vector_store)
                
                result = chain._format_context([])
                assert result == "No relevant context found."
    
    def test_format_context_with_documents(self):
        """Test context formatting with documents"""
        with patch('src.rag_chain.Config') as mock_config:
            mock_config.OPENAI_API_KEY = "test-key"
            mock_config.OPENAI_MODEL = "gpt-3.5-turbo"
            mock_config.TEMPERATURE = 0.7
            mock_config.MAX_TOKENS = 1000
            mock_config.validate.return_value = True
            
            with patch('src.rag_chain.ChatOpenAI'):
                from src.rag_chain import RAGChain
                
                mock_vector_store = Mock()
                chain = RAGChain(mock_vector_store)
                
                # Create mock documents
                doc1 = Mock()
                doc1.page_content = "Content 1"
                doc1.metadata = {"source": "doc1.pdf"}
                
                doc2 = Mock()
                doc2.page_content = "Content 2"
                doc2.metadata = {"source": "doc2.pdf"}
                
                result = chain._format_context([doc1, doc2])
                
                assert "Content 1" in result
                assert "Content 2" in result
                assert "doc1.pdf" in result
                assert "doc2.pdf" in result
    
    def test_clear_history(self):
        """Test clearing conversation history"""
        with patch('src.rag_chain.Config') as mock_config:
            mock_config.OPENAI_API_KEY = "test-key"
            mock_config.OPENAI_MODEL = "gpt-3.5-turbo"
            mock_config.TEMPERATURE = 0.7
            mock_config.MAX_TOKENS = 1000
            mock_config.validate.return_value = True
            
            with patch('src.rag_chain.ChatOpenAI'):
                from src.rag_chain import RAGChain
                
                mock_vector_store = Mock()
                chain = RAGChain(mock_vector_store)
                
                # Add some history
                chain.conversation_history = [
                    {"role": "user", "content": "test"},
                    {"role": "assistant", "content": "response"}
                ]
                
                chain.clear_history()
                
                assert len(chain.conversation_history) == 0


class TestIntegration:
    """Integration tests (require API key)"""
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_full_pipeline(self):
        """Test full RAG pipeline (requires API key)"""
        # This would be a full integration test
        # Skipped by default as it requires API key
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
