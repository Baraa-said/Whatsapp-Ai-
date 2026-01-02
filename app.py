"""
WhatsApp AI RAG Chatbot - Streamlit Interface
A WhatsApp-style chat interface for the RAG chatbot
"""

import streamlit as st
import os
import sys
import tempfile
from datetime import datetime
from typing import Any, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.document_loader import DocumentLoader
from src.vector_store import VectorStore
from src.rag_chain import RAGChain


# Page configuration
st.set_page_config(
    page_title="WhatsApp AI Chatbot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for WhatsApp-like styling
st.markdown("""
<style>
    /* Main chat container */
    .main {
        background-color: #e5ddd5;
    }
    
    /* Chat message styling */
    .user-message {
        background-color: #dcf8c6;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 5px 0;
        max-width: 80%;
        margin-left: auto;
        text-align: right;
    }
    
    .bot-message {
        background-color: white;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 5px 0;
        max-width: 80%;
    }
    
    /* Header styling */
    .chat-header {
        background-color: #075e54;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    /* Timestamp styling */
    .timestamp {
        font-size: 0.7em;
        color: #999;
        margin-top: 5px;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f0f0f0;
    }
    
    /* Custom button styling */
    .stButton>button {
        background-color: #25d366;
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 20px;
    }
    
    .stButton>button:hover {
        background-color: #128c7e;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    
    if "document_count" not in st.session_state:
        st.session_state.document_count = 0


def render_chat_header():
    """Render WhatsApp-style chat header"""
    st.markdown("""
    <div class="chat-header">
        <h2>💬 WhatsApp AI Assistant</h2>
        <p style="margin: 0; font-size: 0.9em;">RAG-powered • Always online</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with document upload and settings"""
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg", width=50)
        st.title("📁 Knowledge Base")
        
        # API Key input
        st.subheader("🔑 API Configuration")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=Config.OPENAI_API_KEY,
            help="Enter your OpenAI API key"
        )
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            Config.OPENAI_API_KEY = api_key
        
        st.divider()
        
        # Document upload section
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, DOCX, Markdown"
        )
        
        if uploaded_files:
            if st.button("🚀 Process Documents", use_container_width=True):
                process_documents(uploaded_files)
        
        st.divider()
        
        # Status section
        st.subheader("📊 Status")
        if st.session_state.documents_loaded:
            st.success(f"✅ {st.session_state.document_count} documents loaded")
        else:
            st.warning("⚠️ No documents loaded yet")
        
        # Load existing knowledge base
        if st.button("📂 Load Existing KB", use_container_width=True):
            load_existing_knowledge_base()
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.rag_chain:
                st.session_state.rag_chain.clear_history()
            st.rerun()
        
        # About section
        st.divider()
        st.subheader("ℹ️ About")
        st.markdown("""
        **WhatsApp AI RAG Chatbot**
        
        This chatbot uses:
        - 🧠 OpenAI GPT for responses
        - 📚 FAISS for vector search
        - 🔗 LangChain for RAG pipeline
        
        Upload documents to create your knowledge base!
        """)


def process_documents(uploaded_files: Any) -> None:
    """Process uploaded documents and create vector store"""
    if not Config.OPENAI_API_KEY:
        st.error("❌ Please enter your OpenAI API key first!")
        return
    
    with st.spinner("Processing documents..."):
        try:
            # Save uploaded files temporarily
            temp_paths: List[str] = []
            for uploaded_file in uploaded_files:
                suffix: str = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    temp_paths.append(tmp.name)
            
            # Load and process documents
            loader = DocumentLoader()
            documents = loader.load_and_split(temp_paths)
            
            # Create vector store
            st.session_state.vector_store = VectorStore()
            st.session_state.vector_store.create_from_documents(documents)
            st.session_state.vector_store.save()
            
            # Initialize RAG chain
            st.session_state.rag_chain = RAGChain(st.session_state.vector_store)
            
            # Update state
            st.session_state.documents_loaded = True
            st.session_state.document_count = len(uploaded_files)
            
            # Cleanup temp files
            for path in temp_paths:
                os.unlink(path)
            
            st.success(f"✅ Successfully processed {len(uploaded_files)} documents!")
            
            # Add welcome message
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"👋 Hello! I've learned from {len(uploaded_files)} documents. Ask me anything about them!",
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")


def load_existing_knowledge_base():
    """Load an existing knowledge base from disk"""
    if not Config.OPENAI_API_KEY:
        st.error("❌ Please enter your OpenAI API key first!")
        return
    
    try:
        st.session_state.vector_store = VectorStore()
        
        if st.session_state.vector_store.exists():
            st.session_state.vector_store.load()
            st.session_state.rag_chain = RAGChain(st.session_state.vector_store)
            st.session_state.documents_loaded = True
            st.session_state.document_count = "?"  # Unknown when loading existing
            
            st.success("✅ Knowledge base loaded successfully!")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": "👋 I've loaded the existing knowledge base. How can I help you?",
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            st.rerun()
        else:
            st.warning("⚠️ No existing knowledge base found. Please upload documents first.")
            
    except Exception as e:
        st.error(f"❌ Error loading knowledge base: {str(e)}")


def render_chat_messages():
    """Render chat messages in WhatsApp style"""
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message["content"])
                st.caption(message.get("timestamp", ""))
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])
                st.caption(message.get("timestamp", ""))


def handle_user_input():
    """Handle user input and generate response"""
    if prompt := st.chat_input("Type a message..."):
        # Add user message
        timestamp = datetime.now().strftime("%H:%M")
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": timestamp
        })
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
            st.caption(timestamp)
        
        # Generate response
        if st.session_state.rag_chain:
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.rag_chain.query(prompt, include_sources=True)
                        response = result["answer"]
                        
                        st.markdown(response)
                        
                        # Show sources in expander
                        if result.get("sources"):
                            with st.expander("📚 Sources"):
                                for i, source in enumerate(result["sources"], 1):
                                    st.markdown(f"**{i}. {source['source']}**")
                                    st.caption(source["content"])
                        
                        response_timestamp = datetime.now().strftime("%H:%M")
                        st.caption(response_timestamp)
                        
                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "timestamp": response_timestamp
                        })
                        
                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "timestamp": datetime.now().strftime("%H:%M")
                        })
        else:
            # No documents loaded
            no_docs_msg = "⚠️ Please upload documents first using the sidebar to start chatting!"
            with st.chat_message("assistant", avatar="🤖"):
                st.warning(no_docs_msg)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": no_docs_msg,
                "timestamp": datetime.now().strftime("%H:%M")
            })


def main():
    """Main application function"""
    initialize_session_state()
    render_sidebar()
    render_chat_header()
    render_chat_messages()
    handle_user_input()


if __name__ == "__main__":
    main()
