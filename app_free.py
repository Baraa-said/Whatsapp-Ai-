"""
WhatsApp AI RAG Chatbot - Streamlit Interface (Ollama/Free Version)
Uses local Ollama LLM instead of OpenAI - completely FREE!
"""
from __future__ import annotations

import streamlit as st
import os
import sys
import tempfile
from datetime import datetime
from typing import List, Any, Optional
from types import ModuleType

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="WhatsApp AI Chatbot (Free)",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for WhatsApp-like styling
st.markdown("""
<style>
    .main { background-color: #e5ddd5; }
    .chat-header {
        background-color: #075e54;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .stButton>button {
        background-color: #25d366;
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 20px;
    }
    .stButton>button:hover { background-color: #128c7e; }
</style>
""", unsafe_allow_html=True)


# ============ RAG Components (Simplified for Ollama) ============

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Ollama module reference with proper typing
_ollama_module: Optional[ModuleType] = None
_ollama_available: bool = False

try:
    import ollama as _ollama_import
    _ollama_module = _ollama_import
    _ollama_available = True
except ImportError:
    pass

OLLAMA_AVAILABLE: bool = _ollama_available


class SimpleRAG:
    """Simple RAG implementation using free models"""
    
    def __init__(self) -> None:
        # Use free HuggingFace embeddings (runs locally)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store: Optional[FAISS] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
    
    def load_documents(self, file_paths: List[str]) -> int:
        """Load and process documents"""
        all_docs: List[Any] = []
        for path in file_paths:
            try:
                if path.endswith('.pdf'):
                    loader = PyPDFLoader(path)
                else:
                    loader = TextLoader(path, encoding='utf-8')
                docs = loader.load()
                all_docs.extend(docs)
            except Exception:
                st.warning(f"Could not load {path}")
        
        if all_docs:
            chunks = self.text_splitter.split_documents(all_docs)
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            return len(chunks)
        return 0
    
    def search(self, query: str, k: int = 3) -> str:
        """Search for relevant context"""
        if not self.vector_store:
            return ""
        
        docs: List[Any] = self.vector_store.similarity_search(query, k=k)  # type: ignore[reportUnknownMemberType]
        context: str = "\n\n".join([doc.page_content for doc in docs])
        return context
    
    def query_ollama(self, question: str, context: str) -> str:
        """Query Ollama LLM"""
        if not OLLAMA_AVAILABLE or _ollama_module is None:
            return self._fallback_response(question, context)
        
        prompt = f"""You are a helpful WhatsApp AI assistant. Answer the question based on the context provided.
        
Context:
{context}

Question: {question}

Answer concisely and helpfully. If the context doesn't contain relevant information, say so."""

        try:
            response = _ollama_module.chat(
                model='llama3.2',  # or 'mistral', 'llama2', etc.
                messages=[{'role': 'user', 'content': prompt}]
            )
            return str(response['message']['content'])
        except Exception:
            # If Ollama not running, use fallback
            return self._fallback_response(question, context)
    
    def _fallback_response(self, question: str, context: str) -> str:
        """Simple keyword-based response when LLM unavailable"""
        if not context:
            return "❌ No relevant information found in the documents. Please upload documents first."
        
        # Simple extractive response
        sentences: List[str] = context.replace('\n', ' ').split('.')
        relevant: List[str] = [s.strip() for s in sentences if any(
            word.lower() in s.lower() 
            for word in question.lower().split() 
            if len(word) > 3
        )]
        
        if relevant:
            return "📄 Based on the documents:\n\n" + ". ".join(relevant[:3]) + "."
        else:
            return f"📄 Here's what I found:\n\n{context[:500]}..."
    
    def chat(self, question: str) -> str:
        """Main chat function"""
        context = self.search(question)
        return self.query_ollama(question, context)


# ============ Streamlit App ============

def initialize_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag" not in st.session_state:
        st.session_state.rag = None
    if "docs_loaded" not in st.session_state:
        st.session_state.docs_loaded = False


def render_chat_header() -> None:
    st.markdown("""
    <div class="chat-header">
        <h2>💬 WhatsApp AI Assistant</h2>
        <p style="margin: 0; font-size: 0.9em;">🆓 Free Version (Local AI) • Always online</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar() -> None:
    with st.sidebar:
        st.title("📁 Knowledge Base")
        
        # Ollama status
        st.subheader("🤖 AI Status")
        if OLLAMA_AVAILABLE and _ollama_module is not None:
            try:
                _ollama_module.list()
                st.success("✅ Ollama connected")
            except Exception:
                st.warning("⚠️ Ollama not running - using basic mode")
                st.info("Run `ollama serve` and `ollama pull llama3.2`")
        else:
            st.warning("⚠️ Ollama not installed - using basic mode")
            st.info("Install: `pip install ollama`")
        
        st.divider()
        
        # Document upload
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("🚀 Process Documents", use_container_width=True):
                process_documents(uploaded_files)
        
        st.divider()
        
        # Status
        if st.session_state.docs_loaded:
            st.success("✅ Documents loaded!")
        else:
            st.info("📤 Upload documents to start")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        st.markdown("""
        **Free Version Features:**
        - 🆓 No API key needed
        - 💻 Runs 100% locally
        - 🔒 Your data stays private
        - 📚 Uses HuggingFace embeddings
        """)


def process_documents(uploaded_files: Any) -> None:
    with st.spinner("Processing documents (this may take a minute)..."):
        try:
            # Save uploaded files temporarily
            temp_paths: List[str] = []
            for uploaded_file in uploaded_files:
                suffix: str = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    temp_paths.append(tmp.name)
            
            # Initialize RAG and load documents
            st.session_state.rag = SimpleRAG()
            num_chunks = st.session_state.rag.load_documents(temp_paths)
            
            # Cleanup
            for path in temp_paths:
                os.unlink(path)
            
            st.session_state.docs_loaded = True
            st.success(f"✅ Processed {len(uploaded_files)} files into {num_chunks} chunks!")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"👋 Hi! I've learned from {len(uploaded_files)} documents. Ask me anything!",
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def render_chat_messages() -> None:
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            st.caption(message.get("timestamp", ""))


def handle_user_input() -> None:
    if prompt := st.chat_input("Type a message..."):
        timestamp = datetime.now().strftime("%H:%M")
        
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": timestamp
        })
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
            st.caption(timestamp)
        
        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            if st.session_state.rag and st.session_state.docs_loaded:
                with st.spinner("Thinking..."):
                    response = st.session_state.rag.chat(prompt)
            else:
                response = "⚠️ Please upload some documents first using the sidebar!"
            
            st.markdown(response)
            response_time = datetime.now().strftime("%H:%M")
            st.caption(response_time)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": response_time
            })


def main() -> None:
    initialize_session_state()
    render_sidebar()
    render_chat_header()
    render_chat_messages()
    handle_user_input()


if __name__ == "__main__":
    main()
