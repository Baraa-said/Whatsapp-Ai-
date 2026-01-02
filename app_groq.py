"""
WhatsApp AI RAG Chatbot - Streamlit Interface (Groq FREE Version)
Uses Groq's FREE API with Llama 3 models - No cost!
Get your free API key at: https://console.groq.com/keys
"""
from __future__ import annotations

import streamlit as st
import os
import sys
import tempfile
from datetime import datetime
from typing import List, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="WhatsApp AI Chatbot (Free Groq)",
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
    .api-info {
        background-color: #dcf8c6;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============ RAG Components with Groq ============

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq


class GroqRAG:
    """RAG implementation using FREE Groq API"""
    
    def __init__(self, api_key: str) -> None:
        self.client = Groq(api_key=api_key)
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
    
    def query_groq(self, question: str, context: str) -> str:
        """Query Groq FREE API"""
        prompt = f"""You are a helpful WhatsApp AI assistant. Answer the question based on the context provided.
        
Context:
{context}

Question: {question}

Answer concisely and helpfully. If the context doesn't contain relevant information, say so. Use emojis occasionally! 😊"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",  # FREE model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content or "Sorry, I couldn't generate a response."
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def chat(self, question: str) -> str:
        """Main chat function"""
        context = self.search(question)
        if not context:
            return "⚠️ Please upload some documents first so I can help you!"
        return self.query_groq(question, context)


# ============ Streamlit App ============

def initialize_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag" not in st.session_state:
        st.session_state.rag = None
    if "docs_loaded" not in st.session_state:
        st.session_state.docs_loaded = False
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = ""


def render_chat_header() -> None:
    st.markdown("""
    <div class="chat-header">
        <h2>💬 WhatsApp AI Assistant</h2>
        <p style="margin: 0; font-size: 0.9em;">🆓 FREE Version (Groq + Llama 3) • Always online</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar() -> None:
    with st.sidebar:
        st.title("📁 Knowledge Base")
        
        # API Key input
        st.subheader("🔑 Groq API Key (FREE)")
        st.markdown("""
        <div class="api-info">
        Get your FREE API key at:<br>
        <a href="https://console.groq.com/keys" target="_blank">console.groq.com/keys</a>
        </div>
        """, unsafe_allow_html=True)
        
        api_key = st.text_input(
            "Enter Groq API Key",
            type="password",
            value=st.session_state.groq_api_key,
            help="Free tier includes 14,400 requests/day!"
        )
        
        if api_key != st.session_state.groq_api_key:
            st.session_state.groq_api_key = api_key
            st.session_state.rag = None
            st.session_state.docs_loaded = False
        
        if api_key:
            st.success("✅ API Key set!")
        else:
            st.warning("⚠️ Enter your free Groq API key")
        
        st.divider()
        
        # Document upload
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True
        )
        
        if uploaded_files and api_key:
            if st.button("🚀 Process Documents", use_container_width=True):
                process_documents(uploaded_files, api_key)
        elif uploaded_files and not api_key:
            st.info("👆 Enter API key first")
        
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
        **🆓 Groq Free Tier:**
        - ✅ 14,400 requests/day
        - ✅ Llama 3.1 8B model
        - ✅ Super fast responses
        - ✅ No credit card needed
        """)


def process_documents(uploaded_files: Any, api_key: str) -> None:
    with st.spinner("Processing documents..."):
        try:
            # Save uploaded files temporarily
            temp_paths: List[str] = []
            for uploaded_file in uploaded_files:
                suffix: str = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    temp_paths.append(tmp.name)
            
            # Initialize RAG and load documents
            st.session_state.rag = GroqRAG(api_key)
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
            elif not st.session_state.groq_api_key:
                response = "⚠️ Please enter your free Groq API key in the sidebar!"
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
