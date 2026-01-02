"""
WhatsApp AI RAG Chatbot - Simple Demo Version
Works WITHOUT any API key - uses basic keyword search
Perfect for testing the interface!
"""

import streamlit as st
import os
from datetime import datetime
from typing import List, Dict, Any
import re

# Page configuration
st.set_page_config(
    page_title="WhatsApp AI Chatbot (Demo)",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    }
    .stButton>button:hover { background-color: #128c7e; }
</style>
""", unsafe_allow_html=True)


class SimpleSearchEngine:
    """Simple keyword-based search engine - no API needed!"""
    
    def __init__(self):
        self.documents: List[Dict[str, str]] = []
        self.chunks: List[Dict[str, str]] = []
    
    def add_document(self, content: str, filename: str):
        """Add a document to the search index"""
        self.documents.append({"content": content, "source": filename})
        
        # Split into chunks
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            if len(para.strip()) > 50:  # Skip very short chunks
                self.chunks.append({
                    "content": para.strip(),
                    "source": filename
                })
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """Search for relevant chunks using keyword matching"""
        if not self.chunks:
            return []
        
        query_words = set(re.findall(r'\w+', query.lower()))
        query_words = {w for w in query_words if len(w) > 2}  # Remove short words
        
        scored_chunks: List[tuple[float, Dict[str, str]]] = []
        for chunk in self.chunks:
            chunk_words = set(re.findall(r'\w+', chunk["content"].lower()))
            
            # Calculate relevance score
            matches = query_words & chunk_words
            score = len(matches) / (len(query_words) + 0.1)
            
            if score > 0:
                scored_chunks.append((score, chunk))
        
        # Sort by score and return top results
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]
    
    def generate_response(self, query: str) -> str:
        """Generate a response based on search results"""
        results = self.search(query)
        
        if not results:
            return "🤔 I couldn't find relevant information about that in the uploaded documents. Try asking something else or upload more documents!"
        
        # Build response from relevant chunks
        response_parts: List[str] = ["📄 **Based on the documents:**\n"]
        
        sources: set[str] = set()
        for result in results[:3]:
            content = result["content"]
            source = result["source"]
            sources.add(source)
            
            # Clean up the content
            if len(content) > 300:
                content = content[:300] + "..."
            
            response_parts.append(f"\n{content}\n")
        
        response_parts.append(f"\n\n📚 *Sources: {', '.join(sources)}*")
        
        return "".join(response_parts)


# Initialize session state
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "engine" not in st.session_state:
        st.session_state.engine = SimpleSearchEngine()
    if "docs_loaded" not in st.session_state:
        st.session_state.docs_loaded = False


def render_header():
    st.markdown("""
    <div class="chat-header">
        <h2>💬 WhatsApp AI Assistant</h2>
        <p style="margin: 0; font-size: 0.9em;">🆓 Demo Version • No API Key Needed!</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.title("📁 Knowledge Base")
        
        st.success("✅ No API key required!")
        st.info("This demo uses keyword search to find answers in your documents.")
        
        st.divider()
        
        # Document upload
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload TXT or MD files",
            type=["txt", "md"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("🚀 Process Documents", use_container_width=True):
                process_files(uploaded_files)
        
        # Load sample documents
        st.divider()
        if st.button("📚 Load Sample Documents", use_container_width=True):
            load_sample_docs()
        
        st.divider()
        
        # Status
        if st.session_state.docs_loaded:
            st.success(f"✅ {len(st.session_state.engine.chunks)} chunks indexed")
        else:
            st.warning("📤 Upload or load sample documents")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        st.markdown("""
        **Demo Features:**
        - 🆓 100% Free
        - 🔒 Runs locally
        - 📝 Keyword-based search
        - 📄 Supports TXT & MD
        """)


def process_files(uploaded_files: Any) -> None:
    """Process uploaded files"""
    st.session_state.engine = SimpleSearchEngine()
    
    for f in uploaded_files:
        content = f.getvalue().decode('utf-8')
        st.session_state.engine.add_document(content, f.name)
    
    st.session_state.docs_loaded = True
    st.success(f"✅ Loaded {len(uploaded_files)} files!")
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"👋 I've loaded {len(uploaded_files)} documents! Ask me anything about them.",
        "timestamp": datetime.now().strftime("%H:%M")
    })
    st.rerun()


def load_sample_docs():
    """Load sample documents from data folder"""
    sample_dir = os.path.join(os.path.dirname(__file__), "data", "documents")
    
    st.session_state.engine = SimpleSearchEngine()
    count = 0
    
    if os.path.exists(sample_dir):
        for filename in os.listdir(sample_dir):
            if filename.endswith(('.txt', '.md')):
                filepath = os.path.join(sample_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    st.session_state.engine.add_document(content, filename)
                    count += 1
                except Exception:
                    st.warning(f"Could not load {filename}")
    
    if count > 0:
        st.session_state.docs_loaded = True
        st.success(f"✅ Loaded {count} sample documents!")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"👋 I've loaded {count} sample documents! Try asking:\n- 'What is RAG?'\n- 'How many sick days do I get?'\n- 'What is the price of Model X Pro?'",
            "timestamp": datetime.now().strftime("%H:%M")
        })
        st.rerun()
    else:
        st.warning("No sample documents found")


def render_messages():
    """Render chat messages"""
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            st.caption(msg.get("timestamp", ""))


def handle_input():
    """Handle user input"""
    if prompt := st.chat_input("Type a message..."):
        timestamp = datetime.now().strftime("%H:%M")
        
        # User message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": timestamp
        })
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
            st.caption(timestamp)
        
        # Bot response
        with st.chat_message("assistant", avatar="🤖"):
            if st.session_state.docs_loaded:
                response = st.session_state.engine.generate_response(prompt)
            else:
                response = "⚠️ Please upload documents or click 'Load Sample Documents' first!"
            
            st.markdown(response)
            resp_time = datetime.now().strftime("%H:%M")
            st.caption(resp_time)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": resp_time
            })


def main():
    init_session()
    render_sidebar()
    render_header()
    render_messages()
    handle_input()


if __name__ == "__main__":
    main()
