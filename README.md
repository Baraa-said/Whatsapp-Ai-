# 🤖 WhatsApp AI RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot with a WhatsApp-style interface. Built with Python, OpenAI, FAISS, and Streamlit for a final assignment project.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a **RAG (Retrieval-Augmented Generation)** chatbot that:
1. Ingests documents (PDF, TXT, DOCX, Markdown)
2. Creates vector embeddings using OpenAI
3. Stores embeddings in FAISS vector database
4. Retrieves relevant context for user queries
5. Generates intelligent responses using GPT-3.5/4

The interface simulates a WhatsApp chat experience using Streamlit.

## ✨ Features

- 📄 **Multi-format Document Support**: PDF, TXT, DOCX, Markdown
- 🧠 **RAG Pipeline**: Context-aware responses based on your documents
- 💬 **WhatsApp-style UI**: Familiar chat interface
- 🔍 **Semantic Search**: FAISS vector similarity search
- 💾 **Persistent Storage**: Save and load knowledge bases
- 📚 **Source Attribution**: See which documents informed each response
- 🔄 **Conversation Memory**: Maintains chat context

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                   (Streamlit WhatsApp UI)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG PIPELINE                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │  Document   │───▶│   Vector    │───▶│    RAG Chain        │ │
│  │   Loader    │    │   Store     │    │  (Query + Generate) │ │
│  └─────────────┘    └─────────────┘    └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │    OpenAI     │ │    FAISS      │ │   LangChain   │
    │  Embeddings   │ │  Vector DB    │ │   Framework   │
    │    + LLM      │ │               │ │               │
    └───────────────┘ └───────────────┘ └───────────────┘
```

### RAG Flow Diagram

```
User Query ──▶ Embedding ──▶ Vector Search ──▶ Retrieve Context
                                                     │
                                                     ▼
Response ◀── LLM Generation ◀── Prompt + Context ◀──┘
```

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- OpenAI API key

### Step 1: Clone the Repository

```bash
git clone https://github.com/Baraa-said/Whatsapp-Ai-.git
cd Whatsapp-Ai-
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-api-key-here
```

## 📖 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Chatbot

1. **Enter API Key**: Input your OpenAI API key in the sidebar
2. **Upload Documents**: Use the file uploader to add your documents
3. **Process Documents**: Click "Process Documents" to create the knowledge base
4. **Start Chatting**: Type your questions in the chat input

### Example Queries

```
"What is the main topic of the documents?"
"Summarize the key points about [topic]"
"What does the document say about [specific subject]?"
```

## 📁 Project Structure

```
Whatsapp-Ai-/
├── app.py                      # Streamlit application (main entry)
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
│
├── src/                       # Source code package
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration settings
│   ├── document_loader.py    # Document processing module
│   ├── vector_store.py       # FAISS vector database module
│   └── rag_chain.py          # RAG pipeline implementation
│
├── data/                      # Data directory
│   ├── documents/            # Sample documents (optional)
│   └── vector_store/         # Persisted vector database
│
├── docs/                      # Documentation
│   ├── architecture.md       # Detailed architecture docs
│   └── report.md             # Project report
│
└── tests/                     # Unit tests
    └── test_rag.py           # Test cases
```

## ⚙️ How It Works

### 1. Document Ingestion
```python
# Documents are loaded and split into chunks
loader = DocumentLoader()
chunks = loader.load_and_split(["document.pdf"])
```

### 2. Vector Embedding
```python
# Chunks are converted to vectors using OpenAI embeddings
vector_store = VectorStore()
vector_store.create_from_documents(chunks)
```

### 3. Query Processing
```python
# User query is embedded and similar chunks are retrieved
relevant_docs = vector_store.similarity_search(query)
```

### 4. Response Generation
```python
# Retrieved context + query sent to LLM for response
rag_chain = RAGChain(vector_store)
response = rag_chain.query("What is the document about?")
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `OPENAI_MODEL` | LLM model to use | `gpt-3.5-turbo` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-ada-002` |
| `CHUNK_SIZE` | Document chunk size | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |

### Customization

Edit `src/config.py` to modify:
- Number of retrieved documents (`TOP_K_RESULTS`)
- LLM temperature (`TEMPERATURE`)
- System prompt (`SYSTEM_PROMPT`)

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src
```

## 📊 Performance Considerations

- **Chunk Size**: Larger chunks = more context but slower retrieval
- **Top-K**: More results = better coverage but more tokens used
- **Model Choice**: GPT-4 is more accurate but slower and costlier

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI](https://openai.com/) for GPT and embedding models
- [LangChain](https://langchain.com/) for the RAG framework
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Streamlit](https://streamlit.io/) for the web interface

---

**Made with ❤️ for AI/ML Learning**
