# RAG Chatbot Architecture Documentation

## System Architecture Overview

This document provides a comprehensive overview of the WhatsApp AI RAG Chatbot architecture.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                           WhatsApp AI RAG Chatbot                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │
┌─────────────────────────────────────┴─────────────────────────────────────┐
│                                                                           │
│                         PRESENTATION LAYER                                │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                                                                     │ │
│  │                    Streamlit Web Application                        │ │
│  │                                                                     │ │
│  │  ┌─────────────┐  ┌─────────────────┐  ┌──────────────────────┐   │ │
│  │  │   Sidebar   │  │   Chat Window   │  │   Message History    │   │ │
│  │  │  - Upload   │  │  - User Input   │  │   - Conversation     │   │ │
│  │  │  - Settings │  │  - AI Response  │  │   - Sources          │   │ │
│  │  │  - Status   │  │  - Timestamps   │  │   - Timestamps       │   │ │
│  │  └─────────────┘  └─────────────────┘  └──────────────────────┘   │ │
│  │                                                                     │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│                          APPLICATION LAYER                                │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                                                                     │ │
│  │                        RAG Pipeline                                 │ │
│  │                                                                     │ │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐ │ │
│  │  │              │    │              │    │                      │ │ │
│  │  │   Document   │───▶│   Vector     │───▶│      RAG Chain       │ │ │
│  │  │    Loader    │    │    Store     │    │                      │ │ │
│  │  │              │    │              │    │  ┌────────────────┐  │ │ │
│  │  │  - PDF       │    │  - FAISS     │    │  │   Retriever    │  │ │ │
│  │  │  - TXT       │    │  - Embeddings│    │  │                │  │ │ │
│  │  │  - DOCX      │    │  - Persist   │    │  │   ┌────────┐   │  │ │ │
│  │  │  - Markdown  │    │              │    │  │   │ Search │   │  │ │ │
│  │  │              │    │              │    │  │   └────────┘   │  │ │ │
│  │  └──────────────┘    └──────────────┘    │  │                │  │ │ │
│  │                                          │  └────────────────┘  │ │ │
│  │                                          │                      │ │ │
│  │                                          │  ┌────────────────┐  │ │ │
│  │                                          │  │   Generator    │  │ │ │
│  │                                          │  │                │  │ │ │
│  │                                          │  │   ┌────────┐   │  │ │ │
│  │                                          │  │   │  LLM   │   │  │ │ │
│  │                                          │  │   └────────┘   │  │ │ │
│  │                                          │  │                │  │ │ │
│  │                                          │  └────────────────┘  │ │ │
│  │                                          └──────────────────────┘ │ │
│  │                                                                     │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│                           DATA LAYER                                      │
│                                                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │                 │  │                 │  │                         │  │
│  │    Document     │  │    Vector       │  │     Conversation        │  │
│  │    Storage      │  │    Database     │  │     History             │  │
│  │                 │  │                 │  │                         │  │
│  │  /data/docs/    │  │  FAISS Index    │  │  Session State          │  │
│  │                 │  │  (Local/Memory) │  │  (In-Memory)            │  │
│  │                 │  │                 │  │                         │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│                        EXTERNAL SERVICES                                  │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                                                                  │    │
│  │                          OpenAI API                              │    │
│  │                                                                  │    │
│  │  ┌────────────────────┐    ┌────────────────────────────────┐   │    │
│  │  │                    │    │                                │   │    │
│  │  │  Embeddings API    │    │         Chat API               │   │    │
│  │  │                    │    │                                │   │    │
│  │  │  text-embedding-   │    │     gpt-3.5-turbo / gpt-4     │   │    │
│  │  │  ada-002           │    │                                │   │    │
│  │  │                    │    │                                │   │    │
│  │  └────────────────────┘    └────────────────────────────────┘   │    │
│  │                                                                  │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Document Loader (`src/document_loader.py`)

Responsible for loading and preprocessing documents.

**Capabilities:**
- Load multiple file formats (PDF, TXT, DOCX, Markdown)
- Split documents into chunks
- Maintain metadata (source, file type)

**Flow:**
```
Documents → Load → Split → Chunks (with metadata)
```

### 2. Vector Store (`src/vector_store.py`)

Manages document embeddings and similarity search.

**Capabilities:**
- Create embeddings using OpenAI
- Store vectors in FAISS
- Perform similarity search
- Persist and load from disk

**Flow:**
```
Chunks → Embedding → FAISS Index → Save/Load
```

### 3. RAG Chain (`src/rag_chain.py`)

Implements the retrieval-augmented generation pipeline.

**Capabilities:**
- Query processing
- Context retrieval
- Response generation
- Conversation memory

**Flow:**
```
Query → Embed → Search → Retrieve → Prompt → LLM → Response
```

## Data Flow

### Document Ingestion Flow

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Upload  │───▶│   Load   │───▶│  Split   │───▶│  Embed   │───▶│  Store   │
│  Files   │    │Documents │    │ Chunks   │    │ Vectors  │    │  FAISS   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### Query Processing Flow

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  User    │───▶│  Embed   │───▶│ Similar  │───▶│  Build   │───▶│ Generate │
│  Query   │    │  Query   │    │  Search  │    │  Prompt  │    │ Response │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                     │                               │
                                     ▼                               ▼
                              ┌──────────┐                    ┌──────────┐
                              │ Relevant │                    │   AI     │
                              │  Chunks  │                    │ Response │
                              └──────────┘                    └──────────┘
```

## Key Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| LLM | OpenAI GPT-3.5/4 | Response generation |
| Embeddings | text-embedding-ada-002 | Vector embeddings |
| Vector DB | FAISS | Similarity search |
| Framework | LangChain | RAG orchestration |
| UI | Streamlit | Web interface |

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: API authentication
- `OPENAI_MODEL`: LLM model selection
- `EMBEDDING_MODEL`: Embedding model
- `CHUNK_SIZE`: Document chunk size
- `CHUNK_OVERLAP`: Overlap between chunks

### Tunable Parameters
- `TOP_K_RESULTS`: Number of retrieved chunks
- `TEMPERATURE`: LLM creativity level
- `MAX_TOKENS`: Response length limit

## Security Considerations

1. **API Key Management**: Keys stored in environment variables
2. **Data Privacy**: Documents processed locally
3. **Input Validation**: Sanitized user inputs

## Scalability Notes

- FAISS supports millions of vectors
- Can switch to cloud vector DB (Pinecone, Weaviate)
- LLM calls can be rate-limited
- Consider caching for frequently asked questions
