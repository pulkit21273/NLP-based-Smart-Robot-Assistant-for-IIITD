# 🤖 NLP-Based Smart Robot Assistant for IIITD

> **Retrieval-Augmented Generation (RAG) | Large Language Models (LLMs) | LangChain | Qdrant Vector Database | Python | Local LLM Deployment**

A resource-efficient AI assistant designed for academic and administrative query answering using **Retrieval-Augmented Generation (RAG)**. The system combines **LangChain**, **Qdrant**, **LLMs**, and **semantic search** to provide accurate, context-aware responses while running entirely on local hardware with limited resources.

---

## 🚀 Highlights

* ✅ Retrieval-Augmented Generation (RAG) pipeline
* ✅ LangChain-based document ingestion and retrieval
* ✅ Qdrant Vector Database for semantic search
* ✅ Local LLM inference using quantized GGUF models
* ✅ Context-aware conversational memory
* ✅ Offline deployment (No internet required)
* ✅ Streamlit chat interface
* ✅ Automated web crawling and document processing
* ✅ LSTM-based greeting intent classifier
* ✅ Optimized for low-RAM environments (8GB RAM)

---

## 🏗️ System Architecture

```text
User Query
    │
    ▼
Greeting Classifier (LSTM)
    │
 ┌──┴───────────────┐
 │                  │
Greeting        Information Query
 │                  │
 ▼                  ▼
Direct LLM    History-Aware Retriever
Response              │
                       ▼
                Qdrant Vector DB
                       │
                       ▼
                Relevant Documents
                       │
                       ▼
                   LLM (Phi-2)
                       │
                       ▼
                  Final Response
```

---

## 🧠 Tech Stack

### LLMs

* Phi-2 (4-bit Quantized GGUF)
* llama-cpp-python

### RAG Components

* LangChain
* Qdrant Vector Database
* Sentence Transformers
* Recursive Character Text Splitter

### Machine Learning

* PyTorch
* LSTM Classifier
* TF-IDF Vectorization

### Backend

* Python
* Docker

### Frontend

* Streamlit

---

## 🔍 Key Features

### 1. Retrieval-Augmented Generation (RAG)

Instead of relying solely on model knowledge, the assistant retrieves relevant information from institutional documents before generating responses.

**Benefits**

* Reduced hallucinations
* Up-to-date knowledge
* Domain-specific answers
* Lower training costs compared to fine-tuning

---

### 2. Context-Aware Conversations

A history-aware retriever reformulates follow-up questions into standalone queries, allowing the chatbot to maintain conversational context.

**Example**

User:

> What is the SG policy?

User:

> What happens if I violate it?

The system automatically understands that *"it"* refers to the SG policy.

---

### 3. Semantic Search with Qdrant

Documents are converted into embeddings and stored inside Qdrant.

Pipeline:

```text
Documents
    │
    ▼
Text Chunking
    │
    ▼
Embeddings
    │
    ▼
Qdrant Vector Store
    │
    ▼
Similarity Search
```

---

### 4. Local LLM Deployment

To operate under limited hardware constraints:

* 4-bit quantized GGUF models
* llama-cpp-python inference
* No GPU requirement
* Offline functionality

This enables deployment on machines with approximately 8GB RAM.

---

### 5. Automated Knowledge Base Generation

A custom crawler:

* Scrapes institutional webpages
* Downloads PDF, DOCX, and TXT files
* Extracts structured content
* Preserves metadata
* Generates RAG-ready datasets

---

## 📂 Project Structure

```bash
.
├── crawler/
│   ├── crawl.py
│
├── rag/
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│
├── classifier/
│   ├── lstm_classifier.py
│   ├── greetings.csv
│
├── models/
│   └── phi2.gguf
│
├── ui/
│   └── streamlit_app.py
│
├── data/
│   ├── documents/
│   └── processed/
│
├── requirements.txt
└── README.md
```

---

## ⚙️ RAG Pipeline

### Step 1: Data Collection

* Crawl institutional website
* Download approved documents
* Extract textual content

### Step 2: Document Processing

* Cleaning
* Chunking
* Metadata preservation

### Step 3: Embedding Generation

Model:

```python
sentence-transformers/all-MiniLM-L6-v2
```

### Step 4: Vector Storage

Store embeddings in:

```text
Qdrant
```

### Step 5: Retrieval

Retrieve top-k relevant chunks using semantic similarity.

### Step 6: Response Generation

Combine:

* User query
* Chat history
* Retrieved context

Generate final response through the LLM.

---

## 📊 Performance Optimizations

### Memory Optimization

* Quantized GGUF models
* Local inference
* Lightweight embedding model

### Retrieval Optimization

```python
Chunk Size = 500
Chunk Overlap = 100
```

### Conversation Optimization

* History-aware retriever
* Query reformulation
* Context preservation

---

## 🎯 Example Queries

### Academic Queries

```text
What is the SG policy?
```

```text
What is the plagiarism policy?
```

```text
How do semester leaves work?
```

### Administrative Queries

```text
What is the admission withdrawal procedure?
```

```text
Who should I contact for academic grievances?
```

---

## 📈 Results

The system successfully:

* Retrieves policy-related information
* Maintains conversational context
* Operates without internet connectivity
* Runs efficiently on resource-constrained hardware
* Provides relevant responses through semantic retrieval

Additionally, the greeting classifier achieved:

```text
Accuracy: 97.83%
```

on the test dataset.

---

## 🔮 Future Improvements

* Larger local LLMs
* Voice-to-voice interaction
* Hybrid search (BM25 + Vector Search)
* Agentic workflows
* Multi-modal document support
* Integration with physical robotic systems
* Advanced reranking pipelines

---

## 🧑‍💻 Skills Demonstrated

### Generative AI

* Retrieval-Augmented Generation (RAG)
* Prompt Engineering
* Local LLM Deployment
* Quantization

### LLM Frameworks

* LangChain
* llama-cpp-python

### Vector Databases

* Qdrant
* Embedding Pipelines

### Machine Learning

* PyTorch
* LSTM Networks
* TF-IDF

### Software Engineering

* Python
* Docker
* Streamlit
* Web Scraping
* System Design
