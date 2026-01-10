# Cortex - Intelligent RAG Chatbot

🚀 **Live Demo**: [https://junaidariie.github.io/Cortex/](https://junaidariie.github.io/Cortex/)

## Architecture Overview

![Cortex Architecture]()
<img width="390" height="333" alt="image" src="https://github.com/user-attachments/assets/fe1f31ec-0e5c-4164-8fa9-4e0743e4f488" />


## Overview

Cortex is an advanced conversational AI system that combines Retrieval-Augmented Generation (RAG) with real-time web search capabilities. Built with FastAPI and LangGraph, it provides intelligent responses using multiple LLM models and supports document ingestion, voice interactions, and streaming responses.

## ✨ Features

- **🤖 Multi-Model Support**: Choose from GPT-4, Groq models (Kimi2, Llama4, Qwen3), and more
- **📚 RAG Capabilities**: Upload and query PDF documents with intelligent context retrieval
- **🌐 Real-time Web Search**: Get up-to-date information using Tavily search integration
- **🎤 Speech-to-Text**: Convert audio to text using Whisper models
- **🗣️ Text-to-Speech**: Generate natural speech using Edge TTS
- **💬 Streaming Responses**: Real-time response streaming for better user experience
- **🧠 Memory Management**: Persistent conversation history with thread-based sessions
- **📄 Document Processing**: Automatic PDF ingestion and vectorization

## 🏗️ Architecture

### Core Components

1. **FastAPI Backend** (`app.py`): Main API server handling HTTP requests
2. **RAG Engine** (`RAG.py`): LangGraph-based conversation flow with state management
3. **Document Ingestion** (`data_ingestion.py`): PDF processing and vector store creation
4. **Utilities** (`utils.py`): Speech processing and audio generation

### Technology Stack

- **Backend**: FastAPI, Python 3.11+
- **AI/ML**: LangChain, LangGraph, OpenAI, Groq
- **Vector Database**: FAISS
- **Search**: Tavily API
- **Speech**: Whisper (STT), Edge TTS (TTS)
- **Document Processing**: PyPDF, RecursiveCharacterTextSplitter

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API Key
- Groq API Key
- Tavily API Key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd try_rag_bot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv myvenv
   source myvenv/bin/activate  # On Windows: myvenv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   GROQ_API_KEY=your_groq_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

5. **Run the application**
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

## 📖 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```http
GET /
```
**Response:**
```json
{
  "status": "running",
  "message": "Bot is ready"
}
```

#### 2. Chat Endpoint
```http
POST /chat
```
**Request Body:**
```json
{
  "query": "Your question here",
  "thread_id": "user_session_id",
  "use_rag": false,
  "use_web": false,
  "model_name": "gpt"
}
```

**Parameters:**
- `query` (string): User's question or message
- `thread_id` (string): Session identifier for conversation history
- `use_rag` (boolean): Enable document-based retrieval
- `use_web` (boolean): Enable real-time web search
- `model_name` (string): LLM model choice (`gpt`, `kimi2`, `lamma4`, `qwen3`, `gpt_oss`)

**Response:** Server-Sent Events (SSE) stream

#### 3. Document Upload
```http
POST /upload
```
**Request:** Multipart form data with PDF file
**Response:**
```json
{
  "message": "File received. Processing started in background.",
  "filename": "document.pdf"
}
```

#### 4. Speech-to-Text
```http
POST /stt
```
**Request:** Multipart form data with audio file
**Response:**
```json
{
  "text": "Transcribed text",
  "segments": [...],
  "language": "en"
}
```

#### 5. Text-to-Speech
```http
POST /tts
```
**Request Body:**
```json
{
  "text": "Text to convert to speech",
  "voice": "en-US-AriaNeural"
}
```
**Response:** Audio file (MP3)

## 🔧 Configuration

### Available Models

| Model Name | Provider | Description |
|------------|----------|-------------|
| `gpt` | OpenAI | GPT-4.1-nano (default) |
| `kimi2` | Groq | Moonshot Kimi K2 Instruct |
| `gpt_oss` | Groq | OpenAI GPT OSS 120B |
| `lamma4` | Groq | Meta Llama 4 Scout |
| `qwen3` | Groq | Qwen 3 32B |

### Voice Options (TTS)

- `en-US-AriaNeural` (default)
- `en-US-JennyNeural`
- `en-GB-SoniaNeural`
- And many more Edge TTS voices

## 💡 Usage Examples

### Basic Chat
```python
import requests

response = requests.post("http://localhost:8000/chat", json={
    "query": "Hello, how are you?",
    "thread_id": "user123",
    "use_rag": False,
    "use_web": False,
    "model_name": "gpt"
})
```

### RAG-Enabled Query
```python
# First upload a document
files = {"file": open("document.pdf", "rb")}
requests.post("http://localhost:8000/upload", files=files)

# Then query with RAG enabled
response = requests.post("http://localhost:8000/chat", json={
    "query": "What does the document say about AI?",
    "thread_id": "user123",
    "use_rag": True,
    "use_web": False,
    "model_name": "gpt"
})
```

### Web Search Query
```python
response = requests.post("http://localhost:8000/chat", json={
    "query": "What's the latest news about AI?",
    "thread_id": "user123",
    "use_rag": False,
    "use_web": True,
    "model_name": "gpt"
})
```

## 🗂️ Project Structure

```
try_rag_bot/
├── app.py                 # FastAPI main application
├── RAG.py                 # LangGraph RAG implementation
├── data_ingestion.py      # Document processing and vectorization
├── utils.py               # Speech utilities (STT/TTS)
├── vectorstore/           # FAISS vector database storage
│   └── db_faiss/
├── uploads/               # Temporary audio file storage
├── outputs/               # Generated audio files
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables
└── README.md             # This file
```

## 🔒 Security Considerations

- API keys are stored in environment variables
- Temporary files are cleaned up after processing
- Input validation on all endpoints
- File upload restrictions (PDF only for documents)

## 🚀 Deployment

### Local Development
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (Optional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Junaid** - [GitHub Profile](https://github.com/junaidariie)

## 🙏 Acknowledgments

- OpenAI for GPT models
- Groq for fast inference
- LangChain community for the framework
- Tavily for web search capabilities

---

**Live Demo**: [https://junaidariie.github.io/Cortex/](https://junaidariie.github.io/Cortex/)
