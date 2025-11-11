# Industrial Voice Assistant Agent

This project is an advanced voice assistant system designed to answer questions about industrial machinery. The system uses an **LLM (Large Language Model)-based **Agent** architecture with **Tool Calling** capabilities and can perform both document search (RAG) and database query (SQL) operations.

## 🎯 Project Features

### Main Components

- **🤖 LLM-Based Agent**: Intelligent agent with ReAct (Reasoning + Acting) architecture built using LangChain and LangGraph
- **🔧 Tool Calling**: The agent analyzes questions and automatically selects and uses appropriate tools
- **📚 RAG (Retrieval-Augmented Generation)**: Performs semantic searches in machine documents using a vector database (Qdrant)
- **💾 SQL Agent**: Retrieves data from the database by translating natural language questions into SQL queries
- **🎤 Speech-to-Text (STT)**: Offline voice recognition using the Whisper model
- **🔊 Text-to-Speech (TTS)**: Turkish voice-over with Google TTS
- **🧠 Intent Classification**: Automatically classifies user questions into RAG or SQL categories

## 🏗️ Architecture

### Agent Architecture

The project includes a **ReAct Agent** built using **LangGraph**. The agent works as follows:

1. **Think**: Analyzes the user's question
2. **Act**: Selects and uses the appropriate tool
3. **Observe**: Evaluates the tool's result
4. **Respond**: Creates the final answer and presents it to the user

### Tool Calling System

The agent can use the following tools:

- **`search_specific_machine_documents`**: Performs semantic searches on documents belonging to a specific machine
- **`query_database_for_machine_logs`**: Querys the database by translating natural language questions into SQL queries

### RAG Pipeline

1. **Document Loading**: Loads `.docx` files from the `documents/` folder
2. **Chunking**: Text is broken into semantic chunks 3. Embedding: Each chunk is converted to a vector using the `intfloat/multilingual-e5-large` model.
4. Vector Database: Stored in Qdrant and filterable by machine name.

### SQL Agent

- Analyzes the user's natural language query.
- Understands the database schema and generates the appropriate SQL query.
- Executes the query and presents the results to the user in understandable language.

## 📋 Requirements

### Python Libraries

bash
pip install langchain
pip install langchain-openai
pip install langgraph
pip install sentence-transformers
pip install qdrant-client
pip install pyodbc
pip install speechrecognition
pip install gtts
pip install pygame
pip install openai-whisper
pip install python-docx
pip install keyboard
```

### System Requirements

- Python 3.8+
- Microsoft SQL Server (ODBC Driver 17 for SQL Server)
- Microphone access
- Local LLM server (LM Studio, Jan, etc.) or OpenAI API key

### Model Files

- **Whisper Model**: The `large-v3.pt` model should be downloaded to the `get_agent/models/` folder.
- **Embedding Model**: The `intfloat/multilingual-e5-large` is downloaded automatically.

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd get_agent_log
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

*Note: If If the `requirements.txt` file is missing, manually install the packages in the "Python Libraries" section above.*

### 3. Download the Whisper Model

```bash
python download_whisper_model.py
```

This script will download the Whisper `large-v3` model to the `get_agent/models/` folder.

### 4. Prepare the RAG Database

```bash
cd rag
python rag_pipeline.py
python setup_vectordb.py
```

### 5. Configure the Configuration

Update the following settings in the `get_agent/agent_4_5_3.py` file to your environment:

```python
# LLM Settings
LOCAL_LLM_BASE_URL = "---" # Local LLM server address
LOCAL_LLM_MODEL_ID = "---" # Model ID used

# Database Settings
DB_SERVER = r''
DB_DATABASE = ''
DB_USERNAME = ''
DB_PASSWORD = ''

# RAG Settings
RAG_BASE_PATH = r""
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large'
```

## 📖 Usage

### Running the Main Application

```bash
cd get_agent
python agent_4_5_3.py
```

### Commands

- **Machine Selection**: `"select machine 5"` or `"select machine five"` - Determines the active machine
- **Status Check**: `"status"` - Shows the active machine
- **Cleanup**: `"clear"` - Reset the active machine selection
- **Help**: `"help"` - Shows the command list
- **Exit**: `"exit"` or `"close"` - Close the application Closes

### Usage Flow

1. When the application starts, it starts listening to the microphone.
2. Speak while holding down the 'V' key.
3. The system analyzes your question:
- **RAG Questions**: Technical documentation, maintenance procedures, etc. → Requests machine selection.
- **SQL Questions**: Logs, errors, remaining life, etc. → Requires active machine ID.
4. The agent generates the answer using the appropriate tool.
5. The answer is spoken.

### Sample Questions

**RAG Questions:**
- "How is this machine oiled is changed?"
- "What is the periodic maintenance table?"
- "When is machine maintenance performed?"

**SQL Questions:**
- "Show the last 5 error logs"
- "How much blade life remains for Machine 1?"
- "What was the last limit exceeded?"

## 📁 Project Structure

```
get_agent_log/
├── get_agent/
│ ├── agent_4_5_3.py # Main agent application
│ └── models/
│ └── large-v3.pt # Whisper STT model
├── rag/
│ ├── rag_pipeline.py # RAG pipeline (chunking + embedding)
│ ├── setup_vectordb.py # Qdrant database setup
│ ├── query_app.py # RAG query application
│ └── qdrant_db/ # Vector database
├── documents/ # ⚠️ PRIVATE INFORMATION (Should not be added to Git)
│ └── *.docx # Machine user manuals
├── ek/ # Old versions and trials
├── download_whisper_model.py # Whisper model download script
└── README.md # This file
```

## ⚠️ Important Notes

### Documents Folder

**The `documents/` folder contains private and confidential information.** This folder:
- **Should not be added to the Git repository** (Must be added to the `.gitignore` file)
- Contains machine user manuals and technical documentation
- These documents are processed by the RAG system and saved in the vector database.
- **Must never be shared or uploaded to public repositories**

### Security

- Database passwords and API keys should not be hardcoded in the code.
- Environment variables or a secure config file should be used in the production environment.
- A `.gitignore` file should be created and sensitive information should be added.

## 🔧 Technical Details

### LLM and Agent

- **Framework**: LangChain + LangGraph
- **Agent Type**: ReAct Agent (Reasoning + Acting)
- **Memory**: MemorySaver (conversation history is stored)
- **Model**: Native LLM (LM Studio/Jan) or OpenAI API

### RAG System

- **Embedding Model**: `intfloat/multilingual-e5-large` (1024-dimensional vectors)
- **Vector Database**: Qdrant (native)
- **Chunk Size**: 800 characters
- **Chunk Overlap**: 150 characters
- **Similarity Threshold**: 0.70

### Speech Processing

- **STT**: OpenAI Whisper `large-v3` (offline)
- **TTS**: Google Text-to-Speech (gTTS)
- **Language**: Turkish (tr-TR)

## 🤝 Contributing

1. Fork
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push your branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

## 📝 License

This project is a private project. Contact the project owner for license information.

## 👤 Contact

You can open an issue for questions or suggestions.

---

**Note**: This project was developed for use in industrial environments. Perform security and performance tests before using it in a production environment.
