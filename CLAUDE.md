# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Set up environment variables
echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
```

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Application Access
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for course materials with the following core architecture:

### Data Flow
1. **Document Processing**: Course files → Structured parsing → Text chunking with context
2. **Vector Storage**: Chunks → Embeddings → ChromaDB semantic search index
3. **Query Processing**: User query → Claude AI → Tool-based search → Contextual response

### Core Components

**Backend Structure (`backend/`)**:
- `app.py` - FastAPI application with `/api/query` and `/api/courses` endpoints
- `rag_system.py` - Main orchestrator coordinating all components
- `ai_generator.py` - Claude API integration with tool-based search
- `document_processor.py` - Course document parsing and chunking
- `vector_store.py` - ChromaDB interface for semantic search
- `search_tools.py` - Tool definitions for AI-driven course content search
- `session_manager.py` - Conversation history management
- `models.py` - Pydantic data models (Course, Lesson, CourseChunk)
- `config.py` - Configuration settings with environment variable loading

**Frontend (`frontend/`)**:
- `index.html` - Web interface
- `script.js` - Chat functionality and API integration
- `style.css` - UI styling

### Document Format Requirements

Course documents in `docs/` must follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: Introduction
Lesson Link: [optional url]
[lesson content...]

Lesson 1: Next Topic
[lesson content...]
```

### Key Configuration (`config.py`)

- **CHUNK_SIZE**: 800 characters (text chunk size for embeddings)
- **CHUNK_OVERLAP**: 100 characters (overlap between chunks)
- **MAX_RESULTS**: 5 (search results returned)
- **MAX_HISTORY**: 2 (conversation turns remembered)
- **EMBEDDING_MODEL**: "all-MiniLM-L6-v2" (sentence transformer model)
- **ANTHROPIC_MODEL**: "claude-sonnet-4-20250514"

### AI Tool System

The system uses Claude's tool calling to perform semantic search:
- **search_course_content** tool with parameters: `query`, `course_name`, `lesson_number`
- Claude decides when to search based on query content
- Search results are synthesized into natural responses with source attribution

### Session Management

- Each conversation has a unique session_id
- Conversation history limited by MAX_HISTORY setting
- Sessions track user queries and AI responses for context

### Data Processing Pipeline

1. **Chunking Strategy**: Sentence-based with overlap to preserve context
2. **Context Enhancement**: Chunks prefixed with "Course [title] Lesson [num] content:"
3. **Hierarchical Structure**: Course → Lessons → Chunks with metadata preservation
4. **Auto-loading**: Documents from `docs/` loaded on application startup

### Dependencies

Core packages managed via `uv`:
- `anthropic` - Claude AI API client
- `chromadb` - Vector database
- `sentence-transformers` - Text embeddings
- `fastapi` + `uvicorn` - Web framework and server
- `python-dotenv` - Environment variable management

## Development Notes

- No test framework currently implemented
- ChromaDB data stored in `./chroma_db` directory
- Application auto-reloads in development mode (`--reload` flag)
- CORS enabled for all origins in development
- Static files served with no-cache headers for development