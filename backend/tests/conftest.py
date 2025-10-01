import pytest
import os
import sys
from unittest.mock import patch, Mock, MagicMock
from typing import Dict, List, Optional

# Add backend directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def mock_anthropic_api():
    """Automatically mock Anthropic API for all tests"""
    with patch("anthropic.Anthropic") as mock:
        yield mock


@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings"""
    return {
        "MAX_TOOL_ROUNDS": 2,
        "TOOL_TIMEOUT": 30.0,
        "ANTHROPIC_MODEL": "claude-3-sonnet-20240229",
    }

# API Testing Fixtures

@pytest.fixture
def mock_rag_system():
    """Mock RAG system for API tests"""
    mock_system = MagicMock()
    mock_system.query.return_value = (
        "This is a test answer from the RAG system.",
        ["Course: Test Course | Lesson 1: Introduction"]
    )
    mock_system.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Test Course 1", "Test Course 2"]
    }
    mock_system.session_manager = MagicMock()
    mock_system.session_manager.create_session.return_value = "test-session-123"
    mock_system.session_manager.clear_session.return_value = None
    return mock_system

@pytest.fixture
def sample_query_request():
    """Sample query request data"""
    return {
        "query": "What is Python?",
        "session_id": "test-session-123"
    }

@pytest.fixture
def sample_query_response():
    """Sample query response data"""
    return {
        "answer": "Python is a high-level programming language.",
        "sources": ["Course: Python Basics | Lesson 1: Introduction"],
        "session_id": "test-session-123"
    }

@pytest.fixture
def sample_course_stats():
    """Sample course statistics data"""
    return {
        "total_courses": 3,
        "course_titles": [
            "Python Basics",
            "JavaScript Fundamentals",
            "Machine Learning"
        ]
    }

@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    mock_store = MagicMock()
    mock_store.search.return_value = [
        {
            "course_title": "Test Course",
            "lesson_number": 1,
            "lesson_title": "Introduction",
            "text": "Sample course content"
        }
    ]
    return mock_store

@pytest.fixture
def mock_document_processor():
    """Mock document processor for testing"""
    mock_processor = MagicMock()
    mock_processor.process_course_document.return_value = (
        Mock(title="Test Course", instructor="Test Instructor", lessons=[]),
        []
    )
    return mock_processor

@pytest.fixture
def sample_course_data():
    """Sample course data for testing"""
    return {
        "title": "Test Course",
        "link": "https://example.com/course",
        "instructor": "Test Instructor",
        "lessons": [
            {
                "lesson_number": 0,
                "title": "Introduction",
                "link": None,
                "content": "Welcome to the course"
            },
            {
                "lesson_number": 1,
                "title": "Getting Started",
                "link": "https://example.com/lesson1",
                "content": "Let's begin with basics"
            }
        ]
    }

@pytest.fixture
def sample_chunks():
    """Sample course chunks for testing"""
    return [
        {
            "course_title": "Test Course",
            "lesson_number": 0,
            "lesson_title": "Introduction",
            "text": "Welcome to the course"
        },
        {
            "course_title": "Test Course",
            "lesson_number": 1,
            "lesson_title": "Getting Started",
            "text": "Let's begin with basics"
        }
    ]
=======
        "ANTHROPIC_MODEL": "claude-3-sonnet-20240229",
    }
>>>>>>> e850bdf (Add code quality tooling and enforce formatting standards)
