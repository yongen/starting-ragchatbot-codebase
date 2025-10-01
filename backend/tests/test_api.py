import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# Add backend directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app with mocked dependencies"""
    # Import FastAPI and create test app
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional

    # Define request/response models inline
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    # Create test app
    app = FastAPI(title="Test RAG System")

    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Use the mocked RAG system
    rag_system = mock_rag_system

    # Define endpoints inline
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()

            answer, sources = rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def clear_session(session_id: str):
        try:
            rag_system.session_manager.clear_session(session_id)
            return {"message": "Session cleared successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        return {"message": "RAG System API", "status": "running"}

    return app


@pytest.fixture
def client(test_app):
    """Create test client for the app"""
    return TestClient(test_app)


@pytest.mark.integration
class TestQueryEndpoint:
    """Tests for the /api/query endpoint"""

    def test_query_with_session_id(self, client, sample_query_request):
        """Test query endpoint with provided session ID"""
        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == sample_query_request["session_id"]
        assert isinstance(data["sources"], list)

    def test_query_without_session_id(self, client):
        """Test query endpoint without session ID (should create new session)"""
        request_data = {"query": "What is Python?"}
        response = client.post("/api/query", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"  # From mock

    def test_query_with_empty_query(self, client):
        """Test query endpoint with empty query string"""
        request_data = {"query": "", "session_id": "test-123"}
        response = client.post("/api/query", json=request_data)

        # Should still return 200 (handled by RAG system)
        assert response.status_code == 200

    def test_query_missing_required_field(self, client):
        """Test query endpoint with missing required field"""
        request_data = {"session_id": "test-123"}  # Missing 'query'
        response = client.post("/api/query", json=request_data)

        # FastAPI validation should fail
        assert response.status_code == 422

    def test_query_with_invalid_json(self, client):
        """Test query endpoint with invalid JSON"""
        response = client.post(
            "/api/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_query_response_structure(self, client, sample_query_request):
        """Test that query response has correct structure"""
        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()

        # Verify all required fields are present
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data

        # Verify types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

    def test_query_rag_system_error(self, client, mock_rag_system):
        """Test query endpoint when RAG system raises error"""
        # Make the mock raise an exception
        mock_rag_system.query.side_effect = Exception("Database error")

        request_data = {"query": "test query", "session_id": "test-123"}
        response = client.post("/api/query", json=request_data)

        assert response.status_code == 500
        assert "detail" in response.json()


@pytest.mark.integration
class TestCoursesEndpoint:
    """Tests for the /api/courses endpoint"""

    def test_get_course_stats_success(self, client):
        """Test successful retrieval of course statistics"""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

    def test_get_course_stats_response_structure(self, client):
        """Test that course stats response has correct structure"""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        # Verify structure matches mock
        assert data["total_courses"] == 2
        assert len(data["course_titles"]) == 2
        assert "Test Course 1" in data["course_titles"]
        assert "Test Course 2" in data["course_titles"]

    def test_get_course_stats_error(self, client, mock_rag_system):
        """Test course stats endpoint when RAG system raises error"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Vector store error")

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert "detail" in response.json()

    def test_get_course_stats_empty_courses(self, client, mock_rag_system):
        """Test course stats with no courses loaded"""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []


@pytest.mark.integration
class TestSessionEndpoint:
    """Tests for the /api/session/{session_id} endpoint"""

    def test_clear_session_success(self, client):
        """Test successful session clearing"""
        session_id = "test-session-123"
        response = client.delete(f"/api/session/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "Session cleared successfully"

    def test_clear_session_with_special_characters(self, client):
        """Test clearing session with special characters in ID"""
        session_id = "test-session-123-abc_xyz"
        response = client.delete(f"/api/session/{session_id}")

        assert response.status_code == 200

    def test_clear_session_error(self, client, mock_rag_system):
        """Test session clearing when RAG system raises error"""
        mock_rag_system.session_manager.clear_session.side_effect = Exception("Session not found")

        session_id = "nonexistent-session"
        response = client.delete(f"/api/session/{session_id}")

        assert response.status_code == 500
        assert "detail" in response.json()


@pytest.mark.integration
class TestRootEndpoint:
    """Tests for the root / endpoint"""

    def test_root_endpoint(self, client):
        """Test root endpoint returns basic info"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data
        assert data["status"] == "running"


@pytest.mark.integration
class TestCORSHeaders:
    """Tests for CORS configuration"""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are present in responses"""
        response = client.options("/api/query")

        # Check for CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers

    def test_cors_allows_post_requests(self, client):
        """Test that POST requests are allowed via CORS"""
        response = client.options(
            "/api/query",
            headers={"Access-Control-Request-Method": "POST"}
        )

        assert response.status_code in [200, 204]


@pytest.mark.integration
class TestEndToEndScenarios:
    """End-to-end test scenarios"""

    def test_full_query_flow(self, client):
        """Test complete query flow: query -> response with sources"""
        # 1. Query without session (creates new session)
        request1 = {"query": "What is machine learning?"}
        response1 = client.post("/api/query", json=request1)

        assert response1.status_code == 200
        data1 = response1.json()
        session_id = data1["session_id"]

        # 2. Query with existing session
        request2 = {"query": "Tell me more", "session_id": session_id}
        response2 = client.post("/api/query", json=request2)

        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["session_id"] == session_id

        # 3. Clear the session
        response3 = client.delete(f"/api/session/{session_id}")
        assert response3.status_code == 200

    def test_multiple_concurrent_sessions(self, client):
        """Test handling multiple concurrent sessions"""
        # Create multiple queries with different sessions
        sessions = []
        for i in range(3):
            request = {"query": f"Query {i}", "session_id": f"session-{i}"}
            response = client.post("/api/query", json=request)

            assert response.status_code == 200
            data = response.json()
            sessions.append(data["session_id"])

        # Verify all sessions are different
        assert len(set(sessions)) == 3

    def test_check_courses_then_query(self, client):
        """Test checking available courses before querying"""
        # 1. Get course list
        response1 = client.get("/api/courses")
        assert response1.status_code == 200
        courses = response1.json()

        # 2. Query about a course
        request = {
            "query": f"Tell me about {courses['course_titles'][0]}",
        }
        response2 = client.post("/api/query", json=request)

        assert response2.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
