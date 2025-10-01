import pytest
import os
import sys
from unittest.mock import patch

# Add backend directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(autouse=True)
def mock_anthropic_api():
    """Automatically mock Anthropic API for all tests"""
    with patch('anthropic.Anthropic') as mock:
        yield mock

@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings"""
    return {
        "MAX_TOOL_ROUNDS": 2,
        "TOOL_TIMEOUT": 30.0,
        "ANTHROPIC_MODEL": "claude-3-sonnet-20240229"
    }