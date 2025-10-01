import pytest
import unittest.mock as mock
from unittest.mock import Mock, MagicMock, patch, call
import anthropic
from typing import List, Dict, Any
import time

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator, ConversationState, RoundResult, ToolCall
from search_tools import ToolManager


class MockResponseBuilder:
    """Builder for creating complex mock Anthropic API responses"""

    @staticmethod
    def create_text_response(text: str, stop_reason: str = "end_turn"):
        """Create a simple text response"""
        mock_response = Mock()
        mock_text_block = Mock()
        mock_text_block.type = "text"
        mock_text_block.text = text
        mock_response.content = [mock_text_block]
        mock_response.stop_reason = stop_reason
        return mock_response

    @staticmethod
    def create_tool_use_response(tool_calls: List[Dict], text: str = ""):
        """Create response with tool use requests"""
        mock_response = Mock()
        content_blocks = []

        # Add text block if provided
        if text:
            text_block = Mock()
            text_block.type = "text"
            text_block.text = text
            content_blocks.append(text_block)

        # Add tool use blocks
        for i, tool_call in enumerate(tool_calls):
            tool_block = Mock()
            tool_block.type = "tool_use"
            tool_block.id = f"tool_use_{i+1}"
            tool_block.name = tool_call["name"]
            tool_block.input = tool_call["input"]
            content_blocks.append(tool_block)

        mock_response.content = content_blocks
        mock_response.stop_reason = "tool_use"
        return mock_response

    @staticmethod
    def create_sequential_responses(responses: List[Dict]):
        """Create a sequence of responses for multi-round scenarios"""
        mock_responses = []
        for response_config in responses:
            if response_config["type"] == "tool_use":
                mock_responses.append(
                    MockResponseBuilder.create_tool_use_response(
                        response_config["tool_calls"], response_config.get("text", "")
                    )
                )
            else:
                mock_responses.append(
                    MockResponseBuilder.create_text_response(
                        response_config["text"], response_config.get("stop_reason", "end_turn")
                    )
                )
        return mock_responses


class TestAIGeneratorSequentialToolCalling:
    """Comprehensive test suite for sequential tool calling functionality"""

    @pytest.fixture
    def ai_generator(self):
        """Create AIGenerator instance with mocked Anthropic client"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            generator = AIGenerator("test-key", "claude-3-sonnet-20240229")
            generator.client = mock_anthropic.return_value
            return generator

    @pytest.fixture
    def mock_tool_manager(self):
        """Create mocked tool manager with test tools"""
        tool_manager = Mock(spec=ToolManager)
        tool_manager.execute_tool.return_value = "Mock tool result"
        return tool_manager

    @pytest.fixture
    def mock_tools(self):
        """Create mock tool definitions"""
        return [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "course_name": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "get_course_outline",
                "description": "Get course structure",
                "input_schema": {
                    "type": "object",
                    "properties": {"course_title": {"type": "string"}},
                    "required": ["course_title"],
                },
            },
        ]

    # Backwards Compatibility Tests

    def test_single_tool_call_backwards_compatibility(
        self, ai_generator, mock_tool_manager, mock_tools
    ):
        """Verify existing single tool call behavior remains unchanged"""
        # Setup mock responses
        initial_response = MockResponseBuilder.create_tool_use_response(
            [{"name": "search_course_content", "input": {"query": "test query"}}]
        )
        final_response = MockResponseBuilder.create_text_response("Final answer")

        ai_generator.client.messages.create.side_effect = [initial_response, final_response]

        # Execute
        result = ai_generator.generate_response(
            query="test query", tools=mock_tools, tool_manager=mock_tool_manager
        )

        # Verify
        assert result == "Final answer"
        assert ai_generator.client.messages.create.call_count == 2
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="test query"
        )

    def test_no_tools_direct_response(self, ai_generator):
        """Test direct response when no tools are provided"""
        response = MockResponseBuilder.create_text_response("Direct answer")
        ai_generator.client.messages.create.return_value = response

        result = ai_generator.generate_response(query="Simple question")

        assert result == "Direct answer"
        assert ai_generator.client.messages.create.call_count == 1

    def test_tool_available_but_not_used(self, ai_generator, mock_tools, mock_tool_manager):
        """Test when tools are available but AI chooses not to use them"""
        response = MockResponseBuilder.create_text_response("Answer without tools")
        ai_generator.client.messages.create.return_value = response

        result = ai_generator.generate_response(
            query="General question", tools=mock_tools, tool_manager=mock_tool_manager
        )

        assert result == "Answer without tools"
        assert ai_generator.client.messages.create.call_count == 1

    # Sequential Tool Calling Core Tests

    def test_two_round_tool_calling_success(self, ai_generator, mock_tool_manager, mock_tools):
        """Test successful 2-round sequential tool calling"""
        # Setup sequential responses - each round has tool use, then next round should provide final text
        round1_response = MockResponseBuilder.create_tool_use_response(
            [{"name": "get_course_outline", "input": {"course_title": "Python Basics"}}]
        )
        # Second round provides final response with text content
        round2_response = MockResponseBuilder.create_text_response("Complete comparison result")

        ai_generator.client.messages.create.side_effect = [round1_response, round2_response]

        # Mock tool execution results
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline: Lesson 1: Intro, Lesson 2: Functions..."
        ]

        # Execute
        result = ai_generator.generate_response(
            query="Compare Python functions to other languages",
            tools=mock_tools,
            tool_manager=mock_tool_manager,
        )

        # Verify
        assert result == "Complete comparison result"
        assert ai_generator.client.messages.create.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 1

        # Verify tool execution
        mock_tool_manager.execute_tool.assert_called_with(
            "get_course_outline", course_title="Python Basics"
        )

    def test_context_preservation_between_rounds(self, ai_generator, mock_tool_manager, mock_tools):
        """Test that context is preserved correctly between tool calling rounds"""
        # Setup responses
        responses = [
            {
                "type": "tool_use",
                "tool_calls": [
                    {"name": "get_course_outline", "input": {"course_title": "Data Science"}}
                ],
            },
            {
                "type": "tool_use",
                "tool_calls": [
                    {
                        "name": "search_course_content",
                        "input": {"query": "machine learning", "course_name": "Data Science"},
                    }
                ],
            },
            {"type": "text", "text": "Based on the course outline and content search..."},
        ]

        mock_responses = MockResponseBuilder.create_sequential_responses(responses)
        ai_generator.client.messages.create.side_effect = mock_responses

        mock_tool_manager.execute_tool.side_effect = [
            "Outline: ML in lesson 3",
            "ML content: algorithms, models...",
        ]

        # Execute with conversation history
        result = ai_generator.generate_response(
            query="Tell me about machine learning in the data science course",
            conversation_history="Previous: We discussed statistics basics",
            tools=mock_tools,
            tool_manager=mock_tool_manager,
        )

        # Verify context preservation in API calls
        api_calls = ai_generator.client.messages.create.call_args_list

        # First call should include conversation history
        first_call_system = api_calls[0][1]["system"]
        assert "Previous: We discussed statistics basics" in first_call_system

        # Second call should have accumulated message history
        second_call_messages = api_calls[1][1]["messages"]
        assert len(second_call_messages) >= 2  # At least assistant + tool results

    # Termination Condition Tests

    def test_max_rounds_termination(self, ai_generator, mock_tool_manager, mock_tools):
        """Test termination after maximum rounds (2)"""
        # Setup 2 tool use responses to test max limit
        responses = [
            MockResponseBuilder.create_tool_use_response(
                [{"name": "search_course_content", "input": {"query": "test1"}}]
            ),
            MockResponseBuilder.create_tool_use_response(
                [{"name": "search_course_content", "input": {"query": "test2"}}]
            ),
        ]

        ai_generator.client.messages.create.side_effect = responses
        mock_tool_manager.execute_tool.return_value = "Tool result"

        # Execute
        result = ai_generator.generate_response(
            query="Test max rounds", tools=mock_tools, tool_manager=mock_tool_manager
        )

        # Should terminate after 2 rounds
        assert ai_generator.client.messages.create.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 2
        assert result != ""  # Should return some response

    def test_natural_termination_with_text_response(
        self, ai_generator, mock_tool_manager, mock_tools
    ):
        """Test natural termination when AI provides text response"""
        responses = [
            MockResponseBuilder.create_tool_use_response(
                [{"name": "search_course_content", "input": {"query": "test"}}]
            ),
            MockResponseBuilder.create_text_response("Final answer after tool use"),
        ]

        ai_generator.client.messages.create.side_effect = responses
        mock_tool_manager.execute_tool.return_value = "Tool result"

        result = ai_generator.generate_response(
            query="Test natural termination", tools=mock_tools, tool_manager=mock_tool_manager
        )

        assert result == "Final answer after tool use"
        assert ai_generator.client.messages.create.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 1

    def test_no_tool_manager_termination(self, ai_generator, mock_tools):
        """Test behavior when tool_manager is None"""
        response = MockResponseBuilder.create_text_response("Direct response")
        ai_generator.client.messages.create.return_value = response

        result = ai_generator.generate_response(
            query="Test without tool manager",
            tools=mock_tools,
            tool_manager=None,  # No tool manager provided
        )

        # Should use simple response path
        assert result == "Direct response"
        assert ai_generator.client.messages.create.call_count == 1

    # Error Handling Tests

    def test_anthropic_api_error_handling(self, ai_generator, mock_tool_manager, mock_tools):
        """Test handling of Anthropic API errors during sequential calls"""
        # First call succeeds, second fails
        # Create proper APIError mock
        from unittest.mock import Mock

        api_error = Mock(spec=anthropic.APIError)
        api_error.message = "Rate limit exceeded"

        ai_generator.client.messages.create.side_effect = [
            MockResponseBuilder.create_tool_use_response(
                [{"name": "search_course_content", "input": {"query": "test"}}]
            ),
            api_error,
        ]

        mock_tool_manager.execute_tool.return_value = "Tool result"

        # Should return error response gracefully
        result = ai_generator.generate_response(
            query="Test API error", tools=mock_tools, tool_manager=mock_tool_manager
        )

        # Should return an error message but not crash
        assert "error" in result.lower() or "unable" in result.lower()
        assert ai_generator.client.messages.create.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 1

    def test_tool_execution_error_handling(self, ai_generator, mock_tool_manager, mock_tools):
        """Test handling of tool execution errors"""
        response = MockResponseBuilder.create_tool_use_response(
            [{"name": "search_course_content", "input": {"query": "test"}}]
        )
        final_response = MockResponseBuilder.create_text_response("Handled error gracefully")

        ai_generator.client.messages.create.side_effect = [response, final_response]
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")

        result = ai_generator.generate_response(
            query="Test tool error", tools=mock_tools, tool_manager=mock_tool_manager
        )

        # Should handle error and provide response
        assert "Handled error gracefully" in result
        assert mock_tool_manager.execute_tool.call_count == 1

    def test_empty_tool_results_handling(self, ai_generator, mock_tool_manager, mock_tools):
        """Test handling when tool returns empty or None results"""
        response = MockResponseBuilder.create_tool_use_response(
            [{"name": "search_course_content", "input": {"query": "nonexistent"}}]
        )
        final_response = MockResponseBuilder.create_text_response("No results found")

        ai_generator.client.messages.create.side_effect = [response, final_response]
        mock_tool_manager.execute_tool.return_value = ""  # Empty result

        result = ai_generator.generate_response(
            query="Test empty results", tools=mock_tools, tool_manager=mock_tool_manager
        )

        assert result == "No results found"

    # Complex Query Scenarios

    def test_course_comparison_scenario(self, ai_generator, mock_tool_manager, mock_tools):
        """Test complex scenario: comparing two courses"""
        responses = [
            MockResponseBuilder.create_tool_use_response(
                [{"name": "get_course_outline", "input": {"course_title": "Python Basics"}}]
            ),
            MockResponseBuilder.create_text_response(
                "Based on both course outlines, here's the comparison..."
            ),
        ]

        ai_generator.client.messages.create.side_effect = responses
        mock_tool_manager.execute_tool.side_effect = ["Python: Variables, Functions, Classes..."]

        result = ai_generator.generate_response(
            query="Compare Python Basics and JavaScript Fundamentals courses",
            tools=mock_tools,
            tool_manager=mock_tool_manager,
        )

        assert "comparison" in result
        assert ai_generator.client.messages.create.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 1

    def test_drill_down_scenario(self, ai_generator, mock_tool_manager, mock_tools):
        """Test drill-down scenario: outline first, then specific content"""
        responses = [
            MockResponseBuilder.create_tool_use_response(
                [{"name": "get_course_outline", "input": {"course_title": "Machine Learning"}}]
            ),
            MockResponseBuilder.create_text_response("Neural networks are covered in lesson 5..."),
        ]

        ai_generator.client.messages.create.side_effect = responses
        mock_tool_manager.execute_tool.side_effect = [
            "Lesson 1: Intro, Lesson 2: Regression, Lesson 5: Neural Networks..."
        ]

        result = ai_generator.generate_response(
            query="Where are neural networks covered in the Machine Learning course?",
            tools=mock_tools,
            tool_manager=mock_tool_manager,
        )

        assert "lesson 5" in result.lower()
        assert ai_generator.client.messages.create.call_count == 2


class TestConversationState:
    """Test the ConversationState class"""

    def test_conversation_state_initialization(self):
        """Test ConversationState initialization"""
        state = ConversationState()
        assert state.rounds == []
        assert state.all_messages == []
        assert state.tool_call_history == []

    def test_add_round_to_state(self):
        """Test adding rounds to conversation state"""
        state = ConversationState()

        # Create mock round result
        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Test response")]

        round_result = RoundResult(
            round_number=1,
            api_response=mock_response,
            messages_exchanged=[{"role": "assistant", "content": "test"}],
        )

        state.add_round(round_result)

        assert len(state.rounds) == 1
        assert len(state.all_messages) == 1
        assert state.rounds[0].round_number == 1

    def test_get_final_response(self):
        """Test extracting final response from conversation state"""
        state = ConversationState()

        # Create mock response with text content
        mock_response = Mock()
        mock_text_block = Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "Final response text"
        mock_response.content = [mock_text_block]

        round_result = RoundResult(round_number=1, api_response=mock_response)

        state.add_round(round_result)

        final_response = state.get_final_response()
        assert final_response == "Final response text"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
