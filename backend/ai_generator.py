import anthropic
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import time


@dataclass
class ToolCall:
    """Represents a tool call execution"""
    tool_name: str
    tool_id: str
    parameters: Dict[str, Any]
    result: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class RoundResult:
    """Results from a single conversation round"""
    round_number: int
    api_response: Any  # Anthropic response object
    tool_calls_made: List[ToolCall] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    messages_exchanged: List[Dict[str, Any]] = field(default_factory=list)
    has_tool_use: bool = False
    has_text_content: bool = False
    error: Optional[str] = None


class ConversationState:
    """Manages conversation state across multiple rounds"""

    def __init__(self):
        self.rounds: List[RoundResult] = []
        self.all_messages: List[Dict[str, Any]] = []
        self.tool_call_history: List[ToolCall] = []

    def add_round(self, round_result: RoundResult):
        """Add a round result to the conversation state"""
        self.rounds.append(round_result)
        self.all_messages.extend(round_result.messages_exchanged)
        self.tool_call_history.extend(round_result.tool_calls_made)

    def get_current_context(self) -> List[Dict[str, Any]]:
        """Get the current message context for API calls"""
        return self.all_messages.copy()

    def get_final_response(self) -> str:
        """Extract the final response text from the conversation"""
        # Look for text content in the last round's response
        if not self.rounds:
            return "No response generated"

        last_round = self.rounds[-1]
        if last_round.api_response and hasattr(last_round.api_response, 'content'):
            for content_block in last_round.api_response.content:
                if hasattr(content_block, 'type') and content_block.type == 'text':
                    return content_block.text

        # Fallback: look for text in any round
        for round_result in reversed(self.rounds):
            if round_result.api_response and hasattr(round_result.api_response, 'content'):
                for content_block in round_result.api_response.content:
                    if hasattr(content_block, 'type') and content_block.type == 'text':
                        return content_block.text

        return "Unable to extract response text"


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

MULTI-ROUND TOOL USAGE:
- You can make tool calls across multiple conversation rounds (maximum 2 rounds)
- Each round is a separate API call - tools remain available in subsequent rounds
- Use this capability for complex queries requiring multiple searches or analysis steps

SEQUENTIAL REASONING PATTERNS:
- **Information gathering then analysis**: First search for content, then search related courses/lessons
- **Progressive refinement**: Search broadly first, then narrow down based on initial results
- **Cross-reference queries**: "Find topic X in course Y, then find other courses covering same topic"
- **Comparative analysis**: Search multiple sources then synthesize findings

ROUND-SPECIFIC GUIDELINES:
Round 1: Focus on primary information gathering
- Use search_course_content for main query
- Use get_course_outline for structure requests

Round 2 (if needed): Refine, cross-reference, or expand
- Search related content based on Round 1 results
- Compare across different courses/lessons
- Provide synthesis of multiple search results

TERMINATION CONDITIONS:
- Complete your response when you have sufficient information
- Don't use tools unnecessarily - if Round 1 answers the query completely, stop
- Tool failures or errors will terminate the sequence

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Use get_course_outline tool first, then provide complete course structure
- **Course content questions**: Use search_course_content tool first, then answer
- **Complex queries**: Use multiple rounds to gather comprehensive information
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_rounds: int = 2) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports sequential tool calling across multiple rounds.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool calling rounds (default 2)

        Returns:
            Generated response as string
        """

        # Handle simple cases without tools or tool manager
        if not tools or not tool_manager:
            return self._generate_simple_response(query, conversation_history)

        # Execute sequential tool calling rounds
        return self._execute_sequential_rounds(
            query, conversation_history, tools, tool_manager, max_rounds
        )

    def _generate_simple_response(self, query: str, conversation_history: Optional[str] = None) -> str:
        """Generate response without tool usage"""
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }

        response = self.client.messages.create(**api_params)
        return response.content[0].text

    def _execute_sequential_rounds(self, query: str, conversation_history: Optional[str],
                                  tools: List, tool_manager, max_rounds: int) -> str:
        """
        Execute sequential tool calling rounds for complex queries

        Args:
            query: User's question
            conversation_history: Previous conversation context
            tools: Available tools
            tool_manager: Tool execution manager
            max_rounds: Maximum rounds to execute

        Returns:
            Final response string
        """
        conversation_state = ConversationState()

        # Initialize messages with user query
        initial_messages = [{"role": "user", "content": query}]

        try:
            for round_num in range(1, max_rounds + 1):
                # Build messages for this round
                messages = self._build_messages_for_round(
                    initial_messages if round_num == 1 else [],
                    conversation_state
                )

                # Update system content for current round
                round_system_content = self._build_system_content_for_round(
                    conversation_history, round_num, conversation_state.rounds
                )

                # Execute single round
                round_result = self._execute_single_round(
                    messages, round_system_content, tools, tool_manager, round_num
                )

                # Add result to conversation state
                conversation_state.add_round(round_result)

                # Check termination conditions
                should_terminate, _ = self._should_terminate(
                    round_result, round_num, max_rounds
                )

                if should_terminate:
                    break

        except Exception as e:
            # Return best available response from completed rounds
            if conversation_state.rounds:
                return conversation_state.get_final_response()
            return f"Error generating response: {str(e)}"

        # Return final response
        return conversation_state.get_final_response()

    def _execute_single_round(self, messages: List[Dict], system_content: str,
                             tools: List, tool_manager, round_number: int) -> RoundResult:
        """Execute a single conversation round with tool calling"""

        # Prepare API parameters
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
            "tools": tools,
            "tool_choice": {"type": "auto"}
        }

        try:
            # Make API call
            response = self.client.messages.create(**api_params)

            # Initialize round result
            round_result = RoundResult(
                round_number=round_number,
                api_response=response,
                has_tool_use=response.stop_reason == "tool_use",
                has_text_content=any(
                    hasattr(block, 'type') and block.type == "text"
                    for block in response.content
                )
            )

            # Add assistant's response to messages
            assistant_message = {"role": "assistant", "content": response.content}
            round_result.messages_exchanged.append(assistant_message)

            # Handle tool execution if needed
            if response.stop_reason == "tool_use":
                self._execute_tools_for_round(round_result, tool_manager)

            return round_result

        except Exception as e:
            return RoundResult(
                round_number=round_number,
                api_response=None,
                error=str(e)
            )

    def _execute_tools_for_round(self, round_result: RoundResult, tool_manager):
        """Execute all tool calls for a round and add results to messages"""
        tool_results = []

        for content_block in round_result.api_response.content:
            if hasattr(content_block, 'type') and content_block.type == "tool_use":
                start_time = time.time()

                tool_call = ToolCall(
                    tool_name=content_block.name,
                    tool_id=content_block.id,
                    parameters=content_block.input
                )

                try:
                    # Execute the tool
                    result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )

                    tool_call.result = result
                    tool_call.execution_time = time.time() - start_time

                    # Add tool result for next API call
                    tool_result = {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": result
                    }
                    tool_results.append(tool_result)

                except Exception as e:
                    tool_call.error = str(e)
                    tool_call.execution_time = time.time() - start_time

                    # Add error result
                    tool_result = {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Tool execution error: {str(e)}"
                    }
                    tool_results.append(tool_result)

                round_result.tool_calls_made.append(tool_call)

        # Add tool results as user message if any tools were called
        if tool_results:
            tool_results_message = {"role": "user", "content": tool_results}
            round_result.messages_exchanged.append(tool_results_message)
            round_result.tool_results = tool_results

    def _should_terminate(self, round_result: RoundResult, round_num: int, max_rounds: int) -> tuple[bool, str]:
        """Determine if conversation should terminate"""

        # Check for errors
        if round_result.error:
            return True, f"api_error"

        # Check if max rounds reached
        if round_num >= max_rounds:
            return True, "max_rounds"

        # Check if Claude provided text response (natural termination)
        if round_result.has_text_content and not round_result.has_tool_use:
            return True, "natural_completion"

        # Check if no tool calls were made
        if not round_result.has_tool_use:
            return True, "no_tools"

        # Continue if tools were used and we haven't hit limits
        return False, "continue"

    def _build_messages_for_round(self, initial_messages: List[Dict],
                                 conversation_state: ConversationState) -> List[Dict]:
        """Build message list for API call in current round"""
        if conversation_state.rounds:
            # Use accumulated conversation context
            return conversation_state.get_current_context()
        else:
            # First round - use initial messages
            return initial_messages

    def _build_system_content_for_round(self, conversation_history: Optional[str],
                                       round_number: int, previous_rounds: List[RoundResult]) -> str:
        """Build system content with round-specific context"""
        content_parts = [self.SYSTEM_PROMPT]

        if conversation_history:
            content_parts.append(f"\nPrevious conversation:\n{conversation_history}")

        if round_number > 1 and previous_rounds:
            # Add context about previous tool calls in this query
            tool_summary = self._format_tool_call_summary(previous_rounds)
            if tool_summary:
                content_parts.append(f"\nPrevious tool calls in this query:\n{tool_summary}")

        if round_number == 2:
            content_parts.append(
                f"\nThis is round {round_number} of 2. Use tools to refine, "
                f"cross-reference, or expand on previous results if needed."
            )

        return "\n".join(content_parts)

    def _format_tool_call_summary(self, previous_rounds: List[RoundResult]) -> str:
        """Format a summary of previous tool calls for context"""
        summary_parts = []

        for round_result in previous_rounds:
            for tool_call in round_result.tool_calls_made:
                if tool_call.result and not tool_call.error:
                    summary_parts.append(
                        f"- Called {tool_call.tool_name} with {tool_call.parameters}"
                    )

        return "\n".join(summary_parts)

    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text