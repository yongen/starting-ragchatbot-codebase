from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from vector_store import VectorStore, SearchResults


class Tool(ABC):
    """Abstract base class for all tools"""

    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the course content",
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')",
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)",
                    },
                },
                "required": ["query"],
            },
        }

    def execute(
        self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None
    ) -> str:
        """
        Execute the search tool with given parameters.

        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter

        Returns:
            Formatted search results or error message
        """

        # Use the vector store's unified search interface
        results = self.store.search(
            query=query, course_name=course_name, lesson_number=lesson_number
        )

        # Handle errors
        if results.error:
            return results.error

        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."

        # Format and return results
        return self._format_results(results)

    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI with potential links

        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get("course_title", "unknown")
            lesson_num = meta.get("lesson_number")

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Build source for the UI, including lesson link if available
            source = course_title
            if lesson_num is not None:
                source += f" - Lesson {lesson_num}"
                # Try to get the lesson link
                lesson_link = self.store.get_lesson_link(course_title, lesson_num)
                if lesson_link:
                    # Format as clickable link data - frontend will parse this
                    source = f"{source}|{lesson_link}"

            sources.append(source)
            formatted.append(f"{header}\n{doc}")

        # Store sources for retrieval
        self.last_sources = sources

        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for retrieving complete course outlines with lesson details"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": "Get the complete outline of a course including all lessons with titles and links",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_title": {
                        "type": "string",
                        "description": "Course title or partial course name to get outline for (e.g. 'MCP', 'Introduction', 'RAG')",
                    }
                },
                "required": ["course_title"],
            },
        }

    def execute(self, course_title: str) -> str:
        """
        Execute the course outline tool.

        Args:
            course_title: Course title or partial name to search for

        Returns:
            Formatted course outline or error message
        """
        # Step 1: Resolve course name using existing vector search
        resolved_course_title = self.store._resolve_course_name(course_title)
        if not resolved_course_title:
            return f"No course found matching '{course_title}'"

        # Step 2: Get course metadata from course catalog
        try:
            results = self.store.course_catalog.get(ids=[resolved_course_title])
            if not results or not results.get("metadatas") or not results["metadatas"]:
                return f"Course metadata not found for '{resolved_course_title}'"

            metadata = results["metadatas"][0]

            # Step 3: Format the course outline
            return self._format_course_outline(metadata)

        except Exception as e:
            return f"Error retrieving course outline: {str(e)}"

    def _format_course_outline(self, metadata: Dict[str, Any]) -> str:
        """Format course metadata into a readable outline"""
        import json

        course_title = metadata.get("title", "Unknown Course")
        instructor = metadata.get("instructor", "Unknown Instructor")
        course_link = metadata.get("course_link", "No link available")
        lessons_json = metadata.get("lessons_json", "[]")

        # Parse lessons from JSON
        try:
            lessons = json.loads(lessons_json)
        except json.JSONDecodeError:
            lessons = []

        # Build formatted output
        outline_parts = []

        # Course header
        outline_parts.append(f"**Course Title:** {course_title}")
        outline_parts.append(f"**Instructor:** {instructor}")
        outline_parts.append(f"**Course Link:** {course_link}")
        outline_parts.append("")  # Empty line

        # Lessons section
        if lessons:
            outline_parts.append("**Course Lessons:**")
            for lesson in sorted(lessons, key=lambda x: x.get("lesson_number", 0)):
                lesson_num = lesson.get("lesson_number", "Unknown")
                lesson_title = lesson.get("lesson_title", "Untitled Lesson")
                lesson_link = lesson.get("lesson_link", "No link")

                if lesson_link and lesson_link != "No link":
                    outline_parts.append(f"Lesson {lesson_num}: {lesson_title} - {lesson_link}")
                else:
                    outline_parts.append(f"Lesson {lesson_num}: {lesson_title}")
        else:
            outline_parts.append("**Course Lessons:** No lessons found")

        return "\n".join(outline_parts)


class ToolManager:
    """Manages available tools for the AI"""

    def __init__(self):
        self.tools = {}

    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"

        return self.tools[tool_name].execute(**kwargs)

    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, "last_sources") and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, "last_sources"):
                tool.last_sources = []
