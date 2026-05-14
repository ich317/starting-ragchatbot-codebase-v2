"""
Tests for RAGSystem.query() in rag_system.py.

Verifies:
  - query() wires tools correctly into AIGenerator
  - sources flow from ToolManager back to the caller
  - session history is read and written
  - exceptions from AIGenerator (e.g. invalid model → API error) propagate
    unhandled, which is why the frontend sees "query failed" (HTTP 500)
"""

import pytest
from unittest.mock import MagicMock, patch


def _make_rag_system(vector_store):
    """Construct a RAGSystem with a mocked AIGenerator and real tool stack."""
    from rag_system import RAGSystem
    from session_manager import SessionManager
    from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool

    mock_ai = MagicMock()
    mock_ai.generate_response.return_value = "Mocked AI response"

    system = object.__new__(RAGSystem)
    system.config = None
    system.vector_store = vector_store
    system.ai_generator = mock_ai
    system.session_manager = SessionManager(max_history=2)
    system.tool_manager = ToolManager()
    system.search_tool = CourseSearchTool(vector_store)
    system.tool_manager.register_tool(system.search_tool)
    system.outline_tool = CourseOutlineTool(vector_store)
    system.tool_manager.register_tool(system.outline_tool)

    return system, mock_ai


class TestRAGSystemQuery:

    # ── basic wiring ───────────────────────────────────────────────────────────

    def test_query_returns_tuple_of_string_and_list(self, vector_store):
        system, _ = _make_rag_system(vector_store)
        result = system.query("What is Python?")

        assert isinstance(result, tuple)
        assert len(result) == 2
        response, sources = result
        assert isinstance(response, str)
        assert isinstance(sources, list)

    def test_query_response_comes_from_ai_generator(self, vector_store):
        system, mock_ai = _make_rag_system(vector_store)
        mock_ai.generate_response.return_value = "AI says hello"

        response, _ = system.query("Hello?")

        assert response == "AI says hello"

    # ── tool definitions passed to AIGenerator ────────────────────────────────

    def test_query_passes_nonempty_tools_to_ai_generator(self, vector_store):
        system, mock_ai = _make_rag_system(vector_store)
        system.query("What is Python?")

        call_kwargs = mock_ai.generate_response.call_args.kwargs
        tools = call_kwargs.get("tools", [])
        assert len(tools) > 0, "No tool definitions were passed to generate_response"

    def test_query_passes_search_tool_definition(self, vector_store):
        system, mock_ai = _make_rag_system(vector_store)
        system.query("What is Python?")

        call_kwargs = mock_ai.generate_response.call_args.kwargs
        tool_names = [t["name"] for t in call_kwargs["tools"]]
        assert "search_course_content" in tool_names

    def test_query_passes_outline_tool_definition(self, vector_store):
        system, mock_ai = _make_rag_system(vector_store)
        system.query("What is Python?")

        call_kwargs = mock_ai.generate_response.call_args.kwargs
        tool_names = [t["name"] for t in call_kwargs["tools"]]
        assert "get_course_outline" in tool_names

    def test_query_passes_tool_manager_to_ai_generator(self, vector_store):
        system, mock_ai = _make_rag_system(vector_store)
        system.query("What is Python?")

        call_kwargs = mock_ai.generate_response.call_args.kwargs
        assert call_kwargs.get("tool_manager") is system.tool_manager

    # ── sources ───────────────────────────────────────────────────────────────

    def test_query_returns_sources_from_tool_manager(self, vector_store):
        system, _ = _make_rag_system(vector_store)
        # Inject a fake source into the search tool so get_last_sources returns it
        system.search_tool.last_sources = [{"label": "Intro Python - Lesson 1", "url": "https://x.com"}]

        _, sources = system.query("What is Python?")

        assert len(sources) > 0
        assert sources[0]["label"] == "Intro Python - Lesson 1"

    def test_query_resets_sources_after_retrieval(self, vector_store):
        system, _ = _make_rag_system(vector_store)
        system.search_tool.last_sources = [{"label": "Some Course", "url": None}]

        system.query("What is Python?")

        # Sources should be cleared after query() so next call starts fresh
        assert system.search_tool.last_sources == []

    # ── session / conversation history ────────────────────────────────────────

    def test_query_without_session_sends_no_history(self, vector_store):
        system, mock_ai = _make_rag_system(vector_store)
        system.query("What is Python?")  # no session_id

        call_kwargs = mock_ai.generate_response.call_args.kwargs
        assert call_kwargs.get("conversation_history") is None

    def test_query_with_session_sends_history(self, vector_store):
        system, mock_ai = _make_rag_system(vector_store)
        session_id = system.session_manager.create_session()
        system.session_manager.add_exchange(session_id, "prev question", "prev answer")

        system.query("Follow-up question?", session_id=session_id)

        call_kwargs = mock_ai.generate_response.call_args.kwargs
        history = call_kwargs.get("conversation_history")
        assert history is not None
        assert "prev question" in history
        assert "prev answer" in history

    def test_query_saves_exchange_to_session(self, vector_store):
        system, mock_ai = _make_rag_system(vector_store)
        mock_ai.generate_response.return_value = "Python is great"
        session_id = system.session_manager.create_session()

        system.query("What is Python?", session_id=session_id)

        history = system.session_manager.get_conversation_history(session_id)
        assert "What is Python?" in history
        assert "Python is great" in history

    # ── Bug 1: exception propagation (the actual "query failed" path) ─────────

    def test_query_propagates_exception_from_ai_generator(self, vector_store):
        """
        Bug 1 symptom – RAGSystem.query() has no try/except around
        ai_generator.generate_response().  When the Anthropic API rejects the
        invalid model ID, an exception escapes query(), reaches app.py, and
        FastAPI returns HTTP 500 → frontend shows "query failed".
        """
        system, mock_ai = _make_rag_system(vector_store)
        mock_ai.generate_response.side_effect = Exception(
            "model: claude-sonnet-4-20250514: not found"
        )

        with pytest.raises(Exception, match="not found"):
            system.query("What is Python?")

    def test_query_api_error_not_silently_swallowed(self, vector_store):
        """Confirm query() doesn't catch and hide API errors as empty strings."""
        system, mock_ai = _make_rag_system(vector_store)
        mock_ai.generate_response.side_effect = RuntimeError("API failure")

        with pytest.raises(RuntimeError):
            system.query("What is Python?")

    # ── prompt construction ───────────────────────────────────────────────────

    def test_query_wraps_user_question_in_prompt(self, vector_store):
        system, mock_ai = _make_rag_system(vector_store)
        system.query("What is a for loop?")

        call_kwargs = mock_ai.generate_response.call_args.kwargs
        query_sent = call_kwargs.get("query", "")
        assert "What is a for loop?" in query_sent
