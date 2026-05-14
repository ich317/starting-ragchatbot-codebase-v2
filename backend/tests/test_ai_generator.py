"""
Tests for AIGenerator in ai_generator.py.

Bug 1 under investigation:
  config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
  Valid Claude 4 model IDs follow the pattern: claude-{model}-{major}-{minor}
  e.g. "claude-sonnet-4-6", "claude-haiku-4-5-20251001".
  "claude-sonnet-4-20250514" has 8 digits as the version number — it does not
  match any valid release and causes the Anthropic API to reject the request,
  which propagates as an unhandled exception → FastAPI 500 → "query failed".
"""

import os
import re
import pytest
from unittest.mock import MagicMock, patch, call

# conftest adds backend/ to sys.path — all imports work from here
from ai_generator import AIGenerator
from helpers import make_mock_message


FAKE_KEY = "sk-ant-test-fake"
VALID_MODEL = "claude-sonnet-4-6"


def _make_generator(mock_client, model=VALID_MODEL):
    """Construct an AIGenerator whose Anthropic client is already mocked."""
    with patch("ai_generator.anthropic.Anthropic", return_value=mock_client):
        return AIGenerator(FAKE_KEY, model)


class TestModelNameValidation:
    """Bug 1 – the model ID in config.py must be a real Anthropic model."""

    def test_model_name_matches_claude4_version_pattern(self):
        """Model ID must follow 'claude-{type}-{major}-{minor}' with a 1-2 digit minor."""
        from config import config

        # Valid examples:  claude-sonnet-4-6   claude-haiku-4-5-20251001
        # Invalid example: claude-sonnet-4-20250514  (minor = 8 digits)
        pattern = r"^claude-(opus|sonnet|haiku)-4-\d{1,2}(-\d{8})?$"
        assert re.match(pattern, config.ANTHROPIC_MODEL), (
            f"Model ID '{config.ANTHROPIC_MODEL}' does not match a valid Claude 4 "
            f"release pattern (expected e.g. 'claude-sonnet-4-6' or "
            f"'claude-haiku-4-5-20251001').  "
            f"An invalid model ID causes the Anthropic API to reject every request, "
            f"resulting in the 'query failed' error."
        )

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="requires real ANTHROPIC_API_KEY",
    )
    def test_model_name_accepted_by_live_api(self):
        """Live smoke-test: verify the configured model ID is accepted by the API."""
        import anthropic
        from config import config

        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        # Cheapest possible call — 1 token output
        response = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=1,
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert response.stop_reason in ("end_turn", "max_tokens")


class TestToolPassthrough:
    """AIGenerator must forward tool definitions to the Anthropic API."""

    def test_tools_included_in_api_call(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = make_mock_message(text="Direct answer")

        gen = _make_generator(mock_client)
        tools = [
            {
                "name": "search_course_content",
                "description": "Search course material",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            }
        ]

        gen.generate_response("What is Python?", tools=tools)

        kwargs = mock_client.messages.create.call_args.kwargs
        assert "tools" in kwargs
        assert kwargs["tools"] == tools

    def test_tool_choice_auto_when_tools_provided(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = make_mock_message(text="Direct answer")

        gen = _make_generator(mock_client)
        gen.generate_response("query", tools=[{"name": "t", "description": "d", "input_schema": {}}])

        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs.get("tool_choice") == {"type": "auto"}

    def test_no_tools_key_when_tools_not_provided(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = make_mock_message(text="Direct answer")

        gen = _make_generator(mock_client)
        gen.generate_response("query")  # no tools argument

        kwargs = mock_client.messages.create.call_args.kwargs
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs


class TestDirectResponse:
    """When Claude responds directly (no tool use), the text is returned as-is."""

    def test_direct_text_returned_when_stop_reason_end_turn(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = make_mock_message(text="42 is the answer")

        gen = _make_generator(mock_client)
        result = gen.generate_response("query")

        assert result == "42 is the answer"

    def test_only_one_api_call_for_direct_response(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = make_mock_message(text="Direct answer")

        gen = _make_generator(mock_client)
        gen.generate_response("query")

        assert mock_client.messages.create.call_count == 1


class TestToolExecution:
    """When Claude returns stop_reason=tool_use, the tool must be executed and
    a follow-up API call must be made with the tool result."""

    def test_tool_execution_triggered_on_tool_use_stop_reason(self):
        mock_client = MagicMock()
        tool_use_response = make_mock_message(
            stop_reason="tool_use",
            tool_calls=[
                {"id": "tu_001", "name": "search_course_content", "input": {"query": "Python"}}
            ],
        )
        final_response = make_mock_message(text="Here is the Python info.")
        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Lesson 1: Variables..."

        gen = _make_generator(mock_client)
        result = gen.generate_response(
            "Tell me about Python", tools=[], tool_manager=mock_tool_manager
        )

        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="Python"
        )
        assert result == "Here is the Python info."

    def test_two_api_calls_made_when_tool_is_used(self):
        mock_client = MagicMock()
        tool_use_response = make_mock_message(
            stop_reason="tool_use",
            tool_calls=[{"id": "tu_002", "name": "search_course_content", "input": {"query": "q"}}],
        )
        final_response = make_mock_message(text="Final")
        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "tool result"

        gen = _make_generator(mock_client)
        gen.generate_response("query", tools=[], tool_manager=mock_tool_manager)

        assert mock_client.messages.create.call_count == 2

    def test_tool_result_sent_in_follow_up_call(self):
        mock_client = MagicMock()
        tool_use_response = make_mock_message(
            stop_reason="tool_use",
            tool_calls=[{"id": "tu_003", "name": "search_course_content", "input": {"query": "q"}}],
        )
        final_response = make_mock_message(text="Done")
        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "chunk content"

        gen = _make_generator(mock_client)
        gen.generate_response("query", tools=[], tool_manager=mock_tool_manager)

        second_call_kwargs = mock_client.messages.create.call_args.kwargs
        messages = second_call_kwargs["messages"]
        # Last message is the user-role tool_result message
        tool_result_message = messages[-1]
        assert tool_result_message["role"] == "user"
        assert any(
            block.get("type") == "tool_result"
            for block in tool_result_message["content"]
        )

    def test_tool_result_content_matches_tool_output(self):
        mock_client = MagicMock()
        tool_use_response = make_mock_message(
            stop_reason="tool_use",
            tool_calls=[{"id": "tu_004", "name": "search_course_content", "input": {"query": "q"}}],
        )
        mock_client.messages.create.side_effect = [
            tool_use_response,
            make_mock_message(text="Done"),
        ]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "THE TOOL RESULT"

        gen = _make_generator(mock_client)
        gen.generate_response("query", tools=[], tool_manager=mock_tool_manager)

        second_call_kwargs = mock_client.messages.create.call_args.kwargs
        messages = second_call_kwargs["messages"]
        tool_result_message = messages[-1]
        result_block = next(
            b for b in tool_result_message["content"] if b.get("type") == "tool_result"
        )
        assert result_block["content"] == "THE TOOL RESULT"


class TestConversationHistory:
    """Conversation history must be appended to the system prompt."""

    def test_history_included_in_system_param(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = make_mock_message(text="answer")

        gen = _make_generator(mock_client)
        gen.generate_response("query", conversation_history="User: hi\nAssistant: hello")

        kwargs = mock_client.messages.create.call_args.kwargs
        assert "User: hi" in kwargs["system"]
        assert "Assistant: hello" in kwargs["system"]

    def test_no_history_section_when_history_is_none(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = make_mock_message(text="answer")

        gen = _make_generator(mock_client)
        gen.generate_response("query", conversation_history=None)

        kwargs = mock_client.messages.create.call_args.kwargs
        assert "Previous conversation:" not in kwargs["system"]
