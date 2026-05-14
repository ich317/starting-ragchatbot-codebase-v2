from unittest.mock import MagicMock


def make_mock_message(stop_reason="end_turn", text="Test response", tool_calls=None):
    """Build a fake Anthropic Message for use in unit tests."""
    msg = MagicMock()
    msg.stop_reason = stop_reason

    if stop_reason == "end_turn":
        block = MagicMock()
        block.type = "text"
        block.text = text
        msg.content = [block]
    elif stop_reason == "tool_use":
        blocks = []
        for tc in tool_calls or []:
            block = MagicMock()
            block.type = "tool_use"
            block.id = tc.get("id", "tool_use_abc123")
            block.name = tc["name"]
            block.input = tc.get("input", {})
            blocks.append(block)
        msg.content = blocks

    return msg
