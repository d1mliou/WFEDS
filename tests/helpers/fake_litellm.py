"""Reusable fake `litellm` module for tests.

`agent.py`'s `WfedsAgent.chat()` does `import litellm` LOCALLY inside the method
body (not at module level), so `agent.litellm` is not a real, patchable attribute
at rest -- the only reliable interception point is `sys.modules` itself. Install a
fake module there before `chat()` runs its local `import litellm` (which will just
find the fake one already sitting in `sys.modules` and bind that).
"""

from __future__ import annotations

import sys
import types
from unittest.mock import Mock


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments  # JSON string, per real litellm/OpenAI shape


class _FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    """Matches exactly what `WfedsAgent.chat()` reads off `resp.choices[0].message`:
    `.tool_calls` (via `getattr(msg, "tool_calls", None)`), `.content`, and
    `.model_dump(exclude_none=True)` (appended straight into the running messages
    list -- never re-parsed by anything real, so it just needs to be a plain dict)."""

    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        d = {"role": "assistant"}
        if not exclude_none or self.content is not None:
            d["content"] = self.content
        if not exclude_none or self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        return d


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


def make_response(content=None, tool_calls=None):
    """Build a fake litellm completion response.

    `tool_calls`: None for a plain final-text response, or a list of
    `(call_id, function_name, arguments_json_str)` 3-tuples to simulate the model
    requesting one or more tool calls.
    """
    tcs = [_FakeToolCall(*tc) for tc in tool_calls] if tool_calls else None
    return _FakeResponse(_FakeMessage(content=content, tool_calls=tcs))


def install_fake_litellm(monkeypatch, responses):
    """Install a fake `litellm` module into `sys.modules` with a `.completion`
    Mock whose `side_effect` is `responses` -- a list consumed in call order by
    successive `litellm.completion(...)` calls. Include an Exception instance in
    the list to make that particular call raise instead of return (standard
    `unittest.mock` `side_effect`-list behaviour). Returns the Mock so callers can
    assert on `.call_count` / `.call_args_list`.
    """
    fake_module = types.ModuleType("litellm")
    completion_mock = Mock(side_effect=responses)
    fake_module.completion = completion_mock
    monkeypatch.setitem(sys.modules, "litellm", fake_module)
    return completion_mock
