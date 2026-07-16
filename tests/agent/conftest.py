"""Shared fixtures for tests/agent/.

`telegram_bot.pins` and `telegram_bot.agents` are module-level, mutable, in-memory
dicts (chat_id -> pins / WfedsAgent) with no reset hook of their own, so any test
touching `on_location`, `cmd_clear`, `cmd_pins`, `_agent`, etc. would otherwise leak
state into later tests in the same session. The autouse fixture below resets both
before AND after every test in this directory (not just telegram_bot's own tests).
"""

import pytest

import telegram_bot


@pytest.fixture(autouse=True)
def _clear_telegram_bot_globals():
    telegram_bot.pins.clear()
    telegram_bot.agents.clear()
    yield
    telegram_bot.pins.clear()
    telegram_bot.agents.clear()
