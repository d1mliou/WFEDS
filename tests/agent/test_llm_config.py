"""Tests for scripts/agent/llm_config.py -- the LLM provider-preset resolver.

No network calls anywhere in this module (it's pure env/file/string logic), so this
is a full-coverage pure-logic test file, not a partial one.
"""

import os

import pytest

import llm_config


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Every test starts from a clean slate: `_REPO_ROOT` points at an empty
    tmp_path (no real repo `.env` file in play unless a test writes one there
    itself), and none of the provider key env vars / WFEDS_LLM_PRESET are set --
    otherwise the REAL repo `.env` and this developer's real environment would
    leak into these tests and make them non-deterministic."""
    monkeypatch.setattr(llm_config, "_REPO_ROOT", tmp_path)
    for var in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
               "WFEDS_LLM_PRESET"):
        monkeypatch.delenv(var, raising=False)


def test_get_model_unknown_preset_raises_value_error_listing_presets():
    with pytest.raises(ValueError) as excinfo:
        llm_config.get_model("not_a_real_preset")

    msg = str(excinfo.value)
    for name in llm_config.PRESETS:
        assert name in msg


def test_get_model_missing_key_env_names_the_missing_var():
    with pytest.raises(RuntimeError) as excinfo:
        llm_config.get_model("claude")

    assert "ANTHROPIC_API_KEY" in str(excinfo.value)


def test_get_model_explicit_arg_overrides_env_preset(monkeypatch):
    monkeypatch.setenv("WFEDS_LLM_PRESET", "gpt")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-claude-key")

    model, key = llm_config.get_model("claude")

    assert model == llm_config.PRESETS["claude"]["model"]
    assert key == "test-claude-key"


def test_get_model_uses_env_preset_when_no_explicit_arg(monkeypatch):
    monkeypatch.setenv("WFEDS_LLM_PRESET", "gpt")
    monkeypatch.setenv("OPENAI_API_KEY", "test-gpt-key")

    model, key = llm_config.get_model()

    assert model == llm_config.PRESETS["gpt"]["model"]
    assert key == "test-gpt-key"


def test_get_model_falls_back_to_default_preset(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    model, key = llm_config.get_model()

    assert model == llm_config.PRESETS[llm_config.DEFAULT_PRESET]["model"]
    assert key == "test-gemini-key"


def test_load_dotenv_existing_env_wins_over_dotenv_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=from_dotenv\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "from_real_env")

    model, key = llm_config.get_model("gemini-pro")

    assert key == "from_real_env"


def test_load_dotenv_reads_a_key_not_already_in_the_environment(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=from_dotenv_only\n", encoding="utf-8")

    model, key = llm_config.get_model("gemini-pro")

    assert key == "from_dotenv_only"


def test_load_dotenv_does_not_strip_quotes(tmp_path):
    """`_load_dotenv` splits on the first '=' and strips WHITESPACE only -- it
    does NOT strip quote characters from a value like `KEY="abc123"`. Confirm this
    directly against the real behaviour (not an assumption): the env var ends up
    containing the literal quote characters."""
    env_file = tmp_path / ".env"
    env_file.write_text('KEY="abc123"\n', encoding="utf-8")

    try:
        llm_config._load_dotenv()
        assert os.environ.get("KEY") == '"abc123"'
    finally:
        os.environ.pop("KEY", None)   # _load_dotenv mutates os.environ directly;
                                       # monkeypatch can't auto-undo that for us.


def test_load_dotenv_skips_blank_lines_and_comments(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n\n   \nGEMINI_API_KEY=real_value\n# trailing comment\n",
        encoding="utf-8")

    model, key = llm_config.get_model("gemini-pro")

    assert key == "real_value"


def test_load_dotenv_is_a_no_op_when_env_file_absent(tmp_path):
    """`_REPO_ROOT` points at an empty tmp_path with no `.env` at all --
    `_load_dotenv` must not raise, and `get_model` should surface the ordinary
    "missing key" error, not a file-not-found error."""
    assert not (tmp_path / ".env").exists()

    with pytest.raises(RuntimeError) as excinfo:
        llm_config.get_model("gemini-pro")

    assert "GEMINI_API_KEY" in str(excinfo.value)
